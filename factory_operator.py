import re
import json
import asyncio
from typing import Literal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from functools import partial
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from contract_integration import (format_phase_as_implementation_guide, inject_architect_contract_awareness)

import os

_ACTIVE_SUBGRAPH = None

# ============================================================================
# PERSONA TEMPLATE RENDERING
# ============================================================================



_CONTRACT_MANAGER = None  # Global contract manager


# Import everything from our new engine
from contract_integration import (
    initialize_contracts_for_run, ContractManager,
     inject_architect_contract_awareness
)

from factory_engine import (MainState, CoderState, AgentContext, safe_model_call,log_split, log_input, init_input_log, 
    log_terminal, log_file, log_transaction_start, log_transaction_end, 
    log_input_messages, log_output_response,
    prepare_workspace, load_task_config, load_persona_config, parse_agent_report,
    compress_session_ucp, call_model_safely
)

# ═══════════════════════════════════════════════════════════════════════════
# DEBUG INFRASTRUCTURE (MACROS & DIRECT JUMP)
# ═══════════════════════════════════════════════════════════════════════════
AGENT_NONE = ""
AGENT_ARCHITECT = "Architect"
AGENT_TECH_LEAD = "Tech Lead"
AGENT_JUNIOR_DEV = "Junior Dev"
AGENT_INTEGRATOR = "Integrator"
AGENT_SPECIALIST = "Specialist"

# 1️⃣  FLIP THIS to jump directly to any agent at startup
DEBUG_TARGET = AGENT_NONE  # e.g. AGENT_INTEGRATOR, AGENT_SPECIALIST

# 2️⃣  This replaces the agent's task context
DEBUG_TEST_MESSAGE = """
[DEBUG OVERRIDE ACTIVE]
Your only mission right now is to review the injected 
and output a summary of the exact interface requirements and test cases you see.
Do not write code yet. Just confirm you understand the contract.
"""

# 3️⃣  THE debug mission — set this to whatever you want when debugging
#     This single variable feeds ALL agent contexts during a debug run.
DEBUG_MISSION = DEBUG_TEST_MESSAGE.strip()

DEBUG_WORKSPACE_DIR = ""#"./debug_workspace"
# ═══════════════════════════════════════════════════════════════════════════



import datetime
from pathlib import Path


# ============================================================================
# SPECIALIST INTERVENTION TRACKING
# ============================================================================

class InterventionTracker:
    """
    Tracks Specialist interventions to prevent infinite loops.
    """
    
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.interventions = {} 
        self._load_log()
    
    def _load_log(self):
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    self.interventions = json.load(f)
            except:
                self.interventions = {}
    
    def _save_log(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.interventions, f, indent=2)
    
    def can_intervene(self, problem_id: str) -> tuple[bool, str]:
        if problem_id not in self.interventions:
            return True, "First attempt"
        
        history = self.interventions[problem_id]
        if history["status"] == "fixed":
            return False, f"HALT: Problem {problem_id} recurred after Specialist fix."
        if history["status"] == "unfixable":
            return False, f"HALT: Problem {problem_id} previously marked unfixable."
        if history["attempts"] >= 1:
            return False, f"HALT: Problem {problem_id} already attempted {history['attempts']} time(s)."
        
        return True, f"Retry allowed (previous attempts: {history['attempts']})"
    
    def log_intervention(self, problem_id: str, fix_type: str, status: str, changed_files: list, summary: str):
        if problem_id not in self.interventions:
            self.interventions[problem_id] = {"attempts": 0, "status": "pending", "history": []}
        
        self.interventions[problem_id]["attempts"] += 1
        self.interventions[problem_id]["status"] = status
        self.interventions[problem_id]["history"].append({
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "fix_type": fix_type,
            "status": status,
            "changed_files": changed_files,
            "summary": summary
        })
        self._save_log()

_INTERVENTION_TRACKER = None
_SPECIALIST_SCOPED_PAYLOAD = None



# ============================================================================
# CONTROL TOOLS (The Router Intercepts These)
# ============================================================================

@tool
def deploy_coder(task: str):
    """Deploys the Junior Coder/Tech Lead team."""
    # LOGGING ADDED: Logs exactly what the Architect sent
    log_split(
        f"--- [HANDOVER SEND] Architect >> Coder ---", 
        f"PAYLOAD:\n{task}"
    )
    return "DEPLOYING_CODER"

@tool
def deploy_integrator(task: str):
    """Deploys the Integrator."""
    # LOGGING ADDED
    log_split(
        f"--- [HANDOVER SEND] Architect >> Integrator ---", 
        f"PAYLOAD:\n{task}"
    )
    return "DEPLOYING_INTEGRATOR"

@tool
def deploy_specialist(task: str):
    """Deploys the Specialist."""
    # LOGGING ADDED
    log_split(
        f"--- [HANDOVER SEND] Architect >> Specialist ---", 
        f"PAYLOAD:\n{task}"
    )
    return "DEPLOYING_SPECIALIST"

@tool
def mark_phase_done(phase_id: str):
    """
    Mark a development phase as DONE.
    
    Use when subordinate reports SUCCESS with proof:
    - Tech Lead: Tests pass, code works
    - Integrator: Phase goal achieved  
    - Specialist: Fixed and working (even with workarounds)
    
    Trust subordinates. If they show proof, mark it done.
    
    Args:
        phase_id: Phase identifier (e.g., "P001", "P002")
    
    Returns:
        Confirmation message
    """
    global _CONTRACT_MANAGER
    
    # 1. Properly handle uninitialized state
    if _CONTRACT_MANAGER is None:
        return "ERROR: Contract manager not initialized"
    
    try:
        # 2. Actually mark the phase as done first
        _CONTRACT_MANAGER.update_phase_status(phase_id, "done")
        
        try:
                plan_data = _CONTRACT_MANAGER.plan_manager.plan
                phases = plan_data.get("phases", [])
                
                # FIXED: Use 'phase_id' from function args instead of undefined 'current_phase_id'
                current_idx = next((i for i, p in enumerate(phases) if p.get("phase_id") == phase_id), -1)
                
                if current_idx != -1 and current_idx + 1 < len(phases):
                    next_phase_id = phases[current_idx + 1]["phase_id"]
                    print(f"[Factory Operator] ⚙️ Phase completed. Triggering lazy JIT generation for next phase: {next_phase_id}")
                    # Dispatch background task for next phase contract
                    asyncio.create_task(_CONTRACT_MANAGER.ensure_phase_contract(next_phase_id))
        except Exception as e:
                print(f"⚠ Failed to auto-trigger next phase JIT: {e}")

        return f"✓ Phase {phase_id} marked as DONE"
        
    except ValueError as e:
        return f"ERROR: {str(e)}"
    except Exception as e:
        return f"ERROR: Failed to mark phase - {str(e)}"

@tool
def list_phase_status():
    """
    Show which phases are marked DONE.
    
    Use before planning to see what's already completed.
    
    Returns:
        Simple list showing DONE vs NOT DONE for each phase
    """
    global _CONTRACT_MANAGER
    
    if _CONTRACT_MANAGER is None:
        return "ERROR: Contract manager not initialized"
    
    try:
        plan_mgr = _CONTRACT_MANAGER.get_plan_manager()
        all_statuses = plan_mgr.get_all_phase_statuses()
        
        lines = ["Phase Status:"]
        for phase in plan_mgr.plan["phases"]:
            phase_id = phase["phase_id"]
            status = all_statuses.get(phase_id, "")
            marker = "✓ DONE" if status == "done" else "○ NOT DONE"
            lines.append(f"  {marker} - {phase_id}: {phase['phase_name']}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"ERROR: Failed to get status - {str(e)}"



# ============================================================================
# DECISION TRACKING
# ============================================================================
DECISION_COUNTER = {"count": 0}

def get_decision_id(agent_prefix: str) -> str:
    """Generate unique decision ID for tracking."""
    DECISION_COUNTER["count"] += 1
    return f"{agent_prefix}-{DECISION_COUNTER['count']:03d}"



# ============================================================================
# TOOL PERMISSION FILTERS
# ============================================================================

def get_architect_tools(tool_schemas):
    """
    Architect sees:
    1. Delegation tools (Deploy Coder/Integrator) - To give orders.
    2. Read-only tools (List/Read) - To VERIFY state before planning.
    3. Phase marking tools - To mark what's done.
    """
    # 1. Select the specific delegation tools
    delegation_names = {"deploy_coder", "deploy_integrator", "deploy_specialist", "update_ledger"}
    
    # 2. Select read-only tools (New Capability)
    read_names = {"list_files"}  # PATCHED: Architect checks file existence only, never content
    
    # 3. Phase marking tools
    phase_marking_names = {"mark_phase_done", "list_phase_status"}

    selected = []
    for t in tool_schemas:
        if t["name"] in delegation_names or t["name"] in read_names or t["name"] in phase_marking_names:
            selected.append(t)
            
    return selected

def get_tech_lead_tools(tool_schemas):
    """Tech Lead can ONLY read files and update scratchpad - NO WRITE ACCESS."""
    allowed_mcp_tools = [
        "list_files",
        "read_file", 
        "read_file_lines",
        "get_workspace_info"
    ]
    
    filtered = [
        t for t in tool_schemas 
        if t['name'] in allowed_mcp_tools
    ]
    
    # Add control flow tools
    filtered.extend([delegate_task, finish_task, update_scratchpad])
    
    return filtered

def get_junior_dev_tools(tool_schemas):
    """Junior has full code tools EXCEPT workspace initialization."""
    forbidden = ["set_workspace_context"]  # Only operator can call this
    
    return [
        t for t in tool_schemas 
        if t['name'] not in forbidden
    ]

def get_specialist_tools(tool_schemas):
    """Specialist has FULL ACCESS - emergency recovery mode."""
    return tool_schemas  # All tools available

def get_integrator_tools(tool_schemas):
    """Integrator has code tools but NOT module implementation tools."""
    allowed = [
        "write_file",
        "edit_file_replace", 
        "run_command",
        "git_checkpoint",
        "list_files",
        "read_file",
        "read_file_lines",
        "get_workspace_info"
    ]
    
    return [
        t for t in tool_schemas 
        if t['name'] in allowed
    ]

# ============================================================================
# CONTROL TOOLS
# ============================================================================


@tool
def update_ledger(ledger_content: str = ""):
    """[ARCHITECT ONLY] Update project ledger with completed milestones and technical debt."""
    return "LEDGER_UPDATED"

@tool
def delegate_task(plan: str):
    """[TECH LEAD ONLY] Send implementation plan to Junior Developer."""
    return "DELEGATION_SENT"

@tool
def finish_task(summary: str):
    """
    [TECH LEAD ONLY] Mark current task as complete. 
    GUIDANCE: If the task failed or is incomplete, use the structure:
    <failure report> <I tried to achieve>...<but  there was a problem>...
    """
    return "TASK_COMPLETE"

@tool
def update_scratchpad(notes: str):
    """[TECH LEAD ONLY] Update your memory/notes."""
    return "SCRATCHPAD_UPDATED"

# ============================================================================
# LEAF AGENT NODES (Tech Lead + Junior Dev)
# ============================================================================

async def tech_lead_node(state: CoderState, model, personas, tool_schemas):
    """
    Tech Lead Node - Fixed to orchestrate tools without dropping calls.
    """
    decision_id = get_decision_id("TL")
    log_terminal(f"[{decision_id}] Tech Lead → Analyzing...")
    
    global _CONTRACT_MANAGER
    task_description = state.get("techlead_context", {}).get("task_description") or "ask for more instruction situation unclear"
    project_root = state.get("techlead_context", {}).get("project_root", "./")
    current_notes = state.get("scratchpad", "")
    
    current_phase_info_str = ""
    if _CONTRACT_MANAGER:
        current_phase_info_json = await _CONTRACT_MANAGER.get_current_phase_info()
        if current_phase_info_json:
            current_phase_info_str = format_phase_as_implementation_guide(current_phase_info_json)

    log_split(
        f"[{decision_id}] Tech Lead Received Task", 
        f"TASK_DESCRIPTION: {task_description}"
    )

    if current_notes:
        current_notes = f"\n[SCRATCHPAD UPDATE]\n{current_notes}"
    else:
        current_notes = ""

    fixed_data = current_phase_info_str + project_root
    messages = state.get("messages", [])
    
    if current_notes:
        messages.insert(0, HumanMessage(current_notes))
        
    junior_return = state.get("junior_output")
    if junior_return:
        messages.append(junior_return)
        jr_content = str(junior_return.content).lower() if hasattr(junior_return, 'content') else str(junior_return).lower()
        is_success = "status: success" in jr_content or "junior success report" in jr_content
 
        if is_success:
            cert_status = "⚠ Certificate missing on disk."
            try:
                import os
                cert_path = os.path.join(project_root, "test_certificate.json")
                if os.path.exists(cert_path):
                    cert_status = f"✅ Test Certificate verified at: {cert_path}"
            except Exception:
                pass
 
            review_protocol = HumanMessage(content=(
                f"### 🛑 HANDOVER PROTOCOL: JUNIOR FINISHED ###\n"
                f"Result: SUCCESS reported.\n"
                f"Proof: {cert_status}\n\n"
                f"**YOUR INSTRUCTIONS:**\n"
                f"1. REVIEW the code changes using `read_file`.\n"
                f"2. VERIFY the tests by reading `test_certificate.json`.\n"
                f"3. DECIDE:\n"
                f"   - **ACCEPT**: Call `finish_task(summary)`.\n"
                f"   - **REJECT**: Call `delegate_task(instructions)` to send back to Junior.\n"
                f"NOTE: You are now READ-ONLY. Do not edit files."
            ))
            messages.append(review_protocol)
            log_terminal(f"[{decision_id}] 📋 Review Protocol Activated")
        else:
            messages.append(HumanMessage(content=(
                "### ⚠ JUNIOR FAILED ###\n"
                "The Junior Developer failed to complete the task within the iteration limit.\n"
                "Review the error log above. You must RE-PLAN and call `delegate_task` with simpler or clearer instructions."
           )))

    tech_lead_tools = get_tech_lead_tools(tool_schemas)
    response = await safe_model_call(task_description, fixed_data, messages, model.bind_tools(tech_lead_tools), "Tech Lead", state)
    
    new_session_messages = list(messages) + [response]
    
    # Pack the state so we return everything smoothly to the router
    state_updates = {"messages": new_session_messages}

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            log_terminal(f"[{decision_id}] Action: {tc['name']}")
            # We track delegation data, but NO LONGER return early!
            if tc['name'] == 'delegate_task':
                log_terminal(f"[{decision_id}] 🧠 Compressing Tech Lead memory before Junior delegation...")
                # Compress Tech Lead session before handing off (mirrors Architect pattern)
                try:
                    _tl_pre_delegation_summary = await compress_session_ucp(
                        general_goal=task_description,
                        messages=new_session_messages,
                        action_type="delegate_task",
                        affected_files=[],
                        model=model,
                        role_description="a Tech Lead's analysis session before delegating to Junior Developer",
                        recent_count=6,
                        max_words=200,
                        summary_points=[
                            "What the phase contract requires",
                            "What files were inspected and their current state",
                            "Key decisions made about implementation approach",
                            "What specific instructions are being sent to Junior",
                        ],
                    )
                    state_updates["messages"] = [
                        HumanMessage(content=f"### TECH LEAD PRE-DELEGATION SUMMARY ###\n{_tl_pre_delegation_summary}"),
                        response
                    ]
                    log_terminal(f"[{decision_id}] ✓ Tech Lead memory compressed before delegation")
                except Exception as _comp_err:
                    log_terminal(f"[{decision_id}] ⚠ Pre-delegation compression failed: {_comp_err}")
                
                junior_instructions = tc.get("args", {}).get("plan") or tc.get("args", {}).get("instructions") or tc.get("args", {}).get("task") or "Proceed with delegated task."
                state_updates["junior_context"] = {**state.get("junior_context", {}), "task_description": junior_instructions}
    else:
        _resp_text = str(response.content) if hasattr(response, 'content') else ""
        if "delegate_task" in _resp_text.lower() and ("plan" in _resp_text.lower() or "{" in _resp_text):
            import re as _re7
            _plan_match = _re7.search(r'delegate_task\W*\w*\W*(.+)', _resp_text, _re7.DOTALL | _re7.IGNORECASE)
            if _plan_match:
                _extracted_plan = _plan_match.group(1).strip().rstrip('}').strip()
                if len(_extracted_plan) > 50:
                    log_terminal(f"[{decision_id}] ⚠ Extracted text-embedded plan ({len(_extracted_plan)} chars)")
                    state_updates["junior_context"] = {**state.get("junior_context", {}), "task_description": _extracted_plan}

    needs_finish_reminder = False
    if hasattr(response, 'tool_calls') and response.tool_calls:
        has_delegation = any(tc['name'] == 'delegate_task' for tc in response.tool_calls)
        has_finish = any(tc['name'] == 'finish_task' for tc in response.tool_calls)
        if not has_delegation and not has_finish:
            raw_content = response.content or ""
            response_text = " ".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in raw_content) if isinstance(raw_content, list) else str(raw_content)
            if "complete" in response_text.lower() or "done" in response_text.lower():
                needs_finish_reminder = True

    if needs_finish_reminder:
        reminder = HumanMessage(content=(
            "[SYSTEM REMINDER]\n"
            "If all contract functions are implemented and tested, call finish_task(summary).\n"
            "This reports completion to the Architect.\n"
        ))
        state_updates["messages"].append(reminder)
        log_terminal(f"[{decision_id}] 💡 Adding finish_task reminder")
     
    return state_updates

async def compress_junior_context_ucp(modified_files: list, original_task: str, iteration_count: int, last_error: str, junior_journal: str, model) -> str:
    """
    UCP-MSP compliant version of compress_junior_context.
    
    This version ONLY compresses execution details, not task/constraints.
    The caller is responsible for re-injecting persona and task context.
    """
    # Build execution log summary (actions only, not identity)
    execution_summary = f"""FILES MODIFIED:
{', '.join(f or "" for f in modified_files) if modified_files else 'None'}

COMPILATION ATTEMPTS: {iteration_count}/5

JUNIOR EXECUTION JOURNAL:
{junior_journal}

LAST ERROR (truncated):
{last_error[:500] if last_error else 'No error details available'}"""
    
    # API-compliant payload (FIX: SystemMessage + HumanMessage)
    instructions = SystemMessage(content=(
        "You are compressing a Junior Developer's execution log for handoff. "
        "Create a BRIEF summary (max 200 words) covering:\n"
        "1. What implementation approach was attempted\n"
        "2. Which files were modified and why\n"
        "3. What specifically failed in the last build/test\n"
        "4. Recommendation for next step\n"
        "Focus ONLY on actions taken, not the agent's identity or constraints."
    ))
    
    payload = HumanMessage(content=f"""ORIGINAL TASK:
{original_task}

EXECUTION LOG:
{execution_summary}

Provide ONLY the summary text.""")
    
    response = await call_model_safely(model, [instructions, payload])
    return response.content

async def junior_dev_node(state: CoderState, model, session: ClientSession, personas, tool_schemas):
    """
    Junior Dev Node - NESTED SESSION MEMORY
    - Receives task from Tech Lead
    - Operates within Tech Lead's session scope
    - Accumulates memory during execution
    - Returns results to Tech Lead (who retains memory)
    """
    decision_id = get_decision_id("JR")
    log_terminal(f"[{decision_id}] Junior Dev → Implementing...")
    

    junior_ctx = state.get("junior_context", {})
    techlead_ctx = state.get("techlead_context", {})

    task_description = junior_ctx.get("task_description") 
    project_root = junior_ctx.get("project_root") or techlead_ctx.get("project_root") or "."
    base_dir = junior_ctx.get("base_dir") or techlead_ctx.get("base_dir") or "."

    modified_files = []
    test_iteration = 0
    MAX_TEST_ITERATIONS = 10
    build_success = False
    last_test_output = "" 
    build_passed = False  
    last_error = ""

    log_split(
        f"[{decision_id}] Junior Dev Received Task", 
        f"TASK_DESCRIPTION: {task_description}"
    )
    
    # Junior gets FULL code tools
    junior_tools = get_junior_dev_tools(tool_schemas)
    
    # Prepend Contextual Headers to the Task
    context_header = (
        f"## ENVIRONMENT CONTEXT ##\n"
        f"PROJECT_ROOT: {project_root}\n"
        f"WORKING_DIR: {base_dir}\n"
        f"-------------------------\n"
    )
    current_session_messages =[]

    fixed_task = f"{context_header}\nPay attention you received task proceed to accomplish:\n{task_description}"

    while test_iteration < MAX_TEST_ITERATIONS and not build_success:
        test_iteration += 1
        log_terminal(f"[{decision_id}] Attempt {test_iteration}/{MAX_TEST_ITERATIONS}")
        
        cleaned_messages = []
        for m in current_session_messages:
            if hasattr(m, "tool_calls") and m.tool_calls:
                call_strs = [f"Call Tool: {tc['name']} with {tc['args']}" for tc in m.tool_calls]
                new_content = f"{m.content or ''}\n" + "\n".join(call_strs)
                cleaned_messages.append(AIMessage(content=new_content))
            else:
                cleaned_messages.append(m)
        
        if cleaned_messages and isinstance(cleaned_messages[-1], AIMessage):
             cleaned_messages.append(HumanMessage(content=f"Continue implementation or fix errors: {last_error}"))

        response = await safe_model_call(fixed_task, cleaned_messages, model.bind_tools(junior_tools), "Junior Dev", state)
        
        tool_feedback = []
        if response.tool_calls:
            for tc in response.tool_calls:
                try:
                    res = await session.call_tool(tc["name"], arguments=tc["args"])
                    output = res.content[0].text if res.content else "Done"
                    tool_feedback.append(f"Tool '{tc['name']}' Output: {output}")

                    # 2. Track modified files
                    if tc['name'] in ['write_file', 'edit_file', 'edit_file_replace']:
                        affected_file = tc['args'].get('filename') or tc['args'].get('path') or 'unknown'
                        modified_files.append(affected_file)

                    file_modification_actions = ['write_file', 'edit_file_replace']
                    if ( tc['name'] in file_modification_actions ) or (test_iteration >= 7):
                        log_terminal(f"[{decision_id}] 🧠 Junior committed to file operation/ or long memory - compressing memory...")
                        
                        # Reset iteration counter on productive work
                        if test_iteration > 0:
                            log_terminal(f"[{decision_id}] ✓ Productive work detected - resetting iteration counter (was: {test_iteration})")
                            test_iteration = 0
                        
                        try:
                            compression_summary = await compress_session_ucp(
                                general_goal=fixed_task,
                                messages=current_session_messages,
                                action_type=tc['name'],
                                affected_files=[affected_file] if 'affected_file' in locals() else [],
                                model=model,
                                role_description="a Junior Developer's work session after a file modification",
                                recent_count=5,
                                max_words=150,
                                summary_points=[
                                    "What problem was being solved",
                                    "What approach was decided",
                                    "Why this specific file operation was committed",
                                    "Expected outcome",
                                ],
                            )
                            
                            # Update session with compressed history
                            current_session_messages = [
                                HumanMessage(content=f"{context_header}\nPay attention you received task proceed to accomplish:\n{task_description}"),
                                HumanMessage(content=f"### WORK SUMMARY ###\n{compression_summary}"),
                                response  # Current decision
                            ]
                            log_terminal(f"[{decision_id}] ✓ Memory compressed")
                        except Exception as compress_err:
                            log_terminal(f"[{decision_id}] ⚠ Compression failed: {compress_err}")

                    if tc['name'] == 'run_command':
                        cmd = tc['args'].get('command', '').lower()
                        cmd_succeeded = "Exit Code: 0" in output
                        
                        if 'cargo test' in cmd:
                            if cmd_succeeded:
                                build_success = True  # Tests pass — we're done
                                last_test_output = output 
                                build_passed = True
                                log_terminal(f"[{decision_id}] ✅ cargo test PASSED — marking success")
                                # PATCHED v5: Write test certificate to filesystem
                                try:
                                    import json as _json5
                                    cert_data = {
                                        "status": "PASS",
                                        "test_output": output[:2000],
                                        "timestamp": __import__('datetime').datetime.utcnow().isoformat() + "Z",
                                        "files_modified": list(set(modified_files)),
                                        "task": task_description[:500]
                                    }
                                    cert_result = await session.call_tool(
                                        "write_test_certificate",
                                        arguments={"certificate_json": _json5.dumps(cert_data)}
                                    )
                                    log_terminal(f"[{decision_id}] 📜 Test certificate written to filesystem")
                                except Exception as cert_err:
                                    log_terminal(f"[{decision_id}] ⚠ Certificate write failed: {cert_err}")
                            else:
                                last_error = output
                        elif 'cargo build' in cmd or 'cargo check' in cmd:
                            if cmd_succeeded:
                                build_passed = True   # Build OK but tests still needed
                            else:
                                last_error = output

                except Exception as e:
                    tool_feedback.append(f"Tool '{tc['name']}' Failed: {str(e)}")
                    log_terminal(f"[{decision_id}] ❌ Tool Execution Error: {e}")

        feedback_str = "\n".join(tool_feedback)
        if build_passed and not build_success:
            feedback_str += "\n\n[SYSTEM: Build compiled successfully. Now run 'cargo test' to verify correctness before reporting.]"
        current_session_messages.extend([HumanMessage(content=f"Feedback: {feedback_str}")])

    # Final reporting logic
    if not build_success:

        junior_journal = "No journal recorded."
        for msg in reversed(current_session_messages):
            if hasattr(msg, 'content') and "### WORK SUMMARY ###" in str(msg.content):
                junior_journal = str(msg.content).split("### WORK SUMMARY ###")[-1].strip()
                break

        summary = await compress_junior_context_ucp(list(set(modified_files)), task_description, test_iteration, last_error, junior_journal, model)
        
        junior_report = HumanMessage(content=(
            f"Junior Failure Report (Compressed):\n{summary}\n\n"
            f"--- [JUNIOR EXECUTION JOURNAL] ---\n{junior_journal}\n\n"
            f"Last Error Snapshot:\n{last_error}"
        ))
    
    else:
        test_proof = last_test_output[:500] if last_test_output else "No test output captured"
        junior_report = HumanMessage(content=(
            f"Junior Success Report:\n"
            f"STATUS: SUCCESS\n"
            f"Files modified: {', '.join(set(modified_files))}\n"
            f"CARGO TEST OUTPUT:\n{test_proof}"
        ))
    
    return {"junior_messages": current_session_messages , "junior_output" : junior_report}

# ============================================================================
# INTERNAL MEMORY COMPRESSION
# ============================================================================

async def summarize_current_state_ucp(messages: list, model) -> str:
    """
    UCP-MSP compliant version for Architect memory compression.
    
    [INTERNAL ONLY] Compresses Architect's memory into a Save Point.
    ONLY compresses execution history, not persona or task definition.
    """
    # Convert messages to log format (execution only)
    log_text = "\n".join([
        f"{msg.__class__.__name__}: {msg.content if hasattr(msg, 'content') else str(msg)}"
        for msg in messages
    ])
    
    # API-compliant payload (FIX: SystemMessage + HumanMessage)
    instructions = SystemMessage(content=(
        "Analyze the project history. Create a dense 'Internal Status Report' for the Architect. "
        "Summarize: 1. Final Project Goal 2. Completed Milestones 3. Current Logic State 4. Known Issues. "
        "This is for internal grounding. Be technical and objective. "
        "Focus on PROGRESS and STATE, not agent identity."
    ))
    
    payload = HumanMessage(content=f"Synthesize current state from history:\n\n{log_text}")
    
    response = await call_model_safely(model, [instructions, payload])
    return response.content


# ============================================================================
# MAIN AGENT NODES (Architect, Specialist, Integrator)
# ============================================================================

async def architect_node(state: MainState, model, personas, tool_schemas, control_tools=None):
    """
    Architect Node - Main orchestration logic
    """
    
    decision_id = get_decision_id("AR")
    log_terminal(f"[{decision_id}] Architect → Planning...")

    # --- INJECT CONTRACT AWARENESS (JIT ROADMAP UPDATE) ---

    global _CONTRACT_MANAGER
    plan_text = ""
    status_text = ""
    current_phase_id = None
    contract_path = None
    if _CONTRACT_MANAGER:
        try:
            plan_mgr = _CONTRACT_MANAGER.get_plan_manager()
            import json
            plan_text = json.dumps(plan_mgr.plan.get("phases", []), indent=2)
            all_statuses = plan_mgr.get_all_phase_statuses()
            
            lines = ["Phase Status:"]
            for phase in plan_mgr.plan.get("phases", []):
                pid = phase.get("phase_id")
                status = all_statuses.get(pid, "")
                marker = "✓ DONE" if status == "done" else "○ NOT DONE"
                lines.append(f"  {marker} - {pid}: {phase.get('phase_name')}")
            status_text = "\n".join(lines)
            
            current_phase_info = await _CONTRACT_MANAGER.get_current_phase_info()
            if current_phase_info and current_phase_info.get("phase"):
                current_phase_id = current_phase_info["phase"].get("phase_id")
                contract_path = current_phase_info.get("contract_path")
        except Exception as e:
            pass


    current_ledger = state.get("project_ledger", "NO LEDGER YET")

    trigger = "STOP: WRONG INSTRUCTION. ESCALATE FOR CLARIFICATION WITH SUPERIOR NOW."
    
    _arch_task = state.get("architect_context", {}).get("task_description", trigger)

    
    # Update the Architect persona permanently for this node execution
    enriched_task_content = inject_architect_contract_awareness(
        current_phase_id, contract_path, plan_text, status_text
    )
    # ------------------------------------------------------
    
    

    messages = list(state.get("messages", []))
            
    error_output = "CRITICAL: Program operator needs to intervene."

    # 3. Logic to crash the application
    if trigger in messages:
        # This prints the message to stderr and exits with a non-zero status code
        raise SystemExit(error_output)

    # Inject Ledger Context if available
    if current_ledger and current_ledger != "NO LEDGER YET":
        ledger_context = HumanMessage(content=f"[SYSTEM CONTEXT - Current Ledger]\n{current_ledger}")
        messages.append(ledger_context)
    
    architect_tools = get_architect_tools(tool_schemas)
    if control_tools:
        architect_tools = architect_tools + list(control_tools)
    
    # PHASE COMPLETION — deterministic, no LLM verification of code
    if len(messages) > 1:
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            content_lower = last_msg.content.lower()

            is_success = (
                ("coding team report" in content_lower or
                 "integrator report" in content_lower or
                 "specialist report" in content_lower) and
                ("status: success" in content_lower)
            )
            is_specialist_done = (
                "specialist report" in content_lower and
                ("status: success" in content_lower or "fixed" in content_lower)
            )

            if is_success or is_specialist_done:
                # Determine current phase
                phase_id = "UNKNOWN"
                expected_files = []
                if _CONTRACT_MANAGER:
                    try:
                        cpi = await _CONTRACT_MANAGER.get_current_phase_info()
                        if cpi:
                            phase_id = cpi["phase"]["phase_id"]
                            contract = cpi.get("contract", {}) or {}
                            for tc in contract.get("contracts", []):
                                fp = tc.get("module_spec", {}).get("file_path")
                                if fp:
                                    expected_files.append(fp)
                    except Exception:
                        pass

                file_list = ", ".join(expected_files) if expected_files else "(check contract)"
                reminder = HumanMessage(content=(
                    f"[SYSTEM: PHASE COMPLETION — ACTION REQUIRED]\n"
                    f"Subordinate reported SUCCESS for phase {phase_id}.\n"
                    f"Expected files: {file_list}\n"
                    f"You may call list_files to confirm file EXISTENCE only.\n"
                    f"Then call mark_phase_done('{phase_id}') and deploy next phase.\n"
                    f"DO NOT read file contents. Trust subordinate test results.\n"
                ))
                messages.append(reminder)
                log_terminal(f"[{decision_id}] ✓ Phase completion signal — injecting mark_phase_done instruction")
    
    # CALL LLM
    response = await safe_model_call(_arch_task, enriched_task_content, messages, model.bind_tools(architect_tools), "architect", state)
  

    # --- 5. POST-PROCESSING (Delegation logic) ------------------------------
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            # We start with a default state update
            state_update = {"messages": messages + [response]}
            
            for tc in response.tool_calls:
                # 1. Handle Deployments (Set up the sub-agent's context)
                if tc['name'].startswith("deploy_"):
                    target_agent = tc['name'].replace("deploy_", "").upper()
                    log_terminal(f"[{decision_id}] 🧠 Compressing internal memory for {target_agent} deployment...")
                    
                    internal_summary = await summarize_current_state_ucp(messages + [response], model)
                    updated_task_desc = tc['args'].get('task') or tc['args'].get('task_description') or \
                                    tc['args'].get('problem_description') or tc['args'].get('integration_goal') or "Execute."
                    
                    # Compress the Architect's memory
                    state_update["messages"] = [
                        HumanMessage(content=f"### INTERNAL STATE SUMMARY ###\n{internal_summary}"),
                        response
                    ]
                    
                    # Inject the task payload into the correct target context
                    if tc['name'] == 'deploy_coder':
                        state_update["architect_context"] = {**state.get("architect_context", {}), "task_description": updated_task_desc}
                    elif tc['name'] == 'deploy_integrator':
                        state_update["integrator_context"] = {**state.get("integrator_context", {}), "task_description": updated_task_desc}
                    elif tc['name'] == 'deploy_specialist':
                        state_update["specialist_context"] = {**state.get("specialist_context", {}), "task_description": updated_task_desc}

                # 2. Handle Ledger Updates
                elif tc['name'] == 'update_ledger':
                    state_update["project_ledger"] = tc['args'].get('ledger_content', '')
                    
            # We process ALL requested tools in the array before returning!
            state_update["architect_text_only_count"] = 0
            return state_update

    # PATCHED: Track text-only iterations — circuit breaker for self-loop
    MAX_ARCHITECT_TEXT_ONLY = 3
    if not (hasattr(response, "tool_calls") and response.tool_calls):
        arch_count = state.get("architect_text_only_count", 0) + 1
        if arch_count >= MAX_ARCHITECT_TEXT_ONLY:
            force_msg = HumanMessage(content=(
                "[SYSTEM: DECISION REQUIRED — LOOP DETECTED]\n"
                f"You have produced {MAX_ARCHITECT_TEXT_ONLY} consecutive responses without calling any tool.\n"
                "You MUST now call ONE of:\n"
                "  - deploy_coder(task) / deploy_integrator(task) / deploy_specialist(task)\n"
                "  - mark_phase_done(phase_id)\n"
                "  - list_phase_status()\n"
                "  - Output the word FINISHED if all phases are complete\n"
                "Do NOT analyze further. ACT NOW.\n"
            ))
            log_terminal(f"[{decision_id}] ⚠ Architect text-only cap hit ({arch_count}) — forcing decision")
            return {"messages": messages + [response, force_msg], "architect_text_only_count": arch_count}
        return {"messages": messages + [response], "architect_text_only_count": arch_count}
    return {"messages": messages + [response], "architect_text_only_count": 0}

async def main_tools_node(state: MainState, session: ClientSession):
    """
    Executes tools for Architect (read-only operations)
    Appends results to Architect's PERMANENT memory
    """
    decision_id = get_decision_id("MT")
    last_msg = state["messages"][-1]
    tool_feedback = []
    
    # PATCHED: Control tools execute locally, MCP tools go to server
    CONTROL_TOOL_MAP = {
        "mark_phase_done": lambda args: mark_phase_done.invoke(args),
        "list_phase_status": lambda args: list_phase_status.invoke({}),
        "deploy_coder": lambda args: deploy_coder.invoke(args),
        "deploy_integrator": lambda args: deploy_integrator.invoke(args),
        "deploy_specialist": lambda args: deploy_specialist.invoke(args),
        "update_ledger": lambda args: update_ledger.invoke(args)
    }

    if hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            log_terminal(f"[{decision_id}] Tool: {tc['name']}")
            try:
                if tc["name"] in CONTROL_TOOL_MAP:
                    # Execute locally — these are LangChain @tool, not MCP
                    output = str(CONTROL_TOOL_MAP[tc["name"]](tc.get("args", {})))
                else:
                    # Execute via MCP server
                    res = await session.call_tool(tc["name"], arguments=tc["args"])
                    output = res.content[0].text if res.content else "Done"
                
                # File log: Full output
                log_split(f"[{decision_id}] Tool Output: {tc['name']}", output)
                
                # Console: Summary
                log_terminal(f"[{decision_id}] Result: {'✓' if 'Error' not in output else '✗'}")
                
                tool_feedback.append(f"Tool '{tc['name']}' Output: {output}")
            except Exception as e:
                log_terminal(f"[{decision_id}] Result: ✗ FAILED - {str(e)}")
                tool_feedback.append(f"Tool '{tc['name']}' Failed: {str(e)}")

    feedback_str = "\n".join(tool_feedback)
    
    # PERMANENT MEMORY: Append to Architect's history
    current_messages = state.get("messages", [])
    new_messages = list(current_messages) + [HumanMessage(content=feedback_str)]
    
    return {"messages": new_messages}

# ============================================================================
# LEAF WORKFLOW ROUTING (Tech Lead <-> Junior)
# ============================================================================

async def leaf_tools_node(state: CoderState, session: ClientSession):
    """
    Executes tools for Tech Lead
    Returns results to Tech Lead's SESSION memory
    
    PATCHED: Enforces tool whitelist at execution time.
    Tech Lead can only execute read-only MCP tools here.
    Write/execute tools are BLOCKED even if the LLM requests them.
    """
    decision_id = get_decision_id("LT")
    last_msg = state["messages"][-1]
    tool_results = []
    
    # FIX: Move current_messages to top (was referenced before definition)
    current_messages = state.get("messages", [])
    
    # PATCHED: Hard whitelist — must match get_tech_lead_tools() MCP subset
    TECH_LEAD_ALLOWED_MCP = {"list_files", "read_file", "read_file_lines", "get_workspace_info"}
    
    # CONTROL_TOOL_MAP: Tech Lead control tools execute locally
    # (mirrors main_tools_node pattern for Architect's deploy_* tools)
    CONTROL_TOOL_MAP = {
        "delegate_task": lambda args: delegate_task.invoke(args),
        "finish_task": lambda args: finish_task.invoke(args),
        "update_scratchpad": lambda args: update_scratchpad.invoke(args),
    }
    
    if hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            # 1. Handle control flow tools locally (delegate, finish, scratchpad)
            if tc["name"] in CONTROL_TOOL_MAP:
                log_terminal(f"[{decision_id}] Control: {tc['name']}")
                try:
                    output = str(CONTROL_TOOL_MAP[tc["name"]](tc.get("args", {})))
                    log_split(f"[{decision_id}] Control Tool Output: {tc['name']}", output)
                    tool_results.append(f"[{tc['name']}]: {output}")
                except Exception as e:
                    tool_results.append(f"[{tc['name']}] Error: {str(e)}")
                
                # Special: update_scratchpad also updates scratchpad state
                if tc["name"] == "update_scratchpad":
                    new_messages = list(current_messages) + [HumanMessage(content="\n".join(tool_results))]
                    return {"messages": new_messages, "scratchpad": tc["args"].get("notes", "")}
                continue
            
            # 2. Block tools not in Tech Lead's allowed MCP set
            if tc["name"] not in TECH_LEAD_ALLOWED_MCP:
                log_terminal(f"[{decision_id}] ⛔ BLOCKED: {tc['name']} (not in Tech Lead whitelist)")
                # PATCHED v6: Context-aware block message
                _jr_succeeded = any(
                    "junior success report" in str(getattr(m, 'content', '')).lower()
                    or ("status: success" in str(getattr(m, 'content', '')).lower()
                        and "junior" in str(getattr(m, 'content', '')).lower())
                    for m in current_messages[-5:] if hasattr(m, 'content')
                )
                if _jr_succeeded:
                    tool_results.append(
                        f"[{tc['name']}]: ❌ PERMISSION DENIED — Tech Lead is READ-ONLY.\n"
                        f"Junior already reported SUCCESS with passing tests.\n"
                        f"REVIEW_THEN_DECIDE:\n"
                        f"  → If tests passed and code is acceptable: call finish_task(summary) to vouch for SUCCESS\n"
                        f"  → If you found a critical flaw: call delegate_task(plan) with specific fix for Junior\n"
                        f"  You CANNOT fix code yourself. Choose one of the above NOW."
                    )
                else:
                    tool_results.append(
                        f"[{tc['name']}]: ❌ PERMISSION DENIED — Tech Lead is READ-ONLY.\n"
                        f"To modify code: call delegate_task(plan) with a precise fix plan for Junior Dev."
                    )
                continue
            
            log_terminal(f"[{decision_id}] Leaf Tool: {tc['name']}")
            res = await session.call_tool(tc["name"], arguments=tc["args"])
            output = res.content[0].text if res.content else "Done"
            tool_results.append(f"[{tc['name']}]: {output}")
    
    # SESSION MEMORY: Append tool results to Tech Lead's session
    # (current_messages already defined at top of function)
    
    # PATCHED v6: Track blocked writes → escalating decision pressure
    _blocked_write_count = sum(1 for r in tool_results if "PERMISSION DENIED" in r)
    if _blocked_write_count > 0:
        _total_blocks = sum(
            1 for m in current_messages
            if hasattr(m, 'content') and "PERMISSION DENIED" in str(m.content)
        ) + _blocked_write_count
        
        if _total_blocks >= 2:
            tool_results.append(
                "\n[SYSTEM: WRITE BLOCKED TWICE — DECISION REQUIRED]\n"
                "You have attempted to modify files multiple times. "
                "Tech Lead is READ-ONLY by design — this will never succeed.\n"
                "You MUST now call ONE of:\n"
                "  → finish_task(summary) — if tests passed, vouch for success to Architect\n"
                "  → delegate_task(plan) — if code needs fixing, send fix plan to Junior\n"
                "Do NOT attempt to write/edit files again."
            )
            log_terminal(f"[{decision_id}] 🚫 Write blocked {_total_blocks}x — forcing decision")
    
    new_messages = list(current_messages) + [HumanMessage(content="\n".join(tool_results))]
    
    return {"messages": new_messages}


MAX_TECH_LEAD_TEXT_ONLY = 3

async def leaf_nudge_node(state: CoderState, **kwargs):
    """
    Nudge node: increments text_only_count and injects a reminder message.
    Routes back to tech_lead for another attempt.
    """
    current_count = state.get("text_only_count", 0) + 1
    nudge = HumanMessage(content=(
        "[SYSTEM: ACTION REQUIRED]\n"
        "You responded without calling any tool. You MUST use one of:\n"
        "  - delegate_task(plan) — to send work to Junior Dev\n"
        "  - finish_task(summary) — to report completion to Architect\n"
        "  - list_files / read_file / get_workspace_info — to gather information\n"
        "  - update_scratchpad(notes) — to update your working memory\n"
        f"Text-only responses remaining before forced exit: {MAX_TECH_LEAD_TEXT_ONLY - current_count}\n"
    ))
    current_messages = list(state.get("messages", []))
    current_messages.append(nudge)
    return {"messages": current_messages, "text_only_count": current_count}


MAX_TECH_LEAD_TEXT_ONLY = 3


def leaf_router(state: CoderState):
    """
    Routes Tech Lead decisions to Junior, Tools, or Finish.
    Some models (Gemini) emit tool calls as text like "call:delegate_task{plan:...}"
    instead of structured tool_calls. We detect and route these correctly.
    """
    last_msg = state["messages"][-1]
    
    # Tool calls: Precedence ladder (no looping)
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        # Extract all requested tool names into a list
        tool_names = [tc["name"] for tc in last_msg.tool_calls]
        
        # Priority 1: Delegate task (takes precedence over everything)
        if "delegate_task" in tool_names:
            return "junior_dev"
        
        # Priority 2: Finish task (takes precedence over scratchpad)
        elif "finish_task" in tool_names:
            return "leaf_finish"
        
        # Priority 3: Update scratchpad or any other tools
        else:
            return "leaf_tools"
    
    # Fallback logic for text-embedded tool calls
    msg_content = ""
    if hasattr(last_msg, "content") and last_msg.content:
        msg_content = str(last_msg.content) if not isinstance(last_msg.content, str) else last_msg.content

    if msg_content:
        content_lower = msg_content.lower()
        if "delegate_task" in content_lower and ("plan" in content_lower or "{" in msg_content):
            log_terminal(f"[LEAF-ROUTER] ⚠ Detected text-embedded delegate_task — routing to junior_dev")
            return "junior_dev"
        if "finish_task" in content_lower and ("summary" in content_lower or "{" in msg_content):
            log_terminal(f"[LEAF-ROUTER] ⚠ Detected text-embedded finish_task — routing to leaf_finish")
            return "leaf_finish"
    
    # No tool call: check text-only counter
    text_only_count = state.get("text_only_count", 0)
    if text_only_count >= MAX_TECH_LEAD_TEXT_ONLY:
        log_terminal(f"[LEAF-ROUTER] Tech Lead hit {MAX_TECH_LEAD_TEXT_ONLY} text-only responses — forcing exit")
        return "leaf_finish"
    
    # Under limit: nudge back to tech_lead via leaf_tools (will inject nudge message)
    log_terminal(f"[LEAF-ROUTER] Tech Lead text-only response ({text_only_count + 1}/{MAX_TECH_LEAD_TEXT_ONLY}) — nudging")
    return "leaf_nudge"



def main_router(state: MainState):
    """
    The Smart Router (Post-Execution Pattern).
    Evaluates what tools the Architect called AFTER they have run.
    """
    messages = state["messages"]
    
    # 1. Find the last AI message that initiated the tool calls
    # We search backwards because the very last messages are now ToolMessages from main_tools_node
    last_ai_msg = None
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            last_ai_msg = msg
            break
            
    # Fallback: No tool calls found in recent history
    if not last_ai_msg:
        last_msg = messages[-1]
        if hasattr(last_msg, "content") and "FINISHED" in str(last_msg.content):
            return "__end__"
        return "agent"

    # 2. Extract the names of all tools that were requested (and are now finished)
    tool_names = [tc["name"] for tc in last_ai_msg.tool_calls]
    
    # 3. Route to sub-agents if a deployment was in the list
    if "deploy_coder" in tool_names:
        return "enter_leaf"
    elif "deploy_integrator" in tool_names:
        return "enter_integrator"
    elif "deploy_specialist" in tool_names:
        return "enter_expert"
        
    # 4. Default: Return to Architect (e.g., if they only called mark_phase_done or list_files)
    return "agent"

# ============================================================================
# FINAL STATUS REPORT
# ============================================================================

async def print_final_status(session: ClientSession, workspace_paths: dict):
    """Print final build status and file locations (TERMINAL ONLY)."""
    log_terminal("\n" + "="*70)
    log_terminal("FACTORY RUN COMPLETE - FINAL STATUS")
    log_terminal("="*70)
    
    try:
        # Check project structure
        files_result = await session.call_tool("list_files", arguments={"path": "."})
        files_output = files_result.content[0].text if files_result.content else ""
        
        log_terminal("\n📁 PROJECT ROOT FILES:")
        log_terminal(files_output)
        
        # Check for binary
        binary_check = await session.call_tool(
            "run_command", 
            arguments={"command": "ls -lh target/release/ 2>/dev/null || echo 'No release binary found'"}
        )
        binary_output = binary_check.content[0].text if binary_check.content else ""
        
        log_terminal("\n🔨 BUILD ARTIFACTS:")
        log_terminal(binary_output)
        
        # Final verdict
        if "Exit Code: 0" in binary_output and "release" in binary_output:
            log_terminal("\n✅ SUCCESS: Binary built successfully")
            log_terminal(f"📍 Location: {workspace_paths['project_root']}/target/release/")
        elif os.path.exists(os.path.join(workspace_paths['project_root'], "src")):
            log_terminal("\n⚠️  PARTIAL: Source code present, build incomplete")
        else:
            log_terminal("\n❌ FAILURE: No artifacts generated")
        
    except Exception as e:
        log_terminal(f"\n❌ Status check failed: {e}")
    
    log_terminal("="*70)
    log_terminal(f"📂 Workspace: {workspace_paths['base_dir']}")
    log_terminal(f"📄 Full log: {workspace_paths['run_dir']}/input_log.txt")
    log_terminal("="*70 + "\n")

# ============================================================================
# WRAPPER FUNCTIONS - SESSION MEMORY MANAGERS
# ============================================================================

async def run_leaf_wrapper(state: MainState, leaf_wf, model=None):
    """
    Wrapper for Tech Lead + Junior Dev workflow
    
    CRITICAL MEMORY ARCHITECTURE:
    1. Starts with EMPTY messages (fresh session)
    2. Tech Lead accumulates memory DURING session
    3. Returns ONLY SUMMARY to Architect
    4. Tech Lead's full session memory DISCARDED
    """
    log_terminal("--- [Factory] Deploying Coding Team ---")
    
    # EXTRACT PAYLOAD FROM TOOL ARGS
    _arch_ctx = state.get("architect_context", {})
    task_description = _arch_ctx.get("task_description", "No task description provided")
    _project_root = _arch_ctx.get("project_root", "")
    _base_dir = _arch_ctx.get("base_dir", "")
    
    # SESSION INITIALIZATION: Fresh session for Tech Lead
    # Messages start EMPTY - tech_lead_node will initialize with SystemMessage + task
    initial_leaf_state = {
        "messages": [],  # ✓ CRITICAL: Empty = fresh session (no parent memory)
        "junior_messages": [],
        "techlead_context": {
            "agent_id": "tech_lead",
            "task_description": task_description,
            "task_context": {},
            "project_root": _project_root,
            "base_dir": _base_dir,
        },
        "junior_context": {
            "agent_id": "junior_dev",
            "task_description": task_description,
            "task_context": {},
            "project_root": _project_root,
            "base_dir": _base_dir,
        },
        "scratchpad": "",  # ✓ Fresh scratchpad per session
        "iterations": 0,
        "tech_lead_spec": "",
        "junior_output": "",
        "text_only_count": 0,
        "temp_dir": state.get("temp_dir", ""),
        "run_dir": state.get("run_dir", ""),
    }
    
    # Invoke the sub-graph
    final_leaf_state = await leaf_wf.ainvoke(initial_leaf_state)
    
    # MEMORY EXTRACTION: Get only the SUMMARY for Architect
    # The full Tech Lead session memory is DISCARDED (ephemeral)
    final_messages = final_leaf_state.get('messages', [])
    
 
    content = ""
    if final_messages:
        raw_report = final_messages[-1]
        
        # Try to extract from finish_task tool_call args first
        if hasattr(raw_report, 'tool_calls') and raw_report.tool_calls:
            for tc in raw_report.tool_calls:
                if tc.get('name') == 'finish_task':
                    args = tc.get('args', {})
                    if isinstance(args, str):
                        try:
                            import json as _json
                            args = _json.loads(args)
                        except:
                            pass
                    content = args.get('summary', '') if isinstance(args, dict) else str(args)
                    log_terminal(f"[Leaf Wrapper] ✓ Extracted finish_task summary ({len(content)} chars)")
                    break
        
        # Fallback to content if no finish_task found
        if not content:
            content = raw_report.content if hasattr(raw_report, 'content') else str(raw_report)
    
    if not content:
        content = "Empty response: The coding team yielded control without messages."

    cert_proof = None
    junior_ran_this_session = any(
        "junior success report" in str(getattr(m, 'content', '')).lower()
        or "junior failure report" in str(getattr(m, 'content', '')).lower()
        for m in final_messages
        if hasattr(m, 'content')
    )
    try:
        cert_path = os.path.join(state.get("architect_context", {}).get("project_root", ""), "test_certificate.json")
        if os.path.exists(cert_path):
            if junior_ran_this_session:
                with open(cert_path, 'r') as _cf:
                    cert_data = json.loads(_cf.read())
                if cert_data.get("status") == "PASS":
                    cert_proof = cert_data
                    log_terminal(f"[Leaf Wrapper] 📜 Found test certificate: PASS (Junior confirmed this session)")
                    # Clean up certificate after reading
                    os.remove(cert_path)
            else:
                log_terminal(f"[Leaf Wrapper] ⚠ Found test_certificate.json but Junior did NOT run this session — ignoring stale certificate")
    except Exception as cert_err:
        log_terminal(f"[Leaf Wrapper] ⚠ Certificate read failed: {cert_err}")

    # Attempt structured parse
    parsed = parse_agent_report(content)

    if parsed["is_json"] and parsed["is_success"]:
        normalized = (
            f"[Coding Team Report]\n"
            f"STATUS: SUCCESS\n"
            f"SUMMARY: {parsed['content']}\n"
            f"PROOF: {parsed['proof']}\n"
        )
    elif cert_proof is not None:
        # PATCHED v5: Certificate overrides — Junior proved tests pass on disk
        test_snippet = cert_proof.get("test_output", "")[:500]
        files_mod = ", ".join(cert_proof.get("files_modified", []))
        normalized = (
            f"[Coding Team Report]\n"
            f"STATUS: SUCCESS\n"
            f"SUMMARY: Tests passed (certificate recovered from filesystem)\n"
            f"FILES: {files_mod}\n"
            f"PROOF (cargo test output):\n{test_snippet}\n"
        )
    elif parsed["is_json"] and not parsed["is_success"]:
        normalized = (
            f"[Coding Team Report]\n"
            f"STATUS: FAILURE\n"
            f"PROBLEM: {parsed['content']}\n"
            f"GOAL: {parsed['goal']}\n"
        )
    else:
        # Unparseable — Tech Lead exited without structured report
        # Summarize what was actually done in the session for the Architect
        truncated = str(content)[:500]
        _leaf_session_summary = ""
        try:
            _leaf_session_summary = await compress_session_ucp(
                general_goal=task_description,
                messages=final_messages,
                action_type="session_complete",
                affected_files=[],
                model=model,
                role_description="a Tech Lead yielding control back to the Architect",
                recent_count=8,
                max_words=250,
                summary_points=[
                    "What coding task was assigned",
                    "What actions were taken (delegation to Junior, file reads, reviews)",
                    "What was the outcome (code written, tests run, results)",
                    "Current state of the work (what is done, what is not)",
                    "Any blockers or issues for the Architect to know about",
                ],
            )
        except Exception:
            _leaf_session_summary = truncated
        normalized = (
            f"[Coding Team Report]\n"
            f"STATUS: INCOMPLETE\n"
            f"PROBLEM: Tech Lead session ended without structured report.\n"
            f"SESSION WORK DONE:\n{_leaf_session_summary}\n"
            f"RAW CONTEXT (truncated): {truncated}\n"
            f"RECOMMENDATION: Redeploy coding team with same task.\n"
        )

    report_entry = HumanMessage(content=normalized)
    return {"messages": state.get("messages", []) + [report_entry]}

async def run_integrator_wrapper(state: MainState, model, session: ClientSession, personas, tool_schemas):
    """
    Wrapper for Integrator workflow
    
    CRITICAL MEMORY ARCHITECTURE:
    1. Starts with EMPTY session (no messages from Architect)
    2. Integrator accumulates memory DURING session (multi-turn if needed)
    3. Returns ONLY SUMMARY to Architect
    4. Integrator's full session memory DISCARDED
    """
    log_terminal("--- [Factory] Deploying Integrator ---")
    
    decision_id = get_decision_id("IN")
    
    task_description = state.get("integrator_context", {}).get("task_description") 
    
    log_split(
        f"--- [HANDOVER RECEIVE] Integrator << Architect ---",
        f"EXTRACTED PAYLOAD:\n{task_description}"
    )
    
    # SESSION INITIALIZATION: Fresh session for Integrator
    session_messages = []
    
    # Integrator gets code tools but NOT module implementation tools
    integrator_tools = get_integrator_tools(tool_schemas)
    
    global _CONTRACT_MANAGER
    current_phase_info_str = ""
    if _CONTRACT_MANAGER:
        current_phase_info_json = await _CONTRACT_MANAGER.get_current_phase_info()
        if current_phase_info_json:
            current_phase_info_str = format_phase_as_implementation_guide(current_phase_info_json)
            
    project_root = state.get("integrator_context", {}).get("project_root", "./")
    fixed_data = current_phase_info_str + project_root
    
    # ITERATIVE SESSION: Allow Integrator to work until task completion
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Execute Integrator logic
        response = await safe_model_call(
            human_input=task_description, 
            fixed_message=fixed_data, 
            messages=session_messages, 
            model=model.bind_tools(integrator_tools), 
            agent_name="Integrator", 
            agent_id=state.get("agent_id"), 
            personas_config=personas
        )
      
        # Check for completion signal
        if not response.tool_calls or "INTEGRATION_COMPLETE" in str(response.content):
            break
        
        # Execute tools and accumulate feedback (with memory compression)
        tool_feedback = []
        _integrator_wrote_file = False
        _integrator_ran_build = False
        _affected_files_this_iter = []
        for tc in response.tool_calls:
            log_terminal(f"[{decision_id}] Action: {tc['name']}")
            try:
                res = await session.call_tool(tc["name"], arguments=tc["args"])
                output = res.content[0].text if res.content else "Done"
                log_split(f"[{decision_id}] Tool Output: {tc['name']}", output)
                log_terminal(f"[{decision_id}] Result: {'✓' if 'Error' not in output else '✗'}")
                tool_feedback.append(f"[{tc['name']}]: {output}")
                
                # Track file modifications
                if tc['name'] in ('write_file', 'edit_file_replace'):
                    _integrator_wrote_file = True
                    _af = tc['args'].get('filename') or tc['args'].get('path') or 'unknown'
                    _affected_files_this_iter.append(_af)
                
                # Track build/test commands (these produce large outputs)
                if tc['name'] == 'run_command':
                    _cmd = tc['args'].get('command', '').lower()
                    if any(kw in _cmd for kw in ['cargo build', 'cargo test', 'cargo check', 'make', 'npm run', 'pytest', 'go build', 'go test']):
                        _integrator_ran_build = True
                        _affected_files_this_iter.append(f"cmd:{_cmd[:60]}")
                    
            except Exception as e:
                log_terminal(f"[{decision_id}] Result: ✗ FAILED")
                tool_feedback.append(f"[{tc['name']}] Error: {str(e)}")
        
        # MEMORY COMPRESSION: Compress after file writes or build commands
        _should_compress = (_integrator_wrote_file or _integrator_ran_build) and len(session_messages) > 4
        if _should_compress:
            log_terminal(f"[{decision_id}] 🧠 Integrator committed {'file write' if _integrator_wrote_file else 'build'} — compressing memory...")
            try:
                _int_compression = await compress_session_ucp(
                    general_goal=task_description,
                    messages=session_messages + [response],
                    action_type="file_write" if _integrator_wrote_file else "build_command",
                    affected_files=_affected_files_this_iter,
                    model=model,
                    role_description="an Integrator's work session after a file or build operation",
                    recent_count=6,
                    max_words=200,
                    summary_points=[
                        "What integration task was being performed",
                        "Which files were modified or what build was run",
                        "Result of the operation (success/failure, key output)",
                        "Remaining integration steps or blockers",
                    ],
                )
                # Compressed session: keep system prompt + task + summary + current
                session_messages = [
                    HumanMessage(content=task_description),  # Original task
                    HumanMessage(content=f"### INTEGRATOR WORK SUMMARY ###\n{_int_compression}"),
                    response,
                    HumanMessage(content="\n".join(tool_feedback))
                ]
                log_terminal(f"[{decision_id}] ✓ Integrator memory compressed")
            except Exception as _comp_err:
                log_terminal(f"[{decision_id}] ⚠ Compression failed: {_comp_err}")
                feedback_msg = HumanMessage(content="\n".join(tool_feedback))
                session_messages = list(session_messages) + [response, feedback_msg]
        else:
            # Normal accumulation
            feedback_msg = HumanMessage(content="\n".join(tool_feedback))
            session_messages = list(session_messages) + [response, feedback_msg]
    
    # Extract final summary from session (session memory will be discarded)
    raw_summary = response.content if response.content else "Integration work completed"

    # SUMMARIZE what was done across the whole session for the Architect
    _integrator_session_summary = ""
    try:
        _integrator_session_summary = await compress_session_ucp(
            general_goal=task_description,
            messages=session_messages,
            action_type="session_complete",
            affected_files=[],
            model=model,
            role_description="an Integrator yielding control back to the Architect",
            recent_count=8,
            max_words=250,
            summary_points=[
                "What integration goal was assigned",
                "What actions were taken (files written, commands run)",
                "What was the outcome of each major action",
                "Current state of the integration (what works, what does not)",
                "Any remaining work or blockers for the Architect to know about",
            ],
        )
    except Exception as _sum_err:
        log_terminal(f"[{decision_id}] ⚠ Integrator session summarization failed: {_sum_err}")
        _integrator_session_summary = str(raw_summary)[:800]

    normalized_report = (
        f"[Integrator Report]\n"
        f"ITERATIONS: {iteration}/{max_iterations}\n"
        f"STATUS: {'SUCCESS' if 'INTEGRATION_COMPLETE' in str(raw_summary) or iteration < max_iterations else 'INCOMPLETE'}\n"
        f"SUMMARY: {str(raw_summary)[:800]}\n"
        f"SESSION WORK DONE:\n{_integrator_session_summary}\n"
    )

    # Return ONLY summary to Architect (Integrator's session memory discarded)
    architect_messages = state.get("messages", [])
    new_architect_messages = list(architect_messages) + [HumanMessage(content=normalized_report)]
    
    return {"messages": new_architect_messages}


async def run_specialist_wrapper(state: MainState, model, session: ClientSession, personas, tool_schemas):
    """
    Wrapper for Specialist workflow
    
    CRITICAL MEMORY ARCHITECTURE:
    1. Starts with EMPTY session (no messages from Architect)
    2. Specialist accumulates memory DURING session (multi-turn recovery)
    3. Returns ONLY SUMMARY to Architect
    4. Specialist's full session memory DISCARDED
    """
    log_terminal("--- [Factory] Deploying Specialist ---")
    
    decision_id = get_decision_id("SP")
    
    # EXTRACT task from Specialist's own context (PATCHED: was reading architect_context)
    task_description = state.get("specialist_context", {}).get("task_description") \
        or state.get("architect_context", {}).get("task_description", "Emergency recovery needed")
    
    log_split(
        f"--- [HANDOVER RECEIVE] Specialist << Architect ---",
        f"EXTRACTED PAYLOAD:\n{task_description}"
    )
    
    # SESSION INITIALIZATION: Fresh session for Specialist
    session_messages = []
    
    # Specialist gets FULL ACCESS
    specialist_tools = get_specialist_tools(tool_schemas)
    
    # ITERATIVE RECOVERY: Allow Specialist to work until recovery complete
    max_iterations = 10  # Higher than Integrator (emergency recovery may need more attempts)
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Execute Specialist logic
        # --- COMBINED ARCHITECT & TECH LEAD CONTEXT ---
        global _CONTRACT_MANAGER
        _plan_txt = ""
        _status_txt = ""
        _contract_path = None
        _tech_phase_guide = ""
        
        if _CONTRACT_MANAGER:
            try:
                # 1. Architect Knowledge (Plan & Status)
                _pm = _CONTRACT_MANAGER.get_plan_manager()
                if _pm:
                    import json
                    _plan_txt = json.dumps(_pm.plan.get("phases", []), indent=2)
                    _status_txt = json.dumps(_pm.get_all_phase_statuses(), indent=2)
                
                # 2. Tech Lead Knowledge (Current Phase Implementation Guide)
                _cpi = await _CONTRACT_MANAGER.get_current_phase_info()
                if _cpi:
                    from contract_integration import format_phase_as_implementation_guide
                    _tech_phase_guide = format_phase_as_implementation_guide(_cpi)
                    
                    # Locate physical contract file if possible
                    _pid = _cpi.get("phase_id")
                    if _pid:
                        _cdir = _CONTRACT_MANAGER.base_dir / ".ai" / "contracts" / "phases"
                        _ppath = _cdir / f"phase_{_pid}.json"
                        if _ppath.exists():
                            _contract_path = str(_ppath)
            except Exception as _ctx_err:
                log_terminal(f"⚠ Context Sync Warning: {_ctx_err}")

        combined_context = f"--- ARCHITECT VIEW (Project Progress & Specs) ---\n"
        combined_context += f"CURRENT PLAN STATUS:\n{_status_txt}\n\nPHASES ROADMAP:\n{_plan_txt}\n"
        if _contract_path:
            combined_context += f"CURRENT PHASE CONTRACT AVAILABLE AT: {_contract_path}\n"
            
        combined_context += f"\n--- TECH LEAD VIEW (Current Phase & Development) ---\n"
        combined_context += _tech_phase_guide
        
        combined_context += "\n\n--- SPECIALIST DIRECTIVE ---\nYou are the highest technical authority. Use the combined context above to solve this extraordinary problem."

        # Execute Specialist logic (Updated Signature)
        response = await safe_model_call(
            human_input=task_description,
            fixed_message=combined_context,
            messages=session_messages,
            model=model.bind_tools(specialist_tools),
            agent_name="Specialist",
            agent_id=state.get("agent_id"),
            personas_config=personas
        )
        
      
        # Check for completion signal
        if not response.tool_calls or "RECOVERY_COMPLETE" in str(response.content):
            break
        
        # Execute tools and accumulate feedback (with memory compression)
        tool_feedback = []
        _specialist_wrote_file = False
        _affected_files_this_iter = []
        for tc in response.tool_calls:
            log_terminal(f"[{decision_id}] Action: {tc['name']}")
            try:
                res = await session.call_tool(tc["name"], arguments=tc["args"])
                output = res.content[0].text if res.content else "Done"
                log_split(f"[{decision_id}] Tool Output: {tc['name']}", output)
                log_terminal(f"[{decision_id}] Result: {'✓' if 'Error' not in output else '✗'}")
                tool_feedback.append(f"[{tc['name']}]: {output}")
                
                # Track file modifications for compression trigger
                if tc['name'] in ('write_file', 'edit_file_replace'):
                    _specialist_wrote_file = True
                    _af = tc['args'].get('filename') or tc['args'].get('path') or 'unknown'
                    _affected_files_this_iter.append(_af)
                    
            except Exception as e:
                log_terminal(f"[{decision_id}] Result: ✗ FAILED")
                tool_feedback.append(f"[{tc['name']}] Error: {str(e)}")
        
        # MEMORY COMPRESSION: Compress after file writes (mirrors Junior behavior)
        if _specialist_wrote_file and len(session_messages) > 4:
            log_terminal(f"[{decision_id}] 🧠 Specialist committed file operation — compressing memory...")
            
            # Reset iteration counter on productive work
            if iteration > 1:
                log_terminal(f"[{decision_id}] ✓ Productive work detected — resetting iteration counter (was: {iteration})")
                iteration = 1
            
            try:
                _sp_compression = await compress_session_ucp(
                    general_goal=task_description,
                    messages=session_messages + [response],
                    action_type="file_write",
                    affected_files=_affected_files_this_iter,
                    model=model,
                    role_description="a Specialist's emergency recovery session after a file modification",
                    recent_count=6,
                    max_words=200,
                    summary_points=[
                        "What problem was being diagnosed",
                        "Root cause identified (if any)",
                        "What fix was applied in this file operation",
                        "Remaining issues or next steps",
                        "Any CAPABILITY GAPS discovered (missing tools, permissions, dependencies)",
                    ],
                )
                # Compressed session: keep task + summary + current response
                session_messages = [
                    HumanMessage(content=task_description),  # Original task
                    HumanMessage(content=f"### SPECIALIST WORK SUMMARY ###\n{_sp_compression}"),
                    response,
                    HumanMessage(content="\n".join(tool_feedback))
                ]
                log_terminal(f"[{decision_id}] ✓ Specialist memory compressed")
            except Exception as _comp_err:
                log_terminal(f"[{decision_id}] ⚠ Compression failed: {_comp_err}")
                # Fallback: normal accumulation
                feedback_msg = HumanMessage(content="\n".join(tool_feedback))
                session_messages = list(session_messages) + [response, feedback_msg]
        else:
            # Normal accumulation (no file write this iteration)
            feedback_msg = HumanMessage(content="\n".join(tool_feedback))
            session_messages = list(session_messages) + [response, feedback_msg]
    
    # Extract final summary from session (session memory will be discarded)
    summary = response.content if response.content else "Recovery operations completed"
    summary_str = str(summary).lower()

    # PATCHED: Detect Specialist failure → HALT with human notice
    specialist_failed = (
        iteration >= max_iterations or
        "unfixable" in summary_str or
        "cannot fix" in summary_str or
        "unable to resolve" in summary_str or
        "halting" in summary_str
    )

    if specialist_failed:
        halt_msg = (
            "\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  🛑  FACTORY HALTED — SPECIALIST COULD NOT RESOLVE PROBLEM     ║\n"
            "╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Iterations used: {iteration}/{max_iterations}\n"
            f"║  Last Specialist output (truncated):\n"
            f"║  {str(summary)[:300]}\n"
            "║\n"
            "║  The Specialist (highest authority) was unable to fix the\n"
            "║  problem. Automated recovery is exhausted.\n"
            "║\n"
            "║  HUMAN INTERVENTION REQUIRED:\n"
            "║  1. Check the detailed log in the run directory\n"
            "║  2. Review Specialist notes (specialist_notes.md)\n"
            "║  3. Fix the root cause manually, then re-run\n"
            "╚══════════════════════════════════════════════════════════════════╝\n"
        )
        log_terminal(halt_msg)
        log_file(halt_msg)

        # Write halt marker for external tooling
        halt_marker_path = os.path.join(state.get("run_dir", "/tmp"), "HALT_SPECIALIST_FAILURE.txt")
        try:
            with open(halt_marker_path, "w") as f:
                f.write(f"Halted at: {datetime.datetime.utcnow().isoformat()}Z\n")
                f.write(f"Reason: Specialist exhausted or reported unfixable\n")
                f.write(f"Summary: {str(summary)[:1000]}\n")
        except Exception:
            pass

        # Hard exit
        import sys as _sys
        _sys.exit(99)

    # COMPRESS full Specialist session before yielding to Architect
    # (mirrors Integrator pattern)
    _specialist_session_summary = ""
    try:
        _specialist_session_summary = await compress_session_ucp(
            general_goal=task_description,
            messages=session_messages,
            action_type="session_complete",
            affected_files=[],
            model=model,
            role_description="a Specialist yielding control back to the Architect",
            recent_count=8,
            max_words=250,
            summary_points=[
                "What problem was assigned to the Specialist",
                "Root cause diagnosis (what was actually wrong)",
                "What fixes were applied (files changed, commands run)",
                "Current state after intervention (what works now, what does not)",
                "Any remaining issues or recommendations for the Architect",
            ],
        )
        log_terminal(f"[{decision_id}] ✓ Specialist session compressed for Architect handoff")
    except Exception as _sum_err:
        log_terminal(f"[{decision_id}] ⚠ Specialist session summarization failed: {_sum_err}")
        _specialist_session_summary = str(summary)[:800]

    # STRUCTURED RETURN: Capability-gap detection + report normalization
    # Mirrors Junior/TechLead pattern: "seems lacking these capabilities: ..."
    _sp_summary_str = str(summary)
    
    # Detect capability gaps from Specialist's session
    _capability_gaps = []
    for msg in session_messages:
        _msg_text = str(msg.content) if hasattr(msg, 'content') else str(msg)
        # Look for common gap indicators in tool outputs and specialist reasoning
        if any(kw in _msg_text.lower() for kw in [
            'permission denied', 'not found', 'no such file', 'command not found',
            'missing dependency', 'missing module', 'import error', 'modulenotfounderror',
            'not installed', 'unavailable', 'unsupported', 'cannot access'
        ]):
            # Extract the relevant line (truncated)
            for line in _msg_text.split('\n'):
                if any(kw in line.lower() for kw in [
                    'permission denied', 'not found', 'command not found',
                    'missing', 'import error', 'not installed', 'unavailable'
                ]):
                    _capability_gaps.append(line.strip()[:200])
    
    # Deduplicate
    _capability_gaps = list(dict.fromkeys(_capability_gaps))[:5]
    
    # Build structured report
    _gap_section = ""
    if _capability_gaps:
        _gap_lines = "\n".join(f"  - {g}" for g in _capability_gaps)
        _gap_section = (
            f"\nCAPABILITY GAPS DETECTED:\n"
            f"Seems lacking these capabilities:\n{_gap_lines}\n"
        )
    
    normalized_report = (
        f"[Specialist Report]\n"
        f"ITERATIONS: {iteration}/{max_iterations}\n"
        f"STATUS: {'SUCCESS' if not specialist_failed else 'FAILED'}\n"
        f"SUMMARY: {_sp_summary_str[:800]}\n"
        f"SESSION WORK DONE:\n{_specialist_session_summary}\n"
        f"{_gap_section}"
    )
    
    # Return ONLY summary to Architect (Specialist's session memory discarded)
    architect_messages = state.get("messages", [])
    new_architect_messages = list(architect_messages) + [HumanMessage(content=normalized_report)]
    
    return {"messages": new_architect_messages}

# ============================================================================
# MAIN FACTORY EXECUTION
# ============================================================================

async def run_factory():
    import os
    from model_resolver import get_model, ModelRole
    
    persona_config = load_persona_config()
    mission_text, config_dict = load_task_config()

    run_id = config_dict.get("mission_id", "default_run")
    workspace_paths = prepare_workspace(config_dict, run_id)

    if DEBUG_WORKSPACE_DIR:
            print(f"\n[⚙️ DEBUG] OVERRIDING WORKSPACE PATHS TO: {DEBUG_WORKSPACE_DIR}")
            workspace_paths["base_dir"] = DEBUG_WORKSPACE_DIR
            workspace_paths["run_dir"] = DEBUG_WORKSPACE_DIR
            # If your project code is also inside the debug dir, uncomment the next line:
            # workspace_paths["project_root"] = DEBUG_WORKSPACE_DIR

    # Initialize Intervention Tracker
    global _INTERVENTION_TRACKER
    intervention_log = Path(workspace_paths["run_dir"]) / "specialist_interventions.json"
    _INTERVENTION_TRACKER = InterventionTracker(str(intervention_log))
    log_terminal("[Factory] Intervention tracker initialized")
    project_root = workspace_paths["project_root"]
     
    init_input_log(workspace_paths["run_dir"])
    
    print("\n" + "="*70)
    print("TOY CODE FACTORY - AGENTIC BUILD SYSTEM")
    log_terminal("="*70)
    print(f"Mission: {config_dict['goal']}")
    print(f"Language: {config_dict['language']}")
    print(f"Target: {config_dict.get('target', config_dict.get('language', 'rust'))}")
    print(f"Workspace: {workspace_paths['base_dir']}")
    log_terminal("="*70 + "\n")
    
    global _CONTRACT_MANAGER
    log_terminal("[Factory] Initializing contract system...")
    _CONTRACT_MANAGER = await initialize_contracts_for_run(
        workspace_base=workspace_paths["base_dir"],
        task_json_path="task.json",
        language=config_dict.get("language", config_dict.get("target", "rust")),
        generate_all=False  # Only generate current phase
    )
    current_phase_info = await _CONTRACT_MANAGER.get_current_phase_info()
    
    smart_model = get_model(ModelRole.SMART, temperature=0)
    worker_model = get_model(ModelRole.WORKER, temperature=0)
    better_model = smart_model
    
    # Use absolute path to ensure server is found regardless of CWD
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infrastructure.py")
    server_params = StdioServerParameters(command="python", args=[server_script])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            log_terminal("[INIT] Configuring MCP workspace...")
            
            init_result = await session.call_tool(
                "set_workspace_context",
                arguments={
                    "project_root": workspace_paths["project_root"],
                    "temp_dir": workspace_paths["temp_dir"],
                    "run_dir": workspace_paths["run_dir"]
                }
            )
            
            log_terminal(init_result.content[0].text if init_result.content else "Workspace initialized")
            log_terminal("")
            
            mcp_tools = await session.list_tools()
            control_tools = [deploy_coder, deploy_integrator, deploy_specialist, 
                           mark_phase_done, list_phase_status]
            tool_schemas = [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in mcp_tools.tools]
            all_architect_tools = list(mcp_tools.tools) + control_tools
            

            # ── DEBUG DIRECT JUMP ────────────────────────────────────
            if DEBUG_TARGET:
                import sys as _dbg_sys
                print(f"\n<<-- DEBUG-DELIMITER -->>")
                print(f"[⚙️ DEBUG] DIRECT JUMP → {DEBUG_TARGET}")
                print(f"<<-- DEBUG-DELIMITER -->>\n")

                # Build a minimal MainState so the wrapper has what it needs
                _dbg_state = {
                    "messages": [],
                    "architect_context": {
                        "agent_id": "debug",
                        "task_description": DEBUG_MISSION,
                        "task_context": {},
                        "project_root": project_root,
                        "base_dir": workspace_paths["base_dir"],
                    },
                    "specialist_context": {
                        "agent_id": "specialist",
                        "task_description": DEBUG_MISSION,
                        "task_context": {},
                        "project_root": project_root,
                        "base_dir": workspace_paths["base_dir"],
                    },
                    "integrator_context": {
                        "agent_id": "integrator",
                        "task_description": DEBUG_MISSION,
                        "task_context": {},
                        "project_root": project_root,
                        "base_dir": workspace_paths["base_dir"],
                    },
                    "project_ledger": "",
                    "temp_dir": workspace_paths["temp_dir"],
                    "run_dir": workspace_paths["run_dir"],
                    "architect_text_only_count": 0,
                }

                if DEBUG_TARGET == AGENT_INTEGRATOR:
                    result = await run_integrator_wrapper(_dbg_state, model=worker_model, session=session, personas=persona_config, tool_schemas=tool_schemas)
                elif DEBUG_TARGET == AGENT_SPECIALIST:
                    result = await run_specialist_wrapper(_dbg_state, model=better_model, session=session, personas=persona_config, tool_schemas=tool_schemas)
                elif DEBUG_TARGET == AGENT_ARCHITECT:
                    result = await architect_node(_dbg_state, model=better_model, personas=persona_config, tool_schemas=tool_schemas, control_tools=control_tools)
                elif DEBUG_TARGET == AGENT_TECH_LEAD:
                    _dbg_leaf_state = {
                        "messages": [],
                        "junior_messages": [],
                        "techlead_context": {
                            "agent_id": "tech_lead",
                            "task_description": DEBUG_MISSION,
                            "task_context": {},
                            "project_root": project_root,
                            "base_dir": workspace_paths["base_dir"],
                        },
                        "junior_context": {
                            "agent_id": "junior_dev",
                            "task_description": DEBUG_MISSION,
                            "task_context": {},
                            "project_root": project_root,
                            "base_dir": workspace_paths["base_dir"],
                        },
                        "scratchpad": "",
                        "iterations": 0,
                        "tech_lead_spec": "",
                        "junior_output": "",
                        "text_only_count": 0,
                        "temp_dir": workspace_paths["temp_dir"],
                        "run_dir": workspace_paths["run_dir"],
                    }
                    result = await tech_lead_node(_dbg_leaf_state, model=better_model, personas=persona_config, tool_schemas=tool_schemas)
                elif DEBUG_TARGET == AGENT_JUNIOR_DEV:
                    _dbg_coder_state = {
                        "messages": [],
                        "junior_messages": [],
                        "techlead_context": {
                            "agent_id": "tech_lead",
                            "task_description": DEBUG_MISSION,
                            "task_context": {},
                            "project_root": project_root,
                            "base_dir": workspace_paths["base_dir"],
                        },
                        "junior_context": {
                            "agent_id": "junior_dev",
                            "task_description": DEBUG_MISSION,
                            "task_context": {},
                            "project_root": project_root,
                            "base_dir": workspace_paths["base_dir"],
                        },
                        "scratchpad": "",
                        "iterations": 0,
                        "tech_lead_spec": "",
                        "junior_output": "",
                        "text_only_count": 0,
                        "temp_dir": workspace_paths["temp_dir"],
                        "run_dir": workspace_paths["run_dir"],
                    }
                    result = await junior_dev_node(_dbg_coder_state, model=worker_model, session=session, personas=persona_config, tool_schemas=tool_schemas)
                else:
                    print(f"[⚙️ DEBUG] Unknown DEBUG_TARGET: {DEBUG_TARGET}")
                    _dbg_sys.exit(1)

                print(f"\n<<-- DEBUG-DELIMITER -->>")
                print(f"[⚙️ DEBUG] {DEBUG_TARGET} completed naturally.")
                print(f"[⚙️ DEBUG] Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                if isinstance(result, dict):
                    for k, v in result.items():
                        preview = str(v)[:300] if v else "None"
                        print(f"[⚙️ DEBUG]   {k}: {preview}")
                print(f"<<-- DEBUG-DELIMITER -->>\n")
                _dbg_sys.exit(0)
            # ── END DEBUG DIRECT JUMP ────────────────────────────────

            # Build Leaf Workflow (Tech Lead + Junior)
            leaf_wf = StateGraph(CoderState)
            leaf_wf.add_node("tech_lead", partial(tech_lead_node, model=better_model, personas=persona_config, tool_schemas=tool_schemas))
            leaf_wf.add_node("junior_dev", partial(junior_dev_node, model=worker_model, session=session, personas=persona_config, tool_schemas=tool_schemas))
            leaf_wf.add_node("leaf_tools", partial(leaf_tools_node, session=session))
            leaf_wf.add_node("leaf_nudge", leaf_nudge_node)

            def tl_to_tools_router(state: CoderState):
                # Always route to leaf_tools first if tools were called
                last_msg = state["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    return "leaf_tools"
                if "TASK_COMPLETE" in str(getattr(last_msg, "content", "")):
                    return "leaf_finish"
                return "leaf_nudge"

            def tools_to_junior_router(state: CoderState):
                # Evaluate if delegation occurred from the AI message preceding the tool execution
                messages = state.get("messages", [])
                for i in range(len(messages)-1, -1, -1):
                    msg = messages[i]
                    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls'):
                        if any(tc['name'] == 'delegate_task' for tc in msg.tool_calls):
                            return "junior_dev"
                        break
                return "tech_lead"

            leaf_wf.set_entry_point("tech_lead")
            
            # 1. Tech Lead always executes tools immediately 
            leaf_wf.add_conditional_edges("tech_lead", tl_to_tools_router, {
                "leaf_tools": "leaf_tools",
                "leaf_finish": END,
                "leaf_nudge": "leaf_nudge",
                "tech_lead": "tech_lead"
            })
            
            # 2. Tools evaluate if we jump to Junior Dev or go back to Tech Lead
            leaf_wf.add_conditional_edges("leaf_tools", tools_to_junior_router, {
                "junior_dev": "junior_dev",
                "tech_lead": "tech_lead"
            })
            
            # 3. Junior dev closes the loop back to Tech Lead
            leaf_wf.add_edge("junior_dev", "tech_lead")
            leaf_wf.add_edge("leaf_nudge", "tech_lead")
                
            leaf_app = leaf_wf.compile()

            # Build Main Workflow (Architect + Subordinates)
            workflow = StateGraph(MainState)
            workflow.add_node("agent", partial(architect_node, model=better_model, personas=persona_config, tool_schemas=tool_schemas, control_tools=control_tools))
            workflow.add_node("main_tools", partial(main_tools_node, session=session))
            
            # Add wrapper nodes with proper session memory management
            workflow.add_node("coding_specialist", partial(run_leaf_wrapper, leaf_wf=leaf_app, model=better_model))
            workflow.add_node("integrator", partial(run_integrator_wrapper, model=worker_model, session=session, personas=persona_config, tool_schemas=tool_schemas))
            workflow.add_node("specialist", partial(run_specialist_wrapper, model=better_model, session=session, personas=persona_config, tool_schemas=tool_schemas))
            
            workflow.set_entry_point("agent")
            #workflow.set_entry_point("specialist")
            
            # 1. Simple check to see if the Architect called tools or is done
            def agent_to_tools_router(state: MainState):
                last_msg = state["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    return "main_tools"
                if "FINISHED" in str(getattr(last_msg, "content", "")):
                    return "__end__"
                return "agent" # Loops back to Architect if they need a text-only nudge

            # 2. Architect goes to Tools
            workflow.add_conditional_edges("agent", agent_to_tools_router, {
                "main_tools": "main_tools",
                "__end__": END,
                "agent": "agent"
            })
            
            # 3. Tools go to the Router we built in Step 3
            workflow.add_conditional_edges("main_tools", main_router, {
                "enter_leaf": "coding_specialist", 
                "enter_integrator": "integrator",
                "enter_expert": "specialist",
                "agent": "agent",
                "__end__": END
            })
            
            # 4. Sub-agents always return control to the Architect when finished
            workflow.add_edge("coding_specialist", "agent")
            workflow.add_edge("integrator", "agent")
            workflow.add_edge("specialist", "agent")
            
            project_root = workspace_paths["project_root"]
            project_ledger = "NO LEDGER YET"

            # Run workflow
            _agent_ctx_template = {
                "agent_id": "",
                "task_description": mission_text,
                #"task_description": "TESTING: This is an infrastructure test. Just tell me how you are doing. Save it as a text file and verify that you have done so. Yield control; system command chain integrity check.",
                "task_context": {},
                "project_root": project_root,
                "base_dir": workspace_paths["base_dir"],
            }
            inputs = {
                "messages": [], 
                "architect_context": {**_agent_ctx_template, "agent_id": "architect"},
                "specialist_context": {**_agent_ctx_template, "agent_id": "specialist"},
                "integrator_context": {**_agent_ctx_template, "agent_id": "integrator"},
                "project_ledger": project_ledger,
                "temp_dir": workspace_paths["temp_dir"],
                "run_dir": workspace_paths["run_dir"],
                "architect_text_only_count": 0,
            }
            
            # Define path for the SQLite checkpoint database within the run directory
            checkpoint_db_path = Path(workspace_paths["run_dir"]) / "checkpoints.sqlite"


            # Initialize persistent checkpointer (survives process restarts)
            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db_path)) as checkpointer:
                # Check if restoring from checkpoint
                if checkpoint_db_path.exists():
                    log_terminal("⚠️  RESTORATION MODE: Resuming from checkpoint")
                    inputs["messages"].append(
                        HumanMessage(content="[SYSTEM ALERT] Workflow restored from checkpoint. LangGraph has recovered all previous state.")
                    )
                
                async for chunk in workflow.compile(checkpointer=checkpointer).astream(inputs, config={"configurable": {"thread_id": "1"}}):
                    if "agent" in chunk:
                        last_msg = chunk["agent"]["messages"][-1].content
                        parsed = parse_agent_report(last_msg)
                        if parsed["is_json"]:
                            status = "✅" if parsed["is_success"] else "❌"
                            log_terminal(f"--- [Report Received] {status} ---")
                
                # Print final status (indentation moved inside to be safe, though not strictly required)
                await print_final_status(session, workspace_paths)
            

if __name__ == "__main__":
    asyncio.run(run_factory())