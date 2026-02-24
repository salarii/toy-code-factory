import os
import json
import shutil 
import datetime
import asyncio
import subprocess
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict, Literal, Optional, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SYSTEM - TWO PATH ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

# Constants
LOG_DELIM = "\n<<-- DIALOG-DELIMITER -->>\n"
INPUT_LOG_PATH = None
_GLOBAL_LOG_HANDLE = None

# Transaction tracking
_TRANSACTION_COUNTER = 0
_CURRENT_TRANSACTION_ID = None

def _get_transaction_id(agent_name: str) -> str:
    """Generate unique transaction ID with prefix based on agent."""
    global _TRANSACTION_COUNTER
    _TRANSACTION_COUNTER += 1
    
    # Agent prefix mapping
    prefix_map = {
        "architect": "AR",
        "tech_lead": "TL", 
        "junior_dev": "JR",
        "integrator": "IG",
        "specialist": "SP"
    }
    
    agent_lower = agent_name.lower().replace(" ", "_")
    for key, prefix in prefix_map.items():
        if key in agent_lower:
            return f"{prefix}-{_TRANSACTION_COUNTER:03d}"
    
    return f"AG-{_TRANSACTION_COUNTER:03d}"


def log_terminal(message: str):
    """Log ONLY to terminal (PATH 1)."""
    # Direct print to terminal - no file redirection
    print(message)


def log_file(message: str):
    """Log ONLY to file (PATH 2)."""
    if _GLOBAL_LOG_HANDLE:
        # Convert lists or dicts to string to prevent TypeError
        if isinstance(message, (list, dict)):
            message = json.dumps(message, indent=2)
        else:
            message = str(message)
        _GLOBAL_LOG_HANDLE.write(message + "\n")
        _GLOBAL_LOG_HANDLE.flush()


def log_transaction_start(transaction_id: str, agent_name: str):
    """Log transaction header to file only."""
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    
    header = f"""
{'═' * 70}
TRANSACTION: {transaction_id}
AGENT: {agent_name}
TIMESTAMP: {timestamp}
{'═' * 70}
"""
    log_file(header)


def log_transaction_end(transaction_id: str):
    """Log transaction footer to file only."""
    footer = f"""
{'═' * 70}
END TRANSACTION: {transaction_id}
{'═' * 70}

{LOG_DELIM}
"""
    log_file(footer)


def log_input_messages(messages: list, label: str = "INPUT TO LLM"):
    """Log input messages in forensic format (file only)."""
    log_file(f"\n─── [{label}] {'─' * (60 - len(label))}─\n")
    
    for idx, msg in enumerate(messages):
        msg_type = msg.__class__.__name__
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        # Detect layer from content
        layer_label = ""
        if "LAYER 0: IDENTITY" in content or "You are the" in content:
            layer_label = "│ LAYER 0: IDENTITY                                            │"
        elif "LAYER 1: TASK CONTEXT" in content or "TASK CONTEXT" in content:
            layer_label = "│ LAYER 1: TASK CONTEXT                                        │"
        elif "LAYER 2: CONVERSATION" in content or "CONVERSATION" in content:
            layer_label = "│ LAYER 2: CONVERSATION                                        │"
        
        log_file(f"MESSAGE {idx} [{msg_type}]:")
        if layer_label:
            log_file("┌─────────────────────────────────────────────────────────────┐")
            log_file(layer_label)
            log_file("└─────────────────────────────────────────────────────────────┘")
        
        # Truncate very long content for readability
        if len(content) > 2000:
            log_file(f"{content[:1000]}\n... [truncated {len(content) - 2000} chars] ...\n{content[-1000:]}")
        else:
            log_file(content)
        log_file("")  # Blank line separator


def log_output_response(response, label: str = "OUTPUT FROM LLM"):
    """Log LLM response in forensic format (file only)."""
    log_file(f"\n─── [{label}] {'─' * (60 - len(label))}─\n")
    
    response_type = response.__class__.__name__
    log_file(f"RESPONSE TYPE: {response_type}")
    
    # Log tool calls if present
    if hasattr(response, 'tool_calls') and response.tool_calls:
        log_file(f"TOOL CALLS: {len(response.tool_calls)}\n")
        for idx, tc in enumerate(response.tool_calls, 1):
            log_file(f"[TOOL CALL {idx}]:")
            log_file(json.dumps(tc, indent=2))
            log_file("")
    
    # Log reasoning/content
    if hasattr(response, 'content') and response.content:
        log_file("REASONING (text content):")
        content = str(response.content)
        if len(content) > 2000:
            log_file(f"{content[:1000]}\n... [truncated {len(content) - 2000} chars] ...\n{content[-1000:]}")
        else:
            log_file(content)


def log_split(short_msg: str, detailed_content: str):
    """
    Terminal: Shows short_msg only
    File: Shows both short_msg AND detailed_content
    """
    # Terminal path
    log_terminal(short_msg)
    
    # File path
    log_file(short_msg)
    log_file(f"\n--- [DETAILED CONTEXT START] ---\n{detailed_content}\n--- [DETAILED CONTEXT END] ---\n")


# ═══════════════════════════════════════════════════════════════════════════
# AGENT REPORT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════


# --- Journal Compression (added by injection.py) ---
# Known field names that agents use for internal reasoning/journal
_JOURNAL_FIELD_NAMES = [
    "journal", "reasoning", "notes", "scratchpad", "thinking",
    "internal_notes", "thought_process", "analysis_notes",
    "reflection", "observations", "deliberation",
]
_JOURNAL_MAX_CHARS = 300  # Hard cap — keeps it small for downstream


def _compress_journal(raw_data: dict) -> str:
    """
    Extract journal-like fields from agent JSON output and compress
    into a single short summary string.

    Returns empty string if no journal content found.
    """
    if not isinstance(raw_data, dict):
        return ""

    fragments = []
    for key in _JOURNAL_FIELD_NAMES:
        val = raw_data.get(key)
        if not val:
            continue
        # Normalize to string
        if isinstance(val, list):
            val = " | ".join(str(v) for v in val)
        elif isinstance(val, dict):
            # Flatten dict values
            val = " | ".join(f"{k}: {v}" for k, v in val.items())
        else:
            val = str(val)
        val = val.strip()
        if val:
            fragments.append(val)

    if not fragments:
        return ""

    combined = " /// ".join(fragments)

    # Compress: collapse whitespace, trim
    combined = " ".join(combined.split())

    if len(combined) > _JOURNAL_MAX_CHARS:
        combined = combined[:_JOURNAL_MAX_CHARS - 3] + "..."

    return combined



def parse_agent_report(report_str: str) -> Dict[str, Any]:
    """Parse a JSON report string and return metadata."""
    if isinstance(report_str, list):
        report_str = " ".join(str(item) for item in report_str)
    
    cleaned = report_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    try:
        data = json.loads(cleaned)
        is_success = bool(data.get("success"))
        
        return {
            "is_json": True,
            "is_success": is_success,
            "content": data.get("success") if is_success else data.get("problem"),
            "proof": data.get("proof", ""),
            "goal": data.get("goal", ""),
            "journal_summary": _compress_journal(data),
            "raw": data
        }
    except Exception:
        is_success = "SUCCESS" in report_str.upper() and "FAIL" not in report_str.upper()
        return {
            "is_json": False,
            "is_success": is_success,
            "content": report_str,
            "proof": "",
            "goal": "",
            "journal_summary": "",
            "raw": None
        }


def sanitize_ai_message(message: BaseMessage) -> BaseMessage:
    """Removes leaked metadata/signatures from LLM outputs."""
    if not hasattr(message, "content"):
        return message
        
    # If the content is a list of blocks
    if isinstance(message.content, list):
        clean_blocks = []
        for block in message.content:
            if isinstance(block, dict):
                # Strip out the 'extras' key which contains the signature
                if "extras" in block:
                    cleaned_dict = {k: v for k, v in block.items() if k != "extras"}
                    clean_blocks.append(cleaned_dict)
                else:
                    clean_blocks.append(block)
            elif isinstance(block, str):
                clean_blocks.append(block)
                
        # Create a new, clean message (preserving attributes)
        if isinstance(message, AIMessage):
            new_msg = AIMessage(content=clean_blocks)
            if hasattr(message, "tool_calls"):
                new_msg.tool_calls = message.tool_calls
            if hasattr(message, "response_metadata"):
                new_msg.response_metadata = message.response_metadata
            if hasattr(message, "id"):
                new_msg.id = message.id
            return new_msg
        else:
            message.content = clean_blocks
            return message
            
    return message


async def call_model_safely(model, messages):
    """Centralized wrapper that calls the LLM and automatically sanitizes the output."""
    response = await model.ainvoke(messages)
    return sanitize_ai_message(response)




# ═══════════════════════════════════════════════════════════════════════════
# STATE TYPES
# ═══════════════════════════════════════════════════════════════════════════


class AgentContext(TypedDict):
    agent_id: str
    task_description: str
    task_context: dict
    project_root: str
    base_dir: str

class MainState(TypedDict):
    messages: list[BaseMessage]
    architect_context: AgentContext
    specialist_context: AgentContext
    integrator_context: AgentContext
    project_ledger: str
    temp_dir: str
    run_dir: str
    architect_text_only_count: int


class CoderState(TypedDict):
    messages: list[BaseMessage]
    junior_messages: list[BaseMessage]
    techlead_context: AgentContext
    junior_context: AgentContext
    iterations: int
    tech_lead_spec: str
    junior_output: str
    scratchpad: str
    text_only_count: int




# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED SESSION COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════

async def compress_session_ucp(
    general_goal: str,    
    messages: list,
    action_type: str,
    affected_files: list,
    model,
    *,
    role_description: str = "an agent's work session after a file modification",
    recent_count: int = 6,
    max_words: int = 200,
    summary_points: list[str] | None = None,
    content_truncate: int = 300,
) -> str:
    """
    Unified UCP-MSP compliant memory compression for any agent session.

    Replaces the three near-identical functions:
      - compress_junior_session_ucp   (recent=5, words=150, 4 points)
      - compress_specialist_session_ucp (recent=6, words=200, 5 points)
      - compress_integrator_session_ucp (recent=6, words=200, 4 points)

    Parameters
    ----------
    messages : list
        Full session message history.
    action_type : str
        What triggered compression (e.g. "file_write", "build_command").
    affected_files : list
        Files touched in this operation.
    model : LLM
        Model instance for compression call.
    role_description : str
        Human-readable role context injected into the system prompt.
        E.g. "a Junior Developer's work session after a file modification".
    recent_count : int
        How many recent messages to include (from tail).
    max_words : int
        Word budget for the summary.
    summary_points : list[str] | None
        Numbered bullet points for the system prompt. If None, uses a
        sensible default set.
    content_truncate : int
        Max chars per message content in the log excerpt.

    Returns
    -------
    str
        Compressed summary text.
    """
    from langchain_core.messages import SystemMessage as _SM, HumanMessage as _HM
    messages.insert(0, general_goal)  
    recent = messages[-recent_count:] if len(messages) > recent_count else messages
    log_text = "\n".join([
        f"{msg.__class__.__name__}: {str(msg.content)[:content_truncate] if hasattr(msg, 'content') else str(msg)[:content_truncate]}"
        for msg in recent
    ])

    if summary_points is None:
        summary_points = [
            "What problem was being solved",
            "What approach was decided",
            "What was committed in this file operation",
            "Expected outcome or remaining issues",
        ]

    numbered = "\n".join(f"{i+1}. {pt}" for i, pt in enumerate(summary_points))

    instructions = _SM(content=(
        f"You are compressing {role_description}. "
        f"Create a BRIEF summary (max {max_words} words) covering:\n"
        f"{numbered}\n"
        f"Focus on DECISIONS MADE and RESULTS, not the thinking process."
    ))

    payload = _HM(content=(
        f"ACTION COMMITTED: {action_type}\n"
        f"FILES AFFECTED: {', '.join(affected_files) if affected_files else 'Unknown'}\n"
        f"\n"
        f"Session history (last {recent_count} messages):\n"
        f"{log_text}\n"
        f"\n"
        f"Provide ONLY the summary text."
    ))

    try:
        response = await call_model_safely(model, [instructions, payload])
        return response.content
    except Exception as e:
        return f"[Compression failed: {str(e)}] Action: {action_type} on {affected_files}"


# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZATION & LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def init_input_log(run_dir: str):
    global INPUT_LOG_PATH
    global _GLOBAL_LOG_HANDLE
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    INPUT_LOG_PATH = run_dir / "input_log.txt"
    
    f = open(INPUT_LOG_PATH, "w", encoding="utf-8")
    _GLOBAL_LOG_HANDLE = f  
    f.write(f"\n# --- NEW SESSION: {datetime.datetime.now()} ---\n")
    
    # DO NOT redirect stdout/stderr - we want separate terminal/file paths
    # Terminal: Use print() or log_terminal()
    # File: Use log_file() functions
    return INPUT_LOG_PATH


def format_messages_for_log(messages):
    out = []
    for m in messages:
        role = m.__class__.__name__
        content = getattr(m, "content", str(m))
        out.append({"role": role, "content": content})
    return out


def log_input(agent: str, node: str, messages):
    """Legacy logging function - maintained for backward compatibility."""
    if INPUT_LOG_PATH is None: 
        return

    print(f"\n[LOGGING INTERACTION] Agent: {agent} | Node: {node}")

    payload = {
        "agent": agent, "node": node,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "messages": format_messages_for_log(messages)
    }
    with open(INPUT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n" + LOG_DELIM + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# SAFE MODEL CALL - WITH TWO-PATH LOGGING
# ═══════════════════════════════════════════════════════════════════════════


def render_persona_prompt(persona_config: dict, role_name: str, context: dict) -> str:
    """
    Render persona system prompt by expanding {{template}} placeholders.
    
    This is CRITICAL for role separation - without it, agents don't know their constraints.
    
    Args:
        persona_config: The persona configuration dict from personas.json
        role_name: The role key (e.g., "tech_lead", "architect")
        context: Optional context for variable substitution (e.g., project_root)
    
    Returns:
        Fully rendered system prompt with all constraints expanded
    """
    

    # Normalize role_name to match persona config keys
    _normalized = role_name.lower().replace(" ", "_")
    if _normalized in persona_config:
        role_name = _normalized
    elif role_name not in persona_config:
        # Fallback: return generic prompt instead of crashing
        return f"You are the {role_name}. Follow your assigned responsibilities."
    
    role = persona_config[role_name]
    template_str = role.get("system_prompt", "")
    
    if not template_str:
        # PATCHED: Graceful fallback instead of crash (was: raise ValueError)
        # Builds minimal prompt from raw persona fields so agent gets SOME guidance
        _fb_identity = json.dumps(role.get("identity", {}), indent=2)
        _fb_constraints = json.dumps(role.get("constraints", {}), indent=2)
        _fb_output = json.dumps(role.get("output_protocol", {}).get("mandatory_structure", {}), indent=2)
        _fb_frameworks = json.dumps(role.get("cognitive_frameworks", {}), indent=2)
        template_str = (
            f"You are the {role_name}. "
            f"Identity: {_fb_identity}. "
            f"Framework: {_fb_frameworks}. "
            f"Constraints: {_fb_constraints}. "
            f"Output: {_fb_output}."
        )
    
    context = context or {}
    
    # PATCHED: Inject factory context preamble (matches build_agent_messages in factory_engine.py)
    factory_preamble = (
        "🏭 FACTORY CONTEXT: You work in a code factory alongside other specialized agents.\n"
        "\n"
        "This is an assembly line - you may receive imperfect context or unclear instructions "
        "from other agents. This is normal in multi-agent collaboration.\n"
        "\n"
        "YOUR MISSION: Focus on YOUR role. Make reasonable assumptions when context is unclear.\n"
        "Keep the assembly line moving - bias toward action over perfect clarity.\n"
        "\n"
        "Remember: Together you are stronger. Your teammates are counting on you. 💪\n\n"
    )
    template_str = factory_preamble + template_str
    
    # Expand {{identity}}
    if "{{identity}}" in template_str:
        identity_json = json.dumps(role.get("identity", {}), indent=2)
        template_str = template_str.replace("{{identity}}", identity_json)
    
    # Expand {{cognitive_frameworks}}
    if "{{cognitive_frameworks}}" in template_str:
        frameworks_json = json.dumps(role.get("cognitive_frameworks", {}), indent=2)
        template_str = template_str.replace("{{cognitive_frameworks}}", frameworks_json)
    
    # Expand {{constraints}}
    if "{{constraints}}" in template_str:
        constraints_json = json.dumps(role.get("constraints", {}), indent=2)
        template_str = template_str.replace("{{constraints}}", constraints_json)
    
    # Expand {{output_protocol}}
    if "{{output_protocol}}" in template_str:
        output_json = json.dumps(role.get("output_protocol", {}), indent=2)
        template_str = template_str.replace("{{output_protocol}}", output_json)
    
    # Expand {{output_protocol.mandatory_structure}} specifically
    if "{{output_protocol.mandatory_structure}}" in template_str:
        mandatory_json = json.dumps(
            role.get("output_protocol", {}).get("mandatory_structure", {}), 
            indent=2
        )
        template_str = template_str.replace("{{output_protocol.mandatory_structure}}", mandatory_json)
    
    # Expand {{quality_gates}}
    if "{{quality_gates}}" in template_str:
        gates_json = json.dumps(role.get("quality_gates", {}), indent=2)
        template_str = template_str.replace("{{quality_gates}}", gates_json)
    
    # Expand context variables (e.g., {project_root})
    for key, value in context.items():
        placeholder = "{" + key + "}"
        if placeholder in template_str:
            template_str = template_str.replace(placeholder, str(value))
    
    return template_str


async def safe_model_call(human_input, fixed_message, messages, model, agent_name, agent_id=None, personas_config=None):
    """
    Centralized model call with TWO-PATH logging:
    - Terminal: Minimal progress indicators
    - File: Complete forensic transaction log
    
    When agent_id is provided, enforces persona isolation.
    """
    global _CURRENT_TRANSACTION_ID

    persona_str = render_persona_prompt(load_persona_config(), agent_name, [])
    # Generate transaction ID
    transaction_id = _get_transaction_id(agent_name)
    _CURRENT_TRANSACTION_ID = transaction_id
    
    # PATH 1: TERMINAL - Minimal progress indicator
    log_terminal(f"[{transaction_id}] {agent_name} → Thinking...")
    messages = [ SystemMessage(content=persona_str) ] + \
          [ SystemMessage(content=fixed_message) ] +  \
          [HumanMessage(human_input) ]+ \
          messages
    # PATH 2: FILE - Complete transaction log
    if INPUT_LOG_PATH:
        log_transaction_start(transaction_id, agent_name)
        log_input_messages(messages)
    
    # Execute inference
    response = await call_model_safely(model, messages)
    
    # PATH 1: TERMINAL - Result indicator
    content_str = str(response.content) if hasattr(response, 'content') else ""
    success_indicator = "✓" if "success" in content_str.lower() else "→"

    tool_info = ""
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_count = len(response.tool_calls)
        tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
        tool_info = f" (Tools: {', '.join(tool_names[:2])}{'...' if tool_count > 2 else ''})"
    
    log_terminal(f"[{transaction_id}] {agent_name} {success_indicator} Response received{tool_info}")
    
    # PATH 2: FILE - Complete response
    if INPUT_LOG_PATH:
        log_output_response(response)
        log_transaction_end(transaction_id)
    
    return response


# ═══════════════════════════════════════════════════════════════════════════
# AGENT STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════

def load_persona_config(config_path: str = "personas.json") -> Dict[str, Any]:
    """Reads personas from disk on demand."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def load_task_config(file_path="task.json"):
    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"{file_path} missing")
    with open(file_path, 'r') as f:
        config = json.load(f)
    # Derive target from explicit field or fall back to language
    if "target" not in config:
        config["target"] = config.get("language", "rust")
    reqs = "\n".join([f"- {r}" for r in config['requirements']])
    mission_text = (
        f"MISSION: {config['goal']}\n"
        f"LANGUAGE: {config['language']}\n"
        f"TARGET PLATFORM: {config['target']}\n"
        f"PROJECT: {config['project_name']}\n"
        f"REQS:\n{reqs}"
    )
    return mission_text, config


def prepare_workspace(config, run_id: str):
    mode = config.get("mode", "create")
    target = config.get("target", config.get("language", "rust"))
    
    base_dir = f"/workspace/{run_id}/generated"
    project_path = f"{base_dir}/{config['project_name']}"
    temp_path = f"{base_dir}/temp"
    run_path = f"{base_dir}/run"
    
    print(f"--- [Engine] Workspace Prep (Mode: {mode}, Target: {target}) ---")
    print(f"    Project: {project_path}")
    print(f"    Temp: {temp_path}")
    print(f"    Run: {run_path}")
    
    # 1. DESTRUCTIVE SETUP (Mode: Create)
    if mode == "create":
        if os.path.exists(base_dir):
            print(f"--- [Engine] Mode is 'create': Wiping {base_dir} ---")
            shutil.rmtree(base_dir)
        os.makedirs(project_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(run_path, exist_ok=True)
        
    # 2. CONSERVATIVE SETUP (Mode: Not Create)
    else:
        os.makedirs(project_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(run_path, exist_ok=True)
        
    # 3. GIT INITIALIZATION (Idempotent)
    if not os.path.exists(os.path.join(project_path, ".git")):
        print(f"--- [Engine] Initializing Git Repository ---")
        subprocess.run(["git", "init"], cwd=project_path, capture_output=True)
        subprocess.run(["git", "checkout", "-b", "main"], cwd=project_path, capture_output=True)
        
    # 4. PLATFORM SCAFFOLDING (Idempotent, target-driven)
    _scaffold_platform(target, project_path, config)
        
    return {
        "project_root": project_path,
        "temp_dir": temp_path,
        "run_dir": run_path,
        "base_dir": base_dir
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLATFORM SCAFFOLDING REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

_PLATFORM_REGISTRY = {
    "rust": {
        "scaffold_check": "Cargo.toml",
        "scaffold_fn": "_scaffold_rust",
    },
    "python": {
        "scaffold_check": ".gitignore",
        "scaffold_fn": "_scaffold_python",
    },
    "documentation": {
        "scaffold_check": None,
        "scaffold_fn": "_scaffold_documentation",
    },
}


def _scaffold_platform(target: str, project_path: str, config: dict):
    """Dispatch scaffolding based on target platform."""
    entry = _PLATFORM_REGISTRY.get(target)
    
    if entry is None:
        print(f"--- [Engine] WARNING: Unknown platform \'{target}\'. No scaffolding applied. ---")
        print(f"--- [Engine] LLM agents will be informed of the platform via prompts. ---")
        _write_gitignore(project_path, ["__pycache__/", ".idea/", ".vscode/"])
        _git_initial_commit(project_path)
        return
    
    # Idempotency check
    check_file = entry.get("scaffold_check")
    if check_file and os.path.exists(os.path.join(project_path, check_file)):
        print(f"--- [Engine] Platform \'{target}\' already scaffolded (found {check_file}) ---")
        return
    
    # Dispatch to platform-specific function
    fn_name = entry["scaffold_fn"]
    globals()[fn_name](project_path, config)


def _scaffold_rust(project_path: str, config: dict):
    """Rust scaffolding — preserves existing behavior exactly."""
    print(f"--- [Engine] Initializing Rust Project ---")
    subprocess.run(["cargo", "init", "--bin"], cwd=project_path, capture_output=True)
    _write_gitignore(project_path, [
        "# Rust build artifacts", "target/", "Cargo.lock",
        "", "# IDE", ".idea/", ".vscode/"
    ])
    _git_initial_commit(project_path)


def _scaffold_python(project_path: str, config: dict):
    """Python scaffolding — minimal setup, assumes platform has python."""
    print(f"--- [Engine] Initializing Python Project ---")
    _write_gitignore(project_path, [
        "# Python", "__pycache__/", "*.pyc", "*.pyo",
        ".venv/", "venv/", "*.egg-info/", "dist/", "build/",
        "", "# IDE", ".idea/", ".vscode/"
    ])
    # Create minimal project structure
    src_dir = os.path.join(project_path, "src")
    os.makedirs(src_dir, exist_ok=True)
    init_path = os.path.join(src_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")
    _git_initial_commit(project_path)


def _scaffold_documentation(project_path: str, config: dict):
    """Documentation/self mode — copies factory source into workspace for analysis."""
    print(f"--- [Engine] Documentation/Self Mode: Copying factory source ---")
    
    factory_root = os.path.dirname(os.path.abspath(__file__))
    
    SKIP_DIRS = {"__pycache__", ".git", "generated", "node_modules", ".venv", "venv"}
    COPY_EXTENSIONS = {".py", ".json", ".txt", ".md", ".toml", ".yaml", ".yml"}
    
    for item in sorted(os.listdir(factory_root)):
        src = os.path.join(factory_root, item)
        dst = os.path.join(project_path, item)
        
        if item in SKIP_DIRS:
            continue
        
        if os.path.isdir(src):
            # Copy subdirectories (e.g., decomposer/) preserving structure
            if not os.path.exists(dst):
                shutil.copytree(
                    src, dst,
                    ignore=shutil.ignore_patterns(*SKIP_DIRS)
                )
                print(f"    Copied dir: {item}/")
        elif os.path.isfile(src) and os.path.splitext(item)[1] in COPY_EXTENSIONS:
            shutil.copy2(src, dst)
            print(f"    Copied: {item}")
    
    _write_gitignore(project_path, ["__pycache__/", ".idea/", ".vscode/"])
    _git_initial_commit(project_path)


def _write_gitignore(project_path: str, lines: list):
    """Helper: write .gitignore file."""
    gitignore_path = os.path.join(project_path, ".gitignore")
    with open(gitignore_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _git_initial_commit(project_path: str):
    """Helper: stage all and commit initial scaffold."""
    subprocess.run(["git", "add", "."], cwd=project_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "System: Initial scaffold"],
        cwd=project_path, capture_output=True
    )
