from decomposer.template_loader import load_templates
"""
Contract Integration Module
Handles contract generation, storage, and injection into agent contexts
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
# Import from decomposer modules
import sys
sys.path.append('/workspace/decomposer')

from task_decomposer import decompose_task
from contract_generator import generate_phase_contract, ContractStore, validate_contract_completeness
from plan_manager import DevelopmentPlan
from model_resolver import get_model


class ContractManager:
    """
    Manages the lifecycle of contracts in the factory system.
    
    Responsibilities:
    - Idempotent dev_plan.json generation
    - Idempotent contract generation per phase
    - Contract injection into agent contexts
    - Phase progression tracking
    """
    
    def __init__(self, workspace_base: str):
        """
        Initialize contract manager.
        
        Args:
            workspace_base: Base directory (e.g., /workspace/{run_id}/generated)
        """
        self.workspace_base = Path(workspace_base)
        self.factory_dir = self.workspace_base / ".factory"
        self.factory_dir.mkdir(parents=True, exist_ok=True)
        self.dev_plan_path = self.factory_dir / "dev_plan.json"
        self.contracts_dir = self.factory_dir / "contracts"
        self.progress_path = self.factory_dir / "dev_progress.json"
        
        # Ensure directories exist
        self.contracts_dir.mkdir(parents=True, exist_ok=True)
        
        self.plan_manager: Optional[DevelopmentPlan] = None
        self.contract_store = ContractStore(str(self.contracts_dir))
        
    async def ensure_phase_contract(self, phase_id: str, language: str = "rust") -> bool:
                """
                JIT (Just-In-Time) Trigger: Checks if a phase contract exists.
                If not, automatically invokes the LLM to generate it before proceeding.
                """
                contract_path = self.contracts_dir / f"phase_{phase_id}.json"
                if contract_path.exists():
                    return True
                    
                print(f"\n[Factory Engine] ⚙️ LAZY GENERATION TRIGGERED FOR PHASE: {phase_id}")
                
                if not self.plan_manager:
                    self.plan_manager = DevelopmentPlan(str(self.dev_plan_path), str(self.progress_path))
                    
                phase_data = next((p for p in self.plan_manager.plan.get("phases", []) 
                                   if p.get("phase_id") == phase_id), None)
                                   
                if not phase_data:
                    print(f"[Factory Engine] ❌ Error: Phase {phase_id} not found in dev_plan.json")
                    return False
                    
                # Actively trigger the generation
                contract = await generate_phase_contract(phase_data, language)
                self.contract_store.save_contract(contract, f"phase_{phase_id}")
                print(f"[Factory Engine] ✅ Successfully lazy-generated contract for {phase_id}")
                return True
    
    def contracts_exist(self) -> bool:
        """Check if essential contract artifacts already exist."""
        # Simple heuristic: Do we have a plan and at least one contract?
        if not self.dev_plan_path.exists():
            return False
        # Check for any json files in contracts dir
        return any(self.contracts_dir.glob("*.json"))

    async def ensure_dev_plan_exists(self, task_json_path: str, language: str) -> Dict[str, Any]:
        """
        Idempotently generates dev_plan.json if it doesn't exist.
        
        Args:
            task_json_path: Path to task.json
            language: Programming language (e.g., "rust")
            
        Returns:
            Development plan dictionary
        """
        if self.dev_plan_path.exists():
            print(f"[ContractMgr] dev_plan.json already exists at {self.dev_plan_path}")
            with open(self.dev_plan_path, 'r') as f:
                return json.load(f)
        
        print(f"[ContractMgr] Generating dev_plan.json from {task_json_path}...")
        
        plan = await decompose_task(
            task_json_path=task_json_path,
            output_path=str(self.dev_plan_path)
        )
        
        print(f"[ContractMgr] ✓ dev_plan.json generated at {self.dev_plan_path}")
        return plan
    
    def get_plan_manager(self) -> DevelopmentPlan:
        """
        Lazy-loads plan manager.
        
        Returns:
            DevelopmentPlan instance
        """
        if self.plan_manager is None:
            if not self.dev_plan_path.exists():
                raise FileNotFoundError(
                    f"dev_plan.json not found at {self.dev_plan_path}. "
                    "Call ensure_dev_plan_exists() first."
                )
            
            self.plan_manager = DevelopmentPlan(
                plan_path=str(self.dev_plan_path),
                progress_path=str(self.progress_path)
            )
            
            print(f"[ContractMgr] Plan manager initialized")
            self.plan_manager.print_status()
            # Generate contracts for ALL phases upfront
    
        return self.plan_manager
    
    async def ensure_contract_exists(self, phase_id: str, language: str) -> Dict[str, Any]:
        """
        Idempotently generates contract for a phase if it doesn't exist.
        
        Args:
            phase_id: Phase identifier (e.g., "P001")
            language: Programming language
            
        Returns:
            Contract dictionary
        """
        contract_path = self.contracts_dir / f"phase_{phase_id}.json"
        
        if contract_path.exists():
            print(f"[ContractMgr] Contract already exists for {phase_id}")
            with open(contract_path, 'r') as f:
                return json.load(f)
        
        print(f"[ContractMgr] Generating contract for {phase_id}...")
        
        # Get phase from plan
        plan_manager = self.get_plan_manager()
        phase = plan_manager.get_phase_by_id(phase_id)
        
        # Generate contract
        templates_dir = "/workspace/decomposer/templates"
        template_context, _ = load_templates(templates_dir, verbose=False)
        contract = await generate_phase_contract(
            phase=phase,
            language=language
        )
        
        # Validate each task contract within the phase contract
        all_errors = []
        for contract in contract.get("contracts", []):
            errors = validate_contract_completeness(contract)
            all_errors.extend(errors)
        
        if all_errors:
            print(f"[ContractMgr] ⚠ Contract validation warnings:")
            for error in all_errors:
                print(f"  - {error}")
        
        # Save
        self.contract_store.save_contract(contract, f"phase_{phase_id}")
        
        print(f"[ContractMgr] ✓ Contract generated for {phase_id}")
        return contract
    
    async def get_current_phase_info(self, language: str = "rust") -> Optional[Dict[str, Any]]:
        """
        Gets current phase and its contract. Generates the contract if it's missing.
        """
        plan_manager = self.get_plan_manager()
        current_phase = plan_manager.get_current_phase()
        
        if current_phase is None:
            return None
        
        phase_id = current_phase["phase_id"]
        contract_path = self.contracts_dir / f"phase_{phase_id}.json"
        
        # JIT Trigger: If the contract doesn't exist, generate it now
        if not contract_path.exists():
            print(f"[ContractMgr] Contract missing for current phase {phase_id}. Triggering generation...")
            # You can use either ensure_phase_contract or ensure_contract_exists here
            await self.ensure_contract_exists(phase_id, language)
        
        # Now it is guaranteed to exist (unless generation failed)
        try:
            with open(contract_path, 'r') as f:
                contract = json.load(f)
        except FileNotFoundError:
            print(f"[ContractMgr] ERROR: Failed to generate or read contract for {phase_id}")
            return {
                "phase": current_phase,
                "contract": None,
                "contract_path": str(contract_path)
            }
        
        return {
            "phase": current_phase,
            "contract": contract,
            "contract_path": str(contract_path)
        }

    def get_phase_status(self, phase_id: str) -> str:
        """Get current status of a phase."""
        return self.get_plan_manager().get_phase_status(phase_id)

    def update_phase_status(self, phase_id: str, status: str) -> None:
        """Update status of a specific phase."""
        self.get_plan_manager().update_phase_status(phase_id, status)

    def get_all_phase_statuses(self) -> Dict[str, str]:
        """Get status map for all phases."""
        return self.get_plan_manager().get_all_phase_statuses()




    



# ============================================================================
# FACTORY INTEGRATION HELPERS
# ============================================================================

async def initialize_contracts_for_run(
    workspace_base: str,
    task_json_path: str,
    language: str,
    generate_all: bool = True
) -> ContractManager:
    """
    Initialize contract system for a factory run.
    
    Args:
        workspace_base: Base workspace directory
        task_json_path: Path to task.json
        language: Programming language
        generate_all: If True, generates contracts for ALL phases upfront
        
    Returns:
        Initialized ContractManager
    """
    manager = ContractManager(workspace_base)
    with open(task_json_path, 'r') as f:
        task_config = json.load(f)
    # Check if we are in "continue" mode
    if task_config.get("mode") == "continue":
        print("--- [Contract Manager] Mode is 'continue'. Checking for existing contracts... ---")
        
        # We assume if the plan exists, we are good to go
        if manager.dev_plan_path.exists():
             print(f"✓ Found existing plan at {manager.dev_plan_path}")
             # Optional: You could add more robust checks here (like checking if contracts dir is empty)
             return manager
        else:
             print("⚠ Mode is 'continue' but no plan found. Falling back to generation.")
    

    # Step 1: Ensure dev_plan.json exists
    await manager.ensure_dev_plan_exists(task_json_path, language)
    
    # Step 2: Initialize plan manager
    plan_manager = manager.get_plan_manager()
    
    # Step 3: Generate contracts for all phases if requested (Option 1 approach)
    if generate_all:
        print("[ContractMgr] Pre-generating all contracts...")
        
        all_phases = plan_manager.plan["phases"]
        for phase in all_phases:
            phase_id = phase["phase_id"]
            await manager.ensure_contract_exists(phase_id, language)
        
        print(f"[ContractMgr] ✓ All {len(all_phases)} contracts generated")
    else:
        # Just generate contract for current phase
        current_phase = plan_manager.get_current_phase()
        if current_phase:
            await manager.ensure_contract_exists(current_phase["phase_id"], language)
    
    return manager




def inject_architect_contract_awareness(
    current_phase_id: Optional[str],
    contract_path: Optional[str],
    plan_overview: str = "",
    progress_status: str = ""
) -> str:
    """
    Injects mission status and roadmap into the Architect's system prompt.
    Leaves persona definitions and phase implementation details out.
    """
    # Ensure idempotency: remove previous mission status blocks to avoid infinite growth

    plan_text = plan_overview if plan_overview else "No phase plan currently defined."
    status_text = progress_status if progress_status else "No status information currently available."
    
    if current_phase_id:
        phase_notice = (
            f"CURRENT PHASE: {current_phase_id}\n"
            f"Note: Detailed phase specifications exist at {contract_path} and have been handed over to the Tech Lead for implementation."
        )
    else:
        phase_notice = "STATUS: All development phases complete or inactive."

    awareness = f"""<mission_status>
{phase_notice}

--- MISSION EXTENSION: PHASE-BASED PLAN ---
The following is the detailed phase-based roadmap detailing what needs to be done:
{plan_text}

--- CURRENT EXECUTION STATUS ---
The following information outlines what has already been accomplished:
{status_text}
-------------------------------------------
</mission_status>"""

    # Append cleanly to the bottom of the prompt
    return f"\n{awareness}"

"""
Contract Formatter Module
Converts phase contract JSON (from get_current_phase_info) into a detailed,
human/LLM-readable implementation specification block.
"""

from typing import Dict, Any, Optional

def format_phase_as_implementation_guide(phase_info: Dict[str, Any]) -> str:
    """
    Takes the dictionary returned by ContractManager.get_current_phase_info()
    and formats it into a detailed, structured implementation guide suitable
    for injection into an LLM agent's context.

    Args:
        phase_info: Dict with keys "phase", "contract", "contract_path"
                    where "contract" is the full phase contract JSON
                    (e.g. contents of phase_P001.json).

    Returns:
        A formatted multi-line string — the implementation specification guide.
        Returns an error block if contract is missing.
    """
    phase = phase_info.get("phase", {})
    contract = phase_info.get("contract")
    # FIX: Extract contract_path from the dictionary with a fallback
    contract_path = phase_info.get("contract_path", "Unknown path")
    
    if contract is None:
        return (
            "⚠ IMPLEMENTATION GUIDE UNAVAILABLE\n"
            f"Contract missing for phase {phase.get('phase_id', '?')}.\n"
            f"Expected at: {contract_path}"
        )

    return _format_contract(contract)


# ---------------------------------------------------------------------------
# Internal formatting
# ---------------------------------------------------------------------------

def _format_contract(contract: Dict[str, Any]) -> str:
    """Core formatter that builds the phase specification text."""
    sections = []

    # Injecting explicit "Phase Spec" terminology and direct agent instructions
    sections.append(
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║            COMPLETE PHASE SPECIFICATION GUIDE                ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
        "👉 SYSTEM HINT FOR AGENT: This entire document IS the complete \n"
        "Phase Specification. DO NOT make external tool calls to fetch it.\n"
        "All context required for this phase is provided below.\n\n"
    )

    # ── Module Specification ────────────────────────────────────────────
    module = contract.get("module_spec", {})
    if module:
        sections.append(_section_divider("PHASE MODULE SPECIFICATION"))
        sections.append(
            f"Module Name : {module.get('module_name', 'N/A')}\n"
            f"Purpose     : {module.get('purpose', 'N/A')}\n"
        )

        # ── Public Interface ────────────────────────────────────────────
        public_interface = module.get("public_interface", [])
        if public_interface:
            sections.append(_section_divider("PHASE PUBLIC INTERFACE"))

            for i, item in enumerate(public_interface, 1):
                sections.append(_format_interface_item(item, i))

    # ── Test Specifications ─────────────────────────────────────────────
    test_specs = contract.get("test_specifications", [])
    if test_specs:
        sections.append(_section_divider("PHASE TEST SPECIFICATIONS"))

        for category_block in test_specs:
            category = category_block.get("test_category", "unknown")
            cases = category_block.get("test_cases", [])
            sections.append(f"┌─ Category: {category.upper()} ({len(cases)} test(s))\n")

            for tc in cases:
                sections.append(_format_test_case(tc))

            sections.append("\n")

    # ── Acceptance Criteria ─────────────────────────────────────────────
    criteria = contract.get("acceptance_criteria", [])
    if criteria:
        sections.append(_section_divider("PHASE ACCEPTANCE CRITERIA"))
        for j, criterion in enumerate(criteria, 1):
            sections.append(f"  [{j}] {criterion}\n")
        sections.append("\n")

    # ── Footer ──────────────────────────────────────────────────────────
    footer_line = "=" * 62
    sections.append(
        f"{footer_line}\n"
        "END OF COMPLETE PHASE SPECIFICATION\n"
        f"{footer_line}\n"
    )

    return "".join(sections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_divider(title: str) -> str:
    return f"\n{'─' * 62}\n  {title}\n{'─' * 62}\n\n"


def _format_interface_item(item: Dict[str, Any], index: int) -> str:
    """Formats a single struct or function from the public interface."""
    lines = []

    # Determine if it's a struct or function
    struct_name = item.get("struct_name")
    func_name = item.get("function_name")

    if struct_name:
        lines.append(f"  [{index}] STRUCT: {struct_name}\n")
        definition = item.get("definition", "")
        if definition:
            lines.append(f"      Definition : {definition}\n")
        desc = item.get("description", "")
        if desc:
            lines.append(f"      Description: {desc}\n")
        lines.append("\n")

    elif func_name:
        signature = item.get("signature", "N/A")
        lines.append(f"  [{index}] FUNCTION: {func_name}\n")
        lines.append(f"      Signature   : {signature}\n")

        ret_type = item.get("return_type", "")
        if ret_type:
            lines.append(f"      Returns     : {ret_type}\n")

        ret_desc = item.get("return_description", "")
        if ret_desc:
            lines.append(f"      Return Desc : {ret_desc}\n")

        complexity = item.get("complexity", "")
        if complexity:
            lines.append(f"      Complexity  : {complexity}\n")

        # Parameters
        params = item.get("parameters", [])
        if params:
            lines.append("      Parameters:\n")
            for p in params:
                p_name = p.get("name", "?")
                p_type = p.get("type", "?")
                p_desc = p.get("description", "")
                p_valid = p.get("validation", "")
                lines.append(f"        • {p_name} ({p_type})\n")
                if p_desc:
                    lines.append(f"          Description : {p_desc}\n")
                if p_valid:
                    lines.append(f"          Validation  : {p_valid}\n")

                valid_ex = p.get("examples_valid", [])
                if valid_ex:
                    lines.append(f"          Valid examples   : {', '.join(str(e) for e in valid_ex)}\n")

                invalid_ex = p.get("examples_invalid", [])
                if invalid_ex:
                    lines.append(f"          Invalid examples : {', '.join(str(e) for e in invalid_ex)}\n")

        # Error cases
        errors = item.get("error_cases", [])
        if errors:
            lines.append("      Error Cases:\n")
            for ec in errors:
                cond = ec.get("condition", "?")
                etype = ec.get("error_type", "?")
                msg = ec.get("message", "?")
                lines.append(f"        ✗ WHEN: {cond}\n")
                lines.append(f"          TYPE: {etype}\n")
                lines.append(f"          MSG : \"{msg}\"\n")

        lines.append("\n")
    else:
        # Fallback for unknown interface items
        lines.append(f"  [{index}] INTERFACE ITEM:\n")
        for k, v in item.items():
            lines.append(f"      {k}: {v}\n")
        lines.append("\n")

    return "".join(lines)


def _format_test_case(tc: Dict[str, Any]) -> str:
    """Formats a single test case."""
    lines = []
    name = tc.get("test_name", "unnamed_test")
    desc = tc.get("description", "")
    priority = tc.get("priority", "normal")
    setup = tc.get("setup", "None")
    inp = tc.get("input", "")
    expected = tc.get("expected_output", "")
    assertions = tc.get("assertions", [])

    priority_marker = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
        priority, "⚪"
    )

    lines.append(f"  │  {priority_marker} {name}  [{priority}]\n")
    if desc:
        lines.append(f"  │    Description : {desc}\n")
    if setup and setup != "None":
        lines.append(f"  │    Setup       : {setup}\n")
    if inp:
        lines.append(f"  │    Input       : {inp}\n")
    if expected:
        lines.append(f"  │    Expected    : {expected}\n")
    if assertions:
        lines.append(f"  │    Assertions  :\n")
        for a in assertions:
            lines.append(f"  │      ✓ {a}\n")
    lines.append("  │\n")

    return "".join(lines)

