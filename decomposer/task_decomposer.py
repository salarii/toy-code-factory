"""
Task Decomposer - Converts natural language task.json into structured dev_plan.json

This module uses standardized LLM interfaces to break down project descriptions into
granular, phase-based development plans with precise function signatures
and test specifications.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
# Logic now handled via model_resolver
# Removed direct provider import
from langchain_core.messages import HumanMessage, SystemMessage
import sys
from model_resolver import get_model, ModelRole
from decomposer.template_loader import load_templates
# ============================================================================
# DECOMPOSITION PROMPT TEMPLATE
# ============================================================================

DECOMPOSITION_PROMPT = """You are an expert software architect specializing in project decomposition.

YOUR TASK:
Analyze the provided project specification and break it into hierarchical development phases.
Start from lowest abstraction (data structures, core algorithms) to highest (integration, UI).

INPUT SPECIFICATION:
{task_spec}

{template_context}

DECOMPOSITION RULES:

1. PHASE HIERARCHY (abstraction levels):
   - Level 1: Core data structures and memory management
   - Level 2: Fundamental algorithms and operations
   - Level 3: Business logic and domain-specific features
   - Level 4: Integration, error handling, and orchestration
   - Level 5: User interfaces and external APIs

2. TASK GRANULARITY:
   - Each task represents ONE module/file
   - Break complex modules into multiple tasks if needed
   - Dependencies must be explicit and acyclic

3. FUNCTION SPECIFICATIONS:
   - Every function needs precise signature
   - Include parameter types and return types
   - Language-specific conventions (e.g., Rust: Result<T, E>, Python: Optional[T])

4. TEST REQUIREMENTS:
   - EVERY function must have test cases
   - Include: happy path, edge cases, error conditions
   - Specify expected inputs and outputs

5. DEPENDENCIES:
   - List phase IDs that must complete first
   - Ensure no circular dependencies

OUTPUT FORMAT (JSON):
{{
    "mission_id": "{mission_id}",
    "metadata": {{
        "generated_at": "ISO-8601 timestamp",
        "model_used": "model-name",
        "language": "{language}",
        "estimated_phases": "number"
    }},
    "phases": [
        {{
            "phase_id": "P001",
            "phase_name": "Clear descriptive name",
            "abstraction_level": 1,
            "description": "What this phase accomplishes",
            "dependencies": [],
            "tasks": [
                {{
                    "task_id": "T001",
                    "module_name": "file_or_module_name",
                    "file_path": "src/module.ext",
                    "description": "What this module does",
                    "functions": [
                        {{
                            "function_id": "F001",
                            "name": "function_name",
                            "signature": "complete function signature with types",
                            "purpose": "Why this function exists",
                            "parameters": [
                                {{
                                    "name": "param_name",
                                    "type": "data_type",
                                    "description": "What this parameter is for"
                                }}
                            ],
                            "return_type": "return type specification",
                            "test_cases": [
                                {{
                                    "test_id": "TC001",
                                    "category": "valid_input | edge_case | error_handling",
                                    "description": "What this test verifies",
                                    "input": "test input",
                                    "expected_output": "expected result",
                                    "priority": "critical | high | medium | low"
                                }}
                            ]
                        }}
                    ]
                }}
            ]
        }}
    ]
}}

CRITICAL REQUIREMENTS:
- Return ONLY valid JSON (no markdown, no explanation)
- Every phase must have at least one task
- Every task must have at least one function
- Every function must have at least 2 test cases (happy + error)
- Dependencies must reference existing phase IDs
- Abstraction levels must increase monotonically (or stay same within parallel tracks)

Begin decomposition now.
"""


# ============================================================================
# CORE DECOMPOSER FUNCTIONS
# ============================================================================

async def decompose_task(
    task_json_path: str,
    output_path: str = "dev_plan.json",
    temperature: Optional[float] = None
) -> Dict[str, Any]:
    """
    Main entry point: Converts natural language task.json into structured dev_plan.json
    
    Args:
        task_json_path: Path to input task specification
        output_path: Where to save generated plan
        temperature: Model temperature (low for consistency)
        
    Returns:
        Dictionary containing the development plan
        
    Raises:
        FileNotFoundError: If task_json_path doesn't exist
        json.JSONDecodeError: If task.json is malformed
        ValueError: If model fails to generate valid plan
    """
    
    # Load task specification
    print(f"[Decomposer] Loading task from: {task_json_path}")
    # Locate templates relative to script or current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(script_dir, "templates")
    if not os.path.isdir(templates_dir):
        templates_dir = os.path.join(os.getcwd(), "decomposer", "templates")

    template_context, _ = load_templates(templates_dir, verbose=False)
    template_context = template_context or ""

    task_spec = {}
    try:
        with open(task_json_path, 'r') as f:
            task_spec = json.load(f) or {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load task specification: {e}")
    
    # Validate required fields
    required_fields = ["mission_id", "goal", "language"]
    for field in required_fields:
        if field not in task_spec:
            raise ValueError(f"Task specification missing required field: {field}")
    
    # Initialize LLM
    print(f"[Decomposer] Initializing model for DECOMPOSER role...")
    model = get_model(ModelRole.DECOMPOSER, temperature=temperature)
    
    # Prepare decomposition prompt
    prompt_content = DECOMPOSITION_PROMPT.format(
        task_spec=json.dumps(task_spec, indent=2),
        mission_id=task_spec["mission_id"],
        language=task_spec["language"],
        template_context=template_context
    )
    
    messages = [
        SystemMessage(content="You are a software architecture expert."),
        HumanMessage(content=prompt_content)
    ]
    
    # Call LLM for decomposition
    print(f"[Decomposer] Generating development plan...")
    response = await model.ainvoke(messages)
    
    # Parse response
    content = response.content
    if isinstance(content, list):
        # Extract and join text from all content parts
        response_text = "".join(
            [part if isinstance(part, str) else part.get("text", "") for part in content]
        ).strip()
    else:
        response_text = content.strip()
    
    # Clean JSON (remove markdown code blocks if present)
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])  # Remove first and last line
    
    try:
        dev_plan = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"[Decomposer] ERROR: Model returned invalid JSON")
        print(f"Response: {response_text[:500]}...")
        raise ValueError(f"LLM failed to generate valid JSON: {e}")
    
    # Add metadata
    dev_plan["metadata"]["generated_at"] = datetime.utcnow().isoformat() + "Z"
    dev_plan["metadata"]["model_used"] = getattr(model, "model", "unknown-model")
    
    # Validate plan structure
    _validate_plan(dev_plan)
    
    # Save to file
    print(f"[Decomposer] Saving plan to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(dev_plan, f, indent=2)
    
    # Print summary
    phase_count = len(dev_plan["phases"])
    task_count = sum(len(p["tasks"]) for p in dev_plan["phases"])
    func_count = sum(
        len(f["functions"]) 
        for p in dev_plan["phases"] 
        for t in p["tasks"] 
        for f in [t]
    )
    
    print(f"\n[Decomposer] ✓ Plan Generated")
    print(f"  Phases: {phase_count}")
    print(f"  Tasks: {task_count}")
    print(f"  Functions: {func_count}")
    print(f"  Saved: {output_path}\n")
    
    return dev_plan


def _validate_plan(plan: Dict[str, Any]) -> None:
    """
    Validates development plan structure.
    
    Raises:
        ValueError: If plan structure is invalid
    """
    
    # Check required top-level keys
    required_keys = ["mission_id", "metadata", "phases"]
    for key in required_keys:
        if key not in plan:
            raise ValueError(f"Plan missing required key: {key}")
    
    # Validate phases
    if not isinstance(plan["phases"], list) or len(plan["phases"]) == 0:
        raise ValueError("Plan must have at least one phase")
    
    phase_ids = set()
    
    for i, phase in enumerate(plan["phases"]):
        # Check phase structure
        required_phase_keys = ["phase_id", "phase_name", "abstraction_level", "tasks"]
        for key in required_phase_keys:
            if key not in phase:
                raise ValueError(f"Phase {i} missing required key: {key}")
        
        # Track phase IDs for dependency validation
        phase_ids.add(phase["phase_id"])
        
        # Validate dependencies reference existing phases
        if "dependencies" in phase:
            for dep_id in phase["dependencies"]:
                if dep_id not in phase_ids:
                    # Allow forward references if phases are ordered
                    pass  # We'll validate after all phases processed
        
        # Validate tasks
        if not phase["tasks"]:
            raise ValueError(f"Phase {phase['phase_id']} has no tasks")
        
        for task in phase["tasks"]:
            if "functions" not in task or not task["functions"]:
                raise ValueError(f"Task {task.get('task_id')} has no functions")
            
            for func in task["functions"]:
                if "test_cases" not in func or len(func["test_cases"]) < 2:
                    raise ValueError(
                        f"Function {func.get('name')} needs at least 2 test cases"
                    )
    
    print("[Decomposer] ✓ Plan validation passed")




# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Example usage of task decomposer"""
    
    
    
    if len(sys.argv) < 2:
        print("Usage: python task_decomposer.py <task.json> [output.json]")
        sys.exit(1)
    
    task_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "dev_plan.json"
    
    try:
        plan = await decompose_task(task_path, output_path=output_path)
        print(f"\n✓ Success! Development plan created at: {output_path}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
