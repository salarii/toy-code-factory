"""
Contract Generator - Creates formal contracts from development phases

This module takes phases/tasks from dev_plan.json and generates detailed
contracts with function signatures, parameter specs, and test requirements.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from model_resolver import get_model, ModelRole
from decomposer.template_loader import load_templates

# ============================================================================
# CONTRACT GENERATION PROMPT
# ============================================================================

CONTRACT_GENERATION_PROMPT = """You are a technical specification expert creating formal development contracts.

YOUR TASK:
Convert the provided phase/task into a precise, testable contract that developers can follow exactly.

INPUT PHASE:
{phase_json}

LANGUAGE: {language}

{template_context}

CONTRACT REQUIREMENTS:

1. MODULE SPECIFICATION:
   - Propose clear module/file name
   - Define exact file path
   - State module purpose in 1-2 sentences
   - List all public interfaces

2. FUNCTION SIGNATURES:
   - Use language-specific conventions ({language})
   - Include complete type annotations
   - Specify all parameters with types and descriptions
   - Define return types (including error types)
   - List all possible error cases

3. PARAMETER VALIDATION:
   - For each parameter, specify constraints
   - Provide example valid values
   - List invalid values that should be rejected

4. TEST SPECIFICATIONS:
   - Categorize tests: valid_input, edge_cases, error_handling, integration
   - Each test needs: name, input, expected output, priority
   - Cover ALL code paths
   - Include performance tests for complex operations

5. ACCEPTANCE CRITERIA:
   - Concrete, measurable completion criteria
   - Code quality requirements (linting, documentation)
   - Performance benchmarks if applicable

OUTPUT FORMAT (JSON):
{{
    "contract_id": "C-{phase_id}-{task_id}",
    "phase_id": "{phase_id}",
    "task_id": "{task_id}",
    "templates_used": bool(template_context),
    "generated_at": "ISO-8601 timestamp",
    "module_spec": {{
        "module_name": "name",
        "file_path": "path/to/file.ext",
        "purpose": "What this module does",
        "dependencies": ["list", "of", "imports"],
        "public_interface": [
            {{
                "function_name": "name",
                "signature": "complete signature",
                "parameters": [
                    {{
                        "name": "param",
                        "type": "type",
                        "description": "purpose",
                        "validation": "constraints",
                        "examples_valid": ["val1", "val2"],
                        "examples_invalid": ["bad1", "bad2"]
                    }}
                ],
                "return_type": "return type",
                "return_description": "what it returns",
                "error_cases": [
                    {{
                        "condition": "when this happens",
                        "error_type": "ErrorType",
                        "message": "error message"
                    }}
                ],
                "complexity": "O(n) or description"
            }}
        ]
    }},
    "test_specifications": [
        {{
            "test_category": "valid_input | edge_case | error_handling | integration | performance",
            "test_cases": [
                {{
                    "test_name": "descriptive_name",
                    "description": "what this verifies",
                    "setup": "test setup steps",
                    "input": "test input",
                    "expected_output": "expected result",
                    "assertions": ["list", "of", "checks"],
                    "priority": "critical | high | medium | low"
                }}
            ]
        }}
    ],
    "acceptance_criteria": [
        "Criterion 1",
        "Criterion 2"
    ],
    "estimated_complexity": "simple | moderate | complex",
    "estimated_time": "time estimate"
}}

LANGUAGE-SPECIFIC CONVENTIONS:

Rust:
- Use Result<T, E> for fallible functions
- Prefer &str over String for parameters
- Use Option<T> for nullable returns
- Include lifetime annotations if needed
- Follow ownership rules in signatures

Python:
- Use type hints (Optional[T], List[T], Dict[K, V])
- Specify exception types in error cases
- Use dataclasses/TypedDict for complex types
- Include docstrings format in contract

JavaScript/TypeScript:
- Use TypeScript types for all signatures
- Specify Promise<T> for async functions
- Include interface definitions
- Use union types for alternatives

CRITICAL: Return ONLY valid JSON, no markdown, no explanation.
"""


# ============================================================================
# CORE CONTRACT GENERATION
# ============================================================================

async def generate_phase_contract(
    phase: Dict[str, Any],
    language: str,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Generate detailed contract for a development phase.
    
    Args:
        phase: Phase dictionary from dev_plan.json
        language: Programming language (for conventions)
        model_name: LLM model to use
        temperature: Model temperature (0 for deterministic)
        
    Returns:
        Phase contract containing multiple task contracts
    """
    # Load templates to get the required context string
    script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(script_dir, "templates")
    template_context, _ = load_templates(templates_dir, verbose=False)
    template_context = template_context or ""
    print(f"[ContractGen] Generating contract for phase: {phase['phase_id']}")
    
    model = get_model(ModelRole.DECOMPOSER)
    
    # Generate contract for each task in phase
    contracts = []
    
    for task in phase["tasks"]:
        prompt_content = CONTRACT_GENERATION_PROMPT.format(
            phase_json=json.dumps(task, indent=2),
            language=language,
            phase_id=phase["phase_id"],
            task_id=task["task_id"],
            template_context=template_context
        )
        
        messages = [
            SystemMessage(content="You are a technical specification expert."),
            HumanMessage(content=prompt_content)
        ]
        
        response = await model.ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            # Extract and join text from all content parts
            response_text = "".join(
                [part if isinstance(part, str) else part.get("text", "") for part in content]
            ).strip()
        else:
            response_text = content.strip()
        
        # Clean JSON
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])
        
        try:
            contract = json.loads(response_text)
            contract["generated_at"] = datetime.utcnow().isoformat() + "Z"
            
            # Add phase completion tracking fields
            contract["phase_done"] = False
            contract["phase_done_by"] = None
            contract["phase_done_at"] = None
            contract["reopen_history"] = []

            # Ensure contract has required keys for validation
            if "contract_id" not in contract:
                contract["contract_id"] = f"C-{phase['phase_id']}-{task['task_id']}"
           
            contracts.append(contract)
            
            print(f"  ✓ Contract generated for task: {task['task_id']}")
            
        except json.JSONDecodeError as e:
            print(f"  ✗ Failed to parse contract for task: {task['task_id']}")
            raise ValueError(f"Invalid JSON from LLM: {e}")
    
    # Combine into phase contract
    phase_contract = {
        "phase_id": phase["phase_id"],
        "phase_name": phase["phase_name"],
        "contracts": contracts,
        "templates_used": bool(template_context),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "phase_done": False,
        "phase_done_by": None,
        "phase_done_at": None,
        "reopen_history": []
    }
    
    return phase_contract




# ============================================================================
# CONTRACT STORAGE
# ============================================================================

class ContractStore:
    """
    Manages storage and retrieval of development contracts.
    """
    
    def __init__(self, contracts_dir: str = "contracts"):
        self.contracts_dir = contracts_dir
        os.makedirs(contracts_dir, exist_ok=True)
    
    def save_contract(self, contract: Dict[str, Any], contract_id: str):
        """Save contract to file"""
        filepath = f"{self.contracts_dir}/{contract_id}.json"
        with open(filepath, 'w') as f:
            json.dump(contract, f, indent=2)
        print(f"[ContractStore] Saved: {filepath}")
    
    def load_contract(self, contract_id: str) -> Dict[str, Any]:
        """Load contract from file"""
        filepath = f"{self.contracts_dir}/{contract_id}.json"
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_contracts(self) -> List[str]:
        """List all stored contract IDs"""
        return [
            f.replace(".json", "") 
            for f in os.listdir(self.contracts_dir) 
            if f.endswith(".json")
        ]

 
# ============================================================================
# CONTRACT VALIDATION
# ============================================================================

def validate_contract_completeness(contract: Dict[str, Any]) -> List[str]:
    """
    Validates contract has all required elements.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ["contract_id", "module_spec", "test_specifications", "acceptance_criteria", 
                    "phase_done", "phase_done_by", "phase_done_at", "reopen_history"]
     
    for key in required_keys:
        if key not in contract:
            errors.append(f"Missing required key: {key}")
    
    # Validate phase completion field types
    if "phase_done" in contract and not isinstance(contract["phase_done"], bool):
        errors.append("phase_done must be boolean")
    
    if "reopen_history" in contract and not isinstance(contract["reopen_history"], list):
        errors.append("reopen_history must be list")

    # Validate module spec
    if "module_spec" in contract:
        module = contract["module_spec"]
        if "public_interface" not in module or not module["public_interface"]:
            errors.append("Module spec missing public_interface")
        
        for func in module.get("public_interface", []):
            if "signature" not in func:
                errors.append(f"Function {func.get('function_name')} missing signature")
            if "parameters" not in func:
                errors.append(f"Function {func.get('function_name')} missing parameters")
            if "return_type" not in func:
                errors.append(f"Function {func.get('function_name')} missing return_type")
    
    # Validate test specifications
    if "test_specifications" in contract:
        if not contract["test_specifications"]:
            errors.append("No test specifications provided")
        
        for test_cat in contract["test_specifications"]:
            if "test_cases" not in test_cat or not test_cat["test_cases"]:
                errors.append(f"Test category {test_cat.get('test_category')} has no test cases")
    
    # Validate acceptance criteria
    if "acceptance_criteria" in contract:
        if not contract["acceptance_criteria"]:
            errors.append("No acceptance criteria defined")
    
    return errors


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Example CLI usage"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python contract_generator.py <dev_plan.json> <phase_id> <language>")
        print("  python contract_generator.py <dev_plan.json> all <language>")
        sys.exit(1)
    
    plan_path = sys.argv[1]
    phase_id = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "python"
    
    # Load plan
    with open(plan_path, 'r') as f:
        plan = json.load(f)
    
    store = ContractStore()
    
    if phase_id == "all":
        # Generate contracts for all phases
        for phase in plan["phases"]:
            contract = await generate_phase_contract(phase, language)
            store.save_contract(contract, f"phase_{phase['phase_id']}")
            print(f"✓ Generated contract for phase: {phase['phase_id']}")
    else:
        # Generate contract for specific phase
        phase = next((p for p in plan["phases"] if p["phase_id"] == phase_id), None)
        if not phase:
            print(f"Error: Phase {phase_id} not found")
            sys.exit(1)
        
        contract = await generate_phase_contract(phase, language)
        store.save_contract(contract, f"phase_{phase_id}")
        
        # Validate
        errors = validate_contract_completeness(contract)
        if errors:
            print("\n⚠ Validation warnings:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\n✓ Contract validation passed")


if __name__ == "__main__":
    asyncio.run(main())