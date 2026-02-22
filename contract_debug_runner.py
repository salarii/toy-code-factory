import asyncio
import os
import json
import shutil


from contract_integration import (
    initialize_contracts_for_run, 
    ContractManager, 
    format_phase_as_implementation_guide
)

async def run_debugger():
    # 1. Setup a dummy workspace for the debugger
    workspace_dir = "./debug_workspace"
    
    #shutil.rmtree(workspace_dir, ignore_errors=True) # Comment this line out to persist the workspace between runs.

    os.makedirs(workspace_dir, exist_ok=True)
    
    task_json_path = os.path.join(workspace_dir, "task.json")
    if not os.path.exists(task_json_path):
        # Provide the strict, full-format task definition
        task_content = {
            "mission_id": "dump_arg",
            "mode": "continue",
            "language": "rust",
            "project_name": "dump_arg",
            "goal": "Implement a Rust CLI tool to parse arguments out of string",
            "requirements": [
                "A CLI tool that parses numerical input arguments (e.g., '4 6') and persists them into a local file called arg.txt."
            ],
            "usage_example": "parse_app  4.3  1.1 garbage1 garbage2 3.4",
            "target": "rust"
        }
        
        with open(task_json_path, "w") as f:
            # indent=4 makes the dumped file readable during manual inspection
            json.dump(task_content, f, indent=4)

    print("\n=== STEP 1: Debugging Initialization & Plan Generation ===")
    # Place a breakpoint on the line below to step into initialize_contracts_for_run
    manager = await initialize_contracts_for_run(
        workspace_base=workspace_dir,
        task_json_path=task_json_path,
        language="rust",
        generate_all=False  # Set to False so we can test JIT generation later
    )

    print("\n=== STEP 2: Debugging JIT Contract Generation (Writing) ===")
    # Place a breakpoint on the line below to step into ensure_phase_contract
    # Ensure "P001" matches a phase_id generated in your dev_plan.json
    await manager.ensure_phase_contract("P001", "rust")

    print("\n=== STEP 3: Debugging Contract Reading & Formatting ===")
    # Place a breakpoint on the line below to step into get_current_phase_info
    phase_info = manager.get_current_phase_info()
    
    if phase_info:
        # Place a breakpoint on the line below to step into the formatter
        formatted_guide = format_phase_as_implementation_guide(phase_info)
        print("\nSuccessfully formatted guide length:", len(formatted_guide))
    else:
        print("\nNo current phase info found.")


    print("\n=== STEP 4: Debugging Status Transitions ===")
    # 4a. Mark P001 as finished
    manager.update_phase_status("P001", "completed")
    print(f"Status of P001 after marking finished: {manager.get_phase_status('P001')}")
    
    # 4b. Mark P001 as unfinished (rolling back)
    manager.update_phase_status("P001", "pending")
    print(f"Status of P001 after marking unfinished: {manager.get_phase_status('P001')}")
    
    # 4c. Mark P001 as finished again
    manager.update_phase_status("P001", "completed")
    print(f"Status of P001 after marking finished again: {manager.get_phase_status('P001')}")

    print("\n=== STEP 5: Triggering Next Phase Generation ===")
    # Because P001 is now "completed", get_current_phase_info() should automatically 
    # resolve to the next incomplete phase (e.g., P002) based on your dev_plan.json.
    next_phase_info = manager.get_current_phase_info()
    
    if next_phase_info and next_phase_info.get("phase"):
        next_phase_id = next_phase_info["phase"]["phase_id"]
        print(f"\nNext phase identified as: {next_phase_id}")
        print(f"Triggering JIT generation for {next_phase_id}...")
        
        # Place a breakpoint here to step into the JIT generation of the NEXT phase
        await manager.ensure_phase_contract(next_phase_id, "rust")
        print(f"✅ Successfully generated/verified contract for {next_phase_id}!")
    else:
        print("\nNo next phase found. (Did the dev_plan.json only have 1 phase?)")


if __name__ == "__main__":
    asyncio.run(run_debugger())