from fastmcp import FastMCP
import sys
import os
import platform
import json
import re
import subprocess
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Create the server
mcp = FastMCP("ToyFactoryInfrastructure")

# ============================================================================
# GLOBAL WORKSPACE CONTEXT
# ============================================================================
# These are set by the factory operator BEFORE deploying any agents
_PROJECT_ROOT = None
_TEMP_DIR = None
_RUN_DIR = None

def _get_absolute_path(filename: str) -> str:
    """
    Convert filename to absolute path within PROJECT_ROOT.
    Enforces that all file operations happen in the correct workspace.
    """
    if not _PROJECT_ROOT:
        raise RuntimeError(
            "CRITICAL: PROJECT_ROOT not initialized!\n"
            "Factory operator MUST call set_workspace_context() before deploying agents."
        )
    
    # If already absolute, validate it's within project root
    if os.path.isabs(filename):
        abs_path = os.path.normpath(filename)
        # Allow files in project, temp, or run directories
        allowed_roots = [_PROJECT_ROOT, _TEMP_DIR, _RUN_DIR]
        if not any(abs_path.startswith(root) for root in allowed_roots):
            raise ValueError(
                f"Path '{filename}' is outside allowed workspace!\n"
                f"Allowed roots: {allowed_roots}"
            )
        return abs_path
    
    # Otherwise, resolve relative to PROJECT_ROOT
    return os.path.normpath(os.path.join(_PROJECT_ROOT, filename))


@mcp.tool()
def register_subgraph_callback(callback_id: str) -> str:
    """[OPERATOR ONLY] Register callback for agent deployment. Internal use only."""
    global _SUBGRAPH_CALLBACK
    _SUBGRAPH_CALLBACK = callback_id
    return "✓ Callback registered"

# ============================================================================
# ARCHITECT TOOLS
# ============================================================================

@mcp.tool()
def update_ledger(ledger_content: str) -> str:
    """[ARCHITECT ONLY] Update project ledger with completed milestones, blockers, and technical debt."""
    if not _PROJECT_ROOT:
        return "❌ Workspace not initialized"
    
    ledger_path = os.path.join(_PROJECT_ROOT, "LEDGER.md")
    with open(ledger_path, "w") as f:
        f.write(ledger_content)
    
    return "✓ Ledger updated"

# ============================================================================
# TECH LEAD TOOLS
# ============================================================================

@mcp.tool()
def update_scratchpad(content: str) -> str:
    """[TECH LEAD ONLY] Update planning scratchpad to track planned vs remaining work."""
    if not _TEMP_DIR:
        return "❌ Workspace not initialized"
    
    path = os.path.join(_TEMP_DIR, "scratchpad.md")
    with open(path, "w") as f:
        f.write(content)
    
    return "✓ Scratchpad updated"

@mcp.tool()
def set_workspace_context(project_root: str, temp_dir: str, run_dir: str) -> str:
    """
    [OPERATOR ONLY] Initialize workspace paths before deploying agents.
    This MUST be the first tool call after MCP server starts.
    """
    global _PROJECT_ROOT, _TEMP_DIR, _RUN_DIR
    
    _PROJECT_ROOT = os.path.abspath(project_root)
    _TEMP_DIR = os.path.abspath(temp_dir)
    _RUN_DIR = os.path.abspath(run_dir)
    
    # Change MCP server's CWD to project root so cargo commands work correctly
    os.chdir(_PROJECT_ROOT)
    
    sys.__stderr__.write(f"\n[MCP INIT] Workspace locked to: {_PROJECT_ROOT}\n")
    
    return (
        f"✓ Workspace initialized successfully\n\n"
        f"PROJECT_ROOT: {_PROJECT_ROOT}\n"
        f"TEMP_DIR: {_TEMP_DIR}\n"
        f"RUN_DIR: {_RUN_DIR}\n"
        f"CWD: {os.getcwd()}\n\n"
        f"All agents will now operate within this workspace."
    )

@mcp.tool()
def get_workspace_info() -> str:
    """Get current workspace configuration and usage rules."""
    if not _PROJECT_ROOT:
        return (
            "❌ ERROR: Workspace not initialized!\n"
            "The factory operator must call set_workspace_context() first."
        )
    
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 CRITICAL: READ BEFORE PROCEEDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 REAL-TIME ENVIRONMENT:
PROJECT_ROOT: {_PROJECT_ROOT} (CWD)
TEMP_DIR: {_TEMP_DIR} (For experiments)
RUN_DIR: {_RUN_DIR} (Final binaries)

⚠️ VERIFICATION MANDATE:
The structure below is a TARGET REFERENCE, not a list of existing files.
Before writing any code, you MUST run `list_files` to see what is actually on disk.
If you assume a file exists without checking, you WILL fail.

📂 ARCHITECTURAL BLUEPRINT (Target Structure):
{_PROJECT_ROOT}/
├── Cargo.toml          <-- (MUST EXIST) Main project manifest
├── .gitignore          <-- (MUST EXIST) Already hides target/
└── src/
    ├── main.rs         <-- (MUST EXIST) Entry point
    ├── lib.rs          <-- (OPTIONAL) For shared logic
    └── [module].rs     <-- (TO BE CREATED) matrix.rs, parser.rs, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛑 HARD CONSTRAINTS (Zero Tolerance):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NO NESTED PROJECTS: 
   You are already INSIDE a Cargo project. 
   ❌ NEVER run 'cargo new' or 'cargo init'.
   ❌ NEVER create a subdirectory containing a 'Cargo.toml'.
   ✅ ONLY write .rs files into 'src/'.

2. PATHING:
   ✅ Use relative paths from ROOT (e.g., 'src/matrix.rs').
   ❌ NEVER use absolute paths or 'mkdir' for project structure.

3. RUST MODULE RULES:
   To add a module (e.g., 'parser'):
   1. Create 'src/parser.rs'.
   2. Add 'mod parser;' at the top of 'src/main.rs'.
   ❌ DO NOT create a separate folder with its own Cargo.toml.

4. INFRASTRUCTURE:
   If the environment is broken (e.g., Cargo.toml is missing), 
   DO NOT try to fix it. This is a system failure. 
   Stop and report it to the Lead Architect.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR CURRENT WORKING DIRECTORY: {os.getcwd()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

@mcp.tool()
def setup_sim_env(env_name: str, packages: list[str]) -> str:
    """Creates an isolated Python venv and installs specific physics/math libraries."""
    import subprocess
    try:
        res = subprocess.run(f"python -m venv {env_name}", shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            return f"VENV ERROR: {res.stderr}"
        pkg_str = " ".join(packages)
        res_pip = subprocess.run(f"./{env_name}/bin/pip install {pkg_str}", shell=True, capture_output=True, text=True)
        return f"STDOUT: {res_pip.stdout}\nSTDERR: {res_pip.stderr}"
      
    except Exception as e:
        return f"Env Setup Failed: {str(e)}"

@mcp.tool()
def get_resource_usage() -> str:
    """Get CPU and Memory usage to determine if the system can handle a simulation."""
    import psutil
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    return f"System Load: CPU {cpu}%, RAM {mem}%. Ready for task: {cpu < 80}"


@mcp.tool()
def check_dependency(binary_name: str) -> str:
    """Check if a specific tool or library is installed and return its path."""
    import shutil
    path = shutil.which(binary_name)
    if path:
        return f"SUCCESS: {binary_name} found at {path}"
    return f"FAILED: {binary_name} is not in the system PATH."

@mcp.tool()
def run_command(command: str) -> str:
    """
    Execute shell command in PROJECT_ROOT.
    All cargo/git commands automatically run in the correct workspace.
    """
    if not _PROJECT_ROOT:
        return "❌ ERROR: Workspace not initialized. Cannot run commands."
    
    sys.__stderr__.write(f"\n[MCP] Executing in {_PROJECT_ROOT}: {command}\n")
    
    import subprocess
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT  # ALWAYS run in project root
        )
        
        sys.__stderr__.write(f"[MCP] Exit code: {res.returncode}\n")
        
        return (
            f"Command: {command}\n"
            f"Exit Code: {res.returncode}\n"
            f"─── STDOUT ───\n{res.stdout}\n"
            f"─── STDERR ───\n{res.stderr}"
        )
    except Exception as e:
        sys.__stderr__.write(f"[MCP] CRITICAL ERROR: {str(e)}\n")
        return f"❌ SYSTEM FAILURE: {str(e)}"
@mcp.tool()
def list_files(path: str = ".") -> str:
    """List files in a directory within the workspace. Use '.' for project root."""
    try:
        abs_path = _get_absolute_path(path)
        files = os.listdir(abs_path)
        
        if not files:
            return f"Directory '{path}' is empty."
        
        # Show relative paths from PROJECT_ROOT for clarity
        rel_path = os.path.relpath(abs_path, _PROJECT_ROOT) if _PROJECT_ROOT else path
        return f"Contents of '{rel_path}':\n" + "\n".join(f"  {f}" for f in sorted(files))
    except Exception as e:
        return f"Error listing '{path}': {str(e)}"

@mcp.tool()
def write_file(filename: str, content: str) -> str:
    """
    Write content to a file within the workspace.
    Use relative paths from PROJECT_ROOT (e.g., 'src/main.rs').
    """
    try:
        abs_path = _get_absolute_path(filename)
        parent_dir = os.path.dirname(abs_path)
        
        # Create parent directories if needed
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        with open(abs_path, "w") as f:
            f.write(content)
        
        # Show relative path for cleaner output
        rel_path = os.path.relpath(abs_path, _PROJECT_ROOT) if _PROJECT_ROOT else filename
        sys.__stderr__.write(f"[MCP] File written: {abs_path}\n")
        
        return f"✓ Successfully created: {rel_path}\n  Full path: {abs_path}"
    except Exception as e:
        return f"❌ Error writing '{filename}': {str(e)}"


@mcp.tool()
def get_system_status() -> str:
    """Returns the current OS and working directory of the factory."""
    return f"Factory running on {platform.system()} {platform.release()}. Path: {os.getcwd()}"

@mcp.tool()
def edit_file_replace(filename: str, old_snippet: str, new_snippet: str) -> str:
    """Replace exact text block in a file. Requires exact whitespace match."""
    try:
        abs_path = _get_absolute_path(filename)
        
        with open(abs_path, "r") as f:
            content = f.read()
        
        if old_snippet not in content:
            return (
                f"❌ ERROR: old_snippet not found in '{filename}'\n"
                f"Ensure exact whitespace/indentation match.\n"
                f"Tip: Use read_file_lines to verify the exact text."
            )
        
        new_content = content.replace(old_snippet, new_snippet, 1)
        
        with open(abs_path, "w") as f:
            f.write(new_content)
        
        rel_path = os.path.relpath(abs_path, _PROJECT_ROOT) if _PROJECT_ROOT else filename
        return f"✓ Code block updated in: {rel_path}"
    except Exception as e:
        return f"❌ Edit failed for '{filename}': {str(e)}"

@mcp.tool()
def git_checkpoint(commit_message: str) -> str:
    """
    Run 'git add .' and 'git commit' in PROJECT_ROOT.
    Call this AFTER every successful code change.
    """
    if not _PROJECT_ROOT:
        return "❌ ERROR: Workspace not initialized."
    
    import subprocess
    try:
        # Stage all changes
        subprocess.run(["git", "add", "."], cwd=_PROJECT_ROOT, check=True, capture_output=True)
        
        # Commit
        res = subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if res.returncode == 0:
            # Get commit hash
            hash_res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True
            )
            commit_hash = hash_res.stdout.strip()
            return f"✓ GIT COMMIT: {commit_message}\n  Hash: {commit_hash}"
        elif "nothing to commit" in res.stdout:
            return "⚠ GIT: No changes to commit (working tree clean)"
        else:
            return f"❌ GIT ERROR:\n{res.stderr}"
    except Exception as e:
        return f"❌ GIT FAILED: {str(e)}"

@mcp.tool()
def write_test_certificate(certificate_json: str) -> str:
    """
    [JUNIOR DEV ONLY] Write test certificate to PROJECT_ROOT.
    This file serves as proof-of-work that Tech Lead and the wrapper can read.
    Called automatically by the factory after cargo test passes.
    """
    if not _PROJECT_ROOT:
        return "❌ ERROR: Workspace not initialized."
    
    cert_path = os.path.join(_PROJECT_ROOT, "test_certificate.json")
    
    try:
        # Validate it's actual JSON
        json.loads(certificate_json)
        with open(cert_path, "w") as f:
            f.write(certificate_json)
        sys.__stderr__.write(f"[MCP] Test certificate written: {cert_path}\n")
        return f"✓ Test certificate written to test_certificate.json"
    except Exception as e:
        return f"❌ Certificate write failed: {str(e)}"

@mcp.tool()
def read_file(filename: str) -> str:
    """Read file content from the workspace. Returns last 1000 chars to save context."""
    try:
        abs_path = _get_absolute_path(filename)
        with open(abs_path, "r") as f:
            content = f.read()
        
        # Truncate if too long
        if len(content) > 1000:
            return f"[Showing last 1000 chars of '{filename}']\n...\n{content[-1000:]}"
        
        return content
    except Exception as e:
        return f"❌ Error reading '{filename}': {str(e)}"
    
@mcp.tool()
def read_file_lines(filename: str, start: int = 0, count: int = 50) -> str:
    """Read specific line range from a file in the workspace."""
    try:
        abs_path = _get_absolute_path(filename)
        with open(abs_path, "r") as f:
            lines = f.readlines()
        
        requested_lines = lines[start : start + count]
        if not requested_lines:
            return f"End of file reached (total lines: {len(lines)})"
        
        header = f"Lines {start} to {start + len(requested_lines) - 1} of '{filename}':\n"
        return header + "".join(requested_lines)
    except Exception as e:
        return f"❌ Error reading '{filename}': {str(e)}"
    


    
@mcp.tool()
def archive_task_context(task_name: str) -> str:
    """Mark a sub-task as complete and archived."""
    return f"✓ STATUS: '{task_name}' is locked and archived. Moving to next step."

if __name__ == "__main__":

   # Silence the banner and logs by directing them to stderr
   import logging
   import sys
   mcp.run(transport="stdio") # Ensure transport is explicit