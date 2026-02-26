"""
Plan Manager - Manages development plan lifecycle and phase consumption

This module handles loading, tracking progress, and consuming phases
from dev_plan.json one-by-one.
"""

import json
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
import os

class DevelopmentPlan:
    """
    Manages dev_plan.json lifecycle with phase consumption tracking.
    
    Key operations:
    - Load plan from JSON
    - Track which phases are complete
    - Get current (next unprocessed) phase
    - Consume phase (mark complete and move to next)
    - Save progress state
    """
    
    # Canonical status for "finished" and all legacy synonyms we accept
    DONE_STATUS = "done"
    DONE_ALIASES = frozenset({"done", "completed"})

    def _is_done(self, phase_id: str) -> bool:
        """Return True if phase_id is marked as finished (accepts 'done' or 'completed')."""
        return self.progress.get("phase_status", {}).get(phase_id, "") in self.DONE_ALIASES

    def __init__(self, plan_path: str, progress_path: Optional[str] = None):
        """
        Initialize plan manager.
        
        Args:
            plan_path: Path to dev_plan.json
            progress_path: Path to progress tracking file (default: dev_progress.json)
            contracts_dir: Directory where contract files are stored (default: contracts)
        """
        self.plan_path = Path(plan_path)
        self.progress_path = Path(progress_path or (self.plan_path.parent / "dev_progress.json"))
        self.contracts_dir = self.plan_path.parent / "contracts"
        
        # Load plan
        with open(self.plan_path, 'r') as f:
            self.plan = json.load(f)
        
        # Load or initialize progress
        if self.progress_path.exists():
            with open(self.progress_path, 'r') as f:
                self.progress = json.load(f)
            print(f"[PlanManager] Loaded progress from: {self.progress_path}")
        else:
            self.progress = self._initialize_progress()
            print(f"[PlanManager] Initialized new progress tracker")
        
        self._validate_consistency()
    
    def _initialize_progress(self) -> Dict[str, Any]:
        """Create new progress tracking structure.
        
        phase_status is the SINGLE source of truth for phase completion.
        Derived views (completed list, remaining list, percentage) are
        computed on-the-fly via helper properties — not stored.
        """
        return {
            "mission_id": self.plan["mission_id"],
            "started_at": datetime.utcnow().isoformat() + "Z",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "phase_history": [],
            "phase_status": {p["phase_id"]: p.get("status", "") for p in self.plan["phases"]}
        }

    # ------------------------------------------------------------------
    # Derived properties (computed from phase_status, never stored)
    # ------------------------------------------------------------------

    @property
    def phases_completed(self) -> List[str]:
        """Phase IDs with status 'completed' — derived from phase_status."""
        return [pid for pid in self.progress.get("phase_status", {})
                if self._is_done(pid)]

    @property
    def phases_remaining(self) -> List[str]:
        """Phase IDs not yet completed — derived from phase_status."""
        return [pid for pid in self.progress.get("phase_status", {})
                if not self._is_done(pid)]

    @property
    def completion_percentage(self) -> float:
        """Percentage complete — derived from phase_status."""
        total = len(self.plan.get("phases", []))
        if total == 0:
            return 0.0
        return (len(self.phases_completed) / total) * 100

    @property
    def current_phase_id(self) -> Optional[str]:
        """ID of first non-completed phase in plan order, or None."""
        for phase in self.plan.get("phases", []):
            pid = phase["phase_id"]
            if self.progress.get("phase_status", {}).get(pid) not in self.DONE_ALIASES:
                return pid
        return None
    
    def _load_progress(self):
        """Re-read progress from disk so we pick up any external changes."""
        if self.progress_path.exists():
            with open(self.progress_path, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = self._initialize_progress()

    def _validate_consistency(self):
        """Ensure progress file matches plan file"""
        if self.progress["mission_id"] != self.plan["mission_id"]:
            raise ValueError(
                f"Mission ID mismatch: "
                f"plan={self.plan['mission_id']}, "
                f"progress={self.progress['mission_id']}"
            )
    

    def get_current_phase(self) -> Optional[Dict[str, Any]]:
        """
        Returns the first phase not marked 'completed' in phase_status.
        phase_status is the single source of truth.
        
        Returns:
            Phase dictionary or None if all phases complete
        """
        self._load_progress()
        for phase in self.plan["phases"]:
            phase_id = phase["phase_id"]
            if not self._is_done(phase_id):
                print(f"[PlanManager] Current phase: {phase_id} - {phase['phase_name']}")
                return phase
                
        print("[PlanManager] All phases completed!")
        return None

    def architect_mark_phase_done(
        self, 
        phase_id: str, 
        actor: str = "architect"
    ) -> Optional[Dict[str, Any]]:
        """
        Mark a phase as complete. Only callable by Architect persona.
        
        Args:
            phase_id: ID of phase to mark complete
            actor: Role performing action (must be "architect")
            
        Returns:
            Next phase dict or None if all complete
            
        Raises:
            PermissionError: If actor is not "architect"
            ValueError: If phase_id not found or already done
            FileNotFoundError: If contract file doesn't exist
        """
        # Validate actor
        if actor != "architect":
            raise PermissionError(f"Only Architect can mark phases done. Actor '{actor}' is not authorized.")
        
        # Validate phase exists
        _ = self.get_phase_by_id(phase_id)
        
        # Load contract file
        contract_path = self.contracts_dir / f"phase_{phase_id}.json"
        if not contract_path.exists():
            raise FileNotFoundError(f"Contract file not found: {contract_path}")
        
        # Check if already done (phase_status is single source of truth)
        if self._is_done(phase_id):
            raise ValueError(f"Phase {phase_id} is already marked done. Use architect_reopen_phase() to reopen it first.")
        
        # Mark complete
        completed_at = datetime.utcnow().isoformat() + "Z"
        
        # Update phase_status (single source of truth)
        if "phase_status" not in self.progress:
            self.progress["phase_status"] = {}
        self.progress["phase_status"][phase_id] = self.DONE_STATUS
        self.progress["last_updated"] = completed_at

        # Add to history (audit trail)
        self.progress["phase_history"].append({
            "phase_id": phase_id,
            "completed_at": completed_at,
            "marked_by": actor
        })
        
        # Get next phase (derived from phase_status)
        next_phase = self.get_current_phase()
        
        # Save progress
        self.save_progress()
        
        print(f"[PlanManager] ✓ Phase {phase_id} marked done by {actor} at {completed_at}")
        print(f"[PlanManager] Progress: {self.completion_percentage:.1f}%")
        
        return next_phase
    
    def architect_reopen_phase(
        self, 
        phase_id: str, 
        reason: str,
        requested_by: str = "expert",
        actor: str = "architect"
    ) -> Dict[str, Any]:
        """
        Reopen a completed phase for rework. Only callable by Architect.
        
        Args:
            phase_id: Phase to reopen (e.g., "P001")
            reason: Why phase is being reopened (required)
            requested_by: Who requested reopening ("expert", "techlead", "architect")
            actor: Role performing action (must be "architect")
            
        Returns:
            Reopened phase dictionary
            
        Raises:
            PermissionError: If actor is not "architect"
            ValueError: If phase not currently done, or reason is empty
        """
        # Validate actor
        if actor != "architect":
            raise PermissionError(f"Only Architect can reopen phases. Actor '{actor}' is not authorized.")
        
        # Validate reason
        if not reason.strip():
            raise ValueError("Reason required for reopening")
        
        # Validate phase exists
        phase = self.get_phase_by_id(phase_id)
        
        # Load contract file
        contract_path = self.contracts_dir / f"phase_{phase_id}.json"
        if not contract_path.exists():
            raise FileNotFoundError(f"Contract file not found: {contract_path}")
        
        # Check if currently done (phase_status is single source of truth)
        if not self._is_done(phase_id):
            raise ValueError(f"Phase {phase_id} is not marked done. Cannot reopen a phase that isn't closed.")
        
        # Create reopen event
        reopened_at = datetime.utcnow().isoformat() + "Z"
        reopen_event = {
            "reopened_at": reopened_at,
            "reopened_by": actor,
            "reason": reason,
            "requested_by": requested_by,
            "phase_id": phase_id
        }
        
        if "reopen_history" not in self.progress:
            self.progress["reopen_history"] = {}
        if phase_id not in self.progress["reopen_history"]:
            self.progress["reopen_history"][phase_id] = []
            
        self.progress["reopen_history"][phase_id].append(reopen_event)
        
        # Update phase_status (single source of truth)
        if "phase_status" not in self.progress:
            self.progress["phase_status"] = {}
        self.progress["phase_status"][phase_id] = "reopened"
        self.progress["last_updated"] = reopened_at
        
        # Save progress
        self.save_progress()
        
        print(f"[PlanManager] ⚠ Phase {phase_id} reopened by {actor} at {reopened_at}")
        print(f"[PlanManager] Reason: {reason}")
        print(f"[PlanManager] Requested by: {requested_by}")
        
        return phase
    
    def get_phase_done_status(self, phase_id: str) -> Dict[str, Any]:
        """
        Read completion status of a specific phase from contract file.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Dictionary with keys:
            - phase_done: bool
            - phase_done_by: str or None
            - phase_done_at: str (ISO-8601) or None
            - reopen_count: int (len of reopen_history)
            
        Raises:
            FileNotFoundError: If phase contract doesn't exist
        """
        is_done = self._is_done(phase_id)
        
        done_by = None
        done_at = None
        if is_done:
            for event in reversed(self.progress.get("phase_history", [])):
                if event["phase_id"] == phase_id and "marked_by" in event:
                    done_by = event["marked_by"]
                    done_at = event["completed_at"]
                    break
                    
        return {
            "phase_done": is_done,
            "phase_done_by": done_by,
            "phase_done_at": done_at,
            "reopen_count": len(self.progress.get("reopen_history", {}).get(phase_id, []))
        }
    
    def consume_phase(self, phase_id: str) -> Optional[Dict[str, Any]]:
        """
        DEPRECATED: Use architect_mark_phase_done() instead.
        
        This method is kept for backward compatibility but should not be used.
        """
        print("[PlanManager] WARNING: consume_phase() is deprecated. Use architect_mark_phase_done() instead.")
        return self.architect_mark_phase_done(phase_id, actor="architect")
    
    def get_phase_by_id(self, phase_id: str) -> Dict[str, Any]:
        """
        Retrieve specific phase by ID.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Phase dictionary
            
        Raises:
            ValueError: If phase_id not found
        """
        for phase in self.plan["phases"]:
            if phase["phase_id"] == phase_id:
                return phase
        
        raise ValueError(f"Phase {phase_id} not found in plan")
    
    
    def get_phase_status(self, phase_id: str) -> str:
        """Get current status of a phase."""
        if "phase_status" not in self.progress:
            self.progress["phase_status"] = {}
        return self.progress["phase_status"].get(phase_id, "")

    def update_phase_status(self, phase_id: str, status: str) -> None:
        """Update status of a specific phase and auto-save."""
        _ = self.get_phase_by_id(phase_id)  # Validate phase exists
        if "phase_status" not in self.progress:
            self.progress["phase_status"] = {}
        old_status = self.progress["phase_status"].get(phase_id, "")
        self.progress["phase_status"][phase_id] = status
        self.progress["last_updated"] = datetime.utcnow().isoformat() + "Z"
        print(f"[PlanManager] Phase {phase_id} status: '{old_status}' → '{status}'")
        self.save_progress()

    def get_all_phase_statuses(self) -> Dict[str, str]:
        """Get status map for all phases."""
        if "phase_status" not in self.progress:
            self.progress["phase_status"] = {}
        return dict(self.progress["phase_status"])

    
    
    def save_progress(self, custom_path: Optional[str] = None):
        """
        Persist current progress to disk.
        
        Args:
            custom_path: Optional override for save location
        """
        save_path = Path(custom_path) if custom_path else self.progress_path
        
        with open(save_path, 'w') as f:
            json.dump(self.progress, f, indent=2)
            
    def print_status(self):
        """Pretty-print current status"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("DEVELOPMENT PLAN STATUS")
        print("="*60)
        print(f"Mission: {summary['mission_id']}")
        print(f"Progress: {summary['completion_percentage']:.1f}% complete")
        print(f"Phases: {summary['completed_phases']}/{summary['total_phases']} done")
        print(f"Current: {summary['current_phase'] or 'COMPLETE'}")
        print(f"Started: {summary['started_at']}")
        print(f"Updated: {summary['last_updated']}")
        print("="*60 + "\n")
        
        # Show phases with status from contract files
        print("PHASE STATUS (from contract files):\n")
        for phase in self.plan["phases"]:
            phase_id = phase["phase_id"]
            contract_path = self.contracts_dir / f"phase_{phase_id}.json"
            
            if not contract_path.exists():
                print(f" ○ {phase_id}: {phase['phase_name']} - [No contract yet]")
            else:
                try:
                    status_info = self.get_phase_done_status(phase_id)
                    if status_info["phase_done"]:
                        done_at = status_info['phase_done_at'][:19] if status_info['phase_done_at'] else 'unknown'
                        reopen_note = f" (reopened {status_info['reopen_count']}x)" if status_info['reopen_count'] > 0 else ""
                        print(f" ✓ {phase_id}: {phase['phase_name']} - [Done at {done_at}]{reopen_note}")
                    else:
                        print(f" ○ {phase_id}: {phase['phase_name']} - [In progress]")
                except Exception as e:
                    print(f" ? {phase_id}: {phase['phase_name']} - [Error: {e}]")
        
        print()
        
        # Show reopen history if any phases have been reopened
        print("REOPEN HISTORY:\n")
        has_reopens = False
        for phase in self.plan["phases"]:
            phase_id = phase["phase_id"]
            reopen_history = self.progress.get("reopen_history", {}).get(phase_id, [])
            if reopen_history:
                    has_reopens = True
                    print(f" {phase_id}: {phase['phase_name']}")
                    for event in reopen_history:
                        print(f"   - {event['reopened_at'][:19]}: {event['reason']}")
                        print(f"     Requested by: {event.get('requested_by', 'unknown')}")
        
        if not has_reopens:
            print(" (No phases have been reopened)")
        print()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary dict for status display"""
        return {
            "mission_id": self.plan["mission_id"],
            "completion_percentage": self.completion_percentage,
            "completed_phases": len(self.phases_completed),
            "total_phases": len(self.plan["phases"]),
            "current_phase": self.current_phase_id,
            "started_at": self.progress["started_at"],
            "last_updated": self.progress["last_updated"]
        }

if __name__ == "__main__":
    # CLI interface
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plan_manager.py <command> [args]")
        print("Commands:")
        print("  status              - Show full status")
        print("  next                - Get next phase to work on")
        print("  complete <phase_id> - Mark phase as complete (DEPRECATED)")
        print("  reset               - Reset all progress")
        print("  get-status <id>     - Get specific phase status")
        print("  set-status <id> <s> - Set specific phase status")
        print("  list-statuses       - List all phases and statuses")
        sys.exit(1)
        
    manager = DevelopmentPlan("dev_plan.json")
    command = sys.argv[1]
    
    if command == "status":
        manager.print_status()
    
    elif command == "next":
        phase = manager.get_current_phase()
        if phase:
            print(f"Next phase: {phase['phase_id']}")
            print(f"Name: {phase['phase_name']}")
            print(f"Description: {phase['description']}")
        else:
            print("All phases complete!")
    
    elif command == "complete":
        if len(sys.argv) < 3:
            print("Error: Specify phase_id to complete")
            sys.exit(1)
        phase_id = sys.argv[2]
        next_phase = manager.consume_phase(phase_id)
        print(f"\nNext phase: {next_phase['phase_id'] if next_phase else 'NONE'}")
    
    elif command == "reset":
        confirm = input("Reset progress? This cannot be undone. (yes/no): ")
        if confirm.lower() == "yes":
            # This requires implementing reset logic or deleting the file manually
            # For now just printing a warning as it wasn't in the provided snippets
            print("Reset functionality not fully implemented in this snippet.")
        else:
            print("Cancelled")
    
    elif command == "get-status":
        if len(sys.argv) < 3:
            print("Error: Specify phase_id")
            sys.exit(1)
        phase_id = sys.argv[2]
        print(f"Phase {phase_id} status: '{manager.get_phase_status(phase_id)}'")

    elif command == "set-status":
        if len(sys.argv) < 4:
            print("Error: Specify phase_id and status")
            sys.exit(1)
        manager.update_phase_status(sys.argv[2], sys.argv[3])
        print(f"✓ Phase {sys.argv[2]} status updated to '{sys.argv[3]}'")

    elif command == "list-statuses":
        statuses = manager.get_all_phase_statuses()
        print("\nPhase Statuses:\n" + "="*60)
        for pid, status in statuses.items():
            print(f"{pid}: {status}")
        print("="*60)
    
    else:
        print(f"Unknown command: {command}")