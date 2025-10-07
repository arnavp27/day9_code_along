import json
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class MemoryManager:
    """Manages persistent storage of conversation history and workflow states."""
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.session_file = self.memory_dir / "sessions.json"
        self.current_session_id = None
        
    def create_session(self, question: str) -> str:
        """Create a new session and return session ID."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = session_id
        
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "states": [],
            "final_answer": None,
            "metadata": {
                "total_iterations": 0,
                "search_used": False,
                "reflection_count": 0
            }
        }
        
        self._save_session(session_data)
        print(f"📝 Created session: {session_id}")
        return session_id
    
    def log_state(self, node_name: str, state: Dict[str, Any]):
        """Log a state transition."""
        if not self.current_session_id:
            return
        
        session_data = self._load_session(self.current_session_id)
        
        state_entry = {
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "iteration": state.get("iteration_count", 0),
            "state_snapshot": {
                "has_draft": bool(state.get("draft_answer")),
                "has_evaluation": bool(state.get("evaluation")),
                "has_search": bool(state.get("search_results")),
                "reflection_count": len(state.get("reflections", [])),
                "evaluation_score": state.get("evaluation", {}).get("score")
            }
        }
        
        session_data["states"].append(state_entry)
        session_data["metadata"]["total_iterations"] = state.get("iteration_count", 0)
        session_data["metadata"]["search_used"] = bool(state.get("search_results"))
        session_data["metadata"]["reflection_count"] = len(state.get("reflections", []))
        
        self._save_session(session_data)
    
    def save_final_answer(self, answer: str, full_state: Dict[str, Any]):
        """Save the final answer and complete state."""
        if not self.current_session_id:
            return
        
        session_data = self._load_session(self.current_session_id)
        session_data["final_answer"] = answer
        session_data["complete_state"] = {
            "question": full_state.get("question"),
            "final_answer": full_state.get("final_answer"),
            "iterations": full_state.get("iteration_count"),
            "reflections": full_state.get("reflections", []),
            "evaluation": full_state.get("evaluation")
        }
        
        self._save_session(session_data)
        print(f"💾 Saved final answer to session: {self.current_session_id}")
    
    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get recent session history."""
        all_sessions = []
        
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                data = json.load(f)
                all_sessions = data.get("sessions", [])
        
        # Sort by timestamp, most recent first
        all_sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_sessions[:limit]
    
    def print_session_summary(self, session_id: str = None):
        """Print a summary of a session."""
        if session_id is None:
            session_id = self.current_session_id
        
        if not session_id:
            print("❌ No session to summarize")
            return
        
        session_data = self._load_session(session_id)
        
        print("\n" + "="*60)
        print(f"📊 SESSION SUMMARY: {session_id}")
        print("="*60)
        print(f"Question: {session_data['question']}")
        print(f"Timestamp: {session_data['timestamp']}")
        print(f"\nMetadata:")
        print(f"  - Total Iterations: {session_data['metadata']['total_iterations']}")
        print(f"  - Search Used: {session_data['metadata']['search_used']}")
        print(f"  - Reflections: {session_data['metadata']['reflection_count']}")
        
        print(f"\nWorkflow Trace ({len(session_data['states'])} states):")
        for state in session_data['states']:
            print(f"  [{state['timestamp'].split('T')[1][:8]}] {state['node']} "
                  f"(Iteration {state['iteration']})")
        
        if session_data.get("final_answer"):
            print(f"\nFinal Answer: {session_data['final_answer'][:200]}...")
        
        print("="*60 + "\n")
    
    def _save_session(self, session_data: Dict):
        """Save session data to persistent storage."""
        all_sessions = []
        
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                data = json.load(f)
                all_sessions = data.get("sessions", [])
        
        # Update or add session
        updated = False
        for i, session in enumerate(all_sessions):
            if session["session_id"] == session_data["session_id"]:
                all_sessions[i] = session_data
                updated = True
                break
        
        if not updated:
            all_sessions.append(session_data)
        
        with open(self.session_file, 'w') as f:
            json.dump({"sessions": all_sessions}, f, indent=2)
    
    def _load_session(self, session_id: str) -> Dict:
        """Load a specific session."""
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                data = json.load(f)
                for session in data.get("sessions", []):
                    if session["session_id"] == session_id:
                        return session
        
        return {
            "session_id": session_id,
            "states": [],
            "metadata": {}
        }
    
    def export_session(self, session_id: str = None, output_file: str = None):
        """Export a session to a separate JSON file."""
        if session_id is None:
            session_id = self.current_session_id
        
        session_data = self._load_session(session_id)
        
        if output_file is None:
            output_file = self.memory_dir / f"session_{session_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"📤 Exported session to: {output_file}")
        return output_file