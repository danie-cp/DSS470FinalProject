# session_persistence.py
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class SessionPersistence:
   """Handles saving and loading sessions with token data."""
  
   def __init__(self, sessions_dir: str = None):
       """Initialize persistence manager."""
       if sessions_dir is None:
           sessions_dir = Path(__file__).resolve().parent / "learner_sessions"
       self.sessions_dir = Path(sessions_dir)
       self.sessions_dir.mkdir(parents=True, exist_ok=True)
  
   def save_session(
       self,
       user_id: str,
       session_data: Dict,
       profile_data: Dict,
       token_data: Dict
   ) -> str:
       """Save a complete session with profile and token data."""
       session_file = self.sessions_dir / f"{user_id}_session.json"
      
       complete_session = {
           "user_id": user_id,
           "saved_at": datetime.now().isoformat(),
           "session": session_data,
           "profile": profile_data,
           "tokens": token_data
       }
      
       with open(session_file, 'w') as f:
           json.dump(complete_session, f, indent=2, default=str)
      
       return str(session_file)
  
   def load_session(self, user_id: str) -> Dict:
       """Load a saved session."""
       session_file = self.sessions_dir / f"{user_id}_session.json"
      
       if not session_file.exists():
           return {}
      
       with open(session_file, 'r') as f:
           return json.load(f)
  
   def get_all_sessions(self) -> List[Dict]:
       """Get all saved sessions."""
       sessions = []
       for session_file in self.sessions_dir.glob("*_session.json"):
           with open(session_file, 'r') as f:
               sessions.append(json.load(f))
       return sessions
  
   def delete_session(self, user_id: str) -> bool:
       """Delete a saved session."""
       session_file = self.sessions_dir / f"{user_id}_session.json"
       if session_file.exists():
           session_file.unlink()
           return True
       return False

