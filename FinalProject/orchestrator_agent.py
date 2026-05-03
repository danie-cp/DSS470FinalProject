"""
Orchestrator Agent Module

This module defines the main orchestration agent that coordinates the entire
learning system. It receives user input and intelligently routes to the
appropriate specialized agents based on the learner's current stage:

1. ASSESSMENT STAGE: Route to LevelPlacementAgent
   - Conduct initial Python code assessment
   - Determine starting proficiency level
2. TRAINING STAGE: Route to ErrorTrainingAgent
   - Generate personalized lessons
   - Adapt to student's proficiency level
   - Provide targeted error correction
3. PROFILE UPDATE STAGE: Route to UpdateProfileAgent
   - Analyze lesson performance
   - Update learner profile
   - Track progress and recommend next steps

The orchestrator is the single point of entry for users and manages the
entire learning session lifecycle, providing friendly guidance and seeking
appropriate input at each stage.

Session Stages:
- NEW: User just started
- ASSESSED: Level determined, ready for training
- TRAINING: Working through error lessons
- PROFILE_UPDATED: Session complete, ready for next cycle
- CONTINUOUS: Ongoing learning sessions

TOKEN TRACKING:
The orchestrator maintains a token ledger throughout the session and includes
token usage statistics in the final session summary provided by UpdateProfileAgent.
"""
from session_persistence import SessionPersistence  
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
from abc import ABC

from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

class SessionStage(Enum):
  """Enumeration of learning session stages."""
  NEW = "new"  # User just started
  GATHERING_NAME = "gathering_name"  # Asking user for their name
  ASSESSED = "assessed"  # Level placement complete
  TRAINING = "training"  # In error training lesson
  PROFILE_UPDATED = "profile_updated"  # Session complete
  CONTINUOUS = "continuous"  # Ongoing learning
  FOLLOWUP_TASK = "followup_task"  # Working on follow-up task

class RoutingDecision(Enum):
  """Agent routing decisions."""
  LEVEL_PLACEMENT = "level_placement"
  ERROR_TRAINING = "error_training"
  UPDATE_PROFILE = "update_profile"
  GATHER_INFO = "gather_info"
  COMPLETE_SESSION = "complete_session"
  SKILL_BUILDING = "skill_building"
  EVALUATE_CORRECTION = "evaluate_correction"
  FOLLOWUP_TASK = "followup_task"  # Handle follow-up task responses
  PYTHON_EXPLANATION = "python_explanation"  # Python concept explanation

class TokenLedger:
  """
  Tracks token usage across the learning session.
  
  Aggregates token counts from all specialized agents and maintains
  a running total throughout the session lifecycle.
  """
  
  def __init__(self):
      """Initialize the token ledger."""
      self.total_tokens: int = 0
      self.prompt_tokens: int = 0
      self.completion_tokens: int = 0
      self.entries: List[Dict] = []
      self.start_time: str = datetime.now().isoformat()
  
  def record_token_usage(
      self,
      agent_name: str,
      prompt_tokens: int = 0,
      completion_tokens: int = 0,
      activity_description: str = ""
  ) -> None:
      """
      Record token usage from an agent call.
      
      Args:
          agent_name (str): Name of the agent that used tokens
          prompt_tokens (int): Tokens used in the prompt
          completion_tokens (int): Tokens used in the completion
          activity_description (str): Description of the activity
      """
      total = prompt_tokens + completion_tokens
      
      entry = {
          "timestamp": datetime.now().isoformat(),
          "agent": agent_name,
          "prompt_tokens": prompt_tokens,
          "completion_tokens": completion_tokens,
          "total_tokens": total,
          "activity": activity_description
      }
      
      self.entries.append(entry)
      self.total_tokens += total
      self.prompt_tokens += prompt_tokens
      self.completion_tokens += completion_tokens
  
  def get_summary(self) -> Dict:
      """
      Get a summary of token usage for the session.
      
      Returns:
          Dict: Token usage summary
      """
      return {
          "total_tokens": self.total_tokens,
          "prompt_tokens": self.prompt_tokens,
          "completion_tokens": self.completion_tokens,
          "session_duration": datetime.now().isoformat(),
          "entry_count": len(self.entries),
          "entries": self.entries
      }
  
  def format_for_display(self) -> str:
      """
      Format token usage for user-friendly display.
      
      Returns:
          str: Formatted token usage report
      """
      return f"""
📊 SESSION TOKEN USAGE REPORT:
├─ Total Tokens Used: {self.total_tokens:,}
├─ Prompt Tokens: {self.prompt_tokens:,}
├─ Completion Tokens: {self.completion_tokens:,}
└─ Total API Calls: {len(self.entries)}
"""

class OrchestratorAgent(ABC):
  """
  Main orchestration agent that coordinates the learning system.
  This agent serves as the central hub that:
  - Receives all user input
  - Determines current session stage
  - Routes to appropriate specialized agents
  - Publishes formatted output to user
  - Maintains session state and context
  - Provides friendly guidance throughout the learning process
  - Tracks token usage across all agent interactions
  
  IMPORTANT: This agent teaches PYTHON EXCLUSIVELY. All lessons, explanations,
  and guidance are Python-specific. Users asking about other programming languages
  will be redirected to focus on Python.
  
  Features:
  - Intelligent session stage tracking
  - Multi-agent routing and coordination
  - Friendly, supportive user communication
  - Session persistence and recovery
  - Error handling and fallback strategies
  - Comprehensive activity logging
  - Python-exclusive teaching focus
  - Token usage tracking and reporting
  
  Attributes:
      name (str): The name/role of the agent
      openai_client (OpenAI): The OpenAI client instance
      llm (ChatOpenAI): The language model for output formatting
      sessions (Dict): Active user sessions
      level_placement_agent: Reference to LevelPlacementAgent
      error_training_agent: Reference to ErrorTrainingAgent
      update_profile_agent: Reference to UpdateProfileAgent
      token_ledgers (Dict): Token ledgers per session
  """
  # LLM Configuration
  LLM_MODEL = "gpt-4o-mini"
  TEMPERATURE = 0.6  # Balanced for friendly yet professional tone
  
  # ===== PYTHON-EXCLUSIVE MESSAGING =====
  WELCOME_AND_NAME_PROMPT = """
╔══════════════════════════════════════════════════════════════════════╗
║                PYTHON LEARNING ASSISTANT                            ║
║                 (Python 3 Focus)                                     ║
║                                                                      ║
║     Welcome to your personalized Python tutor! 🐍                    ║
║                                                                      ║
║  Before we get started, I'd like to know your name! 😊               ║
║                                                                      ║
║  Please tell me your name (or type 'skip' to proceed without one):   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
  
  WELCOME_MESSAGE = """
╔══════════════════════════════════════════════════════════════════════╗
║           🐍 PYTHON LEARNING ASSISTANT 🐍                            ║
║                                                                      ║
║     Welcome to your personalized Python tutor!                       ║
║                                                                      ║
║  I specialize in teaching PYTHON programming (Python 3) through:     ║
║  • Custom proficiency assessments for Python code                    ║
║  • Targeted Python error training and lessons                        ║
║  • Personalized Python learning paths                                ║
║  • Progress tracking and Python recommendations                      ║
║                                                                      ║
║  Note: I teach Python exclusively. For other languages, you'll       ║
║  need to consult a different tutor.                                  ║
║                                                                      ║
║              Let's learn Python together!                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

📝 You can:

 OPTION 1: Share Python code for assessment
 • Paste Python code you've written
 • Share Python code with errors you're struggling with
 • I'll assess your Python proficiency level and create personalized lessons

 OPTION 2: Ask for Python help from scratch
 • "Help me write a Python function that..."
 • "I want to create a Python program for..."
 • "How do I implement this in Python..."
 • I'll provide Python-specific guidance and examples

 OPTION 3: Ask about Python concepts
 • "What is a list comprehension?"
 • "Explain decorators in Python"
 • "What's the difference between a list and tuple?"
 • I'll explain Python concepts in clear, level-appropriate ways

 What would you like to learn about Python?
"""
  
  def __init__(
        self,
        name: str = "Python Learning Orchestrator",
        level_placement_agent=None,
        error_training_agent=None,
        update_profile_agent=None
    ):
        """
        Initialize the Orchestrator Agent (Python-exclusive).
   
        Args:
            name (str): The name or role identifier for this agent
            level_placement_agent: Instance of LevelPlacementAgent
            error_training_agent: Instance of ErrorTrainingAgent
            update_profile_agent: Instance of UpdateProfileAgent
   
        Raises:
            KeyError: If OPENAI_API_KEY environment variable is not set
        """
        self.name = name
   
        # Initialize OpenAI client
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise KeyError("OPENAI_API_KEY environment variable not set")
            self.openai_client = OpenAI(api_key=api_key)
        except KeyError as e:
            raise KeyError(f"Failed to initialize OpenAI client: {e}")
   
        # Initialize LLM for output formatting
        self.llm = ChatOpenAI(
            model=self.LLM_MODEL,
            temperature=self.TEMPERATURE
        )
   
        # Store agent references
        self.level_placement_agent = level_placement_agent
        self.error_training_agent = error_training_agent
        self.update_profile_agent = update_profile_agent
   
        # Session management
        self.sessions: Dict[str, Dict] = {}
        self.activity_log: List[Dict] = []
        
        # Token ledger management
        self.token_ledgers: Dict[str, TokenLedger] = {}
        
        # Session persistence
        from session_persistence import SessionPersistence
        # Get absolute path to learner_sessions folder from project root
        project_root = Path(__file__).resolve().parent.parent
        sessions_dir = project_root / "learner_sessions"
        self.persistence = SessionPersistence(sessions_dir=str(sessions_dir))

  def start_new_session(self, user_id: str, user_name: str = "") -> Dict:
      """
      Start a new Python learning session for a user.
      """
      session = {
          "user_id": user_id,
          "user_name": user_name,
          "start_time": datetime.now().isoformat(),
          "stage": SessionStage.GATHERING_NAME,
          "level": None,
          "level_assessment": None,
          "current_lesson": None,
          "lesson_performance": None,
          "profile": None,
          "interaction_count": 0,
          "history": [],
          # Continuous learning fields
          "followup_options": [],
          "selected_followup": None,
          "learning_path": [],
          "completed_topics": [],
          "current_skill_focus": None,
          "skill_progression_step": 0,
          "last_evaluation": None,
          # Python-specific tracking
          "language": "Python",  # Always Python
          "python_version": "Python 3"
      }

      self.sessions[user_id] = session
      
      # Initialize token ledger for this session
      self.token_ledgers[user_id] = TokenLedger()

      self._log_activity(
          user_id,
          "SESSION_STARTED",
          "New Python learning session started. Gathering user name."
      )

      return {
          "user_id": user_id,
          "message": self.WELCOME_AND_NAME_PROMPT,
          "next_action": "WAITING_FOR_NAME"
      }

  def get_session(self, user_id: str) -> Optional[Dict]:
      """
      Retrieve a session.
   
      Args:
          user_id (str): Unique identifier for the user
       
      Returns:
          Optional[Dict]: Session information or None
      """
      return self.sessions.get(user_id)

  def process_user_input(
      self,
      user_id: str,
      user_input: str
  ) -> Dict:
      """
      Main entry point: Process user input and route to appropriate agent.
      All instruction and guidance is Python-exclusive.
   
      Args:
          user_id (str): Unique identifier for the user
          user_input (str): The user's input
       
      Returns:
          Dict: Formatted output for user and next action
      """
      # Get or create session
      session = self.get_session(user_id)
      if not session:
          self.start_new_session(user_id)
          session = self.sessions[user_id]
          return {
              "message": self.WELCOME_MESSAGE,
              "next_action": "WAITING_FOR_CODE_INPUT"
          }
   
      # Increment interaction count
      session["interaction_count"] += 1

      # Global exit handling: save and complete the session whenever the user wants to leave
      if self._user_wants_to_exit(user_input):
          routing_decision = RoutingDecision.COMPLETE_SESSION
          agent_response = self._route_to_agent(routing_decision, session, user_input)
          return self._format_and_publish_output(session, routing_decision, agent_response)

      if session["stage"] == SessionStage.GATHERING_NAME:
          routing_decision = RoutingDecision.GATHER_INFO
          self._process_user_name(session, user_input)
          session["stage"] = SessionStage.NEW
          # FIX: Generate agent_response for the name gathering stage
          agent_response = self._gather_more_info(session, user_input)
      else:
          # 🔒 SCOPE VALIDATION: Check if user input is within scope BEFORE routing
          # This prevents hallucinations and off-topic responses
          is_in_scope = self._is_within_scope(user_input)
         
          if is_in_scope is False:
              # User input is OUT of scope - return scope warning without routing to agent
              return self._format_scope_warning(session, user_input)
         
          # Determine routing based on session stage
          routing_decision = self._determine_routing(session, user_input)
   
          # Route to appropriate agent and get response
          agent_response = self._route_to_agent(
              routing_decision, session, user_input
          )
   
      # Format and return output to user
      return self._format_and_publish_output(
          session, routing_decision, agent_response
      )

  def _process_user_name(self, session: Dict, user_input: str) -> str:
      """
      Process user's name input and update session.
 
      Args:
          session (Dict): Current session
          user_input (str): User's input (the name or 'skip')
     
      Returns:
          str: The processed name
      """
      name_input = user_input.strip()
   
      # Check if user wants to skip providing a name
      if name_input.lower() in ['skip', 'no', 'none', '']:
          session["user_name"] = session["user_id"]
          self._log_activity(
              session["user_id"],
              "NAME_SKIPPED",
              "User skipped providing a name"
          )
      else:
          session["user_name"] = name_input
          self._log_activity(
              session["user_id"],
              "NAME_PROVIDED",
              f"User provided name: {name_input}"
          )
   
      return session["user_name"]

  def _determine_routing(
      self,
      session: Dict,
      user_input: str
  ) -> RoutingDecision:
      """
      Determine which agent to route to based on session state.
   
      Args:
          session (Dict): Current session
          user_input (str): User's input
       
      Returns:
          RoutingDecision: Which agent to use
      """
      stage = SessionStage(session["stage"])
   
      # First interaction: gather user's name
      if stage == SessionStage.GATHERING_NAME:
          # Process the name and transition to NEW stage
          self._process_user_name(session, user_input)
          session["stage"] = SessionStage.NEW
          return RoutingDecision.GATHER_INFO
 
      # Second interaction: check if code or help request
      elif stage == SessionStage.NEW:
          if self._is_ready_for_lesson(user_input):
              # User typed 'ready' - treat as code input without re-analyzing
              return RoutingDecision.LEVEL_PLACEMENT
          elif self._is_code_input(user_input):
              return RoutingDecision.LEVEL_PLACEMENT
          elif self._is_python_explanation_request(user_input):
              return RoutingDecision.PYTHON_EXPLANATION
          elif self._is_help_request(user_input):
              # Skip assessment for help requests, go directly to training
              session["stage"] = SessionStage.ASSESSED
              session["level"] = "Beginner"  # Assume beginner for help requests
              return RoutingDecision.ERROR_TRAINING
          else:
              # Not clear, gather more info
              return RoutingDecision.GATHER_INFO
 
      # Level determined: route to error training
      elif stage == SessionStage.ASSESSED:
          # Check if user is asking for a concept explanation instead
          if self._is_python_explanation_request(user_input):
              return RoutingDecision.PYTHON_EXPLANATION
          return RoutingDecision.ERROR_TRAINING
 
      # During training: check if lesson is complete
      elif stage == SessionStage.TRAINING:
          # Check for concept explanation requests during training
          if self._is_python_explanation_request(user_input):
              return RoutingDecision.PYTHON_EXPLANATION
          # If user has submitted answer, evaluate correctness
          if self._is_lesson_response(user_input):
              # Evaluate if the code is correct
              evaluation = self._evaluate_code_correctness(user_input, session)
              if evaluation["is_correct"]:
                  # Code is correct, proceed to profile update
                  return RoutingDecision.UPDATE_PROFILE
              else:
                  # Code is incorrect, provide feedback and stay in training
                  return RoutingDecision.EVALUATE_CORRECTION
          else:
              # Otherwise, stay in training
              return RoutingDecision.GATHER_INFO
 
      # Profile updated: start next cycle or complete session
      elif stage == SessionStage.PROFILE_UPDATED:
          # Check for concept explanation requests
          if self._is_python_explanation_request(user_input):
              return RoutingDecision.PYTHON_EXPLANATION
          # This stage is now deprecated - profile updates go to CONTINUOUS
          return RoutingDecision.GATHER_INFO
 
      # Continuous: check what user needs after lesson completion
      elif stage == SessionStage.CONTINUOUS:
          # Check for concept explanation requests during continuous learning
          if self._is_python_explanation_request(user_input):
              return RoutingDecision.PYTHON_EXPLANATION
          if self._user_wants_continue(user_input):
              # Reset to assessed stage for next lesson
              session["stage"] = SessionStage.ASSESSED
              return RoutingDecision.ERROR_TRAINING
          elif self._user_wants_to_exit(user_input):
              return RoutingDecision.COMPLETE_SESSION
          elif self._is_successful_correction(user_input, session):
              # User has successfully corrected an error - suggest skill-building task
              return RoutingDecision.SKILL_BUILDING
          elif user_input.strip().lower() == "new":
              # User wants to start fresh with new code for assessment
              session["stage"] = SessionStage.NEW
              return RoutingDecision.LEVEL_PLACEMENT
          elif user_input.strip().lower() == "skip":
              # User wants to skip the follow-up task and continue to next lesson
              session["stage"] = SessionStage.ASSESSED
              return RoutingDecision.ERROR_TRAINING
          else:
              return RoutingDecision.GATHER_INFO

      # Follow-up task: user is working on comprehension questions or coding exercise
      elif stage == SessionStage.FOLLOWUP_TASK:
          if self._user_wants_to_exit(user_input):
              return RoutingDecision.COMPLETE_SESSION
          elif self._user_wants_continue(user_input):
              # User submitted an answer to the follow-up task
              return RoutingDecision.FOLLOWUP_TASK
          else:
              # Stay in follow-up task stage
              return RoutingDecision.FOLLOWUP_TASK
       
  def _route_to_agent(
      self,
      decision: RoutingDecision,
      session: Dict,
      user_input: str
  ) -> Dict:
      """
      Route to the appropriate specialized agent.
   
      Args:
          decision (RoutingDecision): Routing decision
          session (Dict): Current session
          user_input (str): User's input
       
      Returns:
          Dict: Agent response
      """
      try:
          if decision == RoutingDecision.LEVEL_PLACEMENT:
              return self._route_to_level_placement(session, user_input)
       
          elif decision == RoutingDecision.ERROR_TRAINING:
              return self._route_to_error_training(session, user_input)
       
          elif decision == RoutingDecision.UPDATE_PROFILE:
              return self._route_to_update_profile(session, user_input)
       
          elif decision == RoutingDecision.GATHER_INFO:
              return self._gather_more_info(session, user_input)
       
          elif decision == RoutingDecision.COMPLETE_SESSION:
              return self._complete_session(session)
       
          elif decision == RoutingDecision.SKILL_BUILDING:
              return self._route_to_skill_building(session, user_input)
         
          elif decision == RoutingDecision.EVALUATE_CORRECTION:
              return self._route_to_evaluate_correction(session, user_input)
         
          elif decision == RoutingDecision.FOLLOWUP_TASK:
              return self._route_to_followup_task(session, user_input)
          
          elif decision == RoutingDecision.PYTHON_EXPLANATION:
              return self._route_to_python_explanation(session, user_input)
       
          else:
              return {"error": "Unknown routing decision"}
   
      except Exception as e:
          self._log_activity(
              session["user_id"],
              "ROUTING_ERROR",
              f"Error routing to agent: {str(e)}"
          )
          return {"error": f"An error occurred: {str(e)}"}

  def _record_token_usage(
      self,
      user_id: str,
      agent_name: str,
      response: Dict,
      activity_description: str = ""
  ) -> None:
      """
      Record token usage from an agent response.
      
      Args:
          user_id (str): User ID for the session
          agent_name (str): Name of the agent that was called
          response (Dict): Response from the agent (may contain token info)
          activity_description (str): Description of the activity
      """
      if user_id not in self.token_ledgers:
          self.token_ledgers[user_id] = TokenLedger()
      
      ledger = self.token_ledgers[user_id]
      prompt_tokens = 0
      completion_tokens = 0
      
      if isinstance(response, dict):
          prompt_tokens = response.get("prompt_tokens", 0)
          completion_tokens = response.get("completion_tokens", 0)
          usage = response.get("usage", {}) or {}
          if not prompt_tokens:
              prompt_tokens = usage.get("prompt_tokens", 0)
          if not completion_tokens:
              completion_tokens = usage.get("completion_tokens", 0)
      else:
          usage = getattr(response, "usage", None)
          if usage:
              prompt_tokens = getattr(usage, "prompt_tokens", 0) or usage.get("prompt_tokens", 0)
              completion_tokens = getattr(usage, "completion_tokens", 0) or usage.get("completion_tokens", 0)
          llm_output = getattr(response, "llm_output", None)
          if isinstance(llm_output, dict):
              token_usage = llm_output.get("token_usage", {}) or {}
              if not prompt_tokens:
                  prompt_tokens = token_usage.get("prompt_tokens", 0)
              if not completion_tokens:
                  completion_tokens = token_usage.get("completion_tokens", 0)
      
      ledger.record_token_usage(
          agent_name=agent_name,
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
          activity_description=activity_description
      )

  def _route_to_level_placement(self, session: Dict, user_input: str) -> Dict:
      """Route user input to Python level placement assessment."""
      if not self.level_placement_agent:
          return {"error": "Level placement agent not available"}
   
      # Call level placement agent
      assessment = self.level_placement_agent.analyze_code(user_input)
      
      # Record token usage if available
      self._record_token_usage(
          session["user_id"],
          "LevelPlacementAgent",
          assessment,
          "Python code proficiency assessment"
      )
   
      # Update session
      session["stage"] = SessionStage.ASSESSED
      session["level"] = assessment["level"].value
      session["level_assessment"] = assessment
   
      # Create initial profile
      if self.update_profile_agent:
          initial_profile_result = self.update_profile_agent.execute(
              learner_id=session["user_id"],
              lesson_data={"initial_assessment": True, "level": assessment["level"].value},
              user_response="",
              current_level=assessment["level"].value,
              learner_name=session.get("user_name", session["user_id"]),
              token_ledger_data=self.get_token_usage(session["user_id"]) or {}
          )
          
          # Record token usage from profile creation
          self._record_token_usage(
              session["user_id"],
              "UpdateProfileAgent",
              initial_profile_result,
              "Initial Python learner profile creation"
          )
          
          session["profile"] = initial_profile_result.get("profile")
          session["formatted_profile"] = initial_profile_result.get("formatted_display")
   
      self._log_activity(
          session["user_id"],
          "PYTHON_LEVEL_PLACEMENT_COMPLETE",
          f"Python proficiency level assessed: {assessment['level'].value}"
      )
   
      return {
          "agent": "LevelPlacementAgent",
          "assessment": assessment,
          "level": assessment["level"].value,
          "report": self.level_placement_agent.generate_report(assessment),
          "topic": self._route_to_error_training(session, user_input)
      }

  def _route_to_error_training(self, session: Dict, user_input: str) -> Dict:
      """Route user input to Python error training lesson generation."""
      if not self.error_training_agent:
          return {"error": "Error training agent not available"}
   
      current_level = session.get("level", "Intermediate")
   
      # Call error training agent
      result = self.error_training_agent.execute(
          student_level=current_level,
          error_type="general",
          error_description=user_input
      )
      
      # Record token usage
      self._record_token_usage(
          session["user_id"],
          "ErrorTrainingAgent",
          result,
          "Python error training lesson generation"
      )
   
      # Extract lesson
      lesson = result.get("lesson")
      if lesson:
          session["stage"] = SessionStage.TRAINING
          session["current_lesson"] = lesson
          topic = lesson.get("topic")
          if not topic:
              topic = lesson["components"].get("objective")
          if not topic:
              etype = lesson.get("error_type", "")
              edesc = lesson.get("error_description", "")
              topic = f"{etype}: {edesc}" if edesc else etype
          if not topic:
              topic = "Python Programming Lesson"
          session["topic"] = topic
       
          self._log_activity(
              session["user_id"],
              "PYTHON_LESSON_GENERATED",
              f"Python lesson generated for {lesson.get('error_type', 'general')}"
          )
       
          return {
              "agent": "ErrorTrainingAgent",
              "lesson": lesson,
              "formatted_lesson": self.error_training_agent.format_lesson_for_display(lesson)
          }
   
      return {"error": "Could not generate Python lesson"}

  def _route_to_update_profile(self, session: Dict, user_input: str) -> Dict:
      """Route to profile update based on Python lesson performance."""
      if not self.update_profile_agent:
          return {"error": "Update profile agent not available"}

      lesson_data = session.get("current_lesson", {})
      current_level = session.get("level", "Intermediate")
      user_id = session["user_id"]

      # Call update profile agent
      result = self.update_profile_agent.execute(
          learner_id=user_id,
          lesson_data=lesson_data,
          user_response=user_input,
          current_level=current_level,
          learner_name=session.get("user_name", user_id),
          token_ledger_data=self.get_token_usage(user_id) or {}
      )
      
      # Record token usage
      self._record_token_usage(
          user_id,
          "UpdateProfileAgent",
          result,
          "Python learner profile update after lesson"
      )

      # Update session
      session["stage"] = SessionStage.CONTINUOUS
      session["lesson_performance"] = user_input
      session["profile"] = result.get("profile")
      session["formatted_profile"] = result.get("formatted_display")

      self._log_activity(
          user_id,
          "PYTHON_PROFILE_UPDATED",
          f"Python learner profile updated for lesson"
      )

      # Generate follow-up task based on level and error performance
      followup_task = self._generate_followup_task(session, lesson_data)

      return {
          "agent": "UpdateProfileAgent",
          "profile": result.get("profile"),
          "formatted_profile": result.get("formatted_display"),
          "followup_task": followup_task
      }

  def _generate_followup_task(self, session: Dict, lesson_data: Dict) -> Dict:
      """
      Generate a Python-specific follow-up task based on user's level and error performance.
     
      The task is designed to reinforce Python learning through:
      - Comprehension questions (for all levels)
      - Python code writing exercises (adapted to proficiency level)
      - Python error identification challenges
     
      Args:
          session (Dict): Current session data
          lesson_data (Dict): The completed Python lesson data
         
      Returns:
          Dict: Follow-up task with comprehension questions and/or Python coding exercises
      """
      current_level = session.get("level", "Intermediate")
      user_name = session.get("user_name", "Learner")
      error_type = lesson_data.get("error_type", "general")
      topic = session.get("topic", "Python programming")
     
      # Generate Python-specific task
      prompt = ChatPromptTemplate.from_template("""Create a follow-up task for a {level} level Python learner who just completed a lesson on: {topic} (Error type: {error_type})

The task should include:
1. A brief comprehension check (1-2 questions to verify understanding of Python concepts)
2. A practical Python coding exercise appropriate for {level} level

Format the response as:
COMPREHENSION_QUESTIONS:
[1-2 questions about the key Python concept]

CODING_EXERCISE:
[Clear description of what Python code to write]

DIFFICULTY: [Easy/Medium/Hard based on {level} level]

Make sure the Python coding exercise is challenging but achievable for a {level} level student.
""")

      chain = prompt | self.llm | StrOutputParser()
      task_response = chain.invoke({
          "level": current_level,
          "topic": topic,
          "error_type": error_type
      })
      
      # Record token usage for LLM call
      self._record_token_usage(
          session["user_id"],
          "OrchestratorAgent",
          {},  # LLM calls don't return token info directly
          "Python follow-up task generation"
      )

      # Parse the response
      followup_task = {
          "topic": topic,
          "error_type": error_type,
          "level": current_level,
          "full_task": task_response
      }

      # Extract components
      lines = task_response.split('\n')
      current_section = None
      for line in lines:
          line = line.strip()
          if "COMPREHENSION_QUESTIONS:" in line:
              current_section = "comprehension"
              followup_task["comprehension_questions"] = []
          elif "CODING_EXERCISE:" in line:
              current_section = "exercise"
              followup_task["coding_exercise"] = ""
          elif "DIFFICULTY:" in line:
              current_section = "difficulty"
          elif current_section == "comprehension" and line:
              followup_task["comprehension_questions"].append(line)
          elif current_section == "exercise" and line:
              followup_task["coding_exercise"] += line + "\n"
          elif current_section == "difficulty" and line:
              followup_task["difficulty"] = line.replace("DIFFICULTY:", "").strip()

      # Store in session for tracking
      session["followup_task"] = followup_task
      session["followup_attempts"] = 0

      self._log_activity(
          session["user_id"],
          "PYTHON_FOLLOWUP_TASK_GENERATED",
          f"Python follow-up task generated for {topic}"
      )

      return followup_task

  def _route_to_followup_task(self, session: Dict, user_input: str) -> Dict:
      """
      Route to handle user's response to Python follow-up task.
     
      This method evaluates the user's answers to Python comprehension questions
      and/or their Python code for the coding exercise, then provides feedback
      and determines next steps in the continuous learning loop.
     
      Args:
          session (Dict): Current session data
          user_input (str): User's response to the follow-up task
         
      Returns:
          Dict: Feedback on the follow-up task and next steps
      """
      followup_task = session.get("followup_task", {})
      current_level = session.get("level", "Intermediate")
      user_name = session.get("user_name", "Learner")
     
      # Increment follow-up attempts
      session["followup_attempts"] = session.get("followup_attempts", 0) + 1
     
      # Evaluate the user's response
      prompt = ChatPromptTemplate.from_template("""Evaluate this Python learner's response to their follow-up task.

LEARNER LEVEL: {level}
PYTHON TOPIC: {topic}
PYTHON EXERCISE: {exercise}
COMPREHENSION QUESTIONS: {questions}

LEARNER'S RESPONSE:
{response}

Provide feedback on:
1. Whether the comprehension answers are correct (if applicable)
2. Whether the Python code is correct and follows best practices (if code was submitted)
3. A brief encouraging message

Format as:
COMPREHENSION_FEEDBACK: [Correct/Partially Correct/Incorrect] - [brief explanation]
CODE_FEEDBACK: [Correct/Partially Correct/Incorrect/N/A] - [brief explanation]
ENCOURAGEMENT: [A brief encouraging message]
NEXT_STEPS: [What should the learner do next?]
""")

      chain = prompt | self.llm | StrOutputParser()
      evaluation = chain.invoke({
          "level": current_level,
          "topic": followup_task.get("topic", "Python"),
          "exercise": followup_task.get("coding_exercise", "N/A"),
          "questions": "\n".join(followup_task.get("comprehension_questions", [])),
          "response": user_input
      })
      
      # Record token usage
      self._record_token_usage(
          session["user_id"],
          "OrchestratorAgent",
          {},
          "Python follow-up task response evaluation"
      )

      # Parse evaluation
      lines = evaluation.split('\n')
      feedback_dict = {}
      for line in lines:
          if "COMPREHENSION_FEEDBACK:" in line:
              feedback_dict["comprehension"] = line.split(":", 1)[1].strip()
          elif "CODE_FEEDBACK:" in line:
              feedback_dict["code"] = line.split(":", 1)[1].strip()
          elif "ENCOURAGEMENT:" in line:
              feedback_dict["encouragement"] = line.split(":", 1)[1].strip()
          elif "NEXT_STEPS:" in line:
              feedback_dict["next_steps"] = line.split(":", 1)[1].strip()

      # Log the evaluation
      self._log_activity(
          session["user_id"],
          "PYTHON_FOLLOWUP_TASK_EVALUATED",
          f"Python follow-up task evaluated - Attempt #{session.get('followup_attempts', 1)}"
      )

      # Determine if user wants to continue or exit
      wants_continue = self._user_wants_continue(user_input)
      wants_exit = self._user_wants_to_exit(user_input)
     
      if wants_exit:
          # User wants to exit - complete the session
          return self._complete_session(session)
      
  def resume_session(self, user_id: str) -> Optional[Dict]:
      """
    Resume a saved session.
      
      Args:
          user_id (str): User identifier
      
      Returns:
          Optional[Dict]: Previous session data or None
      """
      previous_session = self.persistence.load_session(user_id)
      if previous_session:
          # Restore session state
          session_data = previous_session.get("session", {})
          self.sessions[user_id] = session_data
          
          # Restore token ledger if available
          token_data = previous_session.get("tokens", {})
          if token_data and "entries" in token_data:
              ledger = TokenLedger()
              ledger.total_tokens = token_data.get("total_tokens", 0)
              ledger.prompt_tokens = token_data.get("prompt_tokens", 0)
              ledger.completion_tokens = token_data.get("completion_tokens", 0)
              ledger.entries = token_data.get("entries", [])
              self.token_ledgers[user_id] = ledger
          
          self._log_activity(
              user_id,
              "SESSION_RESUMED",
              "Previous session loaded and resumed"
          )
          return session_data
      return None

  def get_user_token_history(self, user_id: str) -> Dict:
      """
      Get token usage history for user from saved session.
      
      Args:
          user_id (str): User identifier
      
      Returns:
          Dict: Token history and summary
      """
      session = self.persistence.load_session(user_id)
      if session and "tokens" in session:
          token_data = session["tokens"]
          return {
              "history": token_data.get("entries", []),
              "summary": {
                  "total_tokens": token_data.get("total_tokens", 0),
                  "prompt_tokens": token_data.get("prompt_tokens", 0),
                  "completion_tokens": token_data.get("completion_tokens", 0),
                  "entry_count": token_data.get("entry_count", 0)
              }
          }
      return {"history": [], "summary": {}}
     
      # Generate response with options to continue learning Python
      response_message = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    📋 FOLLOW-UP TASK FEEDBACK                         ║
╚══════════════════════════════════════════════════════════════════════╝

{feedback_dict.get('encouragement', 'Great effort!')}

📝 Feedback:
• {feedback_dict.get('comprehension', 'N/A')}
• {feedback_dict.get('code', 'N/A')}

🚀 Next Steps: {feedback_dict.get('next_steps', 'Continue learning Python')}

╔══════════════════════════════════════════════════════════════════════╗
║                    🔄 CONTINUOUS LEARNING LOOP                        ║
╚══════════════════════════════════════════════════════════════════════╝

What would you like to do next?

 • Type 'continue' or 'next' for another personalized Python lesson
 • Type 'new' to start fresh with new Python code for assessment
 • Type 'exit', 'quit', or 'done' to end the session
 • Ask me anything else about Python programming!
"""

      # Update session to CONTINUOUS stage for next decision
      session["stage"] = SessionStage.CONTINUOUS
      session["last_followup_feedback"] = feedback_dict

      return {
          "agent": "OrchestratorAgent",
          "message": response_message,
          "feedback": feedback_dict,
          "followup_task": followup_task,
          "next_action": "CONTINUE_LEARNING"
      }

  def _gather_more_info(self, session: Dict, user_input: str) -> Dict:
      """Gather more information from the user about Python learning."""
      # Check if we just transitioned from GATHERING_NAME stage
      user_name = session.get("user_name", "Friend")
      # After processing name, interaction_count will be 1 and stage will have just changed to NEW
      is_after_name_collection = (session["interaction_count"] == 1 and user_name != session["user_id"])
   
      # If the user just provided their name, give a personalized welcome
      if is_after_name_collection:
          welcome_response = f"""
Great to meet you, {user_name}! 😊

{self.WELCOME_MESSAGE}
"""
          return {
              "agent": "OrchestratorAgent",
              "response": welcome_response,
              "action": "GATHER_MORE_INFO"
          }
   
      # Otherwise, use generic prompt for Python guidance
      prompt = ChatPromptTemplate.from_template("""Based on this Python learning context, generate a friendly response that:
1. Acknowledges the user's input
2. Asks clarifying questions about what they want to learn in Python
3. Keeps the conversation focused on Python programming

Context:
- Stage: {stage}
- User Input: {user_input}
- Session Count: {session_count}
- User Name: {user_name}

Generate a helpful, encouraging response that reinforces we're learning Python.
""")
   
      chain = prompt | self.llm | StrOutputParser()
      response = chain.invoke({
          "stage": session["stage"],
          "user_input": user_input[:200],
          "session_count": session["interaction_count"],
          "user_name": user_name
      })
      
      # Record token usage
      self._record_token_usage(
          session["user_id"],
          "OrchestratorAgent",
          {},
          "Gathering more information about Python learning"
      )
   
      return {
          "agent": "OrchestratorAgent",
          "response": response,
          "action": "GATHER_MORE_INFO"
      }

  def _route_to_python_explanation(self, session: Dict, user_input: str) -> Dict:
      """
      Route to Python concept explanation generation.
      
      Args:
          session (Dict): Current session
          user_input (str): User's question about a Python concept
          
      Returns:
          Dict: Python concept explanation
      """
      current_level = session.get("level", "Beginner")
      user_name = session.get("user_name", "Learner")
      
      # Generate Python-specific explanation
      prompt = ChatPromptTemplate.from_template("""You are a Python educator. A {level} Python learner asks: "{question}"

Provide a clear, level-appropriate explanation that includes:
1. **Definition**: What is this Python concept?
2. **Why it matters**: When and why would you use it in Python?
3. **Example**: Show a simple Python code example
4. **Common Mistakes**: What mistakes do {level} Python learners make with this?
5. **Practice Tip**: A quick Python practice suggestion

Keep the explanation concise, friendly, and focused on practical Python usage.
Format with clear sections and Python code blocks.
""")

      chain = prompt | self.llm | StrOutputParser()
      explanation = chain.invoke({
          "level": current_level,
          "question": user_input
      })
      
      # Record token usage
      self._record_token_usage(
          session["user_id"],
          "OrchestratorAgent",
          {},
          f"Python concept explanation: {user_input[:50]}"
      )

      self._log_activity(
          session["user_id"],
          "PYTHON_CONCEPT_EXPLAINED",
          f"Python concept explanation provided: {user_input[:50]}"
      )

      return {
          "agent": "OrchestratorAgent",
          "message": explanation,
          "next_action": "CONTINUE_LEARNING"
      }

  def _complete_session(self, session: Dict) -> Dict:
      """Generate Python learning session completion message with token usage summary."""
      profile = session.get("profile", {})
      formatted_profile = session.get("formatted_profile", "")
      user_name = session.get("user_name", "Friend")
      user_id = session["user_id"]
      token_summary = ""
      token_data = None
      if self.update_profile_agent:
          token_report_display = self.update_profile_agent.format_token_report_for_display(
              user_id,
              include_agent_breakdown=True
          )
          token_summary = token_report_display
          token_data = self.update_profile_agent.compile_token_report(user_id) or {
              "total_tokens": 0,
              "prompt_tokens": 0,
              "completion_tokens": 0,
              "entry_count": 0,
              "entries": []
          }
          if token_data.get("total_tokens", 0) == 0:
              ledger_summary = self.get_token_usage(user_id) or {}
              if ledger_summary.get("total_tokens", 0) > 0:
                  token_data["total_tokens"] = ledger_summary.get("total_tokens", 0)
                  token_data["prompt_tokens"] = ledger_summary.get("prompt_tokens", 0)
                  token_data["completion_tokens"] = ledger_summary.get("completion_tokens", 0)
                  token_data["entry_count"] = ledger_summary.get("entry_count", 0)
                  token_data["entries"] = ledger_summary.get("entries", [])
                  token_data["session_count"] = ledger_summary.get("session_count", 0)
                  token_data["average_tokens_per_session"] = (
                      token_data["total_tokens"] / token_data["session_count"]
                      if token_data["session_count"] > 0
                      else 0
                  )
      else:
          token_ledger = self.token_ledgers.get(user_id)
          if token_ledger:
              token_summary = token_ledger.format_for_display()
              token_data = token_ledger.get_summary()
          else:
              token_data = {
                  "total_tokens": 0,
                  "prompt_tokens": 0,
                  "completion_tokens": 0,
                  "entry_count": 0,
                  "entries": []
              }
   
      completion_message = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                   🎉 GREAT PYTHON LEARNING SESSION! 🎉               ║
╚══════════════��═══════════════════════════════════════════════════════╝

Hi {user_name},

Thank you for working through this Python learning session! Here's your final Python learning profile:

{formatted_profile}

📊 PYTHON LEARNING SESSION SUMMARY:
• Interactions: {session.get('interaction_count', 0)}
• Python Proficiency Level: {session.get('level', 'N/A')}
• Python Errors Addressed: {profile.get('total_errors_addressed', 0) if profile else 'N/A'}

{token_summary}

When you're ready to continue learning Python, just start a new session and I'll pick up where we left off!

🚀 REMEMBER:
• Practice Python regularly
• Review common Python mistakes
• Try the recommended Python practice problems
• Come back soon for more Python lessons!

Keep coding Python! 🐍

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
      
      # Save session with token data
      self.persistence.save_session(
          user_id,
          session,
          profile,
          token_data
      )
   
      self._log_activity(
          session["user_id"],
          "SESSION_COMPLETED",
          "Python learning session completed and saved"
      )
   
      return {
          "agent": "OrchestratorAgent",
          "message": completion_message,
          "session_status": "COMPLETED"
      }

  def _format_and_publish_output(
      self,
      session: Dict,
      routing_decision: RoutingDecision,
      agent_response: Dict
  ) -> Dict:
      """
      Format agent output for user presentation.
   
      Args:
          session (Dict): Current session
          routing_decision (RoutingDecision): Routing decision made
          agent_response (Dict): Response from routed agent
       
      Returns:
          Dict: Formatted output for user
      """
      output = {
          "user_id": session["user_id"],
          "timestamp": datetime.now().isoformat(),
          "stage": session["stage"]
      }
   
      # Handle errors
      if "error" in agent_response:
          output["message"] = f"❌ Error: {agent_response['error']}"
          output["next_action"] = "RETRY_OR_CONTINUE"
          return output
   
      # Format based on routing decision
      if routing_decision == RoutingDecision.LEVEL_PLACEMENT:
          formatted_profile = session.get("formatted_profile", "")
          output["title"] = "📊 Your Python Proficiency Assessment"
          output["message"] = f"{agent_response.get('report', '')}\n\n{session.get('formatted_profile', '')}"
          output["level"] = agent_response.get("level", "")
          output["next_prompt"] = self._generate_next_prompt(
              session, RoutingDecision.ERROR_TRAINING
          )
          output["next_action"] = "GENERATE_LESSON"
   
      elif routing_decision == RoutingDecision.ERROR_TRAINING:
          output["title"] = "🎓 Your Personalized Python Lesson"
          output["message"] = agent_response.get("formatted_lesson", "")
          output["next_prompt"] = self._generate_next_prompt(
              session, RoutingDecision.UPDATE_PROFILE
          )
          output["next_action"] = "SUBMIT_RESPONSE"
   
      elif routing_decision == RoutingDecision.UPDATE_PROFILE:
          # Get the follow-up task that was generated
          followup_task = agent_response.get("followup_task", {})
         
          # Build the message with follow-up task
          base_message = "Great work! Your Python learning progress has been updated."
         
          # If there's a follow-up task, include it in the message
          if followup_task:
              task_topic = followup_task.get("topic", "Python")
              questions = followup_task.get("comprehension_questions", [])
              exercise = followup_task.get("coding_exercise", "")
              difficulty = followup_task.get("difficulty", "Medium")
             
              task_message = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    🎯 YOUR PYTHON FOLLOW-UP TASK                     ║
╚══════════════════════════════════════════════════════════════════════╝

📚 Topic: {task_topic}
📊 Difficulty: {difficulty}

"""
             
              # Add comprehension questions if available
              if questions:
                  task_message += "💭 PYTHON COMPREHENSION QUESTIONS:\n"
                  for i, q in enumerate(questions, 1):
                      task_message += f"   {i}. {q}\n"
                  task_message += "\n"
             
              # Add coding exercise if available
              if exercise:
                  task_message += f"💻 PYTHON CODING EXERCISE:\n{exercise}\n"
             
              task_message += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Take your time to complete this Python task. When you're ready, share your
answers or Python code and I'll provide feedback!

Type 'done' when you've completed the task, or 'skip' to move on to
the next Python lesson.
"""
             
              output["title"] = "✅ Progress Recorded"
              output["message"] = base_message + task_message
              output["followup_task"] = followup_task
              output["next_action"] = "SUBMIT_FOLLOWUP_TASK"
              output["next_prompt"] = "Complete the follow-up task above, or type 'skip' to continue"
          else:
              output["title"] = "✅ Progress Recorded"
              output["message"] = base_message
              output["next_prompt"] = self._generate_next_prompt(
                  session, RoutingDecision.ERROR_TRAINING
              )
              output["next_action"] = "CONTINUE_OR_COMPLETE"
   
      elif routing_decision == RoutingDecision.GATHER_INFO:
          output["title"] = "💬 Let's Continue Learning Python"
          output["message"] = agent_response.get("response", "")
          output["next_action"] = "WAITING_FOR_INPUT"
   
      elif routing_decision == RoutingDecision.COMPLETE_SESSION:
          output["title"] = "✨ Python Session Complete"
          output["message"] = agent_response.get("message", "")
          output["next_action"] = "SESSION_COMPLETE"
   
      elif routing_decision == RoutingDecision.SKILL_BUILDING:
          output["title"] = "🚀 Python Skill Building Challenge"
          output["message"] = agent_response.get("message", "")
          output["skill_task"] = agent_response.get("skill_task", "")
          output["comprehension_prompt"] = agent_response.get("comprehension_prompt", "")
          output["next_prompt"] = "Share your Python solution when you're done, and I'll provide feedback!"
          output["next_action"] = "SUBMIT_SOLUTION"

      elif routing_decision == RoutingDecision.EVALUATE_CORRECTION:
          output["title"] = "🔍 Python Code Review"
          output["message"] = agent_response.get("message", "")
          output["evaluation"] = agent_response.get("evaluation", {})
          if not agent_response.get("evaluation", {}).get("is_correct", False):
              output["next_prompt"] = "Please try again with your corrected Python code!"
              output["next_action"] = "CONTINUE_TRAINING"
          else:
              output["next_prompt"] = "Excellent! Your Python code is now correct."
              output["next_action"] = "PROCEED_TO_PROFILE_UPDATE"
 
      elif routing_decision == RoutingDecision.FOLLOWUP_TASK:
          # Get the follow-up task from the session
          followup_task = session.get("followup_task", {})
          output["title"] = "📋 Python Follow-Up Task"
          output["message"] = agent_response.get("message", "")
          output["followup_task"] = followup_task
          output["feedback"] = agent_response.get("feedback", {})
          output["next_action"] = "CONTINUE_LEARNING"
          output["next_prompt"] = "Type 'continue' for another Python lesson or 'exit' to end the session"
      
      elif routing_decision == RoutingDecision.PYTHON_EXPLANATION:
          output["title"] = "📚 Python Concept Explanation"
          output["message"] = agent_response.get("message", "")
          output["next_action"] = "CONTINUE_LEARNING"
          output["next_prompt"] = "Do you have any other Python questions? Or type 'continue' for the next lesson!"
   
      # Add helpful navigation
      output["help_text"] = self._generate_help_text(routing_decision)
   
      return output

  @staticmethod
  def _generate_next_prompt(session: Dict, next_stage: RoutingDecision) -> str:
      """Generate helpful prompt for next user action."""
      prompts = {
          RoutingDecision.ERROR_TRAINING: """
✏️  Ready for your personalized Python lesson? I'll continue working with you until your Python error is corrected.

 🔧 If you're sharing Python code, use the multi-line input mode:
 1. Type ':multi' at the prompt
 2. Paste or type your Python code over multiple lines
 3. Type ':end' on its own line when finished
 4. Use ':cancel' to abort and return to the main prompt

 Share your updated Python code or the next Python error message, and I'll provide clarification or suggestions specific to that issue.
""",
          RoutingDecision.UPDATE_PROFILE: """
📝 Please share your response to the Python lesson or your attempt at the Python practice problems.
 This helps me understand what you've learned and update your Python learning profile!
""",
          RoutingDecision.GATHER_INFO: """
🎯 Great work on that Python lesson! What would you like to do next?

 • Type 'continue' or 'next' for another personalized Python lesson
 • Type 'exit', 'quit', or 'done' to end the session
 • Ask me anything else about Python programming!

Your Python learning progress has been saved and you can always come back later.
""",
          RoutingDecision.COMPLETE_SESSION: """
🎯 Would you like to:
 • Continue with another Python lesson ('continue' or 'next')
 • See your full Python learning profile ('profile')
 • End the session ('done' or 'goodbye')
"""
      }
      return prompts.get(next_stage, "What would you like to do next in your Python learning?")

  @staticmethod
  def _generate_help_text(routing_decision: RoutingDecision) -> str:
      """Generate helpful tips based on current Python learning action."""
      help_texts = {
          RoutingDecision.LEVEL_PLACEMENT: "💡 Tip: Share your actual Python code to get an accurate assessment - use ':multi' for multi-line Python snippets!",
          RoutingDecision.ERROR_TRAINING: "💡 Tip: Share your next Python fix or error message and I'll help you iterate until it's correct.",
          RoutingDecision.UPDATE_PROFILE: "💡 Tip: Share your Python attempt even if you're not 100% sure - I'll help!",
          RoutingDecision.GATHER_INFO: "💡 Tip: Feel free to ask Python questions at any time!",
          RoutingDecision.SKILL_BUILDING: "💡 Tip: Complete the Python skill-building task, then share your solution for personalized feedback!",
          RoutingDecision.EVALUATE_CORRECTION: "💡 Tip: I'll keep working with you until your Python code is correct. Share your updated solution!",
          RoutingDecision.PYTHON_EXPLANATION: "💡 Tip: I'm here to explain any Python concepts! Ask me anything about Python.",
      }
      return help_texts.get(routing_decision, "")

  @staticmethod
  def _is_ready_for_lesson(user_input: str) -> bool:
      """Detect if user typed 'ready' to start lesson from previous input."""
      return user_input.strip().lower() == "ready"

  @staticmethod
  def _is_code_input(user_input: str) -> bool:
      """Detect if user input contains actual Python code."""
      # Don't treat 'ready' as code input
      if OrchestratorAgent._is_ready_for_lesson(user_input):
          return False
   
      code_keywords = [
          "def ", "class ", "import ", "from ", "for ", "while ", "if ",
          "try:", "except:", "with ", "lambda ", "print(", "return ",
          "assert ", "yield ", "raise ", "@", "# ", "="
      ]
      lower_input = user_input.lower().strip()
      # Check for code keywords
      has_keywords = any(kw in lower_input for kw in code_keywords)
      # Check for common code patterns (e.g., indentation, colons)
      has_patterns = ":" in lower_input or "    " in user_input or "\t" in user_input
      return has_keywords or has_patterns or len(user_input.split()) > 20

  @staticmethod
  def _is_help_request(user_input: str) -> bool:
      """Detect if user input is a request for help writing Python code from scratch."""
      help_phrases = [
          "help me with", "i want to write", "how do i", "can you help me",
          "i need to", "teach me", "show me how", "write a", "create a",
          "make a", "build a", "implement", "code for"
      ]
      lower_input = user_input.lower()
      return any(phrase in lower_input for phrase in help_phrases)

  @staticmethod
  def _is_python_explanation_request(user_input: str) -> bool:
      """Detect if user is asking for a Python concept definition or explanation."""
      # Question indicators
      question_markers = ["what is", "what's", "explain", "define", "tell me about",
                         "how do i", "how do you", "what does", "difference between",
                         "show me", "teach me", "help me understand"]
      
      # Python-specific terms
      python_terms = [
          "list", "dict", "tuple", "set", "string", "int", "float", "bool",
          "function", "class", "method", "decorator", "lambda", "generator",
          "list comprehension", "dictionary comprehension", "set comprehension",
          "context manager", "with statement", "exception", "error", "try",
          "except", "finally", "import", "module", "package", "namespace",
          "scope", "global", "nonlocal", "variable", "argument", "parameter",
          "iterator", "iterable", "property", "staticmethod", "classmethod",
          "super", "inheritance", "polymorphism", "encapsulation", "abstraction",
          "list slicing", "unpacking", "args", "kwargs", "yield", "async",
          "await", "coroutine", "type", "isinstance", "isinstance", "enumerate",
          "zip", "map", "filter", "reduce", "lambda", "closure", "decorator",
          "metaclass", "mro", "dunder", "magic method", "__init__", "__str__",
          "f-string", "string formatting", "pickle", "json", "debugging", "pdb"
      ]
      
      lower_input = user_input.lower()
      
      # Check for question marker AND Python term (reduce false positives)
      has_question_marker = any(marker in lower_input for marker in question_markers)
      has_python_term = any(term in lower_input for term in python_terms)
      
      return has_question_marker and has_python_term

  @staticmethod
  def _user_wants_continue(user_input: str) -> bool:
      """Detect if user wants to continue learning Python."""
      positive_keywords = [
          "continue", "next", "more", "another", "yes", "yeah",
          "sure", "ok", "okay", "let's", "ready", "go", "again"
      ]
      return any(kw in user_input.lower() for kw in positive_keywords)

  @staticmethod
  def _user_wants_to_exit(user_input: str) -> bool:
      """Detect if user wants to exit the Python learning session."""
      exit_keywords = [
          "exit", "quit", "bye", "goodbye", "done", "finish", "stop",
          "end", "complete", "no more", "that's it", "finished"
      ]
      return any(kw in user_input.lower() for kw in exit_keywords)

  @staticmethod
  def _is_requesting_new_lesson(user_input: str) -> bool:
      """Detect if user is requesting a new Python lesson."""
      keywords = ["lesson", "new", "error", "teach", "learn", "practice"]
      return any(kw in user_input.lower() for kw in keywords)

  @staticmethod
  def _is_requesting_profile(user_input: str) -> bool:
      """Detect if user is requesting their Python learning profile."""
      keywords = ["profile", "progress", "history", "stats", "score", "level"]
      return any(kw in user_input.lower() for kw in keywords)

  @staticmethod
  def _is_lesson_response(user_input: str) -> bool:
      """Detect if user input is a response to a Python lesson."""
      # Check for code responses (contains code-like content)
      if OrchestratorAgent._is_code_input(user_input):
          return True
   
      # Check for lesson response indicators
      response_indicators = [
          "here's my", "my answer", "i tried", "this is what i got",
          "solution:", "answer:", "code:", "fixed:", "corrected:",
          "attempt:", "practice:", "exercise:", "problem:"
      ]
   
      lower_input = user_input.lower()
      return any(indicator in lower_input for indicator in response_indicators)

  def _evaluate_code_correctness(self, user_input: str, session: Dict) -> Dict:
      """
      Evaluate if the user's Python code response is correct for the current lesson.
   
      Args:
          user_input (str): User's Python code response
          session (Dict): Current session data
       
      Returns:
          Dict: Evaluation result with 'is_correct' boolean and 'feedback' string
      """
      current_lesson = session.get("current_lesson", {})
      if not current_lesson:
          return {"is_correct": False, "feedback": "No active Python lesson found."}
   
      lesson_objective = current_lesson.get("components", {}).get("objective", "")
      error_type = current_lesson.get("error_type", "")
      student_level = session.get("level", "Intermediate")
   
      prompt = ChatPromptTemplate.from_template("""Evaluate if this Python student's code response correctly addresses the Python lesson objective.

PYTHON LESSON INFORMATION:
- Error Type: {error_type}
- Learning Objective: {objective}
- Student Level: {student_level}

STUDENT'S PYTHON RESPONSE:
{user_response}

Determine if the Python code is correct and provides a proper solution to the error/problem taught in the Python lesson.

Respond with:
CORRECT: [yes/no]
FEEDBACK: [brief feedback message for the student]

If CORRECT is 'no', provide encouraging feedback that helps them understand what needs to be fixed in their Python code.
If CORRECT is 'yes', provide positive reinforcement about their Python solution.
""")
   
      chain = prompt | self.llm | StrOutputParser()
      evaluation_text = chain.invoke({
          "error_type": error_type,
          "objective": lesson_objective,
          "student_level": student_level,
          "user_response": user_input
      })
      
      # Record token usage
      self._record_token_usage(
          session["user_id"],
          "OrchestratorAgent",
          {},
          "Python code correctness evaluation"
      )
   
      # Parse the response
      lines = evaluation_text.strip().split('\n')
      is_correct = False
      feedback = "Please try again with your corrected Python code."
   
      for line in lines:
          if "CORRECT:" in line:
              correct_str = line.split(":", 1)[1].strip().lower()
              is_correct = correct_str == "yes"
          elif "FEEDBACK:" in line:
              feedback = line.split(":", 1)[1].strip()
   
      return {
          "is_correct": is_correct,
          "feedback": feedback
      }

  def _log_activity(
      self,
      user_id: str,
      activity_type: str,
      description: str
  ) -> None:
      """
      Log activity for monitoring and debugging.
   
      Args:
          user_id (str): User ID
          activity_type (str): Type of activity
          description (str): Activity description
      """
      log_entry = {
          "timestamp": datetime.now().isoformat(),
          "user_id": user_id,
          "activity_type": activity_type,
          "description": description
      }
      self.activity_log.append(log_entry)

  def _is_within_scope(self, user_input: str) -> bool:
      """
      Check if user input is within the scope of Python learning/coding assistance.
      This is Python-EXCLUSIVE - only Python programming is in scope.
     
      Args:
          user_input (str): The user's input to validate
         
      Returns:
          bool: True if in scope (Python), False if out of scope (not Python)
      """
      # Convert to lowercase for matching
      lower_input = user_input.lower().strip()
     
      # Python-exclusive scope keywords
      in_scope_keywords = [
          # Python core
          "python", "py", "pip", "venv", "virtualenv", "conda", "pandas", "numpy",
          "django", "flask", "fastapi", "requests", "beautifulsoup", "selenium",
          # Python concepts
          "def ", "function", "class ", "method", "import ", "from ",
          "variable", "list", "dict", "tuple", "set", "string", "int", "float", "bool",
          "loop", "for ", "while ", "if ", "else", "elif", "try", "except", "finally",
          "return", "yield", "raise", "with ", "lambda", "decorator", "self",
          # Python learning
          "learn", "tutorial", "lesson", "exercise", "practice", "teach", "teaching",
          "beginner", "intermediate", "advanced", "expert",
          # Python debugging
          "error", "bug", "fix", "debug", "exception", "traceback",
          "syntax", "indentation", "nameerror", "typeerror", "valueerror",
          # Python tools
          "jupyter", "notebook", "vscode", "pycharm", "idle", "debugger", "pdb",
          "pytest", "unittest", "pip install", "requirements",
      ]
     
      # Topics that are OUT of scope (non-Python programming)
      out_of_scope_keywords = [
          "javascript", "java ", "c++", "c#", "ruby ", "php ", "go ", "rust ",
          "html ", "css ", "sql ", "react", "angular", "vue", "node",
          "weather", "stock price", "sports", "news", "recipe", "cooking",
          "travel", "hotel", "flight", "music", "movie", "book",
          "health", "medical", "legal", "therapy", "politics", "religion",
      ]
     
      # Check for out-of-scope keywords first
      for keyword in out_of_scope_keywords:
          if keyword in lower_input:
              if keyword == "java " and "javascript" in lower_input:
                  continue
              return False
     
      # Check for in-scope Python keywords
      for keyword in in_scope_keywords:
          if keyword in lower_input:
              return True
     
      # Check for Python code patterns
      if self._is_code_input(user_input):
          return True
     
      # Check for Python help request patterns
      if self._is_help_request(user_input):
          return True
     
      # Check for Python lesson response patterns
      if self._is_lesson_response(user_input):
          return True
     
      # Check for Python concept explanation requests
      if self._is_python_explanation_request(user_input):
          return True
     
      # Check for profile request patterns
      if self._is_requesting_profile(user_input):
          return True
     
      # Default: uncertain - will NOT route to agent, will ask for clarification
      return False

  def _format_scope_warning(self, session: Dict, user_input: str) -> Dict:
      """
      Format a scope warning response to keep user focused on PYTHON LEARNING ONLY.
     
      Args:
          session (Dict): Current session
          user_input (str): User's out-of-scope input
         
      Returns:
          Dict: Formatted output to redirect user to Python
      """
      output = {
          "user_id": session["user_id"],
          "timestamp": datetime.now().isoformat(),
          "stage": session["stage"],
          "title": "🐍 Python Learning Only",
          "scope_warning": True,
          "next_action": "REDIRECT_TO_PYTHON"
      }
     
      # Generate friendly redirect message
      redirect_message = f"""🐍 I specialize in PYTHON programming exclusively.

I noticed you're asking about something outside Python:
"{user_input[:50]}..."

❌ I DON'T teach: JavaScript, Java, C++, C#, Ruby, PHP, or other languages
✅ I ONLY teach: Python (Python 3)

Here's what I CAN help you with in Python:
• 📝 Writing and fixing Python code
• 🐛 Debugging Python errors and exceptions
• 📚 Learning Python concepts and best practices
• 💻 Python practice exercises and lessons
• 🤔 Python concept explanations

What would you like to learn about Python?"""
     
      output["message"] = redirect_message
      output["next_prompt"] = """
Try asking me something like:
• "Help me write a Python function that..."
• "Fix this Python error: [paste your code]"
• "Explain list comprehensions in Python"
• "Give me a Python practice exercise"
"""
     
      return output

  @staticmethod
  def _is_successful_correction(user_input: str, session: Dict) -> bool:
      """
      Detect if user input represents a successful Python error correction.
   
      Args:
          user_input (str): User's input
          session (Dict): Current session data
       
      Returns:
          bool: True if input appears to be a successful correction
      """
      # Check if we have a recent lesson that was about error correction
      current_lesson = session.get("current_lesson", {})
      if not current_lesson:
          return False
   
      # Look for indicators of successful correction
      success_indicators = [
          "fixed", "corrected", "working", "it works", "solved", "done",
          "here's the fix", "this works", "now it runs", "success",
          "completed", "finished", "that's better"
      ]
   
      lower_input = user_input.lower()
   
      # Check for explicit success statements
      has_success_indicator = any(indicator in lower_input for indicator in success_indicators)
   
      # Check if input contains code (likely a correction attempt)
      has_code = OrchestratorAgent._is_code_input(user_input)
   
      # Check for positive feedback about the fix
      positive_feedback = any(word in lower_input for word in [
          "great", "good", "excellent", "perfect", "awesome", "nice",
          "better", "improved", "thank you", "thanks"
      ])
   
      return has_success_indicator or (has_code and positive_feedback)

  def _route_to_skill_building(self, session: Dict, user_input: str) -> Dict:
      """
      Route to Python skill building task generation after successful error correction.
   
      Args:
          session (Dict): Current session
          user_input (str): User's corrected Python code
       
      Returns:
          Dict: Skill building task response
      """
      if not self.error_training_agent:
          return {"error": "Error training agent not available"}
   
      current_lesson = session.get("current_lesson", {})
      current_level = session.get("level", "Intermediate")
   
      # Generate skill-building task
      skill_task_result = self.error_training_agent.generate_skill_building_task(
          student_level=current_level,
          error_type=current_lesson.get("error_type", "general"),
          error_description=current_lesson.get("error_description", "Python code error"),
          corrected_code=user_input
      )
      
      # Record token usage
      self._record_token_usage(
          session["user_id"],
          "ErrorTrainingAgent",
          skill_task_result,
          "Python skill building task generation"
      )
   
      # Extract task and comprehension prompt
      skill_task = skill_task_result.get("task", "")
      comprehension_prompt = skill_task_result.get("comprehension_prompt", "")
   
      self._log_activity(
          session["user_id"],
          "PYTHON_SKILL_BUILDING_TASK_GENERATED",
          f"Python skill task generated for {current_lesson.get('error_type', 'general')} correction"
      )
   
      # Format the message with both task and comprehension prompt
      message = f"🎯 Great job fixing that Python error! Here's a Python skill-building task to help you level up:\n\n{skill_task}\n\n"
   
      if comprehension_prompt:
          message += f"╔══════════════════════════════════════════════════════════════════════╗\n║                    💭 COMPREHENSION CHECK 💭                        ║\n╚══════════════════════════════════════════════════════════════════════╝\n\n{comprehension_prompt}\n\n"
   
      message += "Take 5-15 minutes to complete this Python task, then share your solution for feedback!"
   
      return {
          "agent": "ErrorTrainingAgent",
          "skill_task": skill_task,
          "comprehension_prompt": comprehension_prompt,
          "message": message
      }

  def _route_to_evaluate_correction(self, session: Dict, user_input: str) -> Dict:
      """
      Route to evaluate user's Python code correction attempt during training.
   
      Args:
          session (Dict): Current session
          user_input (str): User's Python code attempt
       
      Returns:
          Dict: Feedback response
      """
      # Evaluate the code correctness
      evaluation = self._evaluate_code_correctness(user_input, session)
   
      # Keep session in TRAINING stage since code is incorrect
      # The user will be prompted to try again
   
      self._log_activity(
          session["user_id"],
          "PYTHON_CODE_EVALUATION",
          f"Python code evaluated as {'correct' if evaluation['is_correct'] else 'incorrect'}"
      )
   
      # Format feedback message
      if not evaluation["is_correct"]:
          feedback_message = f"🤔 {evaluation['feedback']}\n\nPlease try again with your corrected Python code!"
      else:
          feedback_message = f"🎉 {evaluation['feedback']}\n\nGreat job! Your Python code is now correct."
   
      return {
          "agent": "OrchestratorAgent",
          "evaluation": evaluation,
          "message": feedback_message,
          "action": "CONTINUE_TRAINING"
      }

  def get_activity_log(self, user_id: str = None) -> List[Dict]:
      """
      Retrieve activity log.
   
      Args:
          user_id (str): Optional user ID to filter
       
      Returns:
          List[Dict]: Activity log entries
      """
      if user_id:
          return [log for log in self.activity_log if log["user_id"] == user_id]
      return self.activity_log

  def get_token_usage(self, user_id: str) -> Optional[Dict]:
      """
      Retrieve token usage statistics for a session.
      
      Args:
          user_id (str): User ID for the session
          
      Returns:
          Optional[Dict]: Token usage summary or None if session not found
      """
      if user_id not in self.token_ledgers:
          return None
      
      return self.token_ledgers[user_id].get_summary()

  def execute(
      self,
      user_id: str,
      user_input: str,
      user_name: str = "",
      is_new_session: bool = False
  ) -> Dict:
      """
      Execute the orchestrator agent (Python-exclusive).
   
      Main entry point for the Python learning system.
   
      Args:
          user_id (str): Unique identifier for the user
          user_input (str): The user's input
          user_name (str): Optional user name
          is_new_session (bool): Whether to start a new session
       
      Returns:
          Dict: Formatted output for the user
      """
      # Start new session if requested
      if is_new_session or user_id not in self.sessions:
          result = self.start_new_session(user_id, user_name)
          return result
   
      # Process input through the Python learning system
      return self.process_user_input(user_id, user_input)


