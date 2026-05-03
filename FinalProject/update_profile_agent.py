"""
Update Profile Agent Module


This module defines an agent that analyzes user performance and updates
their Python learner profile based on:
- Responses to error training lessons
- Error patterns and improvements
- Level placement criteria
- Learning progress and trajectory


The agent:
- Evaluates lesson performance and comprehension
- Tracks error reduction and skill acquisition
- Assigns or updates proficiency levels
- Provides personalized growth recommendations
- Maintains continuous learning history
- Suggests next topics and challenges
- Reports token usage statistics


This agent is called after each error_training_agent session to maintain
continuous, autonomous learning with real-time profile updates.


Profile Components:
- Learner Metadata: Name, start date, total sessions
- Performance Metrics: Accuracy, improvement, speed
- Current Level: Assigned proficiency level with justification
- Strengths: Areas of demonstrated competency
- Growth Areas: Areas needing improvement
- Recommendations: Personalized next steps
- Learning History: Session records and progression
- Token Usage: Token consumption tracking per session
"""


import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from abc import ABC


from dotenv import load_dotenv
from openai import OpenAI


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()




@dataclass
class TokenUsageEntry:
   """
   Record of token usage for a specific activity.
  
   Attributes:
       timestamp (str): When tokens were used
       agent_name (str): Name of the agent that used tokens
       activity_description (str): What activity used the tokens
       prompt_tokens (int): Input tokens
       completion_tokens (int): Output tokens
       total_tokens (int): Sum of prompt and completion tokens
   """
   timestamp: str
   agent_name: str
   activity_description: str
   prompt_tokens: int
   completion_tokens: int
   total_tokens: int




@dataclass
class TokenUsageSummary:
   """
   Summary of token usage across a session or multiple sessions.
  
   Attributes:
       total_tokens (int): Total tokens used
       prompt_tokens (int): Total input tokens
       completion_tokens (int): Total output tokens
       token_entries (List[TokenUsageEntry]): Detailed token usage records
       session_count (int): Number of sessions tracked
   """
   total_tokens: int
   prompt_tokens: int
   completion_tokens: int
   token_entries: List[TokenUsageEntry]
   session_count: int




@dataclass
class LearnerProfile:
   """
   Comprehensive profile of a Python learner.
   Attributes:
       learner_id (str): Unique identifier for the learner
       learner_name (str): Name of the learner
       current_level (str): Current proficiency level (Novice-Master)
       start_date (str): When the learner started
       total_sessions (int): Number of learning sessions completed
       total_errors_addressed (int): Cumulative errors worked on
       average_comprehension (float): Average comprehension score 0-100
       error_reduction_rate (float): Percentage reduction in similar errors
       session_history (List[Dict]): Records of all sessions
       strengths (List[str]): Demonstrated competencies
       growth_areas (List[str]): Areas needing improvement
       recommendations (List[str]): Personalized growth recommendations
       last_updated (str): Timestamp of last profile update
       next_recommended_topics (List[str]): Suggested learning topics
       token_usage_summary (TokenUsageSummary): Accumulated token usage
   """
   learner_id: str
   learner_name: str
   current_level: str
   start_date: str
   total_sessions: int
   total_errors_addressed: int
   average_comprehension: float
   error_reduction_rate: float
   session_history: List[Dict]
   strengths: List[str]
   growth_areas: List[str]
   recommendations: List[str]
   last_updated: str
   next_recommended_topics: List[str]
   token_usage_summary: TokenUsageSummary = field(default_factory=lambda: TokenUsageSummary(
       total_tokens=0,
       prompt_tokens=0,
       completion_tokens=0,
       token_entries=[],
       session_count=0
   ))




class PerformanceMetric(Enum):
   """Performance metrics for evaluation."""
   EXCELLENT = 90  # 90-100%
   GOOD = 75       # 75-89%
   SATISFACTORY = 60  # 60-74%
   NEEDS_IMPROVEMENT = 45  # 45-59%
   STRUGGLING = 0   # 0-44%




class UpdateProfileAgent(ABC):
   """
   Agent that updates learner profiles based on lesson performance.
   This agent analyzes responses to error training lessons and updates
   comprehensive learner profiles. It tracks progress, identifies strengths
   and growth areas, and provides personalized recommendations for continued
   learning.
   The agent integrates with:
   - LevelPlacementAgent for proficiency criteria
   - ErrorTrainingAgent for lesson content and student responses
   - OrchestratorAgent for token tracking
   Features:
   - Performance evaluation and scoring
   - Proficiency level updates
   - Error pattern analysis
   - Growth recommendation generation
   - Learning history tracking
   - Autonomous profile maintenance
   - Token usage tracking and reporting
   Attributes:
       name (str): The name/role of the agent
       openai_client (OpenAI): The OpenAI client instance
       llm (ChatOpenAI): The language model for analysis
       profiles (Dict): Storage of learner profiles
   """
   # LLM Configuration
   LLM_MODEL = "gpt-4o-mini"
   TEMPERATURE = 0.5  # More analytical than creative
   def __init__(self, name: str = "Profile Update Specialist"):
       """
       Initialize the Update Profile Agent.
   
       Args:
           name (str): The name or role identifier for this agent
   
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
   
       # Initialize LLM for analysis
       self.llm = ChatOpenAI(
           model=self.LLM_MODEL,
           temperature=self.TEMPERATURE
       )
   
       # Profile storage
       self.profiles: Dict[str, LearnerProfile] = {}
   def create_learner_profile(
       self,
       learner_id: str,
       learner_name: str,
       initial_level: str = "Novice"
   ) -> LearnerProfile:
       """
       Create a new learner profile.
   
       Args:
           learner_id (str): Unique identifier for the learner
           learner_name (str): Name of the learner
           initial_level (str): Starting proficiency level
       
       Returns:
           LearnerProfile: Newly created profile
       """
       profile = LearnerProfile(
           learner_id=learner_id,
           learner_name=learner_name,
           current_level=initial_level,
           start_date=datetime.now().isoformat(),
           total_sessions=0,
           total_errors_addressed=0,
           average_comprehension=0.0,
           error_reduction_rate=0.0,
           session_history=[],
           strengths=[],
           growth_areas=[],
           recommendations=[],
           last_updated=datetime.now().isoformat(),
           next_recommended_topics=[],
           token_usage_summary=TokenUsageSummary(
               total_tokens=0,
               prompt_tokens=0,
               completion_tokens=0,
               token_entries=[],
               session_count=0
           )
       )
   
       self.profiles[learner_id] = profile
       return profile
   def get_profile(self, learner_id: str) -> Optional[LearnerProfile]:
       """
       Retrieve a learner profile.
   
       Args:
           learner_id (str): Unique identifier for the learner
       
       Returns:
           Optional[LearnerProfile]: The learner's profile or None
       """
       return self.profiles.get(learner_id)
   def record_token_usage(
       self,
       learner_id: str,
       agent_name: str,
       prompt_tokens: int,
       completion_tokens: int,
       activity_description: str = ""
   ) -> None:
       """
       Record token usage for a learner's session.
      
       This method aggregates token usage from all agents (LevelPlacementAgent,
       ErrorTrainingAgent, OrchestratorAgent, etc.) into the learner's profile
       for transparent reporting.
      
       Args:
           learner_id (str): Unique identifier for the learner
           agent_name (str): Name of the agent that used tokens
           prompt_tokens (int): Number of prompt/input tokens
           completion_tokens (int): Number of completion/output tokens
           activity_description (str): Description of what used tokens
       """
       profile = self.profiles.get(learner_id)
       if not profile:
           return
      
       total = prompt_tokens + completion_tokens
      
       # Create token usage entry
       entry = TokenUsageEntry(
           timestamp=datetime.now().isoformat(),
           agent_name=agent_name,
           activity_description=activity_description,
           prompt_tokens=prompt_tokens,
           completion_tokens=completion_tokens,
           total_tokens=total
       )
      
       # Update summary
       profile.token_usage_summary.total_tokens += total
       profile.token_usage_summary.prompt_tokens += prompt_tokens
       profile.token_usage_summary.completion_tokens += completion_tokens
       profile.token_usage_summary.token_entries.append(entry)
  
   def compile_token_report(
       self,
       learner_id: str,
       include_detailed_entries: bool = False
   ) -> Dict:
       """
       Compile a comprehensive token usage report for a learner.
      
       This method generates a report of all tokens consumed across the
       learning session, broken down by agent and activity.
      
       Args:
           learner_id (str): Unique identifier for the learner
           include_detailed_entries (bool): Whether to include detailed per-activity entries
      
       Returns:
           Dict: Token usage report with summary and optional details
       """
       profile = self.profiles.get(learner_id)
       if not profile:
           return {
               "error": f"Profile not found for learner {learner_id}",
               "total_tokens": 0,
               "prompt_tokens": 0,
               "completion_tokens": 0
           }
      
       summary = profile.token_usage_summary
      
       # Calculate averages
       avg_tokens_per_session = (
           summary.total_tokens / summary.session_count
           if summary.session_count > 0
           else 0
       )
      
       # Aggregate by agent
       agent_totals: Dict[str, Dict] = {}
       for entry in summary.token_entries:
           agent = entry.agent_name
           if agent not in agent_totals:
               agent_totals[agent] = {
                   "prompt_tokens": 0,
                   "completion_tokens": 0,
                   "total_tokens": 0,
                   "call_count": 0
               }
           agent_totals[agent]["prompt_tokens"] += entry.prompt_tokens
           agent_totals[agent]["completion_tokens"] += entry.completion_tokens
           agent_totals[agent]["total_tokens"] += entry.total_tokens
           agent_totals[agent]["call_count"] += 1
      
       report = {
           "learner_id": learner_id,
           "learner_name": profile.learner_name,
           "session_count": summary.session_count,
           "total_tokens_used": summary.total_tokens,
           "total_prompt_tokens": summary.prompt_tokens,
           "total_completion_tokens": summary.completion_tokens,
           "average_tokens_per_session": avg_tokens_per_session,
           "tokens_by_agent": agent_totals,
           "report_generated": datetime.now().isoformat()
       }
      
       if include_detailed_entries:
           report["detailed_entries"] = [
               {
                   "timestamp": entry.timestamp,
                   "agent": entry.agent_name,
                   "activity": entry.activity_description,
                   "prompt_tokens": entry.prompt_tokens,
                   "completion_tokens": entry.completion_tokens,
                   "total_tokens": entry.total_tokens
               }
               for entry in summary.token_entries
           ]
      
       return report
  
   def format_token_report_for_display(
       self,
       learner_id: str,
       include_agent_breakdown: bool = True
   ) -> str:
       """
       Format token report as a user-friendly display string.
      
       Creates a formatted report suitable for displaying to users at the
       end of their learning session.
      
       Args:
           learner_id (str): Unique identifier for the learner
           include_agent_breakdown (bool): Whether to include per-agent breakdown
      
       Returns:
           str: Formatted token usage report
       """
       report = self.compile_token_report(learner_id, include_detailed_entries=False)
      
       if "error" in report:
           return f"Token Report Error: {report['error']}"
      
       output = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     📊 TOKEN USAGE REPORT                            ║
╚══════════════════════════════════════════════════════════════════════╝


SESSION INFORMATION:
├─ Learner: {report['learner_name']}
├─ Total Sessions: {report['session_count']}
└─ Report Generated: {report['report_generated']}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


AGGREGATE TOKEN USAGE:
├─ Total Tokens Used: {report['total_tokens_used']:,}
├─ Prompt Tokens: {report['total_prompt_tokens']:,}
├─ Completion Tokens: {report['total_completion_tokens']:,}
└─ Average per Session: {report['average_tokens_per_session']:,.0f}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
      
       if include_agent_breakdown and report['tokens_by_agent']:
           output += "\nTOKEN USAGE BY AGENT:\n"
           for agent, stats in sorted(
               report['tokens_by_agent'].items(),
               key=lambda x: x[1]['total_tokens'],
               reverse=True
           ):
               output += f"""
{agent}:
 ├─ Total Tokens: {stats['total_tokens']:,}
 ├─ Prompt Tokens: {stats['prompt_tokens']:,}
 ├─ Completion Tokens: {stats['completion_tokens']:,}
 └─ API Calls: {stats['call_count']}
"""
      
       output += "\n╚══════════════════════════════════════════════════════════════════════╝\n"
       return output
  
   def evaluate_lesson_performance(
       self,
       lesson_data: Dict,
       user_response: str,
       student_level: str
   ) -> Dict:
       """
       Evaluate student performance on a lesson.
   
       Args:
           lesson_data (Dict): The lesson from ErrorTrainingAgent
           user_response (str): Student's response/attempt
           student_level (str): Current proficiency level
       
       Returns:
           Dict: Performance evaluation
       """
       prompt = ChatPromptTemplate.from_template("""Evaluate this student's performance on a Python error training lesson.


LESSON INFORMATION:
- Level: {student_level}
- Error Type: {error_type}
- Learning Objective: {objective}


STUDENT RESPONSE:
{user_response}


Evaluate on these criteria:
1. Comprehension (0-100): Understanding of the concept
2. Application (0-100): Ability to apply learning
3. Error Identification (0-100): Ability to spot similar errors
4. Code Quality (0-100): Quality of provided code
5. Explanation Quality (0-100): Clarity of reasoning


Provide response in this format:
COMPREHENSION: [score]
APPLICATION: [score]
ERROR_IDENTIFICATION: [score]
CODE_QUALITY: [score]
EXPLANATION_QUALITY: [score]
STRENGTHS: [key strengths shown]
AREAS_FOR_GROWTH: [areas needing work]
OVERALL_ASSESSMENT: [brief assessment]
READY_FOR_NEXT_LEVEL: [yes/no/maybe]""")
   
       chain = prompt | self.llm | StrOutputParser()
       evaluation_text = chain.invoke({
           "student_level": student_level,
           "error_type": lesson_data.get("error_type", "Unknown"),
           "objective": lesson_data.get("components", {}).get("objective", ""),
           "user_response": user_response
       })
   
       return self._parse_evaluation(evaluation_text)
   @staticmethod
   def _parse_evaluation(evaluation_text: str) -> Dict:
       """Parse evaluation output into structured format."""
       result = {
           "comprehension": 0,
           "application": 0,
           "error_identification": 0,
           "code_quality": 0,
           "explanation_quality": 0,
           "strengths": [],
           "areas_for_growth": [],
           "overall_assessment": "",
           "ready_for_next_level": False
       }
   
       lines = evaluation_text.strip().split('\n')
   
       for line in lines:
           if "COMPREHENSION:" in line:
               try:
                   result["comprehension"] = int(line.split(":", 1)[1].strip())
               except ValueError:
                   pass
           elif "APPLICATION:" in line:
               try:
                   result["application"] = int(line.split(":", 1)[1].strip())
               except ValueError:
                   pass
           elif "ERROR_IDENTIFICATION:" in line:
               try:
                   result["error_identification"] = int(line.split(":", 1)[1].strip())
               except ValueError:
                   pass
           elif "CODE_QUALITY:" in line:
               try:
                   result["code_quality"] = int(line.split(":", 1)[1].strip())
               except ValueError:
                   pass
           elif "EXPLANATION_QUALITY:" in line:
               try:
                   result["explanation_quality"] = int(line.split(":", 1)[1].strip())
               except ValueError:
                   pass
           elif "STRENGTHS:" in line:
               strengths_text = line.split(":", 1)[1].strip()
               result["strengths"] = [s.strip() for s in strengths_text.split(",")]
           elif "AREAS_FOR_GROWTH:" in line:
               growth_text = line.split(":", 1)[1].strip()
               result["areas_for_growth"] = [a.strip() for a in growth_text.split(",")]
           elif "OVERALL_ASSESSMENT:" in line:
               result["overall_assessment"] = line.split(":", 1)[1].strip()
           elif "READY_FOR_NEXT_LEVEL:" in line:
               ready_text = line.split(":", 1)[1].strip().lower()
               result["ready_for_next_level"] = ready_text in ["yes", "true", "ready"]
   
       return result
   def determine_level_progression(
       self,
       current_level: str,
       performance_evaluation: Dict,
       level_criteria: Dict
   ) -> Tuple[str, str]:
       """
       Determine if student should progress to next level.
   
       Args:
           current_level (str): Current proficiency level
           performance_evaluation (Dict): Performance evaluation results
           level_criteria (Dict): Criteria from LevelPlacementAgent
       
       Returns:
           Tuple[str, str]: (new_level, justification)
       """
       levels = ["Novice", "Beginner", "Intermediate", "Proficient", "Master"]
       current_idx = levels.index(current_level) if current_level in levels else 0
   
       # Calculate average performance
       avg_score = (
           performance_evaluation.get("comprehension", 0) +
           performance_evaluation.get("application", 0) +
           performance_evaluation.get("error_identification", 0) +
           performance_evaluation.get("code_quality", 0) +
           performance_evaluation.get("explanation_quality", 0)
       ) / 5
   
       # Determine progression
       if avg_score >= 90 and performance_evaluation.get("ready_for_next_level"):
           if current_idx < len(levels) - 1:
               new_level = levels[current_idx + 1]
               justification = f"Excellent performance ({avg_score:.1f}%) with demonstrated mastery. Ready for {new_level} challenges."
           else:
               new_level = current_level
               justification = "Master level achieved - continue advancing skills."
       elif avg_score >= 75:
           new_level = current_level
           justification = f"Good performance ({avg_score:.1f}%) at current level. Continue strengthening skills before progression."
       elif avg_score >= 60:
           new_level = current_level
           justification = f"Satisfactory performance ({avg_score:.1f}%). More practice needed at this level."
       else:
           if current_idx > 0:
               new_level = levels[current_idx - 1]
               justification = f"Below-target performance ({avg_score:.1f}%). Recommending review at {new_level} level."
           else:
               new_level = current_level
               justification = f"Needs support ({avg_score:.1f}%). Additional Novice-level practice recommended."
   
       return new_level, justification
   def update_learner_profile(
       self,
       learner_id: str,
       lesson_data: Dict,
       user_response: str,
       current_level: str,
       level_criteria: Dict = None,
       token_ledger_data: Dict = None
   ) -> LearnerProfile:
       """
       Update a learner's profile based on lesson performance.
   
       Args:
           learner_id (str): Unique identifier for the learner
           lesson_data (Dict): The lesson from ErrorTrainingAgent
           user_response (str): Student's response/attempt
           current_level (str): Current proficiency level
           level_criteria (Dict): Optional criteria from LevelPlacementAgent
           token_ledger_data (Dict): Optional token ledger data from OrchestratorAgent
       
       Returns:
           LearnerProfile: Updated profile
       """
       # Get or create profile
       if learner_id not in self.profiles:
           profile = self.create_learner_profile(
               learner_id, learner_id, current_level
           )
       else:
           profile = self.profiles[learner_id]
   
       # Evaluate performance
       evaluation = self.evaluate_lesson_performance(
           lesson_data, user_response, current_level
       )
   
       # Determine level progression
       new_level, progression_justification = self.determine_level_progression(
           current_level, evaluation, level_criteria or {}
       )
   
       # Generate growth recommendations
       recommendations = self._generate_recommendations(
           new_level, evaluation, lesson_data
       )
   
       # Generate next recommended topics
       next_topics = self._recommend_next_topics(new_level, evaluation)
   
       # Calculate metrics
       avg_performance = (
           evaluation.get("comprehension", 0) +
           evaluation.get("application", 0) +
           evaluation.get("error_identification", 0) +
           evaluation.get("code_quality", 0) +
           evaluation.get("explanation_quality", 0)
       ) / 5
   
       # Update profile
       profile.total_sessions += 1
       profile.total_errors_addressed += 1
       profile.average_comprehension = (
           (profile.average_comprehension * (profile.total_sessions - 1) + avg_performance)
           / profile.total_sessions
       )
       profile.current_level = new_level
       profile.strengths = list(set(profile.strengths + evaluation.get("strengths", [])))
       profile.growth_areas = evaluation.get("areas_for_growth", [])
       profile.recommendations = recommendations
       profile.last_updated = datetime.now().isoformat()
       profile.next_recommended_topics = next_topics
   
       # Update token usage summary session count
       profile.token_usage_summary.session_count = profile.total_sessions
      
       # Process token ledger data if provided from OrchestratorAgent
       if token_ledger_data:
           self._ingest_orchestrator_token_data(learner_id, token_ledger_data)
   
       # Add to session history
       session_record = {
           "session_number": profile.total_sessions,
           "timestamp": datetime.now().isoformat(),
           "error_type": lesson_data.get("error_type", "Unknown"),
           "performance": avg_performance,
           "level_before": current_level,
           "level_after": new_level,
           "evaluation": evaluation,
           "progression_note": progression_justification
       }
       profile.session_history.append(session_record)
   
       return profile
  
   def _ingest_orchestrator_token_data(
       self,
       learner_id: str,
       token_ledger_data: Dict
   ) -> None:
       """
       Ingest token usage data from OrchestratorAgent's token ledger.
      
       This method receives aggregated token data from the orchestrator
       and incorporates it into the learner's profile.
      
       Args:
           learner_id (str): Unique identifier for the learner
           token_ledger_data (Dict): Token data from OrchestratorAgent ledger
       """
       profile = self.profiles.get(learner_id)
       if not profile:
           return
      
       # Extract data from orchestrator token ledger
       total_tokens = token_ledger_data.get("total_tokens", 0)
       prompt_tokens = token_ledger_data.get("prompt_tokens", 0)
       completion_tokens = token_ledger_data.get("completion_tokens", 0)
       entries = token_ledger_data.get("entries", [])
      
       # Update summary
       profile.token_usage_summary.total_tokens = total_tokens
       profile.token_usage_summary.prompt_tokens = prompt_tokens
       profile.token_usage_summary.completion_tokens = completion_tokens
      
       # Convert orchestrator entries to TokenUsageEntry objects
       for entry in entries:
           token_entry = TokenUsageEntry(
               timestamp=entry.get("timestamp", datetime.now().isoformat()),
               agent_name=entry.get("agent", "Unknown"),
               activity_description=entry.get("activity", ""),
               prompt_tokens=entry.get("prompt_tokens", 0),
               completion_tokens=entry.get("completion_tokens", 0),
               total_tokens=entry.get("total_tokens", 0)
           )
           profile.token_usage_summary.token_entries.append(token_entry)
   def _generate_recommendations(
       self,
       current_level: str,
       evaluation: Dict,
       lesson_data: Dict
   ) -> List[str]:
       """Generate personalized growth recommendations."""
       prompt = ChatPromptTemplate.from_template("""Based on a {level} student's performance, generate 3-4 specific,
actionable recommendations for Python learning growth.


Performance Scores:
- Comprehension: {comprehension}
- Application: {application}
- Error Identification: {error_identification}


Strengths: {strengths}
Growth Areas: {growth_areas}


Recommendations should:
- Be specific and actionable
- Build on demonstrated strengths
- Address identified growth areas
- Match the student's proficiency level
- Progress toward mastery


Format as a bulleted list.""")
   
       chain = prompt | self.llm | StrOutputParser()
       recommendations_text = chain.invoke({
           "level": current_level,
           "comprehension": evaluation.get("comprehension", 0),
           "application": evaluation.get("application", 0),
           "error_identification": evaluation.get("error_identification", 0),
           "strengths": ", ".join(evaluation.get("strengths", [])),
           "growth_areas": ", ".join(evaluation.get("areas_for_growth", []))
       })
   
       return [r.strip() for r in recommendations_text.split('\n') if r.strip()]
   def _recommend_next_topics(
       self,
       current_level: str,
       evaluation: Dict
   ) -> List[str]:
       """Generate recommended next learning topics."""
       prompt = ChatPromptTemplate.from_template("""Suggest 3-5 specific Python topics for a {level} student to learn next.


Current Strengths: {strengths}
Areas for Growth: {growth_areas}


Topics should:
- Build on current strengths
- Address identified growth areas
- Be progressively more challenging
- Be relevant to practical Python development
- Align with {level}-level expectations


Format as a bulleted list with brief descriptions.""")
   
       chain = prompt | self.llm | StrOutputParser()
       topics_text = chain.invoke({
           "level": current_level,
           "strengths": ", ".join(evaluation.get("strengths", [])),
           "growth_areas": ", ".join(evaluation.get("areas_for_growth", []))
       })
   
       return [t.strip() for t in topics_text.split('\n') if t.strip()]
   def format_profile_for_display(
       self,
       profile: LearnerProfile,
       initial_assessment: bool = False,
       include_token_report: bool = False
   ) -> str:
       """
       Format learner profile for user-friendly display.
   
       Args:
           profile (LearnerProfile): The learner profile to display
           initial_assessment (bool): Whether this is the initial assessment display
           include_token_report (bool): Whether to include token usage report
       
       Returns:
           str: Formatted profile text
       """
       bold = lambda text: f"\033[1m{text}\033[0m"
      
       if initial_assessment:
           # For initial assessment, show only the proficiency level
           output = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    PYTHON LEARNER PROFILE                            ║
╚══════════════════════════════════════════════════════════════════════╝


{bold('🏆 PROFICIENCY LEVEL:')} {profile.current_level.upper()}


╚══════════════════════════════════════════════════════════════════════╝
"""
           return output
      
       output = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    PYTHON LEARNER PROFILE                            ║
╚══════════════════════════════════════════════════════════════════════╝


{bold('👤 LEARNER INFORMATION:')}
Name: {profile.learner_name}
ID: {profile.learner_id}
Started: {profile.start_date}
Sessions Completed: {profile.total_sessions}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


{bold('📊 PROFICIENCY LEVEL:')} {profile.current_level}


Performance Metrics:
Average Comprehension: {profile.average_comprehension:.1f}%
Error Reduction Rate: {profile.error_reduction_rate:.1f}%
Total Errors Addressed: {profile.total_errors_addressed}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━���━━━━━━━━━━━━━━━━━━━━━━━
"""
       if profile.strengths:
           output += f"\n{bold('✅ STRENGTHS:')}\n"
           for strength in profile.strengths[:5]:
               output += f"  • {strength}\n\n"


       if profile.growth_areas:
           output += f"\n{bold('⚠️  GROWTH AREAS:')}\n"
           for area in profile.growth_areas[:5]:
               output += f"  • {area}\n\n"


       if profile.recommendations:
           output += f"\n{bold('💡 PERSONALIZED RECOMMENDATIONS:')}\n"
           for rec in profile.recommendations[:4]:
               output += f"  • {rec}\n\n"


       if profile.next_recommended_topics:
           output += f"\n{bold('🚀 RECOMMENDED NEXT TOPICS:')}\n"
           for topic in profile.next_recommended_topics[:5]:
               output += f"  • {topic}\n\n"
           output += "  Tip: Choose one of these recommended next topics as the basis for the next component of your lesson.\n\n"


       output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━��━━━━━━━━━━━


{bold('📈 LEARNING PROGRESSION:')}
Total Sessions: {profile.total_sessions}
Last Updated: {profile.last_updated}
"""
   
       if profile.session_history:
           output += f"\n  Recent Session:\n"
           last_session = profile.session_history[-1]
           output += f"    Error Type: {last_session.get('error_type', 'N/A')}\n"
           output += f"    Performance: {last_session.get('performance', 0):.1f}%\n"
           output += f"    Level Progression: {last_session.get('level_before')} → {last_session.get('level_after')}\n"
           output += f"    Note: {last_session.get('progression_note', 'N/A')}\n"
      
       # Add token report if requested
       if include_token_report:
           output += "\n" + self.format_token_report_for_display(profile.learner_id)
   
       output += "\n╚══════════════════════════════════════════════════════════════════════╝\n"
       return output
   def get_session_history(self, learner_id: str) -> List[Dict]:
       """
       Retrieve full session history for a learner.
   
       Args:
           learner_id (str): Unique identifier for the learner
       
       Returns:
           List[Dict]: Session history records
       """
       profile = self.profiles.get(learner_id)
       return profile.session_history if profile else []
   def export_profile(self, learner_id: str) -> Dict:
       """
       Export learner profile as dictionary.
   
       Args:
           learner_id (str): Unique identifier for the learner
       
       Returns:
           Dict: Complete profile data
       """
       profile = self.profiles.get(learner_id)
       if not profile:
           return {}
       return asdict(profile)
   def execute(
       self,
       learner_id: str,
       lesson_data: Dict,
       user_response: str,
       current_level: str,
       learner_name: str = "",
       level_criteria: Dict = None,
       token_ledger_data: Dict = None
   ) -> Dict:
       """
       Execute the update profile agent.
   
       This is the main entry point called after ErrorTrainingAgent.
   
       Args:
           learner_id (str): Unique identifier for the learner
           lesson_data (Dict): The lesson from ErrorTrainingAgent
           user_response (str): Student's response/attempt
           current_level (str): Current proficiency level
           learner_name (str): Optional learner name
           level_criteria (Dict): Optional criteria from LevelPlacementAgent
           token_ledger_data (Dict): Optional token ledger data from OrchestratorAgent
       
       Returns:
           Dict: Updated profile and session information with token summary
       """
       # Create profile if doesn't exist
       if learner_id not in self.profiles:
           self.create_learner_profile(
               learner_id,
               learner_name or learner_id,
               current_level
           )
   
       # Update profile with lesson results
       profile = self.update_learner_profile(
           learner_id, lesson_data, user_response,
           current_level, level_criteria, token_ledger_data
       )
      
       # Compile token report for return
       token_report = self.compile_token_report(learner_id, include_detailed_entries=False)
   
       return {
           "agent": self.name,
           "learner_id": learner_id,
           "profile": asdict(profile),
           "formatted_display": self.format_profile_for_display(profile, lesson_data.get("initial_assessment", False)),
           "token_report": token_report,
           "token_report_display": self.format_token_report_for_display(learner_id)
       }

