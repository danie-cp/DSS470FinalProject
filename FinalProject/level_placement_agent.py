"""
Level Placement Assessment Agent Module








This module defines the Level Placement criteria based on the user's input.
This agent will sort the user into one of 5 placements:
- Novice
- Beginner
- Intermediate
- Proficient
- Master








The assessment is based on:
- Syntax Errors: Violations of language grammar and structure
- Logic Errors: Flaws in algorithm and program flow
- Redundancy Errors: Unnecessary code duplication and inefficiency
- Task Complexity: The complexity of the programming task being attempted (1-10 scale)








This agent receives the user's initial input from the Orchestrator Agent
and returns a level placement classification with detailed feedback.
"""








from abc import ABC
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv
from openai import OpenAI








# Load environment variables from .env file
load_dotenv()
















class ProficiencyLevel(Enum):
  """Enumeration of proficiency levels."""
  NOVICE = "Novice"
  BEGINNER = "Beginner"
  INTERMEDIATE = "Intermediate"
  PROFICIENT = "Proficient"
  MASTER = "Master"
















@dataclass
class LevelCriteria:
  """
  Defines the criteria for each proficiency level.
  Attributes:
      level (ProficiencyLevel): The proficiency level
      syntax_error_range (Tuple[int, int]): Min/max count of syntax errors expected
      logic_error_range (Tuple[int, int]): Min/max count of logic errors expected
      redundancy_error_range (Tuple[int, int]): Min/max count of redundancy errors expected
      task_complexity_range (Tuple[int, int]): Min/max task complexity score (1-10)
      description (str): Detailed description of the level
      characteristics (List[str]): Key characteristics of developers at this level
  """
  level: ProficiencyLevel
  syntax_error_range: Tuple[int, int]
  logic_error_range: Tuple[int, int]
  redundancy_error_range: Tuple[int, int]
  task_complexity_range: Tuple[int, int]
  description: str
  characteristics: List[str]
















class LevelPlacementCriteria:
  """
  Defines comprehensive level placement criteria for code assessment.
  This class stores the characteristics and error thresholds for each
  of the 5 proficiency levels.
  """
  LEVEL_DEFINITIONS: Dict[ProficiencyLevel, LevelCriteria] = {
      ProficiencyLevel.NOVICE: LevelCriteria(
          level=ProficiencyLevel.NOVICE,
          syntax_error_range=(5, float('inf')),
          logic_error_range=(4, float('inf')),
          redundancy_error_range=(3, float('inf')),
          task_complexity_range=(1, 2),
          description=(
              "Novice level represents developers who are just beginning their programming journey. "
              "Their code contains frequent fundamental errors and demonstrates minimal understanding of programming concepts."
          ),
          characteristics=[
              "Frequent syntax errors (mismatched brackets, incorrect indentation, missing punctuation)",
              "Misunderstanding of basic language constructs (loops, conditionals, functions)",
              "No understanding of DRY (Don't Repeat Yourself) principle",
              "Massive code duplication and copy-paste programming"
          ]
      ),
      ProficiencyLevel.BEGINNER: LevelCriteria(
          level=ProficiencyLevel.BEGINNER,
          syntax_error_range=(2, 4),
          logic_error_range=(2, 3),
          redundancy_error_range=(2, 3),
          task_complexity_range=(2, 4),
          description=(
              "Beginner level represents developers with basic programming knowledge. "
              "They understand fundamental concepts but still make regular mistakes in implementation."
          ),
          characteristics=[
              "Occasional syntax errors that are usually caught during testing",
              "Basic understanding of control flow but sometimes incorrect logic",
              "Some code duplication, but attempts to avoid it in obvious cases",
              "Functions and variables exist but naming could be clearer"
          ]
      ),
      ProficiencyLevel.INTERMEDIATE: LevelCriteria(
          level=ProficiencyLevel.INTERMEDIATE,
          syntax_error_range=(0, 1),
          logic_error_range=(0, 1),
          redundancy_error_range=(1, 2),
          task_complexity_range=(4, 6),
          description=(
              "Intermediate level represents competent developers. "
              "They write mostly correct code with good structure and minimal errors."
          ),
          characteristics=[
              "Rare syntax errors (code is generally syntactically correct)",
              "Sound logic with proper control flow and edge case handling",
              "Minimal code duplication; follows DRY principle in most places",
              "Clear, descriptive variable and function names"
          ]
      ),
      ProficiencyLevel.PROFICIENT: LevelCriteria(
          level=ProficiencyLevel.PROFICIENT,
          syntax_error_range=(0, 0),
          logic_error_range=(0, 0),
          redundancy_error_range=(0, 1),
          task_complexity_range=(6, 8),
          description=(
              "Proficient level represents skilled developers. "
              "Their code is production-ready with excellent structure and design."
          ),
          characteristics=[
              "No syntax errors; code is syntactically perfect",
              "No logic errors; comprehensive edge case handling",
              "Virtually no code duplication; excellent DRY implementation",
              "Excellent naming conventions across all identifiers"
          ]
      ),
      ProficiencyLevel.MASTER: LevelCriteria(
          level=ProficiencyLevel.MASTER,
          syntax_error_range=(0, 0),
          logic_error_range=(0, 0),
          redundancy_error_range=(0, 0),
          task_complexity_range=(8, 10),
          description=(
              "Master level represents expert developers. "
              "Their code exemplifies excellence across all dimensions: correctness, design, performance, and maintainability."
          ),
          characteristics=[
              "Perfect syntax; code is exemplary",
              "Perfect logic; anticipates edge cases before they occur",
              "Zero code duplication; elegant, DRY solutions",
              "Exceptional naming and documentation"
          ]
      )
  }








  @classmethod
  def get_criteria(cls, level: ProficiencyLevel) -> LevelCriteria:
      """
      Get the criteria for a specific proficiency level.
   
      Args:
          level (ProficiencyLevel): The proficiency level to retrieve criteria for
       
      Returns:
          LevelCriteria: The criteria definition for the specified level
      """
      return cls.LEVEL_DEFINITIONS[level]








  @classmethod
  def get_all_criteria(cls) -> Dict[ProficiencyLevel, LevelCriteria]:
      """
      Get all level criteria definitions.
   
      Returns:
          Dict[ProficiencyLevel, LevelCriteria]: All level criteria
      """
      return cls.LEVEL_DEFINITIONS








  @classmethod
  def assess_level(
      cls,
      syntax_errors: int,
      logic_errors: int,
      redundancy_errors: int,
      task_complexity: int
  ) -> Tuple[ProficiencyLevel, LevelCriteria]:
      """
      Assess the proficiency level based on error counts and task complexity.
   
      Args:
          syntax_errors (int): Number of syntax errors found
          logic_errors (int): Number of logic errors found
          redundancy_errors (int): Number of redundancy errors found
          task_complexity (int): Task complexity score (1-10)
       
      Returns:
          Tuple[ProficiencyLevel, LevelCriteria]: The assessed level and its criteria
      """
      # Check each level in reverse order (Master to Novice)
      for level in reversed(list(ProficiencyLevel)):
          criteria = cls.get_criteria(level)
       
          syntax_min, syntax_max = criteria.syntax_error_range
          logic_min, logic_max = criteria.logic_error_range
          redundancy_min, redundancy_max = criteria.redundancy_error_range
          complexity_min, complexity_max = criteria.task_complexity_range
       
          # Check if the error counts and task complexity match this level's range
          if (syntax_min <= syntax_errors <= syntax_max and
              logic_min <= logic_errors <= logic_max and
              redundancy_min <= redundancy_errors <= redundancy_max and
              complexity_min <= task_complexity <= complexity_max):
              return level, criteria
   
      # Default to Novice if no match (shouldn't happen)
      return ProficiencyLevel.NOVICE, cls.get_criteria(ProficiencyLevel.NOVICE)
















class LevelPlacementAgent(ABC):
  """
  Agent responsible for assessing and assigning proficiency levels.
  This agent receives code samples or descriptions from the Orchestrator Agent,
  analyzes them for syntax errors, logic errors, and redundancy errors,
  and returns a proficiency level assessment with detailed feedback.
  Attributes:
      name (str): The name/role of the agent ("Level Placement Specialist")
      openai_client (OpenAI): The OpenAI client instance for making API calls
  """
  def __init__(self, name: str = "Level Placement Specialist"):
      """
      Initialize the Level Placement Agent.
   
      Args:
          name (str): The name or role identifier for this agent
                     (default: "Level Placement Specialist")
   
      Raises:
          KeyError: If OPENAI_API_KEY environment variable is not set
      """
      self.name = name
   
      # Initialize OpenAI client with API key from environment
      try:
          api_key = os.getenv("OPENAI_API_KEY")
          if not api_key:
              raise KeyError("OPENAI_API_KEY environment variable not set")
          self.openai_client = OpenAI(api_key=api_key)
      except KeyError as e:
          raise KeyError(f"Failed to initialize OpenAI client: {e}")








  def analyze_code(self, code_input: str) -> Dict:
      """
      Analyze code input and return a proficiency level assessment.
   
      This method processes the code input using the OpenAI API to identify
      syntax errors, logic errors, and redundancy errors, then assigns
      an appropriate proficiency level.
   
      Args:
          code_input (str): The code sample or description to analyze
       
      Returns:
          Dict: Assessment results including:
              - level: ProficiencyLevel
              - criteria: LevelCriteria
              - syntax_errors: int
              - logic_errors: int
              - redundancy_errors: int
              - task_complexity: int (1-10 scale)
              - feedback: str
              - recommendations: List[str]
      """
      assessment_prompt = f"""
      You are an expert code reviewer specializing in proficiency assessment.
   
      Analyze the following code or code description and identify:
      1. Syntax Errors: violations of language grammar and structure
      2. Logic Errors: flaws in algorithm and program flow
      3. Redundancy Errors: unnecessary code duplication and inefficiency
      4. Task Complexity: rate the complexity of the programming task on a scale of 1-10
         where 1 = very simple (hello world, basic input/output)
         and 10 = extremely complex (distributed systems, advanced algorithms, novel solutions)
   
      Consider these factors for task complexity:
      - Algorithmic complexity (basic operations vs advanced algorithms)
      - Domain knowledge required (basic vs specialized/expert knowledge)
      - Integration complexity (single function vs multi-system integration)
      - Problem-solving depth (straightforward vs requiring deep analysis)
      - Scale and scope (small script vs large application)
   
      Code/Task to analyze:
      {code_input}
   
      Provide your response in this exact format:
      SYNTAX_ERRORS: [number]
      LOGIC_ERRORS: [number]
      REDUNDANCY_ERRORS: [number]
      TASK_COMPLEXITY: [1-10 score]
      FEEDBACK: [detailed feedback]
      RECOMMENDATIONS: [comma-separated list of improvement recommendations]
      """
   
      try:
          response = self.openai_client.chat.completions.create(
              model="gpt-4",
              messages=[
                  {"role": "system", "content": "You are an expert code proficiency assessor."},
                  {"role": "user", "content": assessment_prompt}
              ],
              temperature=0.5,
              max_tokens=1000
          )
       
          response_text = response.choices[0].message.content
       
          # Parse the response
          syntax_errors = self._extract_value(response_text, "SYNTAX_ERRORS")
          logic_errors = self._extract_value(response_text, "LOGIC_ERRORS")
          redundancy_errors = self._extract_value(response_text, "REDUNDANCY_ERRORS")
          task_complexity = self._extract_value(response_text, "TASK_COMPLEXITY")
          feedback = self._extract_field(response_text, "FEEDBACK")
          recommendations = self._extract_recommendations(response_text, "RECOMMENDATIONS")
       
          # Assess level based on error counts and task complexity
          level, criteria = LevelPlacementCriteria.assess_level(
              syntax_errors, logic_errors, redundancy_errors, task_complexity
          )
       
          return {
              "level": level,
              "criteria": criteria,
              "syntax_errors": syntax_errors,
              "logic_errors": logic_errors,
              "redundancy_errors": redundancy_errors,
              "task_complexity": task_complexity,
              "feedback": feedback,
              "recommendations": recommendations
          }
       
      except Exception as e:
          raise RuntimeError(f"Error during code analysis: {e}")








  def generate_report(self, assessment: Dict) -> str:
      """
      Generate a detailed assessment report from the analysis results.
   
      Args:
          assessment (Dict): The assessment results from analyze_code()
       
      Returns:
          str: A formatted report with all assessment details
      """
      level = assessment["level"]
      criteria = assessment["criteria"]
   
      report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║              PROFICIENCY LEVEL ASSESSMENT REPORT                     ║
╚══════════════════════════════════════════════════════════════════════╝








PROFICIENCY LEVEL: {level.value.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━








ERROR ANALYSIS:
• Syntax Errors:       {assessment['syntax_errors']}
• Logic Errors:        {assessment['logic_errors']}
• Redundancy Errors:   {assessment['redundancy_errors']}








TASK COMPLEXITY: {assessment['task_complexity']}/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━








DESCRIPTION:
{criteria.description}








KEY CHARACTERISTICS:
"""
      for i, char in enumerate(criteria.characteristics, 1):
          report += f"  {i:2d}. {char}\n"
   
      report += f"""
FEEDBACK FROM ANALYSIS:
















RECOMMENDATIONS FOR IMPROVEMENT:
"""
      for i, rec in enumerate(assessment['recommendations'], 1):
          report += f"  {i}. {rec}\n"
   
      report += """
╚══════════════════════════════════════════════════════════════════════╝
"""
      return report








  @staticmethod
  def _extract_value(text: str, key: str) -> int:
      """Extract an integer value from the response text."""
      try:
          for line in text.split('\n'):
              if key in line:
                  return int(line.split(':')[1].strip())
          return 0
      except (IndexError, ValueError):
          return 0








  @staticmethod
  def _extract_field(text: str, key: str) -> str:
      """Extract a text field from the response."""
      try:
          for line in text.split('\n'):
              if key in line:
                  return line.split(':', 1)[1].strip()
          return ""
      except (IndexError, ValueError):
          return ""








  @staticmethod
  def _extract_recommendations(text: str, key: str) -> List[str]:
      """Extract a comma-separated list of recommendations."""
      try:
          for line in text.split('\n'):
              if key in line:
                  recs_text = line.split(':', 1)[1].strip()
                  return [r.strip() for r in recs_text.split(',')]
          return []
      except (IndexError, ValueError):
          return []








  def execute(self, user_input: str) -> Dict:
      """
      Execute the level placement assessment.
   
      Args:
          user_input (str): The code or description from the Orchestrator Agent
       
      Returns:
          Dict: Complete assessment results and report
      """
      assessment = self.analyze_code(user_input)
      report = self.generate_report(assessment)
   
      return {
          "agent": self.name,
          "assessment": assessment,
          "report": report
      }

























