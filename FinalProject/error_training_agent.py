"""
Error Training Agent Module








This module defines an interactive tutoring agent that creates personalized
lessons based on a user's proficiency level and error patterns.








The agent:
- Receives level placement information from LevelPlacementAgent
- Calls ResourceRetrievalAgent for research-based teaching strategies
- Crafts brief, targeted lessons with appropriate complexity
- Adapts teaching style based on proficiency level
- Provides step-by-step error correction and learning
- Summarizesthe previous lesson








Teaching Approaches by Level:
- Novice: Foundational concepts, lots of examples, simple corrections
- Beginner: Guided discovery, pattern recognition, scaffolded practice
- Intermediate: Problem-solving, optimization, design patterns
- Proficient: Advanced techniques, edge cases, system design
- Master: Novel approaches, optimization challenges, mentoring perspectives








The agent interacts directly with users to deliver personalized, adaptive
tutoring experiences grounded in educational research.
"""








import os
from typing import Dict, List, Optional, Tuple
from enum import Enum
from abc import ABC








from dotenv import load_dotenv
from openai import OpenAI








from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser








# Load environment variables
load_dotenv()
















class TeachingStyle(Enum):
  """Enumeration of teaching styles adapted to proficiency levels."""
  DIRECTIVE = "Directive"  # Novice - Direct instruction, clear steps
  GUIDED = "Guided Discovery"  # Beginner - Questions, hints, scaffolding
  SOCRATIC = "Socratic Method"  # Intermediate - Questions to prompt thinking
  COLLABORATIVE = "Collaborative"  # Proficient - Partnership, discussion
  MENTORING = "Mentoring"  # Master - Peer learning, novel challenges
















class LessonComponent(Enum):
  """Components of a lesson structure."""
  OBJECTIVE = "Learning Objective"
  EXPLANATION = "Explanation"
  EXAMPLES = "Examples"
  COMMON_MISTAKES = "Common Mistakes"
  PRACTICE = "Practice Problems"
  RESOURCES = "Research Resources"
  SUMMARY = "Summary"
  NEXT_STEPS = "Next Steps"
















class ErrorTrainingAgent(ABC):
  """
  Interactive tutoring agent that creates personalized lessons based on
  proficiency level and error patterns.
  This agent serves as the primary interface for learners, delivering
  adaptive lessons that target specific errors while respecting the
  learner's current proficiency level.
  Features:
  - Adaptive lesson generation based on proficiency level
  - Integration with ResourceRetrievalAgent for evidence-based strategies
  - Multiple teaching style options
  - Personalized feedback and guidance
  - Progressive difficulty scaling
  - Error-focused instruction
  - Summarize lesson
  Attributes:
      name (str): The name/role of the agent
      openai_client (OpenAI): The OpenAI client instance
      llm (ChatOpenAI): The language model for generating lessons
      resource_agent (ResourceRetrievalAgent): For retrieving teaching strategies
  """
  # LLM Configuration
  LLM_MODEL = "gpt-4o-mini"
  TEMPERATURE = 0.7  # More creative for lesson design
  # Teaching style mapping by proficiency level
  LEVEL_TEACHING_STYLES = {
      "Novice": {
          "level": "Novice",
          "style": TeachingStyle.DIRECTIVE,
          "explanation_depth": "Basic and simplified",
          "example_count": 2,
          "practice_difficulty": "Very Easy",
          "code_annotation": True,
          "step_by_step": True
      },
      "Beginner": {
          "level": "Beginner",
          "style": TeachingStyle.GUIDED,
          "explanation_depth": "Moderate with foundational concepts",
          "example_count": 2,
          "practice_difficulty": "Easy",
          "code_annotation": True,
          "step_by_step": True
      },
      "Intermediate": {
          "level": "Intermediate",
          "style": TeachingStyle.SOCRATIC,
          "explanation_depth": "Moderate to advanced",
          "example_count": 1,
          "practice_difficulty": "Medium",
          "code_annotation": False,
          "step_by_step": False
      },
      "Proficient": {
          "level": "Proficient",
          "style": TeachingStyle.COLLABORATIVE,
          "explanation_depth": "Advanced with theoretical foundations",
          "example_count": 1,
          "practice_difficulty": "Hard",
          "code_annotation": False,
          "step_by_step": False
      },
      "Master": {
          "level": "Master",
          "style": TeachingStyle.MENTORING,
          "explanation_depth": "Expert level with advanced concepts",
          "example_count": 1,
          "practice_difficulty": "Very Hard / Novel",
          "code_annotation": False,
          "step_by_step": False
      }
  }
  def __init__(self, name: str = "Error Training Specialist",
               resource_agent=None):
      """
      Initialize the Error Training Agent.
   
      Args:
          name (str): The name or role identifier for this agent
          resource_agent (ResourceRetrievalAgent): Optional resource agent for
                                                   teaching strategies
   
      Raises:
          KeyError: If OPENAI_API_KEY environment variable is not set
      """
      self.name = name
      self.resource_agent = resource_agent
   
      # Initialize OpenAI client
      try:
          api_key = os.getenv("OPENAI_API_KEY")
          if not api_key:
              raise KeyError("OPENAI_API_KEY environment variable not set")
          self.openai_client = OpenAI(api_key=api_key)
      except KeyError as e:
          raise KeyError(f"Failed to initialize OpenAI client: {e}")
   
      # Initialize LLM for lesson generation
      self.llm = ChatOpenAI(
          model=self.LLM_MODEL,
          temperature=self.TEMPERATURE
      )
  def create_lesson(
      self,
      student_level: str,
      error_type: str,
      error_description: str,
      error_code: str = "",
      topic: str = ""
  ) -> Dict:
      """
      Create a personalized lesson targeting a specific error.
   
      Args:
          student_level (str): Student's proficiency level
          error_type (str): Type of error (syntax, logic, redundancy)
          error_description (str): Description of the error
          error_code (str): The actual error code (optional)
          topic (str): The programming topic related to the error (optional)
       
      Returns:
          Dict: Comprehensive lesson with all components
      """
      # Get teaching approach for this level
      teaching_approach = self.LEVEL_TEACHING_STYLES.get(
          student_level,
          self.LEVEL_TEACHING_STYLES["Intermediate"]
      )








   
      # Retrieve teaching strategies from resource agent
      resource_guidance = None
      if self.resource_agent:
          resource_guidance = self._get_teaching_strategies(
              error_type, student_level, topic
          )
   
      # Generate lesson components based on proficiency level
      lesson = {
          "student_level": student_level,
          "error_type": error_type,
          "teaching_style": teaching_approach["style"].value,
          "components": {}
      }




      lesson["topic"] = self._generate_topic(
          error_type=error_type,
          error_description=error_description,
          user_input=error_description or error_code
              )
   
      # Always include objective and explanation
      lesson["components"]["objective"] = self._generate_objective(
          error_type, student_level, topic, error_description
      )
   
      lesson["components"]["explanation"] = self._generate_explanation(
          error_type, error_description, teaching_approach, resource_guidance, student_level
      )
   
      # Always include common mistakes
      lesson["components"]["common_mistakes"] = self._generate_common_mistakes(
          error_type, student_level
      )
   
      # Include additional components for intermediate/advanced learners
      if student_level in ["Intermediate", "Proficient", "Master"]:
          lesson["components"]["examples"] = self._generate_examples(
              error_type, error_code, teaching_approach
          )
       
          lesson["components"]["practice"] = self._generate_practice_problems(
              error_type, teaching_approach, student_level
          )
       
          if resource_guidance:
              lesson["components"]["resources"] = resource_guidance
       
          lesson["components"]["summary"] = self._generate_summary(
              error_type, teaching_approach
          )
       
          lesson["components"]["next_steps"] = self._generate_next_steps(
              student_level, error_type
          )
     
   
      return lesson
  def _get_teaching_strategies(
      self,
      error_type: str,
      student_level: str,
      topic: str = ""
  ) -> Dict:
      """
      Retrieve teaching strategies from the resource retrieval agent.
   
      Args:
          error_type (str): Type of error to address
          student_level (str): Student's proficiency level
          topic (str): Optional topic for more specific guidance
       
      Returns:
          Dict: Teaching strategies and resources
      """
      if not self.resource_agent:
          return None
   
      try:
          query = f"How to teach {error_type} errors to {student_level} students"
          if topic:
              query += f" in {topic}"
       
          result = self.resource_agent.execute(
              query=f"{topic} | {student_level}" if topic else f"Error handling | {student_level}",
              action="guidance"
          )
          return result
      except Exception as e:
          print(f"[{self.name}] Could not retrieve guidance: {e}")
          return None
     
  def _generate_topic(self, error_type: str, error_description: str, user_input: str = "") -> str:
   """Generate a short topic label summarizing the lesson."""
   prompt = ChatPromptTemplate.from_template("""
                                             Summarize the core topic of this Python lesson in 3–8 words.
                                             Error type: {error_type}
                                             Error description: {error_description}
                                             User input: {user_input}
                                             Examples:
                                             - "Fixing nested loops"
                                             - "Fixing a bubble sort function
                                             - "Understanding variable scope"
                                             - "Correcting off-by-one errors"
                                             - "Using try/except blocks"
                                             Return ONLY the topic text.
                                             """)
   chain = prompt | self.llm | StrOutputParser()
   return chain.invoke({
       "error_type": error_type,
       "error_description": error_description,
       "user_input": user_input[:200]
   })




  def _generate_objective(
      self,
      error_type: str,
      student_level: str,
      topic: str = "",
      user_input: str = ""
  ) -> str:
      """Generate learning objective for the lesson, tailored to Python learning."""
      prompt = ChatPromptTemplate.from_template("""Create a clear, concise learning objective for a {student_level} Python learner
addressing {error_type} errors{topic_ref}.








The objective should:
- Start with action verb (Understand, Identify, Fix, etc.)
- Be specific to Python programming concepts and syntax
- Match the student's proficiency level
- Focus on practical Python application
- Address the specific context from: "{user_input}"








Provide ONLY the learning objective, no other text.""")
   
      topic_ref = f" in {topic}" if topic else ""
   
      chain = prompt | self.llm | StrOutputParser()
      return chain.invoke({
          "student_level": student_level,
          "error_type": error_type,
          "topic_ref": topic_ref,
          "user_input": user_input[:200]  # Limit input length
      })
  def _generate_explanation(
      self,
      error_type: str,
      error_description: str,
      teaching_approach: Dict,
      resource_guidance: Dict = None,
      student_level: str = "Intermediate"
  ) -> str:
      """Generate explanation tailored to proficiency level."""
      depth = teaching_approach["explanation_depth"]
      style = teaching_approach["style"].value
   
      context = ""
      if resource_guidance:
          context = f"\nBased on research guidance: {resource_guidance.get('guidance', '')[:200]}"
   
      # Different prompts based on proficiency level
      if student_level in ["Novice", "Beginner"]:
          prompt = ChatPromptTemplate.from_template("""Create a simple, step-by-step explanation of {error_type} errors
using clear, direct instructions.








Error: {error_description}{context}








Requirements:
- Keep total explanation under 250 words
- Include a MAXIMUM of 3 numbered steps to fix the error
- Use simple language appropriate for {depth} level
- Focus on practical, actionable guidance
- End with a simple tip








Structure:
1. Brief problem statement
2. **Bold each step** like **Check Syntax** and **Use Quotation Marks**
3. One practical tip""")
      else:
          prompt = ChatPromptTemplate.from_template("""Create a focused explanation of {error_type} errors
with tangible steps and understanding checks.








Error: {error_description}{context}








Requirements:
- Keep total explanation under 300 words
- Include a MAXIMUM of 3 numbered tangible steps to fix the error
- Add 2-3 quick understanding checks (questions)
- Cite one key research insight if available
- Use {depth} language
- End with a practical tip








Structure:
1. Brief problem statement
2. **Bold each step** like **Check Syntax** and **Use Quotation Marks**
3. Understanding checks
4. Key takeaway""")
   
      chain = prompt | self.llm | StrOutputParser()
      return chain.invoke({
          "error_type": error_type,
          "error_description": error_description,
          "depth": depth,
          "style": style,
          "context": context
      })
  def _generate_examples(
      self,
      error_type: str,
      error_code: str = "",
      teaching_approach: Dict = None
  ) -> List[str]:
      """Generate examples of the error."""
      example_count = teaching_approach.get("example_count", 2) if teaching_approach else 2
      include_annotations = teaching_approach.get("code_annotation", False) if teaching_approach else False
   
      annotation_note = "Include detailed comments explaining what goes wrong." if include_annotations else ""
   
      prompt = ChatPromptTemplate.from_template("""Generate {count} Python code examples showing {error_type} errors.








Original problematic code:{code_context}








Requirements:
- Create {count} distinct examples
- Show the error clearly
- Include output/error message
{annotation_note}
- Separate examples with "---EXAMPLE---"








Format each example as:
# Example N: [Description]
[Code with error]
# Error: [What goes wrong]""")
   
      code_context = f"\n{error_code}" if error_code else ""
   
      chain = prompt | self.llm | StrOutputParser()
      examples_text = chain.invoke({
          "error_type": error_type,
          "count": example_count,
          "code_context": code_context,
          "annotation_note": annotation_note
      })
   
      # Split examples
      return [ex.strip() for ex in examples_text.split("---EXAMPLE---") if ex.strip()]
  def _generate_common_mistakes(
      self,
      error_type: str,
      student_level: str
  ) -> str:
      """Generate 2-3 key mistakes to avoid, specific to Python learning."""
      prompt = ChatPromptTemplate.from_template("""List 2-3 most common mistakes that {student_level} Python learners make specifically with {error_type} errors in Python programming.








Focus on mistakes that are:
- Common for {student_level} level Python learners
- Directly related to Python syntax, semantics, or best practices
- Specific to the {error_type} error type








Format as bullet points:
• Mistake: [brief description of the Python-specific mistake] → Fix: [Python-specific solution]








Keep it concise and actionable for Python learning.""")
   
      chain = prompt | self.llm | StrOutputParser()
      return chain.invoke({
          "error_type": error_type,
          "student_level": student_level
      })
  def _generate_comprehension_prompt(
      self,
      error_type: str,
      error_description: str,
      student_level: str,
      error_code: str = ""
  ) -> str:
      """Generate a specific comprehension question or code task prompt."""
      code_context = f"Code context:\n{error_code}\n" if error_code else ""
      prompt = ChatPromptTemplate.from_template("""Create one highly specific prompt for a {student_level} Python learner who has just corrected a {error_type} error.








Error description: {error_description}
{code_context}








Deliver a single follow-up item that will shape the learner's next input. Use one of these formats:
- Question: [a focused comprehension question]
- Code Task: [a short, concrete code task the learner can complete]








The prompt should be concise, actionable, and directly tied to the error correction.
""")
      chain = prompt | self.llm | StrOutputParser()
      return chain.invoke({
          "student_level": student_level,
          "error_type": error_type,
          "error_description": error_description,
          "code_context": code_context
      })
  def _generate_practice_problems(
      self,
      error_type: str,
      teaching_approach: Dict,
      student_level: str
  ) -> str:
      """Generate one focused practice problem."""
      difficulty = teaching_approach.get("practice_difficulty", "Medium")
   
      prompt = ChatPromptTemplate.from_template("""Create 1 focused practice problem for {student_level} students
to apply {error_type} error correction.








Difficulty: {difficulty}








Format as:
Practice Problem: [Brief description]
[Code snippet with the error to fix]
Expected fix: [What they should change]








Keep it concise and directly applicable to the lesson.""")
   
      chain = prompt | self.llm | StrOutputParser()
      return chain.invoke({
          "error_type": error_type,
          "student_level": student_level,
          "difficulty": difficulty
      })
  def _generate_summary(
      self,
      error_type: str,
      teaching_approach: Dict
  ) -> str:
      """Generate lesson summary."""
      style = teaching_approach["style"].value
   
      prompt = ChatPromptTemplate.from_template("""Create a brief, memorable summary of how to identify and fix {error_type} errors.








Teaching style: {style}








Use this summary to reinforce key concepts in a memorable way.
Keep it to 3-4 key points maximum.""")
   
      chain = prompt | self.llm | StrOutputParser()
      return chain.invoke({
          "error_type": error_type,
          "style": style
      })
  def _generate_next_steps(
      self,
      student_level: str,
      error_type: str
  ) -> str:
      """Generate 2-3 actionable next steps."""
      prompt = ChatPromptTemplate.from_template("""Suggest 2-3 immediate next steps for a {student_level} student
after learning {error_type} error correction.








Format as bullet points, keep each step actionable and brief.








Examples:
• Practice with similar code patterns
• Apply this fix to your current project
• Learn related error types""")
   
      chain = prompt | self.llm | StrOutputParser()
      return chain.invoke({
          "student_level": student_level,
          "error_type": error_type
      })
  def format_lesson_for_display(self, lesson: Dict) -> str:
      """
      Format a lesson tailored to proficiency level.
   
      Args:
          lesson (Dict): The lesson dictionary from create_lesson()
       
      Returns:
          str: Formatted lesson text
      """
      student_level = lesson['student_level']
      is_advanced = student_level in ["Intermediate", "Proficient", "Master"]
   
      output = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    PERSONALIZED LEARNING LESSON                      ║
╚══════════════════════════════════════════════════════════════════════╝








📚 LEVEL: {lesson['student_level']} | 🎯 ERROR: {lesson['error_type'].upper()}








━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━








🎯 OBJECTIVE:
{lesson['components'].get('objective', 'N/A')}








━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━








📖 HOW TO FIX:
{lesson['components'].get('explanation', 'N/A')}








━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━








⚠️  COMMON PITFALLS:
{lesson['components'].get('common_mistakes', 'N/A')}
"""
   
      # Add advanced sections only for intermediate/advanced learners
      if is_advanced:
          examples = lesson['components'].get('examples', [])
          if examples:
              output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━








📝 EXAMPLE:
{examples[0]}
"""
       
          if 'practice' in lesson['components']:
              output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━








✏️  TRY THIS:
{lesson['components']['practice']}
"""
       
          if 'resources' in lesson['components']:
              guidance = lesson['components']['resources'].get('guidance', '')
              if guidance:
                  key_insight = guidance.split('.')[0] + '.'
                  output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━








🔍 KEY RESEARCH INSIGHT:
{key_insight}
"""
   
      output += "\n╚══════════════════════════════════════════════════════════════════════╝\n"
      return output
  def interactive_session(
      self,
      student_level: str,
      student_code: str = ""
  ) -> Dict:
      """
      Start an interactive tutoring session.
   
      This is the main entry point for interactive tutoring.
   
      Args:
          student_level (str): Student's proficiency level
          student_code (str): Optional student code to analyze
       
      Returns:
          Dict: Session results and lessons
      """
      session = {
          "student_level": student_level,
          "sessions": []
      }
   
      # If student code provided, analyze it
      if student_code:
          analysis = self._analyze_student_code(student_code, student_level)
          session["initial_analysis"] = analysis
       
          # Create lessons for each error found
          for error in analysis.get("errors", []):
              lesson = self.create_lesson(
                  student_level=student_level,
                  error_type=error["type"],
                  error_description=error["description"],
                  error_code=error.get("code_snippet", ""),
                  topic=analysis.get("topic", "")
              )
              session["sessions"].append(lesson)
   
      return session
  def _analyze_student_code(self, code: str, student_level: str) -> Dict:
      """
      Analyze student code to identify errors.
   
      Args:
          code (str): The student's code
          student_level (str): Student's proficiency level
       
      Returns:
          Dict: Analysis of errors found
      """
      prompt = ChatPromptTemplate.from_template("""Analyze this Python code from a {student_level} student.
Identify and categorize errors (syntax, logic, redundancy).








Code:
{code}








Provide response in this format:
TOPIC: [main programming topic]
ERRORS:
1. Type: [syntax/logic/redundancy], Description: [what's wrong], Snippet: [relevant code]
2. (continue for each error found)








If no errors: write "ERRORS: None found"
"""
      )
   
      chain = prompt | self.llm | StrOutputParser()
      analysis_text = chain.invoke({
          "code": code,
          "student_level": student_level
      })
   
      # Parse the analysis
      return self._parse_analysis(analysis_text)
  @staticmethod
  def _parse_analysis(analysis_text: str) -> Dict:
      """Parse the analysis output into structured format."""
      lines = analysis_text.strip().split('\n')
      result = {"errors": []}
      topic = "General Programming"
   
      for line in lines:
          if "TOPIC:" in line:
              topic = line.split(":", 1)[1].strip()
          elif "Type:" in line:
              # Parse error line
              error_parts = line.split(",")
              error = {}
              for part in error_parts:
                  if "Type:" in part:
                      error["type"] = part.split(":", 1)[1].strip()
                  elif "Description:" in part:
                      error["description"] = part.split(":", 1)[1].strip()
                  elif "Snippet:" in part:
                      error["code_snippet"] = part.split(":", 1)[1].strip()
              if error:
                  result["errors"].append(error)
   
      result["topic"] = topic
      return result
  def execute(
      self,
      student_level: str,
      error_type: str = "",
      error_description: str = "",
      error_code: str = "",
      student_code: str = ""
  ) -> Dict:
      """
      Execute the error training agent.
   
      Args:
          student_level (str): Student's proficiency level
          error_type (str): Type of error to teach (optional if student_code provided)
          error_description (str): Description of the error
          error_code (str): Code example showing the error
          student_code (str): Student's code to analyze (overrides other params)
       
      Returns:
          Dict: Lesson(s) and session information
      """
      if student_code:
          # Analyze student code and create lessons
          return self.interactive_session(student_level, student_code)
      elif error_type and error_description:
          # Create specific lesson
          lesson = self.create_lesson(
              student_level=student_level,
              error_type=error_type,
              error_description=error_description,
              error_code=error_code
          )
          return {"lesson": lesson}
      else:
          raise ValueError("Must provide either student_code or (error_type and error_description)")
  def generate_skill_building_task(
      self,
      student_level: str,
      error_type: str,
      error_description: str,
      corrected_code: str = ""
  ) -> Dict[str, str]:
      """
      Generate a specific skill-building task when an error is successfully corrected.
   
      This method creates targeted practice tasks that help users build skills
      at their current proficiency level, focusing on the error type they just fixed.
   
      Args:
          student_level (str): Student's current proficiency level
          error_type (str): Type of error that was corrected
          error_description (str): Description of the error
          corrected_code (str): The corrected code (optional)
       
      Returns:
          Dict[str, str]: Contains 'task' and 'comprehension_prompt' keys
      """
      # Get teaching approach for this level
      teaching_approach = self.LEVEL_TEACHING_STYLES.get(
          student_level,
          self.LEVEL_TEACHING_STYLES["Intermediate"]
      )
   
      prompt = ChatPromptTemplate.from_template("""You are a Python programming tutor. A {student_level} student has just successfully corrected a {error_type} error.








Error description: {error_description}
Corrected code (if provided): {corrected_code}








Generate ONE specific, actionable skill-building task that:
1. Builds on the error they just fixed
2. Is appropriate for their {student_level} level
3. Takes 5-15 minutes to complete
4. Helps them practice the concept without being too easy or difficult
5. Includes a clear success criteria








Format as a single, engaging task description that starts with an action verb.








Examples:
- "Create a function that validates user input using try-except blocks, ensuring it handles at least 3 different input types"
- "Write a script that processes a list of dictionaries, using list comprehensions to filter and transform the data"
- "Implement a class with proper encapsulation that manages a collection of items with add/remove/search methods"








Make it specific and immediately actionable.""")
   
      chain = prompt | self.llm | StrOutputParser()
      task = chain.invoke({
          "student_level": student_level,
          "error_type": error_type,
          "error_description": error_description,
          "corrected_code": corrected_code or "Not provided"
      })
   
      # Generate comprehension prompt
      comprehension_prompt = self._generate_comprehension_prompt(
          error_type=error_type,
          error_description=error_description,
          student_level=student_level,
          error_code=corrected_code
      )
   
      return {
          "task": task,
          "comprehension_prompt": comprehension_prompt
      }

























