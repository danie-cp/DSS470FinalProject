import os
import sys
import json
from pathlib import Path
import time
from datetime import datetime

from typing import Dict, List






# Memory file path
MEMORY_FILE = Path(__file__).parent / "chatbox_memory.json"








def load_memory() -> dict:
   """Load user memory from JSON file."""
   if MEMORY_FILE.exists():
       try:
           with open(MEMORY_FILE, "r") as f:
               return json.load(f)
       except (json.JSONDecodeError, IOError):
           return {"users": {}}
   return {"users": {}}








def save_memory(data: dict) -> None:
   """Save user memory to JSON file."""
   with open(MEMORY_FILE, "w") as f:
       json.dump(data, f, indent=2)








def get_user_memory(user_id: str) -> dict:
   """Get memory for a specific user."""
   memory = load_memory()
   return memory.get("users", {}).get(user_id, {
       "user_id": user_id,
       "last_session_datetime": None,
       "last_topic_discussed": None,
       "performance_in_topic": None,
       "last_updated_level": None
   })








def update_user_memory(
   user_id: str,
   last_topic: str = None,
   performance: float = None,
   level: str = None
) -> None:
   """Update user memory with new session data."""
   memory = load_memory()
 
   if "users" not in memory:
       memory["users"] = {}
 
   if user_id not in memory["users"]:
       memory["users"][user_id] = {
           "user_id": user_id,
           "last_session_datetime": None,
           "last_topic_discussed": None,
           "performance_in_topic": None,
           "last_updated_level": None
       }
 
   user_mem = memory["users"][user_id]
 
   # Update fields if provided
   if last_topic is not None:
       user_mem["last_topic_discussed"] = last_topic
   if performance is not None:
       user_mem["performance_in_topic"] = performance
   if level is not None:
       user_mem["last_updated_level"] = level
 
   # Always update the session datetime
   user_mem["last_session_datetime"] = datetime.now().isoformat()
 
   save_memory(memory)








def get_user_level(user_id: str) -> str:
   """Get the user's current level from memory."""
   user_mem = get_user_memory(user_id)
   return user_mem.get("last_updated_level")








# Ensure the project root is on sys.path so imports work when executing this script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))








from FinalProject.orchestrator_agent import OrchestratorAgent
from FinalProject.level_placement_agent import LevelPlacementAgent
from FinalProject.error_training_agent import ErrorTrainingAgent
from FinalProject.update_profile_agent import UpdateProfileAgent
from FinalProject.resource_retrieval_agent import ResourceRetrievalAgent








from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
















console = Console()
















def display_header() -> None:
  header_text = Text(
      "Python Learning Chatbox\n\n"
      "• Share code for assessment\n"
      "• Ask for help from scratch\n"
      "• Type ':multi' to enter multi-line input mode\n"
      "• In multi-line mode, type ':end' when ready to submit\n"
      "• Type 'exit' to quit at any time\n",
      justify="left",
  )
  console.print(Rule("[bold cyan]Python Learning Chatbox[/]"))
  console.print(Panel(header_text, border_style="bright_blue", expand=False))
















def highlight_keywords(text: str) -> Text:
  # First, process markdown bold **text** and convert to Rich markup
  import re
  processed_text = re.sub(r'\*\*([^*]+)\*\*', r'[bold]\1[/bold]', text)
  styled = Text.from_markup(processed_text)
  keywords = {
      "error": "bold red",
      "errors": "bold red",
      "warning": "bold yellow",
      "mistake": "bold yellow",
      "mistakes": "bold yellow",
      "invalid": "bold red",
      "failed": "bold red",
      "note:": "bold blue",
      "tip:": "bold green",
      "important": "bold magenta",
  }
  lower = processed_text.lower()
  for keyword, style in keywords.items():
      start = 0
      while True:
          index = lower.find(keyword, start)
          if index == -1:
              break
          styled.stylize(style, index, index + len(keyword))
          start = index + len(keyword)
  return styled
















def stream_text(text: str, delay: float = 0.01) -> None:
  """Stream text to the console with a trailing generation effect."""
  from rich.live import Live








  displayed = Text()
  with Live(displayed, console=console, refresh_per_second=60, transient=False) as live:
      for char in text:
          displayed.append(char)
          displayed = highlight_keywords(str(displayed))
          live.update(displayed)
          time.sleep(delay)
















def render_message(message: str, stream: bool = True) -> None:
  if "```" not in message:
      if stream:
          stream_text(message)
      else:
          console.print(highlight_keywords(message), justify="left")
      return








  chunks = message.split("```")
  for index, chunk in enumerate(chunks):
      if index % 2 == 0:
          if chunk.strip():
              if stream:
                  stream_text(chunk)
              else:
                  console.print(highlight_keywords(chunk), justify="left")
      else:
          code = chunk.strip().lstrip("python\n")
          syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
          console.print(syntax)
















def display_response(response: dict) -> None:
  title = response.get("title")
  if title:
      console.print(Rule(f"[bold green]{title}[/]", style="green"))








  message = response.get("message", "")
  if message:
      console.print(Panel.fit(Text("Assistant", style="bold white"), border_style="green", subtitle_align="left"))
      render_message(message)








  next_prompt = response.get("next_prompt")
  if next_prompt:
      console.print(Panel(highlight_keywords(next_prompt), border_style="yellow", title="Next Step"))
















def display_user_message(user_input: str) -> None:
  user_text = Text(user_input, style="bold cyan")
  console.print(Rule("[bold cyan]You[/]", style="cyan"))
  console.print(user_text)
















def display_goodbye() -> None:
  console.print(Rule("[bold magenta]Goodbye![/]"))
  console.print("Thanks for using the Python Learning Chatbox.")
















def collect_multiline_input() -> str:
  console.print(
      Panel(
          Text(
              "Enter your multi-line text or code.\nType ':end', ':done', or ':submit' on its own line when finished.\nType ':cancel' to abort and return to the main prompt.",
              justify="left",
          ),
          title="Multi-line Input Mode",
          border_style="bright_blue",
      )
  )
  lines = []
  while True:
      try:
          line = console.input("[bold cyan]...[/] ")
          if not line:  # Handle empty input
              continue
           
          command = line.strip().lower()
          if command in [":end", ":done", ":submit"]:
              if not lines:  # No content collected
                  console.print("[yellow]No content entered. Use ':cancel' to abort or enter some text first.[/]")
                  continue
              return "\n".join(lines)
          elif command == ":cancel":
              console.print("[yellow]Multi-line input canceled. Returning to main prompt.[/]")
              return ""
          elif command in ["exit", "quit", "bye"]:
              console.print("[yellow]Exiting multi-line input mode.[/]")
              return line
          else:
              lines.append(line)
      except KeyboardInterrupt:
          console.print("\n[yellow]Multi-line input interrupted. Returning to main prompt.[/]")
          return ""
      except EOFError:
          console.print("\n[yellow]End of input reached. Returning to main prompt.[/]")
          return ""
















def get_user_input() -> str:
  try:
      first_line = console.input("\n[bold cyan]You:[/] ")
      if not first_line:  # Handle empty input
          return get_user_input()
       
      command = first_line.strip().lower()
      if command in [":multi", ":multiline", ":code"]:
          console.print("[green]Entering multi-line input mode...[/]")
          multiline = collect_multiline_input()
          # Handle exit commands that were entered during multi-line mode
          if multiline.strip().lower() in ["exit", "quit", "bye"]:
              return multiline
          # Handle cancel - return to main prompt without any input
          if multiline == "":
              console.print("[dim]Returning to main prompt...[/]")
              return get_user_input()
          # Return the collected multi-line input
          return multiline
      return first_line
  except KeyboardInterrupt:
      console.print("\n[yellow]Input interrupted. Type 'exit' to quit or continue...[/]")
      return get_user_input()
  except EOFError:
      console.print("\n[yellow]End of input reached. Type 'exit' to quit.[/]")
      return "exit"
















def main():
   display_header()




   # Initialize agents
   level_agent = LevelPlacementAgent()
   resource_agent = ResourceRetrievalAgent()
   error_agent = ErrorTrainingAgent(resource_agent=resource_agent)
   update_agent = UpdateProfileAgent()




   # Initialize orchestrator
   orch = OrchestratorAgent(
       level_placement_agent=level_agent,
       error_training_agent=error_agent,
       update_profile_agent=update_agent,
   )




   # Get user ID (could be from login or input)
   user_id = "146"




   # Load user memory from persistent storage
   user_mem = get_user_memory(user_id)
   current_level = user_mem.get("last_updated_level")
   last_topic = user_mem.get("last_topic_discussed")
   last_performance = user_mem.get("performance_in_topic")




   # Display welcome back message if returning user
   if current_level:
       console.print(Panel(
           f"Welcome back! Your current level: **{current_level}**\n"
           f"Last topic: {last_topic or 'None'}\n"
           f"Last performance: {last_performance or 'N/A'}%",
           title="Resume Session",
           border_style="cyan"
       ))




   # Start session
   response = orch.execute(user_id, "", user_name="123", is_new_session=True)
   display_response(response)




   # Track current topic and performance during chat
   current_topic = last_topic
   current_performance = last_performance




   # Chat loop
   while True:
       user_input = get_user_input()




       if user_input.lower() in ["exit", "quit", "bye"]:
           # Save user memory before exiting
           update_user_memory(
               user_id=user_id,
               last_topic=current_topic,
               performance=current_performance,
               level=current_level
           )
           display_goodbye()
           break




       display_user_message(user_input)
       response = orch.execute(user_id, user_input)
       display_response(response)




       # Extract topic and performance from session if available
       if hasattr(orch, 'sessions') and user_id in orch.sessions:
           session = orch.sessions[user_id]
           topic = session.get("topic")
           if topic:
               current_topic = topic
           else:
               current_lesson = session.get("current_lesson", {})
               if current_lesson:
                   objective = current_lesson.get("components", {}).get("objective")
                   if objective:
                       current_topic = objective
                   elif "error_type" in current_lesson:
                       etype = current_lesson.get("error_type", "")
                       edesc = current_lesson.get("error_description", "")
                       current_topic = f"{etype}: {edesc}" if edesc else etype
                   else:
                       current_topic = "General Python Lesson"
           profile = session.get("profile")
           if profile:
               if "average_comprehension" in profile:
                   current_performance = profile["average_comprehension"]
         
         
           if session.get("level"):
               current_level = session["level"]




           # Periodically save progress (every 5 interactions)
           if session.get("interaction_count", 0) % 5 == 0:
               update_user_memory(
                   user_id=user_id,
                   last_topic=current_topic,
                   performance=current_performance,
                   level=current_level
               )
















if __name__ == "__main__":
  main()

























