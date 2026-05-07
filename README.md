# Python Learning Chatbox 

A multi-agent, persistent, adaptive Python tutoring system that assesses learner proficiency, provides targeted error correction lessons, and tracks learning progress through intelligent multi-agent orchestration.

## Overview

The Python Learning Assistant is an interactive educational platform built on LangChain and OpenAI APIs that guides users through personalized Python learning journeys. The system intelligently routes users through different learning stages using specialized agents:

1. **Orchestrator** - Central controller; routes user input to correct agent 
2. **Level Assessment** - Analyzes code and assigns proficiency level
3. **Error Training** - Delivers targeted lessons based on assessment
4. **Profile Management** - Tracks progress and provides recommendations after each lesson
5. **Resource Retrieval** - Gathers relevant teaching materials and examples

### Key Features

- 🐍 **Python-Exclusive Teaching** - Focused exclusively on Python 3 programming
- 📊 **Proficiency-Based Routing** - Customized lessons for 5 proficiency levels:
  - Novice
  - Beginner
  - Intermediate
  - Proficient
  - Master
- 💾 **Session Persistence** - Automatically saves and loads learning sessions
- 📈 **Progress Tracking** - Maintains learner profiles with performance metrics
- 🤖 **Multi-Agent Architecture** - Specialized agents for different learning stages
- 📝 **Token Tracking** - Monitors API usage across all agent interactions
  
---

## Project Structure

```
FinalProject/
├── orchestrator_agent.py          # Main coordination agent
├── level_placement_agent.py         # Python proficiency assessment
├── error_training_agent.py          # Personalized lesson delivery
├── update_profile_agent.py          # Progress tracking & recommendations
├── resource_retrieval_agent.py      # Learning material retrieval
└── __init__.py

chatbox.py                           # Interactive user interface
session_persistence.py               # Session storage & management
requirements.txt                     # Python dependencies
pyproject.toml                        # Project configuration
```

---

## System Components

### **1. Orchestrator Agent** (`orchestrator_agent.py`)
The central coordination hub that:
- Manages the complete learning session lifecycle
- Routes users to appropriate specialized agents
- Maintains session state and user context
- Tracks token usage across all agents
- Provides friendly, supportive user guidance
- Persists sessions automatically

**Session Stages:**
- `NEW` → `GATHERING_NAME` → `ASSESSED` → `TRAINING` → `PROFILE_UPDATED` → `CONTINUOUS`

### **2. Level Placement Agent** (`level_placement_agent.py`)
Assesses Python proficiency based on:
- **Syntax Errors** - Language grammar violations
- **Logic Errors** - Algorithm and flow problems
- **Redundancy Errors** - Code duplication and inefficiency
- **Task Complexity** - Problem difficulty (1-10 scale)

Returns a proficiency classification with detailed feedback.

### **3. Error Training Agent** (`error_training_agent.py`)
Delivers adaptive lessons tailored by proficiency level:
- **Novice**: Foundational concepts, abundant examples, simple corrections
- **Beginner**: Guided discovery, pattern recognition, scaffolded practice
- **Intermediate**: Problem-solving, optimization, design patterns
- **Proficient**: Advanced techniques, edge cases, system design
- **Master**: Novel approaches, optimization challenges, mentoring perspectives

Provides step-by-step error correction and learning strategies.

### **4. Update Profile Agent** (`update_profile_agent.py`)
Maintains learner profiles tracking:
- Responses to training lessons
- Error patterns and improvements
- Proficiency level assignments
- Personalized growth recommendations
- Learning history and progression
- Token usage statistics

### **5. Resource Retrieval Agent** (`resource_retrieval_agent.py`)
Gathers research-based teaching materials:
- Relevant Python documentation
- Code examples and patterns
- Best practices and conventions
- Learning resources aligned with proficiency level

### **6. Session Persistence** (`session_persistence.py`)
Manages persistent storage:
- Saves complete sessions
- Stores user profiles and token data
- Loads previous user level 

Saves sessions to the `learner_sessions/` directory with format: `{user_id}_session.json`

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- OpenAI API key

### Step 1: Install Dependencies

Create a virtual environment and install required packages:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

**To get an OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy and paste into your `.env` file

### Step 3: Verify Project Structure

Ensure the following directories exist:
- `FinalProject/` - All agent modules
- `learner_sessions/` - Auto-created on first run, stores session data

---

## Usage

### Running the Interactive Chatbox

Start the Python Learning Assistant chatbox:

```bash
python chatbox.py
```

This launches an interactive terminal-based interface where you can:

**Option 1: Share Python Code for Assessment**
- Paste Python code you've written
- Share code with errors you're struggling with
- Receive a Python proficiency assessment and personalized lessons

**Option 2: Ask for Python Help from Scratch**
- "Help me write a Python function that..."
- "I want to create a Python program for..."
- "How do I implement this in Python..."
- Get step-by-step Python guidance and examples

**Option 3: Ask About Python Concepts**
- "What is a list comprehension?"
- "Explain decorators in Python"
- "What's the difference between a list and tuple?"
- Receive level-appropriate Python concept explanations

### Chatbox in action
**Option 1: Average of 3 Numbers
<img width="2772" height="1536" alt="image" src="https://github.com/user-attachments/assets/5c956d6a-ee82-4270-98fa-b1284337f88c" />

**Option 2: Defining tuples and Lists
:<img width="2770" height="1480" alt="image" src="https://github.com/user-attachments/assets/82e82bf8-3766-4566-9ca1-3c386b912218" />

**Option 3:implement a function to multiply 7*7
<img width="1036" height="570" alt="image" src="https://github.com/user-attachments/assets/8a7ec73f-9e36-4525-91c9-12385465c7ba" />


### Multi-Line Input Mode

For longer code samples or detailed explanations:

```
Type: :multi
[Enter your code/text]
Type: :end
[Submit]
```

### Exit the Program

Type `exit` at any time to quit.

---

## Data Storage & Session Management

### Session Files

Sessions are automatically saved to `learner_sessions/{user_id}_session.json` with the structure:

```json
{
  "user_id": "user_identifier",
  "saved_at": "2026-05-03T10:30:00.123456",
  "session": {
    "user_name": "Name",
    "stage": "training",
    "level": "intermediate",
    ...
  },
  "profile": {
    "strengths": ["..."],
    "growth_areas": ["..."],
    ...
  },
  "tokens": {
    "total_tokens": 5000,
    "prompt_tokens": 3000,
    "completion_tokens": 2000,
    ...
  }
}
```

### Loading Previous Sessions

Sessions are automatically loaded when a returning user starts a new session with their user ID.

### Accessing Saved Sessions

To view all saved sessions programmatically:

```python
from session_persistence import SessionPersistence

sp = SessionPersistence()
all_sessions = sp.get_all_sessions()

# Load a specific session
session = sp.load_session(user_id="146")

# Delete a session
sp.delete_session(user_id="146")
```

---

## Configuration

### Modifying Session Settings

Edit `chatbox.py` line ~550 to change the default user ID:

```python
user_id = "146"  # Change this to use different user
```

### Adjusting Agent LLM Settings

Edit `orchestrator_agent.py` to modify model and temperature:

```python
LLM_MODEL = "gpt-4o-mini"      # Change model version
TEMPERATURE = 0.6              # Adjust creativity (0.0-1.0)
```

---

## API Dependencies

The system uses the following APIs and libraries:

| Component | Purpose |
|-----------|---------|
| **OpenAI GPT-4o-mini** | Language model for all agents |
| **LangChain** | Agent orchestration and prompting |
| **LangChain OpenAI** | OpenAI integration |
| **FAISS** | Vector similarity search |

---

## Token Usage Tracking

The system tracks API token usage across all interactions. A session token report is displayed at session completion:

```
📊 SESSION TOKEN USAGE REPORT:
├─ total_tokens
├─ prompt_tokens
├─ completion_tokens
```

In the furture, we would likke to implement so the token data is saved in session files for billing and analysis purposes.

---

## Future Enhancements

- Sophisticated Database
- Web UI for better accessibility
- Token Behavior influence 

---

## Prompt iteration Improvement
- allow multi line in terminal to allow users to write code.

solution: implement a loop to allow multiple lines of input 
- modified level assessment criteria. Level assessment was originally based on errors which assessed simple codes as master

solution: include a task complexity criteria in the level placement orchestrator 
- out-of-scope conversations were allowed

solution: define out-of-scope words to reject
- continue session: sessions would end after a lesson has been completed

solution:modified orchestrator to allow persistent lessons
- include a long term memory to store user profle

solution: implemented a json memory folder
- chatbox level assesment output too much information. 

solution: shorten the output to reduce token usage

---
## Evaulations and Guardrails
Prompting the objective of the chatbox was not enough to keep the chatbox in scope. We also included out-of-scope keywords the orchestrator agent should reject and output an error message to the user

```python
or keyword in out_of_scope_keywords:
          if keyword in lower_input:
              if keyword == "java " and "javascript" in lower_input:
                  continue
              return False
```
