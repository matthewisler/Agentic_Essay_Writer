# Agentic_Essay_Writer
AI-powered essay writer that plans, drafts, critiques, and revises essays using a multi-agent LangGraph workflow with GPT-3.5 and Tavily search.

# 📝 Essay Writer Agent

An AI-powered essay writing assistant that generates, critiques, and revises essays in a multi-agent loop using LangGraph, LangChain, and OpenAI. It includes optional GUI support for a more interactive experience.

## Features

- 🧠 Essay planning using GPT-3.5
- 📚 Automated research via Tavily API
- ✍️ 5-paragraph essay generation
- 🧑‍🏫 Detailed feedback and critique
- 🔁 Revision loop with a configurable number of cycles
- 🖼️ (Optional) Interactive GUI for writing and revising

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://www.langchain.com/)
- [OpenAI GPT-3.5 Turbo](https://platform.openai.com/)
- [Tavily API](https://www.tavily.com/)
- [Python](https://www.python.org/)
- `.env` management with `python-dotenv`

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/essay-writer-agent.git
cd essay-writer-agent
Set up a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up your environment variables

Create a .env file in the root directory and add your Tavily API key:

env
Copy
Edit
TAVILY_API_KEY=your_api_key_here
Usage
Run the Essay Agent
bash
Copy
Edit
python essay_writer.py
This will generate an essay on a default task ("what is the difference between langchain and langsmith") and iterate through a critique/revision loop.

Optional GUI (if using helper.py)
To launch the GUI (requires helper.py with ewriter() and writer_gui()):

bash
Copy
Edit
python essay_writer.py
If the required GUI modules are not found, it will fallback to CLI-only mode.

Example Output
bash
Copy
Edit
{'plan': '...', 'content': ['...']}
{'draft': '...', 'revision_number': 2}
{'critique': '...'}
...
Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your idea.
