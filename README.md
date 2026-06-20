# ARM (Autonomous Reliable Mechanism)

**ARM** is a modular, high-performance Python backend system functioning as an autonomous AI Software Engineering Agent. Designed to bridge the gap between large language models and practical software engineering execution, ARM autonomously navigates, analyzes, and resolves complex coding tasks.

## 🚀 Overview

Modern software development requires reliable automation. ARM leverages an agentic workflow to act as a virtual software engineer. It doesn't just generate code snippets; it understands repository structures, debugs issues, and implements multi-file solutions. 

The system has been rigorously evaluated against the **SWE-bench Lite** dataset, demonstrating its capability to handle real-world GitHub issues and pull requests autonomously.

## ✨ Key Features

* **Autonomous Task Resolution:** Capable of understanding complex problem statements and independently navigating codebases to implement fixes.
* **Agentic Orchestration:** Built on **LangGraph** to manage complex, multi-step reasoning loops, state management, and tool execution.
* **High-Performance API:** Exposes agent capabilities through a robust, asynchronous **FastAPI** backend, making it easy to integrate into CI/CD pipelines or custom frontends.
* **Production-Ready Modularity:** Clean architecture designed for scalability, allowing for easy swapping of underlying LLMs or execution tools.
* **Benchmarked Reliability:** Evaluated against SWE-bench Lite to ensure consistent, measurable performance on standard software engineering tasks.

## 🛠️ Technology Stack

* **Core Language:** Python 3.10+
* **Framework:** FastAPI
* **Agent Orchestration:** LangGraph / LangChain
* **Deployment & Containerization:** Docker (Recommended)

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10 or higher
* `pip` or `poetry` for dependency management
* API keys for your chosen LLM provider (e.g., OpenAI, Anthropic)

### Local Environment Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/ARM.git](https://github.com/YOUR_GITHUB_USERNAME/ARM.git)
   cd ARM
   ```
   
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
4. Environment Variables:
Create a .env file in the root directory and add your necessary configuration:
   ```bash
   LLM_API_KEY=your_api_key_here
   PORT=8000
   ENVIRONMENT=development
   ```

5. Running the API Server
Start the FastAPI application using Uvicorn:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

Example cURL Request to trigger the agent:
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/agent/solve' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "repository_url": "[https://github.com/example/repo](https://github.com/example/repo)",
  "issue_description": "Fix the race condition in the authentication middleware."
}'
```
