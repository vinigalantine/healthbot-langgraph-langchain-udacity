# Udacity final project HealthBot - AI Agents with LangChain and LangGraph

This is my final project for the Udacity course [AI Agents with LangChain and LangGraph](https://www.udacity.com/course/ai-agents-with-langchain-and-langgraph--cd13764).

The main.py (the application) is based on the `student-starter.ipynb` provided by Udacity.

## Setup

This project uses `venv`. [What is venv?](https://docs.python.org/3/library/venv.html)

### 1. Create the `venv`

```shell
python -m venv .venv
```

### 2. Activate it

Cmd
```bat
.venv\Scripts\activate.bat
```

PowerShell
```shell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

Run this on a brand new `venv` or whenever `requirements.txt` changes (with the `venv` activated):

```shell
pip install -r requirements.txt
```

### 4. Environment variables

Copy `.env.example` to `.env` and fill in your keys:

- `OPENAI_BASE_URL` - only if you use a custom OpenAI-compatible endpoint.
- `OPENAI_API_KEY` 
- `TAVILY_API_KEY`

### 5. Run the application

Run this after `venv` activation

```shell
python main.py
```

### Tavily

An Tavily API key is needed to run this project.

Sign up for a free Tavily account, which will allow you to search the web and get back N pages and textual content for the pages, for free (first 1000 requests are free)

After signing up, you can get your API key here: https://app.tavily.com/home

### OpenAI
An OpenAI API key is needed to run this project.

### Dependencies

Python version: 3.12.4 

| Package | Version |
|---|---|
| langchain-core | 1.8.4 |
| langchain-openai | 1.8.5 |
| langgraph | 1.2.24 |
| tavily-python | 0.8.5 |
