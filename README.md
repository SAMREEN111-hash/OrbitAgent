# OrbitAgent

Agentic AI assistant with web search, code execution, maths, summarisation and fact-checking — built on LangChain and Groq.

Live demo: https://samreen111-hash-orbitagent.streamlit.app

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LangChain](https://img.shields.io/badge/LangChain-latest-green) ![Groq](https://img.shields.io/badge/Groq-Llama3.3-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)

---

## Overview

OrbitAgent is a conversational AI agent that decides which tool to use based on your input. Ask a question, paste an article, give it a maths problem or a Python task — it handles it without you specifying how.

---

## Tools

| Tool | What it does |
|------|-------------|
| Web search | Real-time search via DuckDuckGo — no API key required |
| Calculator | Solves mathematical expressions using LLMMathChain |
| Python REPL | Writes and executes Python code in a sandboxed environment |
| Summariser | Takes any text and returns structured bullet points |
| Fact checker | Searches the web and returns TRUE / FALSE / UNCERTAIN with reasoning |

---

## Features

- Multi-chat with history, favourites, and delete
- Sticky notes panel in sidebar
- Export any conversation as .txt
- Usage stats — total chats and messages
- Configurable memory window, temperature, and model
- Switch between Llama 3.3 70B, Mixtral, and Gemma on the fly
- Debug mode showing agent reasoning steps live

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangChain CONVERSATIONAL_REACT |
| LLM | Groq API — Llama 3.3 70B / Mixtral / Gemma |
| UI | Streamlit |
| Search | DuckDuckGo |
| Memory | ConversationBufferWindowMemory |

---

User input
    |
    v
LangChain agent (CONVERSATIONAL_REACT)
    |
    |-- Web search     (DuckDuckGo)
    |-- Calculator     (LLMMathChain)
    |-- Python REPL    (sandboxed)
    |-- Summariser     (LLMChain + prompt)
    |-- Fact checker   (search + LLM reasoning)
    |
    v
ConversationBufferWindowMemory (k=6)
    |
    v
Groq API — Llama 3.3 70B

## Setup

```bash
git clone https://github.com/SAMREEN111-hash/OrbitAgent.git
cd OrbitAgent
pip install -r requirements.txt
streamlit run app.py
```

Get a free Groq API key at https://console.groq.com — no credit card required. Enter it in the sidebar when the app opens.

---

## Developer

Samreen
https://github.com/SAMREEN111-hash
