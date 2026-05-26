# 🧠 SmartAgent — Agentic AI Assistant

> A fully local, privacy-first agentic AI assistant. No API keys. No cloud. Runs entirely on your machine.

Built by **Samreen** · [SAMREEN111-hash](https://github.com/SAMREEN111-hash)

---

## 🚀 What Makes SmartAgent Different

Most AI assistants send your data to the cloud. SmartAgent runs **100% locally** using open-source LLMs through Ollama — your conversations never leave your machine.

On top of the standard agent features, SmartAgent adds:

| Feature | Description |
|---------|-------------|
| 📄 **Text Summariser** | Paste any long text and get clean bullet-point summary |
| ✅ **Fact Checker** | Give it a claim, it searches the web and gives a verdict |
| 📝 **Sticky Notes** | Save quick notes directly in the sidebar |
| 📤 **Export Chat** | Download any conversation as a .txt file |
| ⭐ **Favourite Chats** | Star important conversations |
| 🗑 **Delete Chats** | Clean up chat history |
| 📊 **Usage Stats** | See how many chats and messages you've sent |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **LangChain** | Agent framework & tool orchestration |
| **Ollama** | Run LLMs locally (Llama3.2, Mistral, Gemma) |
| **Streamlit** | Interactive web UI |
| **DuckDuckGo** | No-API-key real-time web search |
| **LLMMathChain** | Symbolic maths solver |
| **PythonREPL** | Sandboxed code execution |

---

## 🧩 Agent Architecture

```
User Input
    │
    ▼
SmartAgent (LangChain CONVERSATIONAL_REACT)
    │
    ├──► 🔍 Web Search      (DuckDuckGo — no API key)
    ├──► 🧮 Calculator       (LLMMathChain)
    ├──► 🐍 Python REPL      (sandboxed execution)
    ├──► 📄 Summariser       (LLMChain + custom prompt)   ← NEW
    └──► ✅ Fact Checker     (search + LLM reasoning)     ← NEW
    │
    ▼
ConversationBufferWindowMemory (sliding window, k=6)
    │
    ▼
Local LLM via Ollama (Llama3.2 / Mistral / Gemma)
```

---

## ⚡ Quick Start

### 1. Install Ollama
Download from [ollama.ai](https://ollama.ai) and pull a model:
```bash
ollama pull llama3.2
```

### 2. Clone & Setup
```bash
git clone https://github.com/SAMREEN111-hash/SmartAgent.git
cd SmartAgent
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r Source_Code/requirements.txt
```

### 3. Run
```bash
streamlit run Source_Code/Agentic_AI.py
```
Open → http://localhost:8501

---

## 💡 Example Use Cases

| You type | SmartAgent does |
|----------|----------------|
| *"What is the latest news about EVs?"* | Searches web, summarises results |
| *"Calculate compound interest on 50000 at 8% for 5 years"* | Solves with LLMMathChain |
| *"Write a Python function to reverse a linked list"* | Writes and runs the code |
| *"Summarise this article: [paste text]"* | Returns clean bullet points |
| *"Is it true that India landed on the moon in 2023?"* | Fact-checks with web search |

---

## 📁 Project Structure

```
SmartAgent/
├── Source_Code/
│   ├── Agentic_AI.py          # Main application (all features)
│   └── requirements.txt       # Python dependencies
├── Screenshots/               # App screenshots
└── README.md                  # This file
```

---

## 🔧 Configuration

All settings are adjustable in the sidebar:
- **Model** — switch between Llama3.2, Mistral, Gemma
- **Creativity (Temperature)** — 0.0 (focused) to 1.0 (creative)
- **Memory Window** — how many messages the agent remembers (1–10)
- **Debug Mode** — see agent reasoning steps in real time

---

## 🙋 Developer

**Samreen**  
GitHub: [@SAMREEN111-hash](https://github.com/SAMREEN111-hash)

---

*Built with ❤️ using LangChain, Ollama, and Streamlit*
