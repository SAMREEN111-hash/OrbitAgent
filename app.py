# ============================================================
# OrbitAgent — Agentic AI Assistant
# Developer   : Samreen
# GitHub      : https://github.com/SAMREEN111-hash
# Description : A fully deployable agentic AI assistant with
#               web search, code execution, math solving,
#               note-taking, chat export, and summarisation.
#               Powered by Groq (free, no credit card needed).
# ============================================================

import time
import json
import datetime
from typing import Any

import streamlit as st

from langchain_groq import ChatGroq
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🧠 OrbitAgent by Samreen",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
        .main { background-color: #f0f4f8; font-family: 'Segoe UI', sans-serif; }
        .stTextInput>div>div>input {
            border-radius: 20px; padding: 10px 15px; border: 1px solid #ddd;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: #4A90D9; color: white; border: none;
            border-radius: 10px; padding: 10px 15px; font-weight: 600;
            transition: all 0.3s ease; margin-bottom: 10px; width: 100%;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background-color: #357ABD !important;
            transform: translateY(-1px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        }
        .metric-card {
            background: white; border-radius: 12px; padding: 16px;
            border: 1px solid #e0e7ef; margin-bottom: 10px;
        }
        .note-box {
            background: #fffbe6; border-left: 4px solid #f0c040;
            padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
def init_session_state():
    defaults = {
        'chats': {'chat_1': {'title': 'New Chat', 'messages': [], 'created': datetime.datetime.now().strftime("%d %b %Y %H:%M")}},
        'active_chat': 'chat_1',
        'chat_counter': 1,
        'model': 'llama-3.3-70b-versatile',
        'temperature': 0.3,
        'memory_window': 6,
        'debug': False,
        'agent_chain': None,
        'notes': [],
        'total_messages_sent': 0,
        'favourite_chats': [],
        'groq_api_key': '',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────
def get_api_key():
    # Check secrets first (for deployed version), then session state
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return st.session_state.get('groq_api_key', '')

def create_llm():
    api_key = get_api_key()
    if not api_key:
        return None
    return ChatGroq(
        model=st.session_state.get('model', 'llama-3.3-70b-versatile'),
        temperature=st.session_state.get('temperature', 0.3),
        groq_api_key=api_key,
        max_tokens=2048,
    )

# ─────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────
def summarise_text(text: str) -> str:
    try:
        llm = create_llm()
        if not llm:
            return "Error: No API key set."
        prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are a helpful assistant. Summarise the following text "
                "in 3-5 clear bullet points:\n\n{text}\n\nSummary:"
            )
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(text=text)
    except Exception as e:
        return f"Summarisation error: {str(e)}"

def fact_check(claim: str) -> str:
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(f"fact check: {claim}")
        llm = create_llm()
        if not llm:
            return "Error: No API key set."
        prompt = PromptTemplate(
            input_variables=["claim", "results"],
            template=(
                "Based on the following web search results, assess whether "
                "this claim is TRUE, FALSE, or UNCERTAIN:\n\n"
                "Claim: {claim}\n\nSearch Results: {results}\n\n"
                "Verdict (TRUE / FALSE / UNCERTAIN) and brief explanation:"
            )
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(claim=claim, results=results)
    except Exception as e:
        return f"Fact-check error: {str(e)}"

def create_tools(llm):
    search_tool = DuckDuckGoSearchRun()
    python_repl = PythonREPLTool()
    calculator  = LLMMathChain.from_llm(llm=llm)

    return [
        Tool(
            name="web_search",
            func=search_tool.run,
            description="Search the web for current information. Input: a search query string.",
            return_direct=True
        ),
        Tool(
            name="calculator",
            func=calculator.run,
            description="Solve maths problems. Input: a mathematical expression.",
            return_direct=True
        ),
        Tool(
            name="python_repl",
            func=python_repl.run,
            description="Execute Python code. Input: valid Python code.",
            return_direct=True
        ),
        Tool(
            name="summarise",
            func=summarise_text,
            description="Summarise a long text into bullet points. Input: the text to summarise.",
            return_direct=True
        ),
        Tool(
            name="fact_check",
            func=fact_check,
            description="Verify whether a claim is true or false using web search. Input: the claim.",
            return_direct=True
        ),
    ]

def create_agent():
    with st.spinner("🧠 Starting OrbitAgent..."):
        try:
            llm = create_llm()
            if not llm:
                st.error("❌ No Groq API key found. Enter it in the sidebar.")
                return None
            tools = create_tools(llm)
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=st.session_state.get('memory_window', 6),
                return_messages=True,
                output_key="output",
            )
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=st.session_state.get('debug', False),
                memory=memory,
                handle_parsing_errors=True,
                max_iterations=6,
                early_stopping_method="generate",
                return_intermediate_steps=True
            )
            return agent
        except Exception as e:
            st.error(f"❌ Agent error: {str(e)}")
            return None

# ─────────────────────────────────────────
# CHAT HELPERS
# ─────────────────────────────────────────
def new_chat():
    st.session_state.chat_counter += 1
    cid = f"chat_{st.session_state.chat_counter}"
    st.session_state.chats[cid] = {
        'title': 'New Chat',
        'messages': [],
        'created': datetime.datetime.now().strftime("%d %b %Y %H:%M")
    }
    st.session_state.active_chat = cid
    st.session_state.agent_chain = None

def switch_chat(cid):
    st.session_state.active_chat = cid
    st.session_state.agent_chain = None

def delete_chat(cid):
    if len(st.session_state.chats) > 1:
        del st.session_state.chats[cid]
        if st.session_state.active_chat == cid:
            st.session_state.active_chat = list(st.session_state.chats.keys())[-1]
        st.session_state.agent_chain = None

def export_chat_as_text() -> str:
    current = st.session_state.chats[st.session_state.active_chat]
    lines = [f"OrbitAgent Chat Export — {current.get('title', 'Chat')}", "=" * 50, ""]
    for msg in current.get("messages", []):
        role = "You" if msg["role"] == "user" else "OrbitAgent"
        lines.append(f"{role}:\n{msg['content']}\n")
    lines.append("=" * 50)
    lines.append("Exported from OrbitAgent by Samreen | github.com/SAMREEN111-hash")
    return "\n".join(lines)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("🧠 OrbitAgent")
        st.caption("by Samreen")

        # ── API Key input (shown only if not in secrets) ───────
        try:
            st.secrets["GROQ_API_KEY"]
        except Exception:
            with st.expander("🔑 Groq API Key", expanded=not bool(st.session_state.groq_api_key)):
                key_input = st.text_input(
                    "Enter your Groq API key",
                    type="password",
                    placeholder="gsk_...",
                    help="Get a free key at console.groq.com"
                )
                if key_input:
                    st.session_state.groq_api_key = key_input
                    st.session_state.agent_chain = None
                    st.success("✅ API key saved!")
                st.caption("Free at [console.groq.com](https://console.groq.com)")

        st.divider()

        # ── Usage stats ────────────────────────────────────────
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("💬 Chats", len(st.session_state.chats))
        col2.metric("📨 Messages", st.session_state.total_messages_sent)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── New chat ───────────────────────────────────────────
        if st.button("➕ New Chat", type="secondary"):
            new_chat()
            st.rerun()

        # ── Chat list ──────────────────────────────────────────
        st.subheader("🗂 Chats")
        for cid, chat in st.session_state.chats.items():
            is_active = cid == st.session_state.active_chat
            is_fav    = cid in st.session_state.favourite_chats
            label     = f"{'⭐ ' if is_fav else ''}{'▶ ' if is_active else ''}{chat.get('title', 'New Chat')[:22]}"
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(label, key=f"switch_{cid}", type="secondary"):
                    switch_chat(cid)
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{cid}", type="secondary"):
                    delete_chat(cid)
                    st.rerun()

        st.divider()

        # ── Export chat ────────────────────────────────────────
        st.subheader("📤 Export Chat")
        export_text = export_chat_as_text()
        st.download_button(
            label="⬇️ Download as .txt",
            data=export_text,
            file_name=f"orbitagent_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

        # ── Favourite current chat ─────────────────────────────
        cid    = st.session_state.active_chat
        is_fav = cid in st.session_state.favourite_chats
        if st.button("⭐ Unfavourite" if is_fav else "⭐ Favourite this chat", type="secondary"):
            if is_fav:
                st.session_state.favourite_chats.remove(cid)
            else:
                st.session_state.favourite_chats.append(cid)
            st.rerun()

        st.divider()

        # ── Sticky Notes ───────────────────────────────────────
        st.subheader("📝 Quick Notes")
        new_note = st.text_input("Add a note...", key="note_input", placeholder="Type and press Enter")
        if new_note:
            st.session_state.notes.append({
                'text': new_note,
                'time': datetime.datetime.now().strftime("%H:%M")
            })
            st.rerun()
        for i, note in enumerate(st.session_state.notes):
            col_n, col_d = st.columns([5, 1])
            with col_n:
                st.markdown(
                    f'<div class="note-box">📌 {note["text"]} <small style="color:#aaa">({note["time"]})</small></div>',
                    unsafe_allow_html=True
                )
            with col_d:
                if st.button("✕", key=f"delnote_{i}"):
                    st.session_state.notes.pop(i)
                    st.rerun()

        st.divider()

        # ── Settings ───────────────────────────────────────────
        st.subheader("⚙️ Settings")
        model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
        model_labels  = ["Llama 3.3 70B (Best)", "Llama 3.1 8B (Fast)", "Mixtral 8x7B", "Gemma 2 9B"]
        selected = st.selectbox(
            "Model",
            options=model_options,
            format_func=lambda x: model_labels[model_options.index(x)],
            index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0,
            key="model"
        )
        st.slider("Creativity", 0.0, 1.0, value=st.session_state.temperature, step=0.1, key="temperature")
        st.slider("Memory Window", 1, 10, value=st.session_state.memory_window, step=1, key="memory_window")
        st.toggle("Debug Mode", value=st.session_state.debug, key="debug")

        # ── About ──────────────────────────────────────────────
        with st.expander("ℹ️ About OrbitAgent"):
            st.markdown("""
**OrbitAgent** is a deployable agentic AI assistant.

**Tools:**
- 🔍 Web search (DuckDuckGo)
- 🧮 Maths solver
- 🐍 Python execution
- 📄 Text summariser
- ✅ Fact checker

**Features:**
- 📝 Sticky notes
- 📤 Chat export
- ⭐ Favourite chats
- 📊 Usage stats

**Tech:** LangChain · Groq · Streamlit

Built by **Samreen**
[github.com/SAMREEN111-hash](https://github.com/SAMREEN111-hash)
            """)

# ─────────────────────────────────────────
# CHAT RENDERING
# ─────────────────────────────────────────
def format_response(output: str) -> str:
    if not output:
        return ""
    lines, in_code = [], False
    for line in output.split('\n'):
        if line.startswith('```'):
            in_code = not in_code
            lines.append(line)
        elif in_code:
            lines.append(line)
        else:
            lines.append(line)
    return '\n'.join(lines)

def process_response(response: Any) -> str:
    if isinstance(response, dict):
        if 'output' in response:
            return str(response['output'])
        return str(response)
    return format_response(str(response))

def render_chat():
    current  = st.session_state.chats[st.session_state.active_chat]
    messages = current["messages"]

    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.subheader(f"💬 {current.get('title', 'New Chat')}")
        st.caption(f"Started: {current.get('created', '')}")
    with col_clear:
        if st.button("🗑 Clear", key="clear_chat"):
            current['messages'] = []
            current['title'] = 'New Chat'
            st.session_state.agent_chain = None
            st.rerun()

    if not messages:
        st.info("👋 Hi! I'm OrbitAgent by Samreen. I can search the web, run Python code, solve maths, summarise text, and fact-check claims. Ask me anything!")
    else:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask me anything..."):
        api_key = get_api_key()
        if not api_key:
            st.error("❌ Please enter your Groq API key in the sidebar to start chatting.")
            return

        messages.append({"role": "user", "content": prompt})
        st.session_state.total_messages_sent += 1

        if len(messages) == 1:
            current["title"] = prompt[:35] + ("..." if len(prompt) > 35 else "")

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                if not st.session_state.agent_chain:
                    st.session_state.agent_chain = create_agent()
                if not st.session_state.agent_chain:
                    raise Exception("Agent failed to start. Check your API key.")

                with st.spinner("Thinking..."):
                    response = st.session_state.agent_chain.invoke({"input": prompt})
                    full_response = process_response(response)

                st.markdown(full_response)
                messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                err = f"⚠️ {str(e)}"
                st.error(err)
                messages.append({"role": "assistant", "content": err})

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    load_css()
    init_session_state()

    st.title("🧠 OrbitAgent — Agentic AI Assistant")
    st.caption("Web search · Code execution · Maths · Summarise · Fact-check | Built by Samreen")

    col_main, col_side = st.columns([3, 1])
    with col_main:
        render_chat()
    with col_side:
        render_sidebar()

    st.markdown("---")
    st.caption("Powered by Groq · LangChain · Streamlit | Built by Samreen — github.com/SAMREEN111-hash")

    if st.session_state.debug:
        with st.sidebar.expander("🐞 Debug"):
            st.json({
                "active_chat": st.session_state.active_chat,
                "total_chats": len(st.session_state.chats),
                "messages": len(st.session_state.chats[st.session_state.active_chat]['messages']),
                "model": st.session_state.model,
                "temperature": st.session_state.temperature,
                "memory_window": st.session_state.memory_window,
                "agent_ready": st.session_state.agent_chain is not None,
                "notes_count": len(st.session_state.notes),
                "total_msgs_sent": st.session_state.total_messages_sent,
            })

if __name__ == "__main__":
    main()
