# app.py

import streamlit as st
import uuid
import os
import json
from pipeline import RAGPipeline

st.set_page_config(page_title="RAG Chat", layout="wide")

pipeline = RAGPipeline()

CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)


# =====================================================
# Helpers
# =====================================================
def save_chat(chat_id, messages):
    with open(f"{CHAT_DIR}/{chat_id}.json", "w", encoding="utf-8") as f:
        json.dump(messages, f)


def load_chat(chat_id):
    with open(f"{CHAT_DIR}/{chat_id}.json", "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# SESSION INIT
# =====================================================
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# SIDEBAR (multiple chats)
# =====================================================
st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []

st.sidebar.divider()

for file in os.listdir(CHAT_DIR):
    cid = file.replace(".json", "")

    if st.sidebar.button(cid[:8]):
        st.session_state.chat_id = cid
        st.session_state.messages = load_chat(cid)


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =====================================================
# INPUT BAR (bottom)
# =====================================================
if question := st.chat_input("Ask something..."):

    # -------------------------
    # user message
    # -------------------------
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -------------------------
    # assistant response
    # -------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            answer, _ = pipeline.ask(
                question,
                st.session_state.messages
            )

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    save_chat(st.session_state.chat_id, st.session_state.messages)
