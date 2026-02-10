import streamlit as st
import uuid
import os
import json
from pipeline import RAGPipeline

st.set_page_config(page_title="RAG Chat", layout="wide")

pipeline = RAGPipeline()

CHAT_DIR = "chats_t20" #chats
os.makedirs(CHAT_DIR, exist_ok=True)


# Helpers

def chat_path(chat_id):
    return f"{CHAT_DIR}/{chat_id}.json"


def save_chat(chat_id, messages, title):
    data = {
        "title": title,
        "messages": messages
    }
    with open(chat_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_chat(chat_id):
    with open(chat_path(chat_id), "r", encoding="utf-8") as f:
        return json.load(f)


def delete_chat(chat_id):
    os.remove(chat_path(chat_id))


def generate_title(messages):
    for m in messages:
        if m["role"] == "assistant":
            first_line = m["content"].strip().split("\n")[0]
            first_line = first_line.lstrip("#").strip()
            return first_line if first_line else "New Chat"
    return "New Chat"


# SESSION INIT

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "title" not in st.session_state:
    st.session_state.title = "New Chat"


# SIDEBAR

st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.title = "New Chat"

st.sidebar.divider()


for file in os.listdir(CHAT_DIR):
    cid = file.replace(".json", "")
    data = load_chat(cid)
    title = data.get("title", "Chat")

    col1, col2 = st.sidebar.columns([4, 1])

    if col1.button(title, key=f"open_{cid}"):
        st.session_state.chat_id = cid
        st.session_state.messages = data["messages"]
        st.session_state.title = title

    if col2.button("🗑", key=f"del_{cid}"):
        delete_chat(cid)
        st.rerun()


# SHOW TITLE

st.title(st.session_state.title)


# DISPLAY MESSAGES (WITH CITATIONS)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")


# INPUT

if question := st.chat_input("Ask something..."):

    # User message (NO sources)
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            answer, sources = pipeline.ask(
                question,
                st.session_state.messages
            )

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    # SAVE assistant WITH sources
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

    # Auto title once
    if st.session_state.title == "New Chat":
        st.session_state.title = generate_title(
            st.session_state.messages
        )

    # Save chat
    save_chat(
        st.session_state.chat_id,
        st.session_state.messages,
        st.session_state.title
    )
