import streamlit as st
from backend import ingest_document, get_answer

# Page Config
st.set_page_config(page_title="Universal Doc Chatbot", layout="wide")
st.title("Universal Document Intelligence Chatbot")
st.markdown("Ask questions. I'll intelligently decide whether to check your docs or the web.")

# 1. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: Document Upload ---
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Ingesting document..."):
                try:
                    # Ingest and update the persistent DB
                    ingest_document(uploaded_file)
                    st.success("Document processed and saved to knowledge base!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- CHAT INTERFACE ---

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What would you like to know?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Backend handles logic (Web vs Doc) internally
                response = get_answer(prompt)
                st.markdown(response)
                
                # 3. Save Assistant Response
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
