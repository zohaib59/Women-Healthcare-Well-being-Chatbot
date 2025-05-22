import streamlit as st
from langchain_openai import ChatOpenAI  # Replacing Ollama with OpenAI's GPT-4o
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import os
from dotenv import load_dotenv
import io
import json

# Load environment variables
load_dotenv()

# Styling for the app
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 GPT-4o Health Companion")
st.caption("🚀 Your AI-Powered Health Advisor using GPT-4o")

# Sidebar Controls
with st.sidebar:
    st.header("🧰 Controls")

    if st.button("🔄 Clear Chat History"):
        st.session_state.message_log = []
        st.rerun()

    if st.button("💣 End Session"):
        st.session_state.clear()
        st.success("Session ended. Start fresh!")
        st.stop()

    if st.button("📥 Download Chat Log"):
        if "message_log" in st.session_state:
            chat_history = json.dumps(st.session_state.message_log, indent=2)
            st.download_button("Download Log", data=chat_history, file_name="chat_log.json")

    st.divider()
    st.markdown("Built with [LangChain](https://python.langchain.com/) & GPT-4o")

# Initialize GPT-4o
llm_engine = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Define the system prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI expert in Mental Health and Well Being. Provide accurate and detailed insights on the Issue related to Health, Well Being and Growth. Always respond in a precise and informative manner."
)

# Initialize chat log
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Namaste! I am your Health Expert. How can I assist you with your queries today? ⚖️"}
    ]

# Display chat history
for msg in st.session_state.message_log:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
user_query = st.chat_input("Ask your question here...")

# Prompt generation and response
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})

    def generate_prompt_chain():
        prompt_msgs = [system_prompt]
        for msg in st.session_state.message_log:
            if msg["role"] == "user":
                prompt_msgs.append(HumanMessagePromptTemplate.from_template(msg["content"]))
            elif msg["role"] == "ai":
                prompt_msgs.append(AIMessagePromptTemplate.from_template(msg["content"]))
        return ChatPromptTemplate.from_messages(prompt_msgs)

    with st.spinner("🧠 Processing..."):
        try:
            chain = generate_prompt_chain() | llm_engine | StrOutputParser()
            response = chain.invoke({})
        except Exception as e:
            st.error(f"Error: {e}")
            response = "Sorry, I encountered an error while processing your request."

    st.session_state.message_log.append({"role": "ai", "content": response})
    st.rerun()
