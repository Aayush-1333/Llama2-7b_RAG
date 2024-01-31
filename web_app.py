"""
    This web app is created using streamlit
    Created to demonstrate the working of Llama2-7B-hf-chat model
    for RAG (Retreival Augmented Generation) purposes 
"""
import streamlit as st
import random
import time
import os
from rag_model import Llama2_7B_Chat, reset_model

model = None


@st.cache_data
def reset_state() -> None:
    """resets the state of the model"""

    reset_model()  # deletes all the files from the disk
    st.cache_data.clear()  # clear streamlit data cache memory
    st.cache_resource.clear()


def save_chat_to_history(chat_data: dict) -> None:
    """Saves the chat data to streamlit session state"""

    st.session_state.messages.append(chat_data)


def get_llm_reply(mode: str = "default", user_prompt: str = None) -> None:
    """get response from the LLM"""

    global model

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        if mode == "retrieve":
            llm_response, _ = model.ask_llm(
                user_prompt)   # reply and source nodes
        elif mode == "greet":
            llm_response = random.choice(
                [
                    "Hello there! How can I assist you today?",
                    "Hi, user! Is there anything I can help you with?",
                    "Do you need help?",
                ]
            )
        else:
            llm_response = "No files uploaded! Please upload a file :)"

    full_response = ""

    try:
        for chunk in llm_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    except AttributeError as e:
        message_placeholder.markdown("No answer! Error!")

    # Add assistant response to chat history
    save_chat_to_history({"role": "assistant", "content": full_response})


def show_history() -> None:
    """Display chat messages from streamlit chat history"""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


@st.cache_resource
def upload_data() -> None:
    """Upload the data given by the user to a directory"""

    global model

    st.session_state.files_uploaded = True

    upload_time = str(int(time.time()))
    folder_name = "Data_" + upload_time

    if not os.path.exists(folder_name):
        os.system(f"mkdir {folder_name}")
        status = st.text("Please wait your files are beig uploaded...")

        # Write the file to Data directory
        for file in uploaded_files:
            with open(os.path.join(folder_name, file.name), 'wb') as bytes_file:
                bytes_file.write(file.getbuffer())

        # initialize model, create vector_index and start query_engine
        model = Llama2_7B_Chat()
        model.create_index(folder_name)
        model.start_query_engine()

        status.text("Your files have uploaded! You can ask now :)")
        st.session_state.messages.clear()


def accept_input() -> None:
    """Accepts the user's input as query for the LLM"""

    # Accept user input
    if prompt := st.chat_input("Enter query"):
        show_history()
        
        # Add user message to chat history
        save_chat_to_history({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        if "files_uploaded" in st.session_state:
            get_llm_reply(prompt)
        else:
            get_llm_reply()


# Main code
if __name__ == "__main__":
    # Reset model if data exists
    # reset_state()
    
    # Site title
    st.title("Llama2-TalkBot")

    # Sidebar to upload files
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload files", type='pdf', accept_multiple_files=True)
        if uploaded_files:
            upload_data()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        get_llm_reply(mode="greet")

    # Accept user input
    accept_input()
