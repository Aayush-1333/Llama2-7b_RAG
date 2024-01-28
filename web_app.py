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

def reset_button() -> None:
    """resets the state of the model"""

    st.session_state.messages.clear()
    reset_model()
    greet()


def save_chat_to_history(chat_data: dict) -> None:
    """Saves the chat data to streamlit session state"""

    st.session_state.messages.append(chat_data)


def greet() -> None:
    """Greets the user with a welcome message"""

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        llm_response = random.choice(
            [
                "Hello there! How can I assist you today?",
                "Hi, user! Is there anything I can help you with?",
                "Do you need help?",
            ]
        )

    full_response = ""

    for chunk in llm_response.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")

    message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    save_chat_to_history({"role": "assistant", "content": full_response})


def get_reply(user_prompt: str) -> None:
    """get response from the LLM"""

    global model

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        llm_response, _ = model.ask_llm(user_prompt)   # reply and source nodes
    
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


def upload_data() -> None:
    """Upload the data given by the user to a directory"""

    global model
    
    # When files are uploaded by the user start the model
    if uploaded_files:
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

            # create vector_index and start query_engine
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
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        if "files_uploaded" in st.session_state:
            get_reply(prompt)
        else:
            with st.chat_message("assistant"):
                st.markdown("No files uploaded! Please upload a file :)")


# Main code
if __name__ == "__main__":
    # Site title
    st.title("Llama2-TalkBot")

    # Sidebar to upload files
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload files", type='pdf', accept_multiple_files=True)
    upload_data()
        
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        greet()

    # Accept user input
    accept_input()
