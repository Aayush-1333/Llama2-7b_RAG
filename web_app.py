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


# @st.cache_data
# def reset_state() -> None:
#     """resets the state of the model"""

#     reset_model()  # deletes all the files from the disk
#     st.cache_data.clear()  # clear streamlit data cache memory
#     st.session_state.messages.clear()  # clear chat history
#     # st.session_state.files.clear()  # clear files cache data
#     st.session_state.files_uploaded = False  # set files uploaded to False


@st.cache_resource
def load_llm():
    return Llama2_7B_Chat()


def save_chat_to_history(chat_data: dict) -> None:
    """Saves the chat data to streamlit session state"""

    st.session_state.messages.append(chat_data)


def get_llm_reply(model: Llama2_7B_Chat, mode: str = "default", user_prompt: str = None) -> None:
    """get response from the LLM"""
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        if mode == "reply":
            llm_response, _ = model.ask_llm(user_prompt, st.session_state.query_engine)   # reply and source nodes
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
    # print("Streamlit got --->", llm_response)
    
    try:
        for chunk in str(llm_response).split():
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


@st.cache_data
def upload_data(_model: Llama2_7B_Chat) -> None:
    """Upload the data given by the user to a directory"""
    
    st.session_state.files_uploaded = False

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
        user_id = st.session_state.session_user
        _model.create_index(folder_name, user_id)
        st.session_state.query_engine = _model.start_query_engine(user_id)

        status.text("Your files have uploaded! You can ask now :)")
        st.session_state.messages.clear()


# Main code
if __name__ == "__main__":
    # Site title
    st.title("Llama2-TalkBot")
    
    st.write("Initializing Llama2-7B model....")
    model = load_llm()
    st.write("Done!!")
    
    # Sidebar to upload files
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload files", type='pdf', accept_multiple_files=True)
        if uploaded_files:
            upload_data(model)        

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        id = time.time()
        st.session_state.session_user = f"{id}"
        get_llm_reply(model, mode="greet")

    # Accept user input
    if prompt := st.chat_input("Enter query"):
        show_history()
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add user message to chat history
        save_chat_to_history({"role": "user", "content": prompt})

        if "files_uploaded" in st.session_state:
            get_llm_reply(model, mode="reply", user_prompt=prompt)
        else:
            get_llm_reply(model)
