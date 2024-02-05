import streamlit as st
from Llama_chatbot import Llama2_ChatBot
import time


@st.cache_resource
def load_model():
    return Llama2_ChatBot()


def save__chat_to_history(role, content):
    st.session_state.messages.append(
        {"role": role, 
         "content": content
    })


def greet():
    with st.chat_message("assistant"):
        msg = "Hello, How may I help you?"
        st.markdown(msg)
    
    save__chat_to_history("assistant", msg)


def show_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


model = load_model()

# Site title
st.title("Llama2-ChatBot")

# initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []
    greet()
    

# When user asks query
if prompt := st.chat_input("Your query"):
    show_history()
    with st.chat_message("user"):
        st.markdown(prompt)
    save__chat_to_history("user", prompt)
    
    # get reply from the model
    with st.chat_message("assistant"):
        msg_holder = st.markdown("")
        reply = model.get_chat_response(prompt)
        
        print(reply)
        full_response = ""
        reply_list = reply.split('\n')
        
        for response in reply_list:
            for reply in response.split():
                full_response += reply + " " 
                msg_holder.markdown(full_response + "▌")
                time.sleep(0.05)
            full_response += "  \n"
        msg_holder.markdown(full_response)
        print(full_response)
                
    save__chat_to_history("assistant", full_response)
