from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.context import tracing_v2_enabled

import streamlit as st 

from utils.sidebar_config import setup_llm
from utils.llm_models import get_llm_model


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    # Handler for OpenAI streaming
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
    
    # Handlers for Ollama streaming
    def on_llm_start(self, *args, **kwargs) -> None:
        self.text = ""
        self.container.markdown(self.text)

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.text = ""
        self.container.markdown(self.text)

    def on_text(self, text: str, **kwargs) -> None:
        self.text += text
        self.container.markdown(self.text)
        
def initialize_page():
    st.set_page_config(
        page_title="Chat With LLM",
        page_icon="💬"
    )
    # Streamlit UI        
    st.title("💬 Chat With LLM")
    st.subheader("Pick 1 model and chat")

def main():
    initialize_page()
    
    # Get all configuration settings
    config = setup_llm()

    # Access settings as needed
    llm_choice = config['llm_choice']
    model_choice = config['model_choice']
    chat_permission = config['chat_permission']
    param_config = config['parameters']
    if chat_permission:    
        if "messages" not in st.session_state:
            st.session_state["messages"] = [AIMessage(content="I'm a chatbot from Kyanon Digital—basically a wizard trapped in code. What can I conjure up for you today?")]
        
        # Add this near the top of the UI section, after st.title and st.subheader
        if st.button("New Chat"):
            st.session_state.messages = [AIMessage(content="I'm a chatbot from Kyanon Digital—basically a wizard trapped in code. What can I conjure up for you today?")]
            st.rerun()
            
        for msg in st.session_state.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt := st.chat_input():
            st.session_state.messages.append(HumanMessage(content=prompt))
            st.chat_message('human').write(prompt)
            
            with st.chat_message('assistant'):
                stream_handler = StreamHandler(st.empty())
                llm = get_llm_model(
                    chatmodel=llm_choice, 
                    model_name=model_choice, 
                    param=param_config, 
                    stream_handler=stream_handler,
                    )
                with tracing_v2_enabled(project_name="DEPLOY_MIND_Chat_With_LLM"):
                    if llm_choice == "Gemini":
                        for chunk in llm.stream(st.session_state.messages):
                            stream_handler.on_llm_new_token(chunk.content)
                        st.session_state.messages.append(AIMessage(content=stream_handler.text))
                    else:
                        response = llm.invoke(st.session_state.messages)
                        st.session_state.messages.append(AIMessage(content=response.content))
                        
if __name__ == "__main__":
    main()