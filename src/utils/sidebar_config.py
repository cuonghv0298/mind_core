import streamlit as st 
import os 

OLLAMA_MODEL_LIST = ("qwen2.5:14b","gemma3:27b","llava:13b", "qwen2.5-coder:14b", "phi4:latest","phi:latest", "phi3:medium", "deepseek-r1:14b", "deepseek-r1:7b","mistral" )
OPEN_AI_MODEL_LIST = ("gpt-4o-mini", "gpt-4o")
GEMINI_AI_MODEL_LIST = ("gemini-2.0-flash", "gemini-2.0-flash-lite")
OLLAMA_VISION_MODEL_LIST = ("llama3.2-vision")
OPEN_AI_VISION_MODEL_LIST = ("gpt-4.1-nano","gpt-4.1-mini", "gpt-4o")

def setup_llm(mode:str = 'chat'):
    if mode == 'chat':
        ai_list = OPEN_AI_MODEL_LIST
        ollama_list = OLLAMA_MODEL_LIST
    elif mode == 'image':
        ai_list = OPEN_AI_VISION_MODEL_LIST
        ollama_list = OLLAMA_VISION_MODEL_LIST
    """
    Arg:
    - mode helps us define core concept: 
        - chat: text and prompt
        - image: images and vision
        - audio: audio and speech
    Sets up and manages the sidebar configuration for LLM providers and their parameters.
    Returns a dictionary containing all configuration settings.
    """
    config = {}
    
    # LLM Provider Selection
    llm_choice = st.sidebar.radio(
        "Choose LLM Provider",
        ["Ollama","OpenAI", "Gemini"],
        index=0
    )
    config['llm_choice'] = llm_choice
    
    # Provider-specific configuration
    if llm_choice != "Ollama":
        st.sidebar.markdown(f"### {llm_choice} Config")
        config['use_key_default'] = st.sidebar.checkbox("Using Our Key")
        
        if config['use_key_default']:
            config['api_key'] = {"llm_provider": "OpenAI", "api_key": os.environ['OPENAI_API_KEY']} if llm_choice == "OpenAI" else  {"llm_provider": "Gemini", "api_key": os.environ['GOOGLE_API_KEY']} 
            st.sidebar.text_input(
                f"Enter {llm_choice} API Key",
                key="api_key",
                type="password",
                disabled=config['use_key_default']  # Add this line
            )
        else:
            config['api_key'] = st.sidebar.text_input(f"Enter {llm_choice} API Key", key="api_key", type="password")

        if llm_choice == "OpenAI":
            st.sidebar.markdown("[Get an OpenAI API Key](https://platform.openai.com/account/api-keys)")
        else:
            st.sidebar.markdown("[Get an Gemini API Key](https://ai.google.dev/gemini-api/docs/api-key)")
        
        if llm_choice == "Gemini": ai_list = GEMINI_AI_MODEL_LIST
        
        config['model_choice'] = st.sidebar.selectbox(
            f"{llm_choice} Model Provider",
            ai_list,
        )
        config['chat_permission'] = bool(config['api_key'])
        if not config['chat_permission']:
            st.warning(f"Please enter your {llm_choice} API key to get started. Currently, we are supporting you with \"Using Our Key\"", icon="⚠")
    
    else:
        st.sidebar.markdown("### Ollama Config")
        config['model_choice'] = st.sidebar.selectbox(
            "Ollama Model Provider",
            ollama_list,
        )
        config['chat_permission'] = True

    # Model Parameters
    st.sidebar.markdown("### Model Parameters")
    st.sidebar.markdown("""
    **Parameter Guide:**
    - Temperature: Controls randomness (0 = focused, 1 = creative)
    - Top P: Controls diversity of responses (lower = more focused)
    - Max Retries: Number of API retry attempts
    """)
    
    config['parameters'] = {
        "temperature": st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values (0.8-1.0) make output more random, lower values (0.2-0.5) make it more focused"
        ),
        "top_p": st.sidebar.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Limits cumulative probability in token selection. Lower values = more focused output"
        ),
        "max_retries": st.sidebar.number_input(
            "Max Retries",
            min_value=1,
            max_value=5,
            value=2,
            help="Maximum number of times to retry failed API calls"
        )
    }
    
    return config