from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def get_llm_model(chatmodel: str, model_name: str, param: dict, stream_handler=None):
    param['model'] = model_name
    if stream_handler is not None:
        if chatmodel != "Gemini":
            param['streaming'] = True
            param['callbacks'] = [stream_handler]
    if chatmodel == 'OpenAI':
        llm = ChatOpenAI(**param)
    elif chatmodel == 'Ollama':
        llm = ChatOllama(**param)
    elif chatmodel == "Gemini":
        llm = ChatGoogleGenerativeAI(**param)
    else:
        raise ValueError(f"We currently just support OpenAI, Gemini and Ollama, your client is {chatmodel}")
    return llm