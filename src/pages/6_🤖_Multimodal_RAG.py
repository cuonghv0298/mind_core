import os, gc
from langchain.agents import tool
from loguru import logger
from langchain.schema import HumanMessage
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv
import streamlit as st
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import (
    ImageNode,
)
from PIL import Image

import base64
from io import BytesIO
import shutil
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.indices import MultiModalVectorStoreIndex


from langchain_core.runnables.config import RunnableConfig
from llama_index.core import SimpleDirectoryReader
from uuid import uuid4
from llama_index.core.schema import ImageDocument, BaseNode, ImageNode

# Modules
from utils.query_engine import MultimodalQueryEngine
from utils.vector_db import VectorDB
from utils.loaders import load_documents
from langfuse.llama_index import LlamaIndexInstrumentor
import nest_asyncio
from dotenv import load_dotenv

load_dotenv("/datadrive/man.pham/.env")
nest_asyncio.apply()
# load env variable

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")


def _get_session():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    runtime = get_instance()
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("No Streamlit script run context found")
    session_id = ctx.session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session


# Get your keys from the Langfuse project settings page and set them as environment variables

# or pass them as arguments when initializing the instrumentor

instrumentor = LlamaIndexInstrumentor()

# Automatically trace all LlamaIndex operations

instrumentor.start()

# Obserability tools
# langfuse_client = Langfuse()
thread_id = _get_session().id
output_dir = "./data"

# The 'All' guardrail checks for Prompt Injection, Sensitive Topics, and Topic Restriction

langfuseproject_name = "handbook_agent_lang_graph"


# Streamlit UI Setup
st.set_page_config(initial_sidebar_state="collapsed")
ss = st.session_state
st.markdown(
    """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)


def handle_file_upload():

    file = st.file_uploader(
        "**Upload your file to create a document id**",
        type="pdf",
        help="PDF file to be parsed",
    )

    # Define a custom file extractor for image files
    if file is not None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        path = os.path.join(output_dir, file.name)
        # Save uploaded file to disk
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        if file:
            st.success(f"File uploaded successfully", icon="✅")
        else:
            st.warning(
                "Your PDF contains mostly images, which may cause issues later. \
                Please upload a PDF that allows you to copy text directly to your clipboard"
            )
        return path


def save_feedback(index):
    feedback = ss[f"feedback_{index}"]
    trace = ss.get(f"trace_{index}")
    ss.messages_mu[index]["feedback"] = feedback
    if int(feedback) == 0:
        ss.show_comment_box = index
    elif trace:
        ss.show_comment_box = None
        logger.info(
            f"Feedback: {round(feedback, 2)} for question: {ss.messages_mu[index]}"
        )
        # langfuse_client.score(
        #     trace_id=trace,
        #     name="feedback",
        #     value=round(feedback, 2),
        # )


def save_comment(index):
    comment = ss.get(f"comment_{index}", "")
    ss.messages_mu[index]["comment"] = comment
    ss.show_comment_box = None  # Close comment box
    feedback = ss.messages_mu[index]["feedback"]
    trace = ss.get(f"trace_{index}")
    # langfuse_client.score(
    #     trace_id=trace, name="feedback", value=round(feedback, 2), comment=comment
    # )
    logger.info(f"Feedback: {round(feedback, 2)} for question: {ss.messages_mu[index]}")


def plot_images(image_paths):
    from PIL import Image
    from io import BytesIO

    for img_path in image_paths:
        if isinstance(img_path, BytesIO):
            image = Image.open(img_path)
            st.image(image=image, width=200)


def create_query_engine(retriever):
    gpt_4o = OpenAIMultiModal(model="gpt-4o", max_new_tokens=4096, temperature=0.0)
    query_engine = MultimodalQueryEngine(
        retriever=retriever, multi_modal_llm=gpt_4o, streaming=True
    )
    return query_engine


# Define variables in state
if "show_comment_box" not in ss:
    ss.show_comment_box = None
if "collection_name" not in ss:
    ss.collection_name = "LlamaIndex_da9b7bb158e64c93bea491df09894psd"
if "tenant_name" not in ss:
    ss.tenant_name = None
if "trace_id" not in ss:
    ss.trace_id = None
if "retriever" not in ss:
    ss.retriever = None
if "vector_store" not in ss:
    ss.vector_store = None

if "messages_mu" not in ss:
    ss["messages_mu"] = [
        {
            "role": "assistant",
            "content": "Xin chào Archer, tôi có thể giúp gì cho bạn ?",
        }
    ]


# wdm_f7c03e11e4164392869ad49796b65fd1
def refresh():
    ss.messages_mu = [
        {
            "role": "assistant",
            "content": "Xin chào Archer, tôi có thể giúp gì cho bạn ?",
        }
    ]
    st.cache_resource.clear()
    st.cache_data.clear()
    gc.collect()
    st.rerun()


## Sidebar
with st.sidebar:
    if st.button("Làm mới cuộc hội thoại", type="primary"):
        refresh()
    # LLM Provider Selection
    ss.vector_store = st.sidebar.radio(
        "Choose vector store", ["qdrant", "weaviate"], index=0
    )


def create_retriever(index):
    # Create retriever
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tenant",
                operator=FilterOperator.EQ,
                value=ss.tenant_name,
            ),
        ]
    )
    return index.as_retriever(
        similarity_top_k=6,
        image_similarity_top_k=6,
        filters=filters,
    )


st.title("🔱 Poseidon Agent")
col1, col2 = st.columns(2)
with col1:
    file_path = handle_file_upload()
    if file_path is not None:
        from llama_index.readers.file import ImageReader
        from pathlib import Path

        embedding_btn = st.button(
            "Generate Your Document ID",
            key="embedding_btn",
            help="Access Chatbot Read Your PDF to Gen Document ID",
        )

        if embedding_btn:
            with st.spinner("Processing PDF... This may take a moment"):
                # Reset retreiver
                ss.retriever = None

                ss.tenant_name = f"wdm_{uuid4().hex}"
                # Check vector store and tenant
                if ss.vector_store and ss.tenant_name:
                    ## Step 1: Load documents
                    input_doc_path = Path(file_path)
                    documents = load_documents(input_doc_path)
                    ## Step 3: Create tenant
                    vectordb = VectorDB(type=ss.vector_store, tenant=ss.tenant_name)
                    index = vectordb.load_index_from_documents(documents)
                    ss.retriever = create_retriever(index)
                else:
                    st.error("Please select vector store")
            if ss.retriever:
                st.success(
                    f"PDF processed successfully! Your code: {ss.tenant_name}",
                    icon="✅",
                )
                # Reset messages
                ss.messages_mu = [
                    {
                        "role": "assistant",
                        "content": "Xin chào Archer, tôi có thể giúp gì cho bạn ?",
                    }
                ]
            else:
                st.error("PDF processing failed!")
with col2:
    st.text_input(
        "Nhập mã tài liệu của bạn",
        value=ss.tenant_name,
        key="tenant_name",
        help="wdm_e04bc24a3ee44856ae25ebe85b64b973",
        placeholder="wdm_e04bc24a3ee44856ae25ebe85b64b973",
    )
    vectordb = VectorDB(type="qdrant", tenant=ss.tenant_name)
    index = vectordb.load_index()
    ss.retriever = create_retriever(index)
# wdm_8c2d223085174e9fbe9ff8acb731b49a
if ss.tenant_name and ss.retriever:
    query_engine = create_query_engine(ss.retriever)
    messages_col, images_col = st.columns(2)
    ## Display messages
    for i, message in enumerate(ss.messages_mu):
        st.chat_message(message["role"]).write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            ss[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )
            # Show comment box if "👎" is clicked
            if ss.show_comment_box == i:
                st.text_area(
                    "Có thể cải thiện được điều gì?",
                    key=f"comment_{i}",
                    placeholder="Góp ý ở đây...",
                    on_change=save_comment,
                    args=[i],
                )

    ## User input message
    if prompt := st.chat_input(max_chars=3000):
        st.chat_message("user").write(prompt)
        ss.messages_mu.append({"role": "user", "content": prompt})

        try:
            with instrumentor.observe(
                user_id="kendrick", session_id=thread_id
            ) as trace:
                rs = query_engine.query(prompt)
            images = [
                Image.open(BytesIO(base64.b64decode(node.node.image)))
                for node in rs.metadata["image_nodes"]
            ]
            response = rs.response_gen
        except Exception as e:
            logger.error(e)
            response = "Vui lòng hỏi lại"
            images = []

        if isinstance(response, str):
            response = st.chat_message("assistant").write(response)
        else:
            with st.spinner(f"Đang suy nghĩ..."):
                response = st.chat_message("assistant").write_stream(response)

        n_messages = len(ss.messages_mu)

        st.feedback(
            "thumbs",
            key=f"feedback_{n_messages}",
            on_change=save_feedback,
            args=[n_messages],
        )
        ss.messages_mu.append({"role": "assistant", "content": response})
        st.image(images)
        # langfuse_client.flush()
        instrumentor.flush()
