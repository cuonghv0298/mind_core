import streamlit as st
import uuid
import asyncio
import os

from langchain_core.tracers.context import tracing_v2_enabled
from PyPDF2 import PdfReader


from  utils.generate import Generation
from  utils.sidebar_config import setup_llm
from  utils.llm_models import get_llm_model
from  utils.embedding import VectorDB
from  utils import kd_chatbot
import  utils.utils as utils
from  config.llm_config import ChatConfig


def initialize_page():
    file_config = utils.load_config("src/config/file_config.yml")
    prompt_config = utils.load_config(file_config["llm_env"]["prompting_file"])
    model_config = utils.load_config(file_config["llm_env"]["model_config_file"])
    st.set_page_config(page_title="Chat with PDF App", page_icon="📕")
    st.title("📕 Chat with PDF App")
    st.subheader("Chat to find your thoughts in your PDF")
    return_para = {
        "model_config": model_config,
        "prompt_config": prompt_config,
    }
    st.markdown(
        """
    ##### *How to Use:*

    1. **Upload Your PDF**: Start by uploading your PDF document, enabling you to copy text directly to your clipboard. 

    2. **Generate Document ID**: Click on "Generate Your Document ID" to process the PDF and create an ID. Save this ID for future reference.

    3. **Start Chatting**: Enter your document ID and ensure you have the necessary permissions to chat. You can then start asking questions about the content of your PDF.

    4. **Use Support Tools**: Utilize the support tools to look up specific text, generate example questions, or summarize the document.

    5. **New Chat**: Click "New Chat" to start a fresh conversation with the PDF.
    """
    )
    return return_para


def handle_file_upload():
    file = st.file_uploader(
        "**Upload your file to create a document id**",
        type="pdf",
        help="PDF file to be parsed",
    )
    if file is not None:
        reader = PdfReader(file)
        texts = [page.extract_text() for page in reader.pages]
        text = " ".join(texts)
        if len(text) > 2:
            st.success("File uploaded successfully", icon="✅")
        else:
            st.warning(
                "Your PDF contains mostly images, which may cause issues later. \
                Please upload a PDF that allows you to copy text directly to your clipboard"
            )
    return file


def main():
    return_para = initialize_page()
    model_config = return_para.get("model_config")
    prompt_config = return_para.get("prompt_config")

    # Start Generation bot
    generate = Generation(prompt_config)
    # Start VectorDB
    vectordb = VectorDB(model_config)
    # Get all configuration settings
    config = setup_llm()
    # Overload config from streamlit UI
    model_config["condense_question"]["client"] = config["llm_choice"]
    model_config["condense_question"]["model_name"] = config["model_choice"]
    model_config["combine_docs"]["client"] = config["llm_choice"]
    model_config["combine_docs"]["model_name"] = config["model_choice"]
    # Access settings
    chat_permission = config["chat_permission"]

    # initialize st session_state
    if "document_id5" not in st.session_state:
        st.session_state.document_id5 = None
    if "embedding_btn" not in st.session_state:
        st.session_state.embedding_btn = None
    col1, col2 = st.columns(2)
    with col1:
        file = handle_file_upload()
        if file is not None:
            embedding_btn = st.button(
                "Generate Your Document ID",
                key="embedding_btn",
                help="Access Chatbot Read Your PDF to Gen Document ID",
            )

            if embedding_btn:
                with st.spinner("Processing PDF... This may take a moment"):
                    print("file", file)
                    index, tenant_name, documents = (
                        vectordb.embedding_pdf_to_db_by_llamaindex(
                            index_name=os.environ.get("COLLECTION_ID_LLAMAINDEX"),
                            pdf_file=file,
                            
                        )
                    )
                    st.session_state.document_id5 = tenant_name
                st.success(
                    f"PDF processed successfully! Your code: {tenant_name}",
                )

    with col2:
        if st.session_state.document_id5:
            st.info("Please save the document ID in your note for future use.")
        document_id5 = st.text_input(
            "**Enter your document id** ",
            value=(
                st.session_state.document_id5 if st.session_state.document_id5 else ""
            ),
            label_visibility="visible",
            disabled=False,
        )
        st.markdown("**Example ID:**")
        st.code("deploy_mind_0a307e88553f4991af356345e879bb18", language="text")

        st.session_state.document_id5 = document_id5

    if not (st.session_state.document_id5 and chat_permission):
        st.warning("Fill OpenAI API key and Document ID to chat")
        return
    else:
        document_sources, error = vectordb.get_all_source_from_tenant(
            tenant_name=st.session_state.document_id5,
            index_name=os.environ.get("COLLECTION_ID_LLAMAINDEX"),
            multi_tenancy=False,
        )
        st.info(f"Your conversation is based on: **{document_sources[0]}**", icon="ℹ️")
        st.subheader("Support Tools")
        support_tabs = st.tabs(
            ["Look Up Evident Text", "Example question", "Summarization"]
        )
        with support_tabs[0]:
            # Look up evident text outside the chat
            object_id = st.text_input(
                "**Enter object id** ",
                label_visibility="visible",
                disabled=False,
                placeholder="f3c494d7-f205-4738-bcc4-b809c41363ea",
                help="Bot return object_id while answering your question",
            )

            if object_id:
                response = vectordb.get_quote_from_object_id(
                    tenant_name=st.session_state.document_id5,
                    object_id=object_id,
                    index_name=os.environ.get("COLLECTION_ID_LLAMAINDEX"),
                    multi_tenancy=False,
                )
                text = utils.preprocess_text_for_markdown(response.properties["text"])
                try:
                    source = f"File: {response.properties['file_name']}, Page: {response.properties['page_label']}"
                except:
                    source = f"{response.properties['file_name']}"
                with st.expander(f"View Quote Details from {source}"):
                    st.markdown(f"**Content:**\n> {text}")
                    # st.markdown(f"**From source:**\n> {source}")
        with support_tabs[1]:
            # Use a different key for session state
            if "gen_ex_question_triggered" not in st.session_state:
                st.session_state.gen_ex_question_triggered = False

            gen_ex_quesiot_btn = st.button(
                "Generate Example Question",
                help="Randomly generate an example question",
            )

            if gen_ex_quesiot_btn:
                st.session_state.gen_ex_question_triggered = True

            if st.session_state.gen_ex_question_triggered:
                content = vectordb.choice_random_text(
                    tenant_name=st.session_state.document_id5,
                    index_name=os.environ.get("COLLECTION_ID_LLAMAINDEX"),
                    multi_tenancy=True,
                )
                llm = get_llm_model(
                    chatmodel=config["llm_choice"],
                    model_name=config["model_choice"],
                    param=config["parameters"],
                )

                question = generate.generate_question(llm, content)
                text = f'Here is the example question you can ask about the file: \n\n {question["text"]}'
                st.markdown(text)
                # Reset the trigger after displaying the question
                st.session_state.gen_ex_question_triggered = False
        with support_tabs[2]:
            option_map = {
                0: "Stuff",
                1: "Map-Reduce",
            }
            selection_sum_method = st.segmented_control(
                "Summarization methos",
                options=option_map.keys(),
                format_func=lambda option: option_map[option],
                selection_mode="single",
                default=0,
            )
            st.write(
                "Your selected option: "
                f"{None if selection_sum_method is None else option_map[selection_sum_method]}"
            )
            llm = get_llm_model(
                chatmodel=config["llm_choice"],
                model_name=config["model_choice"],
                param=config["parameters"],
            )
            sum_btn = st.button("Summarize your PDF")
            if sum_btn:
                document_contents = vectordb.get_all_docs(
                    tenant_name=st.session_state.document_id5,
                    index_name=os.environ.get("COLLECTION_ID_LLAMAINDEX"),
                    multi_tenancy=False,
                )
                response_sum = generate.summarized_by_stuff(llm, document_contents)
                st.markdown(f"About this PDF: \n\n {response_sum}")

        st.subheader("Start your conversation")
        if "session_id5" not in st.session_state:
            st.session_state.session_id5 = str(uuid.uuid4())
        if "messages5" not in st.session_state:
            st.session_state.messages5 = []

        if st.button("New Chat"):
            st.session_state.session_id5 = []
            st.session_state.session_id5 = str(uuid.uuid4())
            st.rerun()

        with st.chat_message("assistant"):
            st.markdown(
                "Hi Archer, I'm here to assist you in discovering the pieces of memories in your PDF."
            )

        # Display messages in a scrollable container
        with st.expander("Chat History", expanded=True):
            for message in st.session_state.messages5:
                role = "assistant" if message.get("role") == "assistant" else "user"
                with st.chat_message(role):
                    st.markdown(message["content"])

        if question := st.chat_input("Ask your question about the document"):
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages5.append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    config_llm = ChatConfig(prompt_config, model_config)
                    chatbot = kd_chatbot.start_chatbot(
                        config_llm=config_llm,
                        tenant_name=st.session_state.document_id5,
                        index=os.environ.get("COLLECTION_ID_LLAMAINDEX"),
                        debug=True,
                        multi_tenancy=False,
                    )

                    try:
                        with tracing_v2_enabled(
                            project_name="DEPLOY_MIND_Chat_with_PDF"
                        ):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            response, msg = loop.run_until_complete(
                                chatbot.ask(
                                    session_id=st.session_state.session_id5,
                                    question=question,
                                )
                            )
                            loop.close()
                            answer = response.get("chatbot_answer")
                            st.markdown(answer)
                            st.session_state.messages5.append(
                                {"role": "assistant", "content": answer}
                            )
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
