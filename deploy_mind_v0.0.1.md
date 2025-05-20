# Deploy Mind v0.0.1 Documentation

## Introduction
Deploy Mind is an interactive application suite designed to facilitate advanced chat and document interaction experiences using large language models (LLMs). The platform provides multiple interfaces for chatting with LLMs, images, and PDF documents, supporting both single and multi-document workflows. It leverages modern AI and NLP technologies to enable users to extract insights, ask questions, and summarize content from various data sources.

## Overall Architecture
The application is structured as a modular Streamlit-based frontend, with each page dedicated to a specific interaction mode:
- **Chat With LLM**: Direct conversational interface with an LLM, supporting model selection and chat history.
- **Chat With Images**: Allows users to upload and interact with images, integrating image handling and LLM-based analysis.
- **Chat With PDF**: Enables uploading a single PDF, extracting text, generating document IDs, and chatting about the document's content.
- **Chat With PDFs**: Supports multi-PDF workflows, including document ID management, embedding, and advanced support tools (lookup, summarization, example questions).
- **Chat With PDF (New Injection)**: An enhanced PDF chat interface with improved document ID and support tool integration.

Each page initializes its own configuration, manages session state, and provides a tailored user experience for its modality. The backend leverages modular utility functions for LLM setup, model management, embedding, and PDF/image processing.

## Tech Stacks
- **Frontend/UI**: [Streamlit](https://streamlit.io/) for rapid web app development and interactive UI components.
- **LLM Integration**: [LangChain](https://python.langchain.com/) for LLM orchestration, message handling, and streaming callbacks.
- **PDF Processing**: [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF text extraction; custom utilities for document parsing and image extraction.
- **Image Handling**: Native Streamlit file uploaders and session state for persistent image management.
- **Vector Database**: Custom embedding and vector search utilities (VectorDB) for document retrieval and question answering.
- **Configuration Management**: YAML-based config files for model, prompt, and file settings.
- **Other Utilities**: Async support, UUID for unique document/session IDs, and modular utility imports for code reuse.

## Next Steps
- Add Multimodal support.
- Enhance PDF processing for better text and image extraction.
- Optimize performance for large documents and images.
- Established evaluation.
- Add security measures for sensitive data.