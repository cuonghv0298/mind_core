from uuid import uuid4
import random
import os
from typing import Any, Dict, List, Optional, Sequence, Union
from PyPDF2 import PdfReader
import json

import streamlit as st
from weaviate.classes.tenants import Tenant
from langchain.schema import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore as WeaviateLC
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding as Llama_OpenAIEmbedding
from weaviate.classes.query import Filter

import tempfile
import qdrant_client  # Added import for qdrant-client
from qdrant_client import models  # Added import for qdrant models


import utils.utils as utils
from driver import qdrantdb
from driver import weaviatedb
from contextlib import contextmanager


DEFAULT_NAME_ID = "deploy_mind"
db = qdrantdb.QdrantConnector()


@contextmanager
def managed_client():
    client = db.get_client()
    try:
        yield client
    finally:
        client.close()


class VectorDB:
    def __init__(self, model_config):
        self.model_config = model_config
        self.embedding = self.chose_llm_embedding(
            llm_model="OpenAI", model="text-embedding-3-small"
        )
        self.vectorstore = qdrantdb.QdrantConnector()
        # self.client = config_db.get_client()

    # The chose_llm_embedding function is rewirte in config/llm_config.py
    def chose_llm_embedding(self, llm_model=None, model=None):
        if llm_model == None and model == None:
            llm_model = self.model_config["llm_embeding"]["client"]
            model = self.model_config["llm_embeding"]["model_name"]

        if llm_model == "OpenAI":
            embedding = OpenAIEmbeddings(model=model)
        elif llm_model == "Ollama":
            embedding = OllamaEmbeddings(model=model)
        else:
            raise "WE ONLY SUPPORT OpenAIEmbeddings AND OllamaEmbeddings"
        return embedding

    def import_data_to_db(
        self, meta_data, page_content, index_name="", tenant_name="", text_key="text"
    ):
        # As a default we will create a new tenant
        if index_name == "":
            index_name = os.environ.get("COLLECTION_ID")
        if tenant_name == "":
            tenant_name = f"{DEFAULT_NAME_ID}_{uuid4().hex}"
        text_splitter = utils.chunking(method="RecursiveCharacterTextSplitter")
        # client = config_db.get_client()
        embed_model = self.embedding
        doc = Document(
            metadata=meta_data,
            page_content=page_content,
        )
        documents = text_splitter.create_documents(
            [doc.page_content], metadatas=[doc.metadata]
        )

        db = self.vectorstore.load_document_to_vectordb(
            documents=documents,
            index_name=index_name,
            text_key=text_key,
            embedding=embed_model,
            tenant_name=tenant_name,
        )

        return db._index_name, tenant_name, db._text_key, documents

    def embedding_webpage_to_db(
        self, links: List[str], index_name="", tenant_name="", text_key="text"
    ):
        # As a default we will create a new tenant
        if index_name == "":
            index_name = os.environ.get("COLLECTION_ID")
        if tenant_name == "":
            tenant_name = f"{DEFAULT_NAME_ID}_{uuid4().hex}"
        # [ RecursiveCharacterTextSplitter, CharacterTextSplitter]
        text_splitter = utils.chunking(method="RecursiveCharacterTextSplitter")
        # client = config_db.get_client()
        embed_model = self.embedding
        documents = []
        for link in links:
            loader = WebBaseLoader(link)
            docs = loader.load()
            for doc in docs:
                text = utils.reformat_text(doc.page_content)
                doc_splits = text_splitter.create_documents(
                    [text], metadatas=[doc.metadata]
                )
                documents.extend(doc_splits)

        db = self.vectorstore.load_document_to_vectordb(
            documents=documents,
            index_name=index_name,
            text_key=text_key,
            embedding=embed_model,
            tenant_name=tenant_name,
        )
        return db._index_name, tenant_name, db._text_key, documents

    def embedding_pdf_to_db(
        self, pdf_file, index_name="", tenant_name="", text_key="text"
    ):
        if index_name == "":
            index_name = os.environ.get("COLLECTION_ID")
        if tenant_name == "":
            tenant_name = f"{DEFAULT_NAME_ID}_{uuid4().hex}"
        documents = []
        text_splitter = utils.chunking(method="RecursiveCharacterTextSplitter")
        embed_model = self.embedding
        reader = PdfReader(pdf_file)
        number_of_pages = len(reader.pages)
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            text = utils.preprocess_text_for_markdown(text)
            metadata = {
                "source": (
                    pdf_file.name
                    if type(pdf_file) == st.runtime.uploaded_file_manager.UploadedFile
                    else ""
                ),
                "page": i,
            }
            doc_splits = text_splitter.create_documents([text], metadatas=[metadata])
            documents.extend(doc_splits)
        db = self.vectorstore.load_document_to_vectordb(
            documents=documents,
            index_name=index_name,
            text_key=text_key,
            embedding=embed_model,
            tenant_name=tenant_name,
        )
        return db._index_name, tenant_name, db._text_key, documents

    def embedding_pdf_to_db_by_llamaindex(
        self, index_name="", tenant_name="", pdf_file=None
    ):

        if index_name == "":
            index_name = os.environ.get("COLLECTION_ID")
        if tenant_name == "":
            tenant_name = f"{DEFAULT_NAME_ID}_{uuid4().hex}"
        embed_model = self.embedding
        # Create a tempfile
        ## Create with lib tempfile
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", prefix=pdf_file.name[:-4] + "_"
        ) as temp_file:
            # Write the uploaded filße's content to the temporary file
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name
            # Read by SimpleDirectoryReader
            # file_extractor = {
            #     ".pdf": PdfReader,
            # }
            filename_fn = lambda filename: {"file_name": pdf_file.name}
            reader = SimpleDirectoryReader(
                input_files=[temp_file_path],
                file_metadata=filename_fn,
            )
            documents = reader.load_data()
        llama_index_embeddings = Llama_OpenAIEmbedding(
            model=embed_model.model,
        )
        print("going to connect to qdrant db")
        db = self.vectorstore.load_document_to_vectordb_llamaindex(
            documents=documents,
            index_name=index_name,
            embedding=llama_index_embeddings,
            tenant_name=tenant_name,
        )
        return index_name, tenant_name, documents

    def delete_collections_from_vectordb(self, index_name):
        with managed_client() as client:
            client.collections.delete(index_name)

    def delete_tenants_from_vectordb(self, index_name, tenant_name):
        with managed_client() as client:
            multi_collection = client.collections.get(index_name)
            multi_collection.tenants.remove([tenant_name])

    def delete_id_from_vectordb(
        self, index_name=None, text_key="text", tenant_name="Admin", ids=[]
    ):
        with managed_client() as client:
            weaviatevectorstore = WeaviateLC(
                client, index_name=index_name, text_key=text_key, use_multi_tenancy=True
            )
            weaviatevectorstore.delete(ids=ids, tenant=tenant_name)

    def create_collection(self, collection_id):
        with managed_client() as client:
            text_key = "text"
            use_multi_tenancy = True
            WeaviateLC(
                client=client,
                index_name=collection_id,
                text_key=text_key,
                use_multi_tenancy=use_multi_tenancy,
            )
        return collection_id, text_key

    def create_tenant(self, tenant_id: str = "", collection_id=""):
        if collection_id == "":
            collection_id = os.environ.get("COLLECTION_ID")
        if tenant_id == "":
            tenant_id = f"{DEFAULT_NAME_ID}_{uuid4().hex}"
        with managed_client() as client:
            multi_collection = client.collections.get(collection_id)
            # Create new tenant with input collection_id
            multi_collection.tenants.create(tenants=[Tenant(name=tenant_id)])
        return collection_id, tenant_id

    def get_quote_from_object_id(
        self, tenant_name, object_id, index_name="", multi_tenancy=True
    ):
        with managed_client() as client:
            if index_name == "":
                index_name = os.environ.get("COLLECTION_ID")
            if multi_tenancy:
                multi_collection = client.collections.get(index_name)
                my_tenant = multi_collection.with_tenant(tenant_name)
                response = my_tenant.query.fetch_object_by_id(uuid=object_id)
            else:
                collection = client.collections.get(index_name)
                response = collection.query.fetch_object_by_id(uuid=object_id)
        return response

    def get_all_source_from_tenant(
        self, tenant_name, index_name="", multi_tenancy=True, **kwargs
    ):
        error = f"Please recheck your id"
        source = []
        vector_type = "qdrant"
        if index_name == "":
            index_name = os.environ.get("COLLECTION_ID")

        if vector_type is not None and vector_type == "qdrant":
            source, error = self.get_first_point_from_qdrant_collection(
                collection_name=index_name, tenant_name_filter=tenant_name
            )
        else:
            # store by langchain and leverage weaviate multitenancy
            with managed_client() as client:
                if multi_tenancy:
                    multi_collection = client.collections.get(index_name)
                    my_tenant = multi_collection.with_tenant(tenant_name)
                    response = my_tenant.query.fetch_objects(include_vector=False)
                    source = list(
                        set([ob.properties.get("source") for ob in response.objects])
                    )
                    error = ""
                else:
                    collection = client.collections.get(index_name)
                    response = collection.query.fetch_objects(
                        filters=Filter.by_property("tenant_name").equal(tenant_name),
                    )
                    source = list(
                        set([ob.properties.get("file_name") for ob in response.objects])
                    )
                    error = ""
        return source, error

    def choice_random_text(self, tenant_name, index_name="", multi_tenancy=True):
        with managed_client() as client:
            if index_name == "":
                index_name = os.environ.get("COLLECTION_ID")
            if multi_tenancy:
                limit = client.count(
                    collection_name=index_name,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_name",
                                match=models.MatchValue(
                                    value=tenant_name
                                ),
                            ),
                        ]
                    ),
                    exact=True,
                ).count
                point_list = client.scroll(
                    collection_name=index_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_name",
                                match=models.MatchValue(
                                    value=tenant_name
                                ),
                            ),
                        ]
                    ),
                    limit=limit,
                    with_payload=True,
                )
                object = random.choice(point_list[0])
                text_object = json.loads(object.payload['_node_content'])['text']
                text = utils.preprocess_text_for_markdown(text_object)
            else:
                collection = client.collections.get(index_name)
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("tenant_name").equal(tenant_name),
                )
                object = random.choice(response.objects)
                text = utils.preprocess_text_for_markdown(object.properties["text"])
        return text

    def get_all_docs(self, tenant_name, index_name="", multi_tenancy=True):
        with managed_client() as client:
            if index_name == "":
                index_name = os.environ.get("COLLECTION_ID")
            if multi_tenancy:
                limit = client.count(
                    collection_name=index_name,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_name",
                                match=models.MatchValue(
                                    value=tenant_name
                                ),
                            ),
                        ]
                    ),
                    exact=True,
                ).count
                point_list = client.scroll(
                    collection_name=index_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_name",
                                match=models.MatchValue(
                                    value=tenant_name
                                ),
                            ),
                        ]
                    ),
                    limit=limit,
                    with_payload=True,
                )
                
                multi_collection = client.collections.get(index_name)
                my_tenant = multi_collection.with_tenant(tenant_name)
                response = my_tenant.query.fetch_objects(include_vector=False)
                tenant_docs = []
                for point in point_list[0]:
                # for ob in response.objects:
                    node_content_as_dict = json.loads(point.payload['_node_content']) 
                    text = node_content_as_dict['text']
                    metadata = {
                        "source": node_content_as_dict['metadata']['file_name'],
                        "page": node_content_as_dict['metadata']['page_label'],
                    }
                    doc = Document(
                        metadata=metadata,
                        page_content=text,
                    )
                    tenant_docs.append(doc)
            else:
                collection = client.collections.get(index_name)
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("tenant_name").equal(tenant_name),
                )
                tenant_docs = []

                for ob in response.objects:
                    print("OBBBB:", ob)
                    text = ob.properties["text"]
                    metadata = {
                        "source": ob.properties["file_name"],
                        "page": ob.properties["page_label"],
                    }
                    doc = Document(
                        metadata=metadata,
                        page_content=text,
                    )
                    tenant_docs.append(doc)
        return tenant_docs

    def get_first_point_from_qdrant_collection(
        self, collection_name: str, tenant_name_filter: str
    ):
        """
        Retrieves the first point from a Qdrant collection that matches the tenant_name metadata filter.

        Args:
            collection_name (str): The name of the Qdrant collection.
            tenant_name_filter (str): The value to filter the 'tenant_name' metadata by. Defaults to "ahihi".

        Returns:
            The first point (dict) found, or None if no matching point is found or an error occurs.
        """
        sources = []
        error = ""
        try:
            # Initialize Qdrant client
            # Assuming QDRANT_HOST is set in your environment variables
            qdrant_host = os.environ.get("QDRANT_HOST")
            if not qdrant_host:
                print("Error: QDRANT_HOST environment variable not set.")
                return None

            client = qdrant_client.QdrantClient(
                host=qdrant_host, port=6333
            )  # Default Qdrant port

            # Define the filter for metadata
            filter_conditions = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_name",  # Assuming your metadata field is named 'tenant_name'
                        match=models.MatchValue(value=tenant_name_filter),
                    )
                ]
            )

            # Scroll for the first point that matches the filter
            points, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_conditions,
                limit=1,  # We only need the first point
                with_payload=True,  # Include the payload (metadata)
                with_vectors=False,  # Optionally, set to True if you need vectors
            )

            if points:
                sources = [point.payload["file_name"] for point in points]
            else:
                error = f"No points found in collection '{collection_name}' with tenant_name '{tenant_name_filter}'."

        except Exception as e:
            ## add tracing

            error = f"An error occurred while querying Qdrant: {e}"

        return sources, error


def extract_link_objects_from_document(content: str):
    link_objects = []
    links = utils.find_links(content)
    for link in links:
        loader = WebBaseLoader(link)
        docs = loader.load()
        link_content = " ".join(utils.reformat_text(doc.page_content) for doc in docs)
        link_objects.append({"link": link, "content": link_content})
    return link_objects


def verify_document(
    document_content,
    document_metadata,
    document_db_content,
    link_db_link,
    avail_storage,
    content_limit=1000000,
):
    """
    Args:
    - document_content: current content
    - document_metadata: current metadata
    - document_db_content: list content from document database
    - link_db_link: list of link from links database
    - avail_storage: number of words is available to store to vectorDB
    - content_limit: limited number of words for each document_content or link_content (default is 1 million)

    Returns:
    - document_status: str, status after verification
    - document_to_save: dict, object to save
    - link_to_save: list of links to save
    """
    if document_content in document_db_content:
        return "Document exists", None, None

    document_content_word_count = utils.count_words(document_content)

    if document_content_word_count > content_limit:
        return (
            f"The number of words exceeds the {content_limit} limit. The document has {document_content_word_count}",
            None,
            None,
        )

    content_to_save = [document_content]
    save_links = []

    extracted_links = utils.find_links(document_content)
    valid_links = [link for link in extracted_links if link not in link_db_link]

    for link in valid_links:
        loader = WebBaseLoader(link)
        docs = loader.load()
        link_content = " ".join(doc.page_content for doc in docs)
        link_word_count = utils.count_words(link_content)

        if link_word_count < content_limit:
            save_links.append(link)
            content_to_save.append(link_content)

    content_to_save_text = " ".join(content_to_save)
    content_to_save_word_count = utils.count_words(content_to_save_text)

    if content_to_save_word_count > avail_storage:
        if document_content_word_count > avail_storage:
            return (
                f"Out of storage: available storage is {avail_storage} words, document content is {document_content_word_count} words",
                None,
                None,
            )
        else:
            return (
                f"Excessive content from links resulting in out of storage. Only document content will be stored.",
                {"page_content": document_content, "meta_data": document_metadata},
                None,
            )

    return (
        "Pass database health check",
        {"page_content": document_content, "meta_data": document_metadata},
        save_links,
    )
