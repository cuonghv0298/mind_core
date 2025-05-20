from typing import Literal
import os
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import Document
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.schema import ImageDocument, BaseNode
from llama_index.core.schema import TransformComponent
from typing import Any, List
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core import Document, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.program import MultiModalLLMCompletionProgram
from pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from time import sleep
from dotenv import load_dotenv
from llama_index.embeddings.clip import ClipEmbedding

load_dotenv("/datadrive/man.pham/ownllm/.env")


class AttatchTenant(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        # Implement your transformation logic here
        # For example, add tenant information to each node's metadata
        for node in nodes:
            if "tenant" in kwargs:
                node.metadata["tenant"] = kwargs["tenant"]
            else:
                node.metadata["tenant"] = "empty"
        return nodes


class Response(BaseModel):
    description: str = Field(default="", description="Description from image")


class VectorDB:
    """A vector database."""

    QUESTION_GENERATOR_PROMPT = f"""
Based on the given the image, summary the image in brief.
"""
    gemini_llm = GeminiMultiModal(model_name="models/gemini-2.0-flash")

    def __init__(
        self, type: Literal["qdrant", "weaviate"], tenant: str, **kwargs
    ) -> None:
        """Initialize a vector database."""
        self.tenant = tenant
        match type:
            case "qdrant":
                from llama_index.vector_stores.qdrant import QdrantVectorStore
                import qdrant_client

                # Create a local Qdrant vector store
                self.client = qdrant_client.QdrantClient(
                    url=os.environ["QDRANT_HOST"],
                )
                self.text_store = QdrantVectorStore(
                    client=self.client,
                    collection_name="text_collection",
                )
                self.image_store = QdrantVectorStore(
                    client=self.client,
                    collection_name="image_collection",
                )
            case "weaviate":
                import weaviate

                self.client = weaviate.connect_to_custom(
                    http_host=os.environ.get(
                        "WEAVIATE_HOST"
                    ),  # "10.100.224.34",  # URL only, no http prefix
                    http_port=os.environ.get("WEAVIATE_HOST_PORT"),  # "8080",
                    http_secure=False,  # Set to True if https
                    grpc_host=os.environ.get("WEAVIATE_GPC_URL"),  #  "10.100.224.34",
                    grpc_port=os.environ.get(
                        "WEAVIATE_GPC_URL_PORT"
                    ),  # "50051",      # Default is 50051, WCD uses 443
                    grpc_secure=False,  # Edit as needed
                    skip_init_checks=True,
                )
                meta_info = self.client.get_meta()
                print(f"meta_info: {meta_info}")
                self.text_store = WeaviateVectorStore(
                    client=self.client,
                    index_name="LLamaindex_text_collection",
                )
                self.image_store = WeaviateVectorStore(
                    client=self.client,
                    index_name="LLamaindex_image_collection",
                )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.text_store, image_store=self.image_store
        )

    def _ingest_text(self, documents: list[Document]) -> BaseNode:
        """Ingest documents into the vector database."""
        import nest_asyncio
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.extractors import TitleExtractor
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.vector_stores import (
            MetadataFilter,
            MetadataFilters,
            FilterOperator,
        )

        nest_asyncio.apply()
        # Delete all nodes from the vector store for the current tenant
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="tenant", operator=FilterOperator.EQ, value=self.tenant
                ),
                MetadataFilter(
                    key="_node_type", operator=FilterOperator.EQ, value="TextNode"
                ),
            ]
        )
        try:
            self.text_store.delete_nodes(filters=filters)
        except Exception as e:
            pass
        pipeline = IngestionPipeline(
            transformations=[
                AttatchTenant(),
                SentenceSplitter(chunk_size=512),
                OpenAIEmbedding(model="text-embedding-3-small"),
            ],
            vector_store=self.text_store,
        )
        return pipeline.run(
            documents=documents, show_progress=True, num_workers=-1, tenant=self.tenant
        )

    def _ingest_image(
        self, documents: list[ImageDocument], with_context: bool = False
    ) -> list[ImageDocument]:
        """Preprocess images."""
        from llama_index.core.ingestion import IngestionPipeline
        import nest_asyncio
        from llama_index.core.vector_stores import (
            MetadataFilter,
            MetadataFilters,
            FilterOperator,
        )

        nest_asyncio.apply()

        # Delete all nodes from the vector store for the current tenant
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="tenant", operator=FilterOperator.EQ, value=self.tenant
                ),
                MetadataFilter(
                    key="_node_type", operator=FilterOperator.EQ, value="ImageNode"
                ),
            ]
        )
        try:
            self.image_store.delete_nodes(filters=filters)
        except Exception as e:
            pass
        # Calculate the delay based on your rate limit
        rate_limit_per_minute = 15
        delay = 60.0 / rate_limit_per_minute
        for doc in documents:
            doc.metadata["tenant"] = self.tenant
            # Extract description from image
            if with_context:
                llm_program = MultiModalLLMCompletionProgram.from_defaults(
                    output_parser=PydanticOutputParser(Response),
                    image_documents=[doc],
                    prompt_template_str=self.QUESTION_GENERATOR_PROMPT,
                    multi_modal_llm=self.gemini_llm,
                    verbose=True,
                )
                response = llm_program()
                sleep(delay)
                doc.set_content(str(response.description))
        MultiModalVectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=True,
        )
        return documents

    def load_index_from_documents(
        self, documents: list[Document]
    ) -> MultiModalVectorStoreIndex:
        """Load an index from a documents."""
        from time import time

        then = time()
        text_documents = [
            doc for doc in documents if not isinstance(doc, ImageDocument)
        ]
        image_documents = [doc for doc in documents if isinstance(doc, ImageDocument)]
        # text indexing
        text_nodes = self._ingest_text(text_documents)
        # image indexing
        image_nodes = self._ingest_image(image_documents, with_context=False)
        latency = time() - then
        print(
            f"Ingested {len(text_nodes)} text nodes and {len(image_nodes)} image nodes in {latency}s."
        )

        # load index
        return self.load_index()

    def load_index(self) -> MultiModalVectorStoreIndex:
        """Load an index from a vectordb."""
        from qdrant_client import models

        if self.client.collection_exists("text_collection"):
            self.client.create_payload_index(
                collection_name="text_collection",
                field_name="metadata.tenant",
                field_type=models.PayloadSchemaType.KEYWORD,
            )
        if self.client.collection_exists("image_collection"):
            self.client.create_payload_index(
                collection_name="image_collection",
                field_name="metadata.tenant",
                field_type=models.PayloadSchemaType.KEYWORD,
            )
            
        return MultiModalVectorStoreIndex.from_vector_store(
            vector_store=self.text_store,
            image_vector_store=self.image_store,
            show_progress=True,
        )
