import os
from typing import Optional, Any, List, Dict

import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore as WeaviateLC


from llama_index.vector_stores.weaviate import WeaviateVectorStore as WeaviateIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent

from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import models, QdrantClient

import contextlib


QDRANT_HOST = "QDRANT_HOST"
QDRANT_HOST_PORT = "QDRANT_HOST_PORT"
QDRANT_GPC_URL_PORT = "QDRANT_GPC_URL_PORT"

CHUNK_SIZE = 512
CHUNK_OVER_OVERLAP = 10


class AttatchTenant(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            if "tenant_name" in kwargs:
                node.metadata["tenant_name"] = kwargs["tenant_name"]
        return nodes


class QdrantConnector:
    """A connector to Qdrant."""

    def __init__(
        self,
        host: Optional[str] = None,
        host_port: Optional[str] = None,
        gpc_url_port: Optional[str] = None,
    ):
        self.__client = None
        self.host = host or os.environ.get(QDRANT_HOST)
        self.host_port = host_port or os.environ.get(QDRANT_HOST_PORT)
        self.gpc_url_port = gpc_url_port or os.environ.get(QDRANT_GPC_URL_PORT)
        self._connect()

    def _connect(
        self,
        host: Optional[str] = None,
        host_port: Optional[str] = None,
        gpc_url_port: Optional[str] = None,
    ):
        try:
            self.host = host or os.environ.get(QDRANT_HOST)
            self.host_port = host_port or os.environ.get(QDRANT_HOST_PORT)
            self.gpc_url_port = gpc_url_port or os.environ.get(QDRANT_GPC_URL_PORT)
            print(f'host / port / gpc_url_port: {self.host} / {self.host_port} / {self.gpc_url_port}')
            self.__client = QdrantClient(
                host=self.host, port=self.host_port, grpc_port=self.gpc_url_port
            )
        except Exception as e:
            print(str(e))

    # Get the Qdrant client
    def get_client(self):
        if not self.__client:
            self._connect() # connect to Qdrant server default
            
        return self.__client

    # Check connection
    def check_connectionc_status(
        self,
    ) -> bool:
        if not self.__client:
            return False
        else:
            return True 
    # Load document to vector database by llamaindex
    def load_document_to_vectordb_llamaindex(
        self,
        documents,
        index_name,
        embedding,
        tenant_name,
        debug=False,
    ):
        # Create a local Qdrant vector store
        print('connecting to client')
        client = self.get_client() #connect to Qdrant server default
        print("client", client)
        print("index_name", index_name)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=index_name,
        )
        print("vector_store", vector_store)
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVER_OVERLAP
                ),
                AttatchTenant(),
                embedding,
            ],
            vector_store=vector_store,
        )
        print("pipeline", pipeline)
        node = pipeline.run(
            documents=documents,
            show_progress=True,
            num_workers=-1,
            tenant_name=tenant_name,
        )
        # with contextlib.closing(client):
            # nodes = pipeline.run(
            #     documents=documents, show_progress=True, num_workers=-1
            # )
        ## Enable Multi-Tenancy
        if client is not None and client.collection_exists(index_name):
            client.create_payload_index(
                collection_name=index_name,
                field_name="metadata.tenant_name",
                field_type=models.PayloadSchemaType.KEYWORD,
            )

        return index_name, tenant_name, documents
