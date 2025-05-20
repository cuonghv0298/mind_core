def data_ingestion(
    documents, collection_name: str, tenant: str, override: bool = False
):
    """Ingest data into the vectordb."""
    from llama_index.core import (
        Settings,
    )
    import asyncio
    import nest_asyncio
    import time
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.embeddings.openai import OpenAIEmbedding

    # from llama_index.storage.docstore.mongodb import MongoDocumentStore
    from llama_index.core.node_parser import SentenceSplitter
    import os
    from llama_index.vector_stores.weaviate import WeaviateVectorStore

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    # nest_asyncio.apply()

    import weaviate
    from llama_index.core.schema import TransformComponent
    from typing import Any, List
    from llama_index.core.schema import BaseNode

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

    client = weaviate.connect_to_custom(
        http_host=os.environ.get(
            "WEAVIATE_HOST", "weaviatedocker-weaviate-1"
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
    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name=collection_name
    )
    if override:
        client.connect()
        vector_store.delete_index()

    # MONGO_DATABASE_HOST: str = (
    #     "mongodb://mongo1:30001,mongo2:30002,mongo3:30003/?replicaSet=my-replica-set"
    # )
    # docstore = MongoDocumentStore.from_uri(
    #     uri=MONGO_DATABASE_HOST, db_name="llama_index_docstore"
    # )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=0),
            AttatchTenant(),
            OpenAIEmbedding(model="text-embedding-3-small"),
        ],
        vector_store=vector_store,
    )
    then = time.time()
    # loop = asyncio.get_event_loop()
    # nodes = loop.run_until_complete(
    #     pipeline.arun(documents=documents, show_progress=True, num_workers=-1)
    # )

    nodes = pipeline.run(
        documents=documents, show_progress=True, num_workers=-1, tenant=tenant
    )
    end = time.time() - then
    print(f"Latency: {end}")
    print(f"Ingested {len(nodes)} nodes")
    client.close()
    return nodes
