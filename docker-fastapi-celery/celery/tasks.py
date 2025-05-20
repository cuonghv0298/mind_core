import json
import traceback
from time import sleep
from celery import states

from worker import celery_app
from typing import List
from pydantic import BaseModel
from llama_index.core.schema import (
    ImageNode,
    TextNode,
)
from llama_index.core import Document
from models.models import IngestionRequest
import weaviate
import os
from slowapi.errors import RateLimitExceeded

# client = weaviate.connect_to_local(host="10.100.224.34", port="50051")


# print(client.is_ready())

# class ReturnModel(BaseModel):
#     value: str


# create celery worker for 'hello.task' task
@celery_app.task(name="hello.task", bind=True)
def hello_world(self, name):
    try:
        # if name is error
        if name == "error":
            # will raise ZeroDivisionError
            a, b = 1, 0
            a = a / b

        # update task state every 1 second
        for i in range(60):
            sleep(1)
            self.update_state(state="PROGRESS", meta={"done": i, "total": 60})

        # return result
        return {"result": f"hello {name}"}

    # if any error occurs
    except Exception as ex:
        # update task state to failure
        self.update_state(
            state=states.FAILURE,
            meta={
                "exc_type": type(ex).__name__,
                "exc_message": traceback.format_exc().split("\n"),
            },
        )

        # raise exception
        raise ex


# pydantic=True,
@celery_app.task(name="rag.etl", bind=True, retries=3)
def ETL_pipeline(self, data: dict):
    """Ingest data into the vectordb."""
    from llama_index.core import (
        Settings,
    )
    import asyncio
    import nest_asyncio
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.storage.docstore.mongodb import MongoDocumentStore
    from llama_index.core.node_parser import SentenceSplitter
    import os
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    from llama_index.core.schema import TransformComponent
    import weaviate
    from time import time, sleep
    import gc
    import contextlib

    nest_asyncio.apply()
    try:
        if isinstance(data, str):
            data = json.loads(data)

        documents = data.get("documents")
        documents = [Document.model_validate(doc) for doc in documents]
        tenant = data["tenant"]
        collection = data["collection_name"]
        self.update_state(state="INGESTING", meta={"total": len(documents)})

        class AttatchTenant(TransformComponent):
            def __call__(self, nodes, **kwargs):
                for node in nodes:
                    if "tenant_name" in kwargs:
                        node.metadata["tenant_name"] = kwargs["tenant_name"]
                return nodes

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        client = weaviate.connect_to_custom(
            http_host="10.10.193.73",  # "10.100.224.34",  # URL only, no http prefix
            http_port=os.environ.get("WEAVIATE_HOST_PORT"),  # "8080",
            http_secure=False,  # Set to True if https
            grpc_host=os.environ.get("WEAVIATE_GPC_URL"),  #  "10.100.224.34",
            grpc_port=os.environ.get(
                "WEAVIATE_GPC_URL_PORT"
            ),  # "50051",      # Default is 50051, WCD uses 443
            grpc_secure=False,  # Edit as needed
            # skip_init_checks=True,
        )

        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name=collection
        )
        # try:
        #     MONGO_DATABASE_HOST: str = (
        #         "mongodb://llm-core-mongo1:30001,llm-core-mongo2:30002,llm-core-mongo3:30003/?replicaSet=my-replica-set"
        #     )
        #     docstore = MongoDocumentStore.from_uri(
        #         uri=MONGO_DATABASE_HOST, db_name="llama_index_docstore"
        #     )
        # except Exception as e:
        #     print("Error connecting to MongoDB:", e)
        #     docstore = None

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=0),
                AttatchTenant(tenant_name=tenant),
                OpenAIEmbedding(model="text-embedding-3-small"),
            ],
            vector_store=vector_store,
            # docstore=docstore,
        )
        # loop = asyncio.get_event_loop()
        # nodes = loop.run_until_complete(
        #     pipeline.arun(documents=documents, show_progress=True, num_workers=-1)
        # )
        nodes = []
        with contextlib.closing(client):
            nodes = pipeline.run(
                documents=documents, show_progress=True, num_workers=-1
            )
        # gc.collect()
        # return result
        return {"ingested_node": len(nodes)}

    # if any error occurs
    except (ConnectionError, RateLimitExceeded) as e:
        # update task state to failure
        # self.update_state(
        #     state=states.FAILURE,
        #     meta={
        #         "exc_type": type(ex).__name__,
        #         "exc_message": traceback.format_exc().split("\n"),
        #     },
        # )

        # # raise exception
        # raise ex
        raise self.retry(exc=e, countdown=0.1)
