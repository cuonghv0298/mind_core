import os
from typing import Optional, Any, List, Dict

import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore as WeaviateLC


from llama_index.vector_stores.weaviate import WeaviateVectorStore as WeaviateIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import models


import contextlib


WEAVIATE_APIKEY = "WEAVIATE_APIKEY"
WEAVIATE_HOST = "WEAVIATE_HOST"
WEAVIATE_HOST_PORT = "WEAVIATE_HOST_PORT"
WEAVIATE_GPC_URL = "WEAVIATE_GPC_URL"
WEAVIATE_GPC_URL_PORT = "WEAVIATE_GPC_URL_PORT"

WEAVIATE_USER = "WEAVIATE_USER"
WEAVIATE_PWD = "WEAVIATE_PWD"

DEFAULT_INDEX_NAME = "LangChainBot_DFIDXNM"
DEFAULT_TEXT_KEY = "LangChainBot_TK"
RAG_INDEX_NAME = "RAG_INDEX_NAME"
RAG_TEXT_KEY = "RAG_TEXT_KEY"

CONNECT_TYPE: List[str] = [
    "ANONYMOUS",
    "AuthAPIKey",
    "ClientPassword",
    "ClientCredentials",
    "BearerToken",
]

# "ENV_NAME" : "Optional-api-key-name"
ADDITIONAL_HEADERS: Dict[str, str] = {
    "COHERE_API_KEY": "X-Cohere-Api-Key",
    "HUGGINGFACE_API_KEY": "X-HuggingFace-Api-Key",
    "OPENAI_API_KEY": "X-OpenAI-Api-Key",
}

HYBRID_SEARCH_CORE: List[str] = ["langchain_retrievers", "weaviate_client_v3"]
CHUNK_SIZE = 512
CHUNK_OVER_OVERLAP = 10


class AttatchTenant(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            if "tenant_name" in kwargs:
                node.metadata["tenant_name"] = kwargs["tenant_name"]
        return nodes


class WeaviateDB:
    __host: str = None
    __connect_type: str = None
    __connected: bool = False
    __hybrid_search: bool = False
    __hybrid_search_core: str = None
    ##

    def __init__(
        self,
        connect_type: Optional[str] = None,
        hybrid_search: bool = True,
    ) -> None:
        try:
            # get env vars, re-check if connect type does not init with right env
            update_connect_type = self.__getEnvironmentVariables(
                connect_type=connect_type,
                hybrid_search=hybrid_search,
            )
            # try to connect by 3 methods
            if "AuthAPIKey" in update_connect_type and self.__connect_APIKey(
                hybrid_search=hybrid_search
            ):
                self.check_connection()
                print("-------Weaviate Client connect with AuthAPIKey")
            elif (
                "ClientPassword" in update_connect_type
                and self.__connect_ClientPassword(hybrid_search=hybrid_search)
            ):
                print("--------Weaviate Client connect with ClientPassword")
            elif "ANONYMOUS" in update_connect_type and self.__connect_ANONYMOUS(
                hybrid_search=hybrid_search
            ):
                print("Weaviate Client connect with ANONYMOUS")
            elif "localhost" in update_connect_type and self.__connect_with_localhost():
                print("Weaviate Client connect with localhost")
            else:
                print("Can not connect to Weavitae HOST")
                raise Exception(
                    "Weaviate Client can not connect or cannot access by any connect method"
                )
            print("------VectorDB client:", self.__client)
        except Exception as e:
            print(str(e))

    # return connect_type if None or empty env
    # update optional_api_key list if hybrid search is enabled into self.__api_keys

    def __getEnvironmentVariables(
        self,
        connect_type: Optional[str] = None,
        hybrid_search: bool = False,
        rag: bool = True,
    ) -> Optional[List[str]]:
        try:
            update_connect_type = []
            self.__host = (
                os.environ.get(WEAVIATE_HOST) or "localhost:8080"
            )  # host.docker.internal:8080
            # client required connection config
            if self.__host == "localhost:8080":
                if not connect_type or connect_type == "AuthAPIKey":
                    auth = os.environ.get(WEAVIATE_APIKEY) or None
                    if auth:
                        update_connect_type.append("AuthAPIKey")
                        self.__apiKey = auth
                if not connect_type or connect_type == "ClientPassword":
                    user = os.environ.get(WEAVIATE_USER) or None
                    pwd = os.environ.get(WEAVIATE_PWD) or None
                    if user and user != "ANONYMOUS" and pwd:
                        update_connect_type.append("ClientPassword")
                        self.__user = user
                        self.__pwd = pwd
                    else:
                        update_connect_type.append("localhost")
                        self.__user = "ANONYMOUS"
            else:
                update_connect_type.append("ANONYMOUS")
                self.__user = "ANONYMOUS"
            # optional hybrid search retrieval with external api keys
            if hybrid_search:
                self.__api_keys = {}
                #
                cohere_api_key = os.environ.get("COHERE_API_KEY") or None
                if cohere_api_key:
                    self.__api_keys["COHERE_API_KEY"] = cohere_api_key
                #
                huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY") or None
                if huggingface_api_key:
                    self.__api_keys["HUGGINGFACE_API_KEY"] = huggingface_api_key
                #
                openai_api_key = os.environ.get("OPENAI_API_KEY") or None
                if openai_api_key:
                    self.__api_keys["OPENAI_API_KEY"] = openai_api_key
            #
            if rag:
                rag_index_name = os.environ.get("RAG_INDEX_NAME") or DEFAULT_INDEX_NAME
                if rag_index_name:
                    self.__rag_index_name = rag_index_name
                #
                rag_text_key = os.environ.get("RAG_TEXT_KEY") or DEFAULT_TEXT_KEY
                if rag_text_key:
                    self.__rag_text_key = rag_text_key
            # return connect type list to try connecting
            return update_connect_type
        except Exception as e:
            print(str(e))
            return None

    #
    def __connect_with_localhost(
        self,
    ):
        try:
            self.__client = weaviate.connect_to_local()
            self.__connected = True
            self.__connect_type = "localhost"
            self.__hybrid_search = True
            return True
        except Exception as e:
            print(str(e))
            return False

    def __connect_APIKey(self, hybrid_search: bool = False) -> bool:
        if hybrid_search:
            additional_headers = {
                ADDITIONAL_HEADERS[name]: self.__api_keys[name]
                for name in self.__api_keys
            }
            # try to connect with api keys
            if additional_headers:
                try:
                    self.__client = weaviate.Client(
                        url=self.__host,
                        auth_client_secret=weaviate.AuthApiKey(self.__apiKey),
                        additional_headers=additional_headers,
                    )
                    self.__connected = True
                    self.__connect_type = "AuthAPIKey"
                    self.__hybrid_search = True
                    return True
                # except Exception as e:
                #     traceback.print_exc()
                #     print(str(e))
                except:
                    pass
        # if connect fail, try again without
        try:
            if "localhost" in self.__host:
                self.__client = weaviate.Client(
                    url="http://localhost:8080",  # Replace with your endpoint
                )
            else:
                self.__client = weaviate.Client(
                    url=self.__host,
                    auth_client_secret=weaviate.AuthApiKey(self.__apiKey),
                )
            self.__connected = True
            self.__connect_type = "AuthAPIKey"
            self.__hybrid_search = False
            return True
        # except Exception as e:
        #     traceback.print_exc()
        #     print(str(e))
        #     return False
        except:
            pass

    #

    def __connect_ClientPassword(
        self,
        hybrid_search: bool = False,
    ) -> bool:
        if hybrid_search:
            additional_headers = {
                ADDITIONAL_HEADERS[name]: self.__api_keys[name]
                for name in self.__api_keys
            }
            # try to connect with api keys
            if additional_headers:
                try:
                    self.__client = weaviate.Client(
                        url=self.__host,
                        auth_client_secret=weaviate.AuthClientPassword(
                            username=self.__user,
                            password=self.__pwd,
                            # optional, depends on the configuration of your identity provider (not required with WCS)
                            scope="offline_access",
                        ),
                        additional_headers=additional_headers,
                    )
                    print(
                        f"Connect into {self.__host} with AuthClientPassword and enabled hybrid search with {self.__api_keys.keys()}"
                    )
                    self.__connected = True
                    self.__connect_type = "ClientPassword"
                    self.__hybrid_search = True
                    return True
                # except Exception as e:
                #     print(str(e))
                except:
                    pass
        # if connect fail, try again without
        try:
            self.__client = weaviate.Client(
                url=self.__host,
                auth_client_secret=weaviate.AuthClientPassword(
                    username=self.__user,
                    password=self.__pwd,
                    # optional, depends on the configuration of your identity provider (not required with WCS)
                    scope="offline_access",
                ),
                additional_headers=additional_headers,
            )
            print(f"Connect into {self.__host} with AuthClientPassword")
            self.__connected = True
            self.__connect_type = "ClientPassword"
            self.__hybrid_search = False
            return True
        # except Exception as e:
        #     traceback.print_exc()
        #     print(str(e))
        #     print(
        #         f"Can not connect into {self.__host} with AuthClientPassword")
        #     return False
        except:
            pass

    #

    def __connect_ANONYMOUS(
        self,
        hybrid_search: bool = False,
    ) -> bool:
        if hybrid_search:
            # additional_headers = {
            #     ADDITIONAL_HEADERS[name]: self.__api_keys[name] for name in self.__api_keys
            # }
            additional_headers = {}
            # try to connect with api keys
            if additional_headers:
                try:
                    self.__client = weaviate.Client(
                        url=self.__host,
                        auth_client_secret=weaviate.AuthClientPassword(
                            username=self.__user,
                            # optional, depends on the configuration of your identity provider (not required with WCS)
                            scope="offline_access",
                        ),
                        additional_headers=additional_headers,
                    )
                    self.__connected = True
                    self.__connect_type = "ANONYMOUS"
                    self.__hybrid_search = True
                    return True
                except Exception as e:
                    print(f"ANONYMOUS cannot connect with additional_headers: {str(e)}")
            else:
                try:
                    self.__client = weaviate.connect_to_custom(
                        http_host=os.environ.get(
                            WEAVIATE_HOST
                        ),  # "10.100.224.34",  # URL only, no http prefix
                        http_port=os.environ.get(WEAVIATE_HOST_PORT),  # "8080",
                        http_secure=False,  # Set to True if https
                        grpc_host=os.environ.get(WEAVIATE_GPC_URL),  #  "10.100.224.34",
                        grpc_port=os.environ.get(
                            WEAVIATE_GPC_URL_PORT
                        ),  # "50051",      # Default is 50051, WCD uses 443
                        grpc_secure=False,  # Edit as needed
                        skip_init_checks=True,
                    )
                    self.__connected = True
                    self.__connect_type = "ANONYMOUS"
                    self.__hybrid_search = True
                    return True
                except Exception as e:
                    print(
                        f"ANONYMOUS cannot connect without additional_headers: {str(e)}"
                    )

        # if connect fail, try again without additional_headers

    def check_connection(
        self,
    ) -> bool:
        try:
            return self.__client.is_ready()
        except Exception as e:
            print("Connection failed:", str(e))
            return False

    #

    def is_connected(self) -> bool:
        return self.__connected

    #

    def is_hybrid_search(self) -> bool:
        return self.__hybrid_search

    #

    def get_client(self):
        return self.__client

    #
    def get_llamaindex_vectorstore(
        self,
        index_name: Optional[str] = None,
        text_key: Optional[str] = None,
        embedding: Optional[Any] = None,
        as_retriever: bool = True,
        tenant_name: list[str] = None,
        k: int = 4,
    ) -> Optional[WeaviateIndex]:
        pass

    def get_langchain_vectorstore(
        self,
        index_name: Optional[str] = None,
        text_key: Optional[str] = None,
        # embedding : Optional[Any] = OpenAIEmbeddings(),
        embedding: Optional[Any] = None,
        as_retriever: bool = True,
        tentant_name: list[str] = None,
        k: int = 4,
        multi_tenancy: bool = True,
    ) -> Optional[WeaviateLC]:

        try:
            vectorstore = WeaviateLC(
                client=self.__client,
                index_name=index_name or self.__rag_index_name,
                text_key=text_key or self.__rag_text_key,
                embedding=embedding,
                use_multi_tenancy=multi_tenancy,
            )
            if multi_tenancy:
                # return vectorstore.as_retriever(search_kwargs={'tenant': ['tenant1', 'tenant2']})
                return vectorstore.as_retriever(
                    search_kwargs={"tenant": tentant_name, "k": k, "return_uuids": True}
                )
            else:
                return vectorstore.as_retriever(
                    search_kwargs={"k": k, "return_uuids": True}
                )

        except Exception as e:
            print(str(e))
            return None

    def load_document_to_vectordb(
        self,
        documents,
        index_name: Optional[str] = None,
        text_key: Optional[str] = None,
        embedding: Optional[Any] = None,
        tenant_name: list[str] = None,
    ):

        if index_name is not None:
            db = WeaviateLC.from_documents(
                documents,
                embedding,
                client=self.__client,
                tenant=tenant_name,
                index_name=index_name,
            )
        else:
            db = WeaviateLC.from_documents(
                documents, embedding, client=self.__client, tenant=tenant_name
            )
        return db

    def load_document_to_vectordb_llamaindex(
        self,
        documents,
        index_name,
        text_key,
        embedding,
        tenant_name,
        debug=False,
        **kwargs,
    ):
        if "vector_type" in kwargs:
            vector_type = kwargs["vector_type"]
        if vector_type == "qdrant":
            import qdrant_client

            # Create a local Qdrant vector store
            client = qdrant_client.QdrantClient(
                url=os.environ["QDRANT_HOST"],
            )
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=index_name,
            )
        else:
            vector_store = WeaviateIndex(
                weaviate_client=self.__client, index_name=index_name, text_key=text_key
            )

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVER_OVERLAP
                ),
                AttatchTenant(),
                embedding,
            ],
            # docstore=docstore,
            vector_store=vector_store,
            # UPSERTS: This strategy is used to handle both the insertion of new documents and the updating of existing ones. When a document is added to the document store, the UPSERTS strategy checks if the document already exists based on its ID. If the document does not exist, it is inserted. If it does exist and the hash of the document has changed, the document is updated in the document store. This strategy ensures that the document store always contains the most recent version of each document.
            # DUPLICATES_ONLY: This strategy focuses solely on handling duplicates. It checks if a document with the same hash already exists in the document store. If a duplicate is found, the document is not added again. This strategy is useful when you want to avoid storing multiple copies of the same document without updating existing ones.
            # docstore_strategy=DocstoreStrategy.UPSERTS,
        )
        # nodes = []
        node = pipeline.run(
            documents=documents,
            show_progress=True,
            num_workers=-1,
            tenant_name=tenant_name,
        )
        # with contextlib.closing(self.__client):
        #     nodes = pipeline.run(
        #         documents=documents, show_progress=True, num_workers=-1
        #     )
        if debug:
            print("Nodes:", node)

        ## Enable Multi-Tenancy
        if client is not None and client.collection_exists(index_name):
            client.create_payload_index(
                collection_name=index_name,
                field_name="metadata.tenant_name",
                field_type=models.PayloadSchemaType.KEYWORD,
            )

        return index_name, tenant_name, text_key, documents
