import imghdr
from tkinter import image_names
from pydantic import BaseModel
from typing import List
from llama_index.core import Document


class IngestionRequest(BaseModel):
    documents: List[Document]
    collection_name: str
    tenant: str
