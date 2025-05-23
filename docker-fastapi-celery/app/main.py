import json
from typing import List
import os
from pydantic import BaseModel
from fastapi import FastAPI
from llama_index.core import Document
from worker import celery_app
from llama_index.core.schema import (
    ImageNode,
    TextNode,
)
from llama_index.core import Document

from models.models import IngestionRequest

# create fast api application
app = FastAPI()


# item model
class Item(BaseModel):
    name: str


@app.post("/task_hello_world/")
async def create_item(item: Item):
    # celery task name
    task_name = "hello.task"
    # send task to celery
    task = celery_app.send_task(task_name, args=[item.name])

    # return task id and url
    return dict(
        id=task.id,
        url=f"localhost:5000/check_task/{task.id}",
    )


@app.post("/etl_task/")
async def ingestion_pipeline(input: IngestionRequest):
    # celery task name
    task_name = "rag.etl"
    # send task to celery
    task = celery_app.send_task(task_name, args=[input.model_dump()])

    # return task id and url
    return dict(
        id=task.id,
        url=f"http://localhost:5000/check_task/{task.id}",
    )


@app.get("/check_task/{id}")
def check_task(id: str):
    # get celery task from id
    task = celery_app.AsyncResult(id)

    # if task is in success state
    if task.state == "SUCCESS":
        response = {
            "status": task.state,
            "result": task.result,
            "task_id": id,
        }

    # if task is in failure state
    elif task.state == "FAILURE":
        response = json.loads(
            task.backend.get(
                task.backend.get_key_for_task(task.id),
            ).decode("utf-8")
        )
        del response["children"]
        del response["traceback"]

    # if task is in other state
    else:
        response = {
            "status": task.state,
            "result": task.info,
            "task_id": id,
        }

    # return response
    return response
