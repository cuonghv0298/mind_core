import os
from celery import Celery


class CeleryConfig:
    task_serializer = "json"
    result_serializer = "json"
    event_serializer = "json"
    accept_content = [
        "pickle",
        "json",
        "msgpack",
        "yaml",
        "application/x-python-serialize",
    ]
    result_accept_content = ["application/json", "pickle"]
    worker_send_task_events = True


# celery broker and backend urls
CELERY_BROKER_URL = os.getenv("REDISSERVER", "redis://redis_server:6379")
CELERY_RESULT_BACKEND = os.getenv("REDISSERVER", "redis://redis_server:6379")

# create celery application
celery_app = Celery(
    "celery",
    backend=CELERY_BROKER_URL,
    broker=CELERY_RESULT_BACKEND,
)

celery_app.config_from_object(CeleryConfig)
