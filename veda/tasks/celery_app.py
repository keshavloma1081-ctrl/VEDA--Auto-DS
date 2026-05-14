from celery import Celery
from kombu import Queue, Exchange
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_BACKEND = os.getenv("REDIS_BACKEND_URL", "redis://localhost:6379/1")

celery_app = Celery("veda", broker=REDIS_URL, backend=REDIS_BACKEND)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    result_expires=86400,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_max_retries=3,
    task_soft_time_limit=3300,
    task_time_limit=3600,
    task_queues=(
        Queue("high", Exchange("high"), routing_key="high"),
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("low", Exchange("low"), routing_key="low"),
        Queue("dead_letter", Exchange("dead_letter"), routing_key="dead_letter"),
    ),
    task_default_queue="default",
    task_routes={
        "veda.tasks.workflow_tasks.run_workflow_task": {"queue": "default"},
        "veda.tasks.workflow_tasks.run_workflow_priority": {"queue": "high"},
    },
    worker_max_tasks_per_child=100,
    worker_send_task_events=True,
    task_send_sent_event=True,
)