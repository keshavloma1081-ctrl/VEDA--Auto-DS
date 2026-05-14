from celery import Task
from celery.utils.log import get_task_logger
from datetime import datetime
import traceback
import os

from veda.tasks.celery_app import celery_app

logger = get_task_logger(__name__)


class VEDABaseTask(Task):
    abstract = True
    _db = None

    @property
    def db(self):
        if self._db is None:
            from veda.database.models import SessionLocal
            self._db = SessionLocal()
        return self._db

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        if self._db is not None:
            self._db.close()
            self._db = None


@celery_app.task(
    bind=True,
    base=VEDABaseTask,
    name="veda.tasks.workflow_tasks.run_workflow_task",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    track_started=True,
    queue="default"
)
def run_workflow_task(self, job_id, dataset_path, goal, config=None):
    from veda.database.models import Workflow
    from veda.agents.core_pipeline.planner import PlannerAgent
    from veda.core.state import VEDAState

    config = config or {}
    workflow = None

    try:
        workflow = self.db.query(Workflow).filter(
            Workflow.job_id == job_id
        ).first()

        if not workflow:
            raise ValueError(f"Workflow {job_id} not found")

        workflow.status = "running"
        workflow.started_at = datetime.utcnow()
        workflow.current_step = "Initializing"
        self.db.commit()

        self.update_state(
            state="PROGRESS",
            meta={"job_id": job_id, "step": "Initializing", "progress": 0}
        )

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY not configured")

        state = VEDAState(
            dataset_path=dataset_path,
            user_goal=goal,
            job_id=job_id
        )

        planner = PlannerAgent()
        result = planner.execute(state)

        workflow.status = "completed"
        workflow.progress = 100.0
        workflow.result = result
        workflow.completed_at = datetime.utcnow()
        workflow.updated_at = datetime.utcnow()
        self.db.commit()

        logger.info(f"Workflow {job_id} completed")
        return {"job_id": job_id, "status": "completed", "result": result}

    except (FileNotFoundError, EnvironmentError) as exc:
        if workflow:
            workflow.status = "failed"
            workflow.error = str(exc)
            self.db.commit()
        raise

    except Exception as exc:
        retry_in = 60 * (2 ** self.request.retries)

        if self.request.retries < self.max_retries:
            if workflow:
                workflow.current_step = f"Retry {self.request.retries + 1}/{self.max_retries}"
                self.db.commit()
            raise self.retry(exc=exc, countdown=retry_in)
        else:
            if workflow:
                workflow.status = "failed"
                workflow.error = f"Failed after {self.max_retries + 1} attempts: {str(exc)}"
                workflow.error_traceback = traceback.format_exc()
                self.db.commit()
            raise


@celery_app.task(
    bind=True,
    base=VEDABaseTask,
    name="veda.tasks.workflow_tasks.run_workflow_priority",
    max_retries=5,
    default_retry_delay=30,
    acks_late=True,
    queue="high"
)
def run_workflow_priority(self, job_id, dataset_path, goal, config=None):
    return run_workflow_task(self, job_id, dataset_path, goal, config)


@celery_app.task(
    name="veda.tasks.workflow_tasks.cancel_workflow_task",
    queue="high"
)
def cancel_workflow_task(job_id):
    from veda.database.models import SessionLocal, Workflow
    db = SessionLocal()
    try:
        workflow = db.query(Workflow).filter(Workflow.job_id == job_id).first()
        if workflow and workflow.status in ["submitted", "running"]:
            workflow.status = "cancelled"
            workflow.updated_at = datetime.utcnow()
            db.commit()
        return {"status": "cancelled", "job_id": job_id}
    finally:
        db.close()


@celery_app.task(
    name="veda.tasks.workflow_tasks.cleanup_old_workflows",
    queue="low"
)
def cleanup_old_workflows(days_to_keep=30):
    from veda.database.models import SessionLocal, Workflow
    from datetime import timedelta
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days_to_keep)
        deleted = db.query(Workflow).filter(
            Workflow.status.in_(["completed", "failed", "cancelled"]),
            Workflow.created_at < cutoff
        ).delete()
        db.commit()
        return {"deleted_workflows": deleted}
    finally:
        db.close()