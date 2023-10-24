import os
import time

from celery.result import AsyncResult
from decouple import config

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    config("DJANGO_SETTINGS_MODULE", default="naso.settings_production"),
)

app = Celery("naso", backend=config("CELERY_BROKER_URL"))

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


def get_celery_task_state(task_id):
    result = AsyncResult(task_id)
    context = {"state": ""}
    retries = 0
    for _ in range(10):
        try:
            context = {
                "state": result.state,
                "details": result.info,
            }
            if result.info:
                if "epochs" in result.info:
                    context["epochs"] = result.info["epochs"]
                if "accuracy" in result.info:
                    context["accuracy"] = result.info["accuracy"]

            return context
        except AttributeError:
            inspector = app.control.inspect()
            active_tasks = inspector.active()
            if not active_tasks:
                # if there is no active task running we can immmediately return with a closed state
                return {"state": "CLOSED"}

            print("Retrying due to DisabledBackend")
            time.sleep(1)  # Exponential backoff
        retries += 1
    return context


def get_tasks():
    inspector = app.control.inspect()
    active_tasks = inspector.active()
    if active_tasks:
        for task_collection in active_tasks.values():
            training_tasks = sorted(
                list(task_collection), key=lambda d: d["time_start"]
            )

            if training_tasks:
                # sort by time_start
                if training_tasks[-1]["id"]:
                    return {"training_task_id": training_tasks[-1]["id"]}
    return {}


if __name__ == "__main__":
    app.start()
