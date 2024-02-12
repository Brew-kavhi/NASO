import os
import time

from celery.result import AsyncResult
from decouple import config

from celery import Celery

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    config("DJANGO_SETTINGS_MODULE", default="naso.settings"),
)

app = Celery("naso", backend=config("CELERY_BROKER_URL"))

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")
app.conf.update(worker_pool_restarts=True, worker_prefetch_multiplier=1)
app.conf.update(
    task_routes={"runs.views.trial.memory_safe_model_load": {"queue": "start_trials"}}
)

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


def get_celery_task_state(task_id):
    """
    Returns the state of a Celery task with the given task_id.

    Args:
        task_id (str): The ID of the Celery task.

    Returns:
        dict: A dictionary containing the state of the task and any additional details.
              If the task is not found, the state will be "CLOSED".
              If the task is still running, the state will be "PENDING".
              If the task has completed successfully, the state will be "SUCCESS".
              If the task has failed, the state will be "FAILURE".
              Additional details about the task (such as the number of epochs or accuracy)
              may also be included in the dictionary.
    """
    result = AsyncResult(task_id)
    context = {"state": ""}
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
    return context


def kill_celery_task(task_id):
    """
    Kills a Celery task with the given task_id.

    Args:
        task_id (str): The ID of the Celery task.
    """
    app.control.revoke(task_id, terminate=True)


def restart_all_workers():
    response = app.control.broadcast(
        "pool_restart", arguments={"reload": True}, reply=True
    )
    if response:
        print(response)


def get_tasks():
    """
    Returns a list of objects that map all the active Celery task ids.
    """
    inspector = app.control.inspect()
    running_tasks = []
    try:
        active_tasks = inspector.active()
        if active_tasks:
            for task_collection in active_tasks.values():
                training_tasks = sorted(
                    list(task_collection), key=lambda d: d["time_start"]
                )
                for training_task in training_tasks:
                    if training_task and training_task["id"]:
                        running_tasks.append({"training_task_id": training_task["id"]})
        return running_tasks
    except BrokenPipeError:
        return []
    finally:
        return running_tasks


def get_registered_tasks():
    """
    Returns a list of objects that map all the registered Celery task ids.
    """
    try:
        inspector = app.control.inspect()
        registered_tasks = inspector.reserved()
        tasks = []
        if registered_tasks:
            for task_collection in registered_tasks.values():
                # these are the tasks for one worker
                for task in task_collection:
                    is_autokeras = "autokeras" in task["type"]
                    run_id = task["args"][0]
                    tasks.append(
                        {
                            "id": task["id"],
                            "run_id": run_id,
                            "is_autokeras": is_autokeras,
                        }
                    )
        return tasks
    except:
        return []


if __name__ == "__main__":
    app.start()
