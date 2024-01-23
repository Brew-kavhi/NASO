from django.urls import reverse_lazy
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from naso.celery import app, get_celery_task_state
from neural_architecture.models.autokeras import AutoKerasRun, KerasModelRun
from runs.models.training import NetworkTraining


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_workers(request):
    """
    This view returns the workers.

    Args:
        request (Request): The request object.

    Returns:
        Response: The response object with parameter 'WORKERS'
    """
    return Response({"WORKERS": get_workers_information()})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def restart_worker(request, worker):
    """
    This view restarts a worker.

    Args:
        request (Request): The request object.
        worker (str): The name of the worker.

    Returns:
        Response: The response object with parameter 'success'
    """
    response = app.control.broadcast(
        "pool_restart", arguments={"reload": True}, destination=[worker], reply=True
    )
    if response and "ok" in response[0][worker]:
        return Response({"success": True})
    return Response({"success": False})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def is_run_executing(request):
    """
    This view checks if a run is executing.

    Args:
        request (Request): The request object. The body must contain parameters run_type and run_id.

    Returns:
        Response: The response object with parameter 'success' and 'task_id'
    """
    run_id = request.data.get("run_id")
    i = app.control.inspect()
    active_tasks = i.active()

    if not active_tasks:
        return Response({"success": False})
    for _, tasks in active_tasks.items():
        for task in tasks:
            print(task)
            if task["args"][0] == run_id:
                # could be a match:
                return Response({"success": True, "task_id": task["id"]})

    return Response({"success": False})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_task_details(request, task_id):
    """
    This view returns the details of a task.

    Args:
        request (Request): The request object.
        task_id (str): The id of the task.

    Returns:
        Response: The response object with array of details
    """
    try:
        details = get_celery_task_state(task_id)["details"]
        return Response(details)
    except Exception as e:
        return Response({"sucess": False})


def get_workers_information():
    """
    This function returns the workers information.

    Returns:
        list: The workers information, which is array of objects with the following structure:
            {
                "name": str,
                "concurrency": int,
                "tasks": [
                    {
                        "id": str,
                        "name": str,
                        "link": str,
                        "device": str,
                        "power": str
                    }
                ]
            }
    """
    try:
        i = app.control.inspect()
        if not i:
            return []
        stats = i.stats()
        active_tasks = i.active()

        # Process the information to fit your template structure
        workers_info = []
        if not active_tasks or not stats:
            return []
        for worker, tasks in active_tasks.items():
            if not stats[worker]:
                continue
            worker_info = {
                "name": worker,
                "concurrency": stats[worker]["pool"]["max-concurrency"],
                "tasks": [],
            }
            for task in tasks:
                task_details = get_celery_task_state(task["id"])["details"]
                if not task_details:
                    continue
                if "run_id" in task_details:
                    if "autokeras" in task_details and task_details["autokeras"]:
                        run = AutoKerasRun.objects.filter(id=task_details["run_id"])
                        if run.exists():
                            run = run.first()
                            task["name"] = run.model.project_name
                            task["link"] = reverse_lazy(
                                "runs:autokeras:details", kwargs={"pk": run.id}
                            )
                        else:
                            task["name"] = "Undefined"
                            task["link"] = ""
                    elif (
                        "autokeras_trial" in task_details
                        and task_details["autokeras_trial"]
                    ):
                        run = KerasModelRun.objects.filter(id=task_details["run_id"])
                        if run.exists():
                            run = run.first()
                            task["name"] = run.model.name
                            task["link"] = reverse_lazy("runs:list")
                        else:
                            task["link"] = ""
                            task["name"] = "Undefined"
                    else:
                        run = NetworkTraining.objects.filter(id=task_details["run_id"])
                        if run.exists():
                            run = run.first()
                            task["name"] = run.network_config.name
                            task["link"] = reverse_lazy(
                                "runs:details", kwargs={"pk": run.id}
                            )
                        else:
                            task["link"] = ""
                            task["name"] = "Undefined"
                else:
                    task["link"] = ""
                if "gpu" in task_details:
                    worker_info["tasks"].append(
                        {
                            "id": task["id"],
                            "name": task["name"],
                            "link": task["link"],
                            "device": task_details["gpu"]["device"],
                            "power": task_details["gpu"]["power"],
                        }
                    )
                else:
                    worker_info["tasks"].append(
                        {"name": task["name"], "device": "", "power": ""}
                    )
            workers_info.append(worker_info)
        return workers_info
    except BrokenPipeError:
        return []
