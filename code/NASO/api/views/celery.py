from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
import celery

from naso.celery import app, get_celery_task_state


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_workers(request):
    return Response({"WORKERS": get_workers_information()})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def restart_worker(request, worker):
    response = app.control.broadcast(
        "pool_restart", arguments={"reload": True}, destination=[worker], reply=True
    )
    if response and "ok" in response[0][worker]:
        print(response)
        return Response({"success": True})
    else:
        print(response)
        return Response({"success": False})
    return Response({"success": True})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_task_details(request, task_id):
    details = get_celery_task_state(task_id)
    return Response(details)


def get_workers_information():
    i = app.control.inspect()
    stats = app.control.inspect().stats()
    active_tasks = i.active()

    # Process the information to fit your template structure
    workers_info = []
    if not active_tasks:
        return []
    for worker, tasks in active_tasks.items():
        worker_info = {
            "name": worker,
            "concurrency": stats[worker]["pool"]["max-concurrency"],
            "tasks": [],
        }
        for task in tasks:
            task_details = get_celery_task_state(task["id"])["details"]
            if "gpu" in task_details:
                worker_info["tasks"].append(
                    {
                        "name": task["name"],
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
