from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from django.urls import reverse_lazy
from neural_architecture.models.autokeras import AutoKerasRun, KerasModelRun
from runs.models.training import NetworkTraining

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
        return Response({"success": True})
    else:
        return Response({"success": False})
    return Response({"success": True})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def is_run_executing(request):
    run_type = request.data.get("run_type")
    run_id = request.data.get("run_id")
    i = app.control.inspect()
    active_tasks = i.active()

    if not active_tasks:
        return []
    for _, tasks in active_tasks.items():
        for task in tasks:
            print(task)
            if task["args"][0] == run_id:
                # could be a match:
                task_details = get_celery_task_state(task["id"])["details"]
                if (
                    run_type == "autokeras"
                    and "autokeras" in task_details
                    and task_details["autokeras"]
                ):
                    return Response({"success": True, "task_id": task["id"]})
                elif (
                    run_type == "autokeras_trial"
                    and "autokeras_trial" in task_details
                    and task_details["autokeras_trial"]
                ):
                    return Response({"success": True, "task_id": task["id"]})
                else:
                    return Response({"success": True, "task_id": task["id"]})

    return Response({"success": False})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_task_details(request, task_id):
    details = get_celery_task_state(task_id)["details"]
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
            print(task_details)
            if "run_id" in task_details:
                if "autokeras" in task_details and task_details["autokeras"]:
                    run = AutoKerasRun.objects.get(id=task_details["run_id"])
                    task["name"] = run.model.project_name
                    task["link"] = reverse_lazy(
                        "runs:autokeras:details", kwargs={"pk": run.id}
                    )
                elif (
                    "autokeras_trial" in task_details
                    and task_details["autokeras_trial"]
                ):
                    run = KerasModelRun.objects.get(id=task_details["run_id"])
                    task["name"] = run.model.name
                    task["link"] = reverse_lazy("runs:list")
                else:
                    run = NetworkTraining.objects.get(id=task_details["run_id"])
                    task["name"] = run.network_config.name
                    task["link"] = reverse_lazy("runs:details", kwargs={"pk": run.id})
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
