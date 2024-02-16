from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView

from naso.celery import (
    get_celery_task_state,
    get_registered_tasks,
    get_tasks,
    kill_celery_task,
)
from naso.models.page import PageSetup
from neural_architecture.models.autokeras import AutoKerasRun
from neural_architecture.models.model_runs import KerasModelRun
from runs.models.training import NetworkTraining
from inference.models.inference import Inference


class Dashboard(TemplateView):
    """
    A view that renders the dashboard page.

    Attributes:
        template_name (str): The name of the template to be rendered.
        page (PageSetup): An instance of the PageSetup class that sets up the page.
        context (dict): A dictionary containing the context data to be passed to the template.
    """

    template_name = "dashboard/dashboard.html"

    page = PageSetup(title="Dashboard")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        """
        Handles GET requests and returns the rendered template with the context data.

        Args:
            request (HttpRequest): The HTTP request object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            HttpResponse: The HTTP response object containing the rendered template with the context data.
        """

        self.context["celery"] = get_tasks()
        self.context["registered_tasks"] = get_registered_tasks()
        for task in self.context["registered_tasks"]:
            if task["is_autokeras"]:
                run = AutoKerasRun.objects.get(id=task["run_id"])
                task["name"] = run.model.project_name
                task["link"] = reverse_lazy(
                    "runs:autokeras:details", kwargs={"pk": run.id}
                )
            elif "is_autokeras_trial" in task and task["is_autokeras_trial"]:
                run = KerasModelRun.objects.get(id=task["run_id"])
                task["name"] = run.model.name
                task["link"] = reverse_lazy("runs:list")
            else:
                run = NetworkTraining.objects.get(id=task["run_id"])
                task["name"] = run.network_config.name
                task["link"] = reverse_lazy("runs:details", kwargs={"pk": run.id})

        self.context["run"] = []
        for running_tasks in self.context["celery"]:
            if "training_task_id" in running_tasks:
                run_details = get_celery_task_state(running_tasks["training_task_id"])[
                    "details"
                ]
                if run_details:
                    is_autokeras = "autokeras" in run_details
                    if "run_id" in run_details:
                        if is_autokeras and run_details["autokeras"]:
                            run = AutoKerasRun.objects.get(
                                id=run_details["run_id"]
                            ).model
                            self.context["run"].append(
                                {
                                    "name": run.project_name,
                                    "id": run.id,
                                    "link": reverse_lazy(
                                        "runs:autokeras:details",
                                        kwargs={"pk": run_details["run_id"]},
                                    ),
                                    "task_id": running_tasks["training_task_id"],
                                }
                            )
                        elif (
                            "autokeras_trial" in run_details
                            and run_details["autokeras_trial"]
                        ):
                            run = KerasModelRun.objects.get(id=run_details["run_id"])
                            self.context["run"].append(
                                {
                                    "name": run.model.name,
                                    "id": run.id,
                                    "link": reverse_lazy("runs:list"),
                                    "task_id": running_tasks["training_task_id"],
                                }
                            )
                        elif "inference" in run_details and run_details["inference"]:
                            run = Inference.objects.get(id=run_details["run_id"])
                            self.context["run"].append(
                                {
                                    "name": run.name,
                                    "id": run.id,
                                    "link": reverse_lazy(
                                        "inference:details",
                                        kwargs={"pk": run_details["run_id"]},
                                    ),
                                    "task_id": running_tasks["training_task_id"],
                                }
                            )
                        else:
                            run = NetworkTraining.objects.get(
                                id=run_details["run_id"]
                            ).network_config
                            self.context["run"].append(
                                {
                                    "name": run.name,
                                    "id": run.id,
                                    "link": reverse_lazy(
                                        "runs:details",
                                        kwargs={"pk": run_details["run_id"]},
                                    ),
                                    "task_id": running_tasks["training_task_id"],
                                }
                            )
        return self.render_to_response(self.context)


def kill_task(request, task_id):
    """
    A view that kills a task.

    Args:
        request (HttpRequest): The HTTP request object.
        task_id (str): The task id.
    """
    kill_celery_task(task_id)
    return redirect("dashboard:index")
