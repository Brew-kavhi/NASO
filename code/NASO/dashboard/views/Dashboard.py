from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView

from naso.celery import (get_celery_task_state, get_registered_tasks,
                         get_tasks, kill_celery_task)
from naso.models.page import PageSetup
from neural_architecture.models.AutoKeras import AutoKerasRun
from runs.models.Training import NetworkTraining


class Dashboard(TemplateView):
    template_name = "dashboard/dashboard.html"

    page = PageSetup(title="Dashboard")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        self.context["celery"] = get_tasks()
        self.context["registered_tasks"] = get_registered_tasks()
        for task in self.context["registered_tasks"]:
            if task["is_autokeras"]:
                run = AutoKerasRun.objects.get(id=task["run_id"])
                task["name"] = run.model.project_name
                task["link"] = reverse_lazy(
                    "runs:autokeras_details", kwargs={"pk": run.id}
                )
            else:
                run = NetworkTraining.objects.get(id=task["run_id"])
                task["name"] = run.network_config.name
                task["link"] = reverse_lazy("runs:details", kwargs={"pk": run.id})

        if "training_task_id" in self.context["celery"]:
            run_details = get_celery_task_state(
                self.context["celery"]["training_task_id"]
            )["details"]
            print(run_details)
            is_autokeras = "autokeras" in run_details
            if "run_id" in run_details:
                if is_autokeras:
                    run = AutoKerasRun.objects.get(id=run_details["run_id"]).model
                    self.context["run"] = {
                        "name": run.project_name,
                        "id": run.id,
                        "link": reverse_lazy(
                            "runs:autokeras_details", kwargs={"pk": run.id}
                        ),
                    }
                else:
                    run = NetworkTraining.objects.get(
                        id=run_details["run_id"]
                    ).network_config
                    self.context["run"] = {
                        "name": run.name,
                        "id": run.id,
                        "link": reverse_lazy("runs:details", kwargs={"pk": run.id}),
                    }

        return self.render_to_response(self.context)


def kill_task(request, task_id):
    kill_celery_task(task_id)
    return redirect("dashboard:index")
