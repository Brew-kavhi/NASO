from django.views.generic.base import TemplateView
import safedelete
from django.urls import reverse_lazy

from naso.models.page import PageSetup
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import NetworkTraining


class DeletedRuns(TemplateView):
    template_name = "runs/deleted_runs.html"
    page = PageSetup(
        title="Gelöschte Element", description="Endgültig löschen oder Wiederherstellen"
    )
    context = {"page": page.get_context()}

    def get(self, request):
        training_runs = reversed(
            NetworkTraining.objects.deleted_only().only(
                "id",
                "network_config__name",
            )
        )
        autokeras_runs = reversed(
            AutoKerasRun.objects.deleted_only().only(
                "id",
                "model__project_name",
                "metrics",
            )
        )
        self.page.actions = []
        self.page.add_pageaction(reverse_lazy("runs:autokeras:new"), "Neuer Autokeras")
        self.page.add_pageaction(reverse_lazy("runs:new"), "Neuer Tensorflow")
        self.context["page"] = self.page.get_context()
        self.context["network_training_data"] = training_runs
        self.context["autokeras_runs"] = autokeras_runs
        self.context["network_training_ids"] = [
            run.id for run in NetworkTraining.objects.all().only("id")
        ]
        self.context["autokeras_run_ids"] = [
            keras_run.id for keras_run in AutoKerasRun.objects.all().only("id")
        ]
        return self.render_to_response(self.context)


def undelete_run(id: int, run_type: str):
    if run_type == "tensorflow":
        run = NetworkTraining.objects.deleted_only().get(pk=id)
        run.undelete()
        return True
    elif run_type == "autokeras":
        run = AutoKerasRun.objects.deleted_only().get(pk=id)
        run.undelete()
        return True
    return False


def harddelete_run(id: int, run_type: str):
    if run_type == "tensorflow":
        run = NetworkTraining.objects.deleted_only().get(pk=id)
        run.delete(force_policy=safedelete.models.HARD_DELETE)
        return True
    elif run_type == "autokeras":
        run = AutoKerasRun.objects.deleted_only().get(pk=id)
        run.delete(force_policy=safedelete.models.HARD_DELETE)
        return True
    return False
