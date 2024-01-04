from django.views.generic.base import TemplateView

from naso.models.page import PageSetup
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import NetworkTraining


class ComparisonView(TemplateView):
    template_name = "runs/comparison.html"
    page = PageSetup(
        title="Vergleich", description="Metriken zwischen Laufen vergleichen"
    )
    context = {"page": page.get_context()}

    def get(self, request):
        comparison_runs = request.session["comparison"]
        runs = []
        for run_id in comparison_runs:
            model = {}
            if comparison_runs[run_id] == "tensorflow":
                run = NetworkTraining.objects.filter(pk=run_id).first()
                if not run:
                    continue
                model = {
                    "id": run_id,
                    "name": run.network_config.name,
                    "size": run.network_config.size,
                    "memory_usage": run.memory_usage,
                    "energy_consumption": run.get_average_energy_consumption,
                }
            elif comparison_runs[run_id] == "autokeras":
                run = AutoKerasRun.objects.filter(pk=run_id).first()
                if not run:
                    continue
                model = {
                    "id": run_id,
                    "name": run.model.project_name,
                    "size": "-",
                    "memory_usage": run.memory_usage,
                    "energy_consumption": run.get_average_energy_consumption,
                }
            model["rating"] = run.rate
            model["device"] = run.gpu
            model["description"] = run.description
            model["run_type"] = comparison_runs[run_id]
            model["comparison_id"] = run_id
            runs.append(model)
        self.context["runs"] = runs
        return self.render_to_response(self.context)
