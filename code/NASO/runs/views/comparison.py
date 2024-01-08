from django.views.generic.base import TemplateView
from django.urls import reverse_lazy
from api.views.autokeras import get_trial_details

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
        for comparison_id in comparison_runs:
            run_id = comparison_id
            if ":" in run_id:
                run_id = run_id.split(":")[1]
            model = {}
            if comparison_runs[comparison_id] == "tensorflow":
                run = NetworkTraining.objects.filter(pk=run_id).first()
                if not run:
                    continue
                model = {
                    "id": run_id,
                    "name": run.network_config.name,
                    "size": run.network_config.size,
                    "memory_usage": run.memory_usage,
                    "energy_consumption": run.get_average_energy_consumption,
                    "link": reverse_lazy("runs:details", kwargs={"pk": run_id}),
                    "pruning_method": run.network_config.pruning_method,
                    "pruning_schedule": run.network_config.pruning_schedule,
                    "pruning_policy": run.network_config.pruning_policy,
                    "optimizer": run.hyper_parameters.optimizer,
                }
                model["optimizer"].additional_arguments = get_arguments_as_dict(
                    run.hyper_parameters.optimizer.additional_arguments
                )
                if run.network_config.pruning_method:
                    model[
                        "pruning_method"
                    ].additional_arguments = get_arguments_as_dict(
                        run.network_config.pruning_method.additional_arguments
                    )
                if run.network_config.pruning_schedule:
                    model[
                        "pruning_schedule"
                    ].additional_arguments = get_arguments_as_dict(
                        run.network_config.pruning_schedule.additional_arguments
                    )
                if run.network_config.pruning_policy:
                    model[
                        "pruning_policy"
                    ].additional_arguments = get_arguments_as_dict(
                        run.network_config.pruning_policy.additional_arguments
                    )
            elif comparison_runs[comparison_id] == "autokeras":
                run = AutoKerasRun.objects.filter(pk=run_id).first()
                if not run:
                    continue
                model = {
                    "id": run_id,
                    "name": run.model.project_name,
                    "size": "-",
                    "memory_usage": run.memory_usage,
                    "energy_consumption": run.get_average_energy_consumption,
                    "link": reverse_lazy(
                        "runs:autokeras:details", kwargs={"pk": run_id}
                    ),
                    "pruning_method": run.model.pruning_method,
                    "pruning_schedule": run.model.pruning_schedule,
                    "pruning_policy": run.model.pruning_policy,
                }
                if run.model.pruning_method:
                    model[
                        "pruning_method"
                    ].additional_arguments = get_arguments_as_dict(
                        run.model.pruning_method.additional_arguments
                    )
                if run.model.pruning_schedule:
                    model[
                        "pruning_schedule"
                    ].additional_arguments = get_arguments_as_dict(
                        run.model.pruning_schedule.additional_arguments
                    )
                if run.model.pruning_policy:
                    model[
                        "pruning_policy"
                    ].additional_arguments = get_arguments_as_dict(
                        run.model.pruning_policy.additional_arguments
                    )
            elif comparison_runs[comparison_id] == "autokeras_trial":
                [autokeras_id, trial_id] = comparison_id.split("_")
                run = AutoKerasRun.objects.filter(pk=autokeras_id).first()
                if not run:
                    continue
                model["name"] = "Trial " + trial_id + " von " + run.model.project_name
                trial_metrics = run.get_trial_metric(trial_id)
                model["size"] = trial_metrics["model_size"]
                model["energy_consumption"] = trial_metrics["average_energy"]
                model["id"] = comparison_id
                model["link"] = reverse_lazy(
                    "runs:autokeras:trial",
                    kwargs={"run_id": run.id, "trial_id": trial_id},
                )
                model["pruning_method"] = run.model.pruning_method
                model["pruning_schedule"] = run.model.pruning_schedule
                model["pruning_policy"] = run.model.pruning_policy
                if run.model.pruning_method:
                    model[
                        "pruning_method"
                    ].additional_arguments = get_arguments_as_dict(
                        run.model.pruning_method.additional_arguments
                    )
                if run.model.pruning_schedule:
                    model[
                        "pruning_schedule"
                    ].additional_arguments = get_arguments_as_dict(
                        run.model.pruning_schedule.additional_arguments
                    )
                if run.model.pruning_policy:
                    model[
                        "pruning_policy"
                    ].additional_arguments = get_arguments_as_dict(
                        run.model.pruning_policy.additional_arguments
                    )
                model["hyperparameters"] = get_trial_details(run, trial_id)
            model["rating"] = run.rate
            model["device"] = run.compute_device
            model["description"] = run.description
            model["size_on_disk"] = run.size_on_disk
            model["run_type"] = comparison_runs[comparison_id]
            model["comparison_id"] = comparison_id
            runs.append(model)
        self.context["runs"] = runs
        return self.render_to_response(self.context)


def get_arguments_as_dict(arguments):
    argument_dict = {}
    for argument in arguments:
        argument_dict[argument["name"]] = argument["value"]
    return argument_dict
