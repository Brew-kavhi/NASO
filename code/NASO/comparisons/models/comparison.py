from django.db import models
from django.urls import reverse_lazy

from api.views.autokeras import get_trial_details
from inference.models.inference import Inference
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import NetworkTraining, TrainingMetric


class Comparison(models.Model):
    name = models.CharField(max_length=64)
    runs = models.JSONField(default=dict)
    description = models.TextField()

    def __str__(self):
        return self.name

    def get_details(self):
        return get_comparison_details(self.runs)


def get_inference_details(run_id):
    model, metrics = {}, {}

    run = Inference.objects.filter(pk=run_id).first()
    if not run:
        return {}, None
    if run.prediction_metrics.last():
        metrics = run.prediction_metrics.last().metrics[0]["metrics"]

    loss, val_loss, sparsity, memory_consumption, power_consumption = (
        "-",
        "-",
        "-",
        "-",
        "-",
    )
    if "loss" in metrics:
        loss = metrics["loss"]
    if "val_loss" in metrics:
        val_loss = metrics["val_loss"]
    if "sparsity" in metrics:
        sparsity = metrics["sparsity"]
    if "memory_consumption" in metrics:
        memory_consumption = metrics["memory_consumption"]
    if "power_consumption" in metrics:
        power_consumption = metrics["power_consumption"]
    model = {
        "id": run_id,
        "name": run.name,
        "memory_usage": memory_consumption,
        "power_consumption": power_consumption,
        "link": reverse_lazy("runs:details", kwargs={"pk": run_id}),
        "pruning_method": "-",
        "pruning_schedule": "-",
        "pruning_policy": "-",
        "optimizer": "-",
        "prediction_metrics": run.prediction_metrics,
        "loss": loss,
        "val_loss": val_loss,
        "sparsity": sparsity,
    }
    return model, run


def get_tensorflow_details(run_id):
    model = {}
    run = NetworkTraining.objects.filter(pk=run_id).first()
    if not run:
        return {}, None
    metrics = (
        TrainingMetric.objects.filter(
            neural_network=run, epoch__lte=run.fit_parameters.epochs
        )
        .last()
        .metrics[0]["metrics"]
    )
    loss, val_loss, sparsity = "-", "-", "-"
    if "loss" in metrics:
        loss = metrics["loss"]
    if "val_loss" in metrics:
        val_loss = metrics["val_loss"]
    if "sparsity" in metrics:
        sparsity = metrics["sparsity"]
    model = {
        "id": run_id,
        "name": run.network_config.name,
        "size": run.network_config.size,
        "memory_usage": run.memory_usage,
        "power_consumption": run.get_average_power_consumption,
        "link": reverse_lazy("runs:details", kwargs={"pk": run_id}),
        "pruning_method": run.network_config.pruning_method,
        "pruning_schedule": run.network_config.pruning_schedule,
        "pruning_policy": run.network_config.pruning_policy,
        "optimizer": run.hyper_parameters.optimizer,
        "prediction_metrics": run.prediction_metrics,
        "loss": loss,
        "val_loss": val_loss,
        "sparsity": sparsity,
    }
    model["optimizer"].additional_arguments = get_arguments_as_dict(
        run.hyper_parameters.optimizer.additional_arguments
    )
    if run.network_config.pruning_method:
        model["pruning_method"].additional_arguments = get_arguments_as_dict(
            run.network_config.pruning_method.additional_arguments
        )
    if run.network_config.pruning_schedule:
        model["pruning_schedule"].additional_arguments = get_arguments_as_dict(
            run.network_config.pruning_schedule.additional_arguments
        )
    if run.network_config.pruning_policy:
        model["pruning_policy"].additional_arguments = get_arguments_as_dict(
            run.network_config.pruning_policy.additional_arguments
        )
    return model, run


def get_autokeras_details(run_id):
    model = {}
    run = AutoKerasRun.objects.filter(pk=run_id).first()
    if not run:
        return {}, None
    model = {
        "id": run_id,
        "name": run.model.project_name,
        "size": "-",
        "memory_usage": run.memory_usage,
        "power_consumption": run.get_average_power_consumption,
        "link": reverse_lazy("runs:autokeras:details", kwargs={"pk": run_id}),
        "pruning_method": run.model.pruning_method,
        "pruning_schedule": run.model.pruning_schedule,
        "pruning_policy": run.model.pruning_policy,
        "prediction_metrics": run.prediction_metrics,
    }
    if run.model.pruning_method:
        model["pruning_method"].additional_arguments = get_arguments_as_dict(
            run.model.pruning_method.additional_arguments
        )
    if run.model.pruning_schedule:
        model["pruning_schedule"].additional_arguments = get_arguments_as_dict(
            run.model.pruning_schedule.additional_arguments
        )
    if run.model.pruning_policy:
        model["pruning_policy"].additional_arguments = get_arguments_as_dict(
            run.model.pruning_policy.additional_arguments
        )
    return model, run


def get_autokerastrial_details(autokeras_id, trial_id):
    model = {}
    run = AutoKerasRun.objects.filter(pk=autokeras_id).first()
    if not run:
        return {}, None
    model["name"] = "Trial " + trial_id + " von " + run.model.project_name
    trial_metrics = run.get_trial_metric(trial_id)
    model["size"] = trial_metrics["model_size"]
    model["power_consumption"] = trial_metrics["average_power"]
    model["id"] = f"{autokeras_id}_{trial_id}"
    model["link"] = reverse_lazy(
        "runs:autokeras:trial",
        kwargs={"run_id": run.id, "trial_id": trial_id},
    )
    model["pruning_method"] = run.model.pruning_method
    model["pruning_schedule"] = run.model.pruning_schedule
    model["pruning_policy"] = run.model.pruning_policy
    if run.model.pruning_method:
        model["pruning_method"].additional_arguments = get_arguments_as_dict(
            run.model.pruning_method.additional_arguments
        )
    if run.model.pruning_schedule:
        model["pruning_schedule"].additional_arguments = get_arguments_as_dict(
            run.model.pruning_schedule.additional_arguments
        )
    if run.model.pruning_policy:
        model["pruning_policy"].additional_arguments = get_arguments_as_dict(
            run.model.pruning_policy.additional_arguments
        )
    model["hyperparameters"] = get_trial_details(run, trial_id)
    return model, run


def get_comparison_details(comparisons):
    details = []
    for comparison_id in comparisons:
        run_id = comparison_id
        if ":" in run_id:
            run_id = run_id.split(":")[1]
        model = {}
        if comparisons[comparison_id] == "tensorflow":
            model, run = get_tensorflow_details(run_id)
            if run:
                model["rating"] = run.rate
                model["size_on_disk"] = run.size_on_disk
        elif comparisons[comparison_id] == "autokeras":
            model, run = get_autokeras_details(run_id)
            if run:
                model["rating"] = run.rate
                model["size_on_disk"] = run.size_on_disk
        elif comparisons[comparison_id] == "autokeras_trial":
            [autokeras_id, trial_id] = comparison_id.split("_")
            model, run = get_autokerastrial_details(autokeras_id, trial_id)
            if run:
                model["rating"] = run.rate
                model["size_on_disk"] = run.size_on_disk
        elif comparisons[comparison_id] == "inference":
            model, run = get_inference_details(run_id)
            model["rating"] = 0
            if run.network_training:
                model["size_on_disk"] = run.network_training.size_on_disk
                model["size"] = run.network_training.network_config.size
            else:
                model["size_on_disk"] = "0"
                model["size"] = "0"
        if run:
            model["device"] = run.compute_device
            model["dataset"] = str(run.dataset)
            model["description"] = run.description
        else:
            model["size_on_disk"] = "0"
        model["run_type"] = comparisons[comparison_id]
        model["comparison_id"] = comparison_id
        details.append(model)
    return details


def get_arguments_as_dict(arguments):
    argument_dict = {}
    for argument in arguments:
        argument_dict[argument["name"]] = argument["value"]
    return argument_dict
