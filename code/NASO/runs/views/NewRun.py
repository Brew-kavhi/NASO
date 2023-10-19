import json
import random

from django.contrib import messages
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView

from naso.celery import get_tasks
from naso.models.page import PageSetup
from neural_architecture.autokeras import run_autokeras
from neural_architecture.models.AutoKeras import AutoKerasModel, AutoKerasNode, AutoKerasTuner
from neural_architecture.models.Architecture import NetworkConfiguration, NetworkLayer
from neural_architecture.models.AutoKeras import (
    AutoKerasModel,
    AutoKerasNode,
    AutoKerasRun,
    AutoKerasTuner,
)
from neural_architecture.models.Dataset import Dataset
from neural_architecture.neural_net import run_neural_net
from runs.models.Training import (
    EvaluationParameters,
    FitParameters,
    LossFunction,
    Metric,
    NetworkHyperparameters,
    NetworkTraining,
    Optimizer,
)
from runs.forms.NewRunForm import NewRunForm, NewAutoKerasRunForm



class NewRun(TemplateView):
    template_name = "runs/new_tensorflow.html"
    page = PageSetup(title="Experimente", description="Neu")
    page.add_pageaction(reverse_lazy("runs:list"), "Alle Experimente")
    context = {"page": page.get_context()}

    def get_typewise_arguments(self, request_params):
        optimizer_arguments = []
        loss_arguments = []
        metrics_arguments = {}
        for key, value in request_params:
            if key.startswith("optimizer_argument_"):
                argument_name = key[len("optimizer_argument_") :]
                optimizer_argument = {}
                optimizer_argument["name"] = argument_name
                optimizer_argument["value"] = value
                optimizer_arguments.append(optimizer_argument)
            elif key.startswith("loss_argument_"):
                argument_name = key[len("loss_argument_") :]
                loss_argument = {}
                loss_argument["name"] = argument_name
                loss_argument["value"] = value
                loss_arguments.append(loss_argument)
            elif key.startswith("metric_argument_"):
                # this is metric, get the metric key and check if there isa already a metric definition
                metric_id = int(key.split("_")[2])
                argument_name = "_".join(key.split("_")[3:])
                metrics_argument = {}

                metrics_argument["name"] = argument_name
                metrics_argument["value"] = value
                if metric_id not in metrics_arguments:
                    metrics_arguments[metric_id] = [metrics_argument]
                else:
                    metrics_arguments[metric_id].append(metrics_argument)
        return (optimizer_arguments, loss_arguments, metrics_arguments)

    def get(self, request, *args, **kwargs):
        running_task = get_tasks()
        print(running_task)
        if running_task.get("training_task_id"):
            # there is already a task running, cannotr start a second one.
            # vthis would falsify the measurements
            self.context["running_task"] = {
                "url": reverse_lazy("dashboard:index"),
                "title": "Zum Dashboard",
            }
        else:
            # there is no training going on right now, so show form to start a new one
            form = NewRunForm(
                initial={
                    "optimizer": None,  # Assuming 'optimizer' is the field name
                    "loss": None,  # Assuming 'loss' is the field name
                    "metrics": [],  # Assuming 'metrics' is the field name
                }
            )
            
            if 'rerun' in request.GET:
                # this run should be a rerun from an older config, so load the config
                training_id = request.GET.get("rerun")
                training = NetworkTraining.objects.get(pk=training_id)
                form.initial["name"] = training.network_config.name
                form.initial["loss"] = training.hyper_parameters.loss.instance_type
                form.initial[
                    "optimizer"
                ] = training.hyper_parameters.optimizer.instance_type
                form.initial["metrics"] = [
                    metric.instance_type
                    for metric in training.hyper_parameters.metrics.all()
                ]
                form.initial["run_eagerly"] = training.hyper_parameters.run_eagerly
                form.initial[
                    "steps_per_execution"
                ] = training.hyper_parameters.steps_per_execution
                form.initial["jit_compile"] = training.hyper_parameters.jit_compile
                form.initial["epochs"] = training.fit_parameters.epochs
                form.initial["batch_size"] = training.fit_parameters.batch_size
                form.initial["shuffle"] = training.fit_parameters.shuffle
                form.initial[
                    "steps_per_epoch"
                ] = training.fit_parameters.steps_per_epoch
                form.initial["workers"] = training.fit_parameters.workers
                form.initial[
                    "use_multiprocessing"
                ] = training.fit_parameters.use_multiprocessing
                nodes = [
                    {
                        "id": layer.id,
                        "label": f"{layer.layer_type.name} ({layer.id})",
                        "x": random.random() / 10.0,
                        "y": layer.id + random.random() / 5.0,
                        "size": 3,
                        "color": "#008cc2",
                        "naso_type": layer.layer_type.id,
                        "type": "image",
                        "additional_arguments": layer.additional_arguments,
                    }
                    for layer in training.network_config.layers.all()
                ]
                form.load_graph(nodes, training.network_config.connections)
                form.load_optimizer_config(
                    training.hyper_parameters.optimizer.additional_arguments
                )
                form.load_loss_config(
                    training.hyper_parameters.loss.additional_arguments
                )
                form.load_metric_configs(
                    [
                        {
                            "id": metric.instance_type.id,
                            "arguments": metric.additional_arguments,
                        }
                        for metric in training.hyper_parameters.metrics.all()
                    ]
                )

                # load dataset:
                form.initial["dataset"] = training.dataset.name
                form.initial["dataset_is_supervised"] = training.dataset.as_supervised
                self.context.update(form.get_extra_context())
            else:
                self.context = {"page": self.page.get_context()}
            print(self.context)

            self.context["form"] = form

        return self.render_to_response(self.context)

    def post(self, request, *args, **kwargs):
        if "epochs" in request.POST:
            form = NewRunForm(request.POST)
            if form.is_valid():
                # create the arrays for additional_arguments
                (
                    optimizer_arguments,
                    loss_arguments,
                    metrics_arguments,
                ) = self.get_typewise_arguments(request.POST.items())

                # Create Optimizer object
                optimizer_type = form.cleaned_data["optimizer"]
                optimizer, _ = Optimizer.objects.get_or_create(
                    instance_type=optimizer_type,
                    additional_arguments=optimizer_arguments,
                )

                # Create LossFunction object (similarly for Metric objects)
                loss_type = form.cleaned_data["loss"]
                loss_function, _ = LossFunction.objects.get_or_create(
                    instance_type=loss_type, additional_arguments=loss_arguments
                )

                print(loss_type)

                # Handle multiple selected metrics
                selected_metrics = form.cleaned_data["metrics"]

                metrics = []

                for metric_type in selected_metrics:
                    metric_arguments = []
                    if metric_type.id in metrics_arguments:
                        metric_arguments = metrics_arguments[metric_type.id]
                    metric, _ = Metric.objects.get_or_create(
                        instance_type=metric_type, additional_arguments=metric_arguments
                    )
                    metrics.append(metric)

                hyper_parameters = NetworkHyperparameters(
                    run_eagerly=form.cleaned_data["run_eagerly"],
                    steps_per_execution=form.cleaned_data["steps_per_execution"],
                    jit_compile=form.cleaned_data["jit_compile"],
                )
                hyper_parameters.optimizer = optimizer
                hyper_parameters.loss = loss_function

                hyper_parameters.save()

                hyper_parameters.metrics.set(metrics)

                training = NetworkTraining()
                training.hyper_parameters = hyper_parameters

                eval_parameters, _ = EvaluationParameters.objects.get_or_create(
                    steps=form.cleaned_data["steps_per_execution"],
                    batch_size=form.cleaned_data["batch_size"],
                    callbacks=[],
                )
                callbacks = []
                fit_parameters, _ = FitParameters.objects.get_or_create(
                    epochs=form.cleaned_data["epochs"],
                    batch_size=form.cleaned_data["batch_size"],
                    shuffle=form.cleaned_data["shuffle"],
                    steps_per_epoch=form.cleaned_data["steps_per_epoch"],
                    workers=form.cleaned_data["workers"],
                    use_multiprocessing=form.cleaned_data["use_multiprocessing"],
                    callbacks=callbacks,
                )

                network_config = NetworkConfiguration(name=form.cleaned_data["name"])
                network_config.save()

                layers = json.loads(request.POST.get("nodes"))
                connections = json.loads(request.POST.get("edges"))

                node_to_layers = {}
                for layer in layers:
                    if layer["id"] == "input_node":
                        continue
                    print(layer["additional_arguments"])
                    naso_layer, _ = NetworkLayer.objects.get_or_create(
                        layer_type_id=layer["naso_type"],
                        name=layer["id"],
                        additional_arguments=layer["additional_arguments"],
                    )
                    naso_layer.save()
                    node_to_layers[layer["id"]] = naso_layer.id
                    network_config.layers.add(naso_layer)
                # next iterate over the edges and adjust the ids, so we use the
                # naso ids instead of the javacript ones

                # TODO adjust htis here, so we dont modify the edges here.
                # # we need those unqiue ids form the graphs cause naso_ids ar not unique for the layers
                for edge in connections:
                    if not edge["source"] == "input_node":
                        edge["source"] = node_to_layers[edge["source"]]
                    edge["target"] = node_to_layers[edge["target"]]

                network_config.connections = connections
                network_config.save()

                # generate the dataset:
                dataset = form.cleaned_data["dataset"]
                dataset_is_supervised = form.cleaned_data["dataset_is_supervised"]
                data, _ = Dataset.objects.get_or_create(
                    name=dataset, as_supervised=dataset_is_supervised
                )

                training.dataset = data
                training.network_config = network_config
                training.fit_parameters = fit_parameters
                training.evaluation_parameters = eval_parameters

                training.save()

                run_neural_net.delay(training.id)
                messages.add_message(
                    request, messages.SUCCESS, "Training wurde gestartet."
                )
                return redirect("dashboard:index")


class NewAutoKerasRun(TemplateView):
    template_name = "runs/new_autokeras.html"
    page = PageSetup(title="Autokeras", description="Neu")
    # page.add_pageaction(reverse_lazy("runs:list"), "Alle Experimente")
    context = {"page": page.get_context()}

    def get_typewise_arguments(self, request_params):
        tuner_arguments = []
        for key, value in request_params:
            if key.startswith("tuner_argument_"):
                argument_name = key[len("tuner_argument_") :]
                tuner_argument = {}
                tuner_argument["name"] = argument_name
                tuner_argument["value"] = value
                tuner_arguments.append(tuner_argument)
            
        return tuner_arguments

    def get(self, request, *args, **kwargs):
        running_task = get_tasks()
        print(running_task)
        if running_task.get("training_task_id"):
            # there is already a task running, cannotr start a second one.
            # vthis would falsify the measurements
            self.context["running_task"] = {
                "url": reverse_lazy("dashboard:index"),
                "title": "Zum Dashboard",
            }
        else:
            # there is no training going on right now, so show form to start a new one
            form = NewAutoKerasRunForm()

            if "rerun" in request.GET:
                # this run should be a rerun from an older config, so load the config
                autokeras_run_id = request.GET.get("rerun")
                autokeras_run = AutoKerasRun.objects.get(pk=autokeras_run_id)
                form.initial["name"] = autokeras_run.model.project_name
                form.initial["max_model_size"] = autokeras_run.model.max_model_size
                form.initial["objective"] = autokeras_run.model.objective
                form.initial["max_trials"] = autokeras_run.model.max_trials
                form.initial["directory"] = autokeras_run.model.directory

                nodes = [
                    {
                        "id": layer.id,
                        "label": f"{layer.name} ({layer.id})",
                        "x": random.random() / 10.0,
                        "y": layer.id + random.random() / 5.0,
                        "size": 3,
                        "color": "#008cc2",
                        "naso_type": layer.node_type.id,
                        "type": "image",
                        "additional_arguments": layer.additional_arguments,
                    }
                    for layer in autokeras_run.model.blocks.all()
                ]
                form.load_graph(nodes, autokeras_run.model.connections)
                form.load_tuner_config(autokeras_run.model.tuner)

                # load dataset:
                form.initial["dataset"] = autokeras_run.dataset.name
                form.initial[
                    "dataset_is_supervised"
                ] = autokeras_run.dataset.as_supervised
                self.context.update(form.get_extra_context())
            else:
                self.context = {"page": self.page.get_context()}

            self.context["form"] = form

        return self.render_to_response(self.context)

    def post(self, request, *args, **kwargs):
        form = NewAutoKerasRunForm(request.POST)
        if form.is_valid():
            # create the arrays for additional_arguments
            tuner_arguments = self.get_typewise_arguments(request.POST.items())

            # Create Optimizer object
            tuner_type = form.cleaned_data["tuner"]
            tuner, _ = AutoKerasTuner.objects.get_or_create(
                tuner_type=tuner_type,
                additional_arguments=tuner_arguments,
            )

            model = AutoKerasModel.objects.create(
                project_name=form.cleaned_data["name"],
                max_trials=form.cleaned_data["max_trials"],
                directory=form.cleaned_data["directory"],
                tuner=tuner,
                objective=form.cleaned_data["objective"],
                max_model_size=form.cleaned_data["max_model_size"],
            )
            layers = json.loads(request.POST.get("nodes"))
            connections = json.loads(request.POST.get("edges"))

            node_to_layers = {}
            for layer in layers:
                autokeras_layer, _ = AutoKerasNode.objects.get_or_create(
                    node_type_id=layer["naso_type"],
                    name=layer["id"],
                    additional_arguments=layer["additional_arguments"],
                )
                autokeras_layer.save()
                node_to_layers[layer["id"]] = autokeras_layer.id
                model.blocks.add(autokeras_layer)

            dataset = form.cleaned_data["dataset"]
            dataset_is_supervised = form.cleaned_data["dataset_is_supervised"]
            data, _ = Dataset.objects.get_or_create(
                name=dataset, as_supervised=dataset_is_supervised
            )

            model.node_to_layer_id = node_to_layers
            model.connections = connections
            model.save()

            run = AutoKerasRun.objects.create(
                dataset=data,
                model=model,
            )

            run_autokeras.delay(run.id)
            messages.add_message(request, messages.SUCCESS, "Training wurde gestartet.")
            return redirect("dashboard:index")
