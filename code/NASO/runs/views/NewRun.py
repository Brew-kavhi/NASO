import json
import random

from django.contrib import messages
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils import timezone as tz
from django.views.generic.base import TemplateView
from django.views.generic.edit import CreateView, DeleteView

from naso.celery import get_tasks
from naso.models.page import PageSetup
from neural_architecture.models.Architecture import NetworkConfiguration, NetworkLayer
from neural_architecture.neural_net import run_neural_net
from runs.forms.NewRunForm import NewRunForm
from runs.models.Training import (EvaluationParameters, FitParameters,
                                  LossFunction, Metric, NetworkHyperparameters,
                                  NetworkTraining, Optimizer)


class NewRun(TemplateView):
    template_name = "runs/run_new.html"
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
                training_id  = request.GET.get('rerun')
                training = NetworkTraining.objects.get(pk = training_id)
                form.initial['name'] = training.network_config.name
                form.initial['loss'] = training.hyper_parameters.loss.instance_type
                form.initial['optimizer'] = training.hyper_parameters.optimizer.instance_type
                form.initial['metrics'] = [metric.instance_type for metric in training.hyper_parameters.metrics.all()]
                form.initial['run_eagerly'] = training.hyper_parameters.run_eagerly
                form.initial['steps_per_execution'] = training.hyper_parameters.steps_per_execution
                form.initial['jit_compile'] = training.hyper_parameters.jit_compile
                form.initial['epochs']  = training.fit_parameters.epochs
                form.initial['batch_size'] = training.fit_parameters.batch_size
                form.initial['shuffle'] = training.fit_parameters.shuffle
                form.initial['steps_per_epoch'] = training.fit_parameters.steps_per_epoch
                form.initial['workers'] = training.fit_parameters.workers
                form.initial['use_multiprocessing'] = training.fit_parameters.use_multiprocessing
                nodes = [{
                    'id': layer.id,
                    'label': f"{layer.layer_type.name} ({layer.id})",
                    'x': random.random()/10.0,
                    'y': layer.id + random.random()/5.0,
                    'size': 3,
                    'color': '#008cc2',
                    'type': layer.layer_type.id,
                    'additional_arguments': layer.additional_arguments,
                } for layer in training.network_config.layers.all()]
                form.load_graph(nodes,training.network_config.connections)
                form.load_optimizer_config(training.hyper_parameters.optimizer.additional_arguments)
                form.load_loss_config(training.hyper_parameters.loss.additional_arguments)
                form.load_metric_configs([{'id':metric.instance_type.id, 'arguments': metric.additional_arguments} for metric in training.hyper_parameters.metrics.all()])
                self.context.update(form.get_extra_context())

                
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
                #     {
                #         "module_name": "neural_architecture.NetworkCallbacks.CeleryUpdateCallback",
                #         "class_name": "CeleryUpdateCallback",
                #     }
                # ]
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
                        layer_type_id=layer["type"],
                        name=layer["id"],
                        additional_arguments=layer["additional_arguments"],
                    )
                    naso_layer.save()
                    node_to_layers[layer['id']] = naso_layer.id
                    network_config.layers.add(naso_layer)
                # next iterate over the edges and adjust the ids, so we use the
                # naso ids instead of the javacript ones

                for edge in connections:
                    if not edge['source'] == 'input_node':
                        edge['source'] = node_to_layers[edge["source"]]
                    edge['target'] = node_to_layers[edge["target"]]

                network_config.connections = connections
                network_config.save()

                training.network_config = network_config
                training.fit_parameters = fit_parameters
                training.evaluation_parameters = eval_parameters

                training.save()

                run_neural_net.delay(training.id)
                messages.add_message(
                    request, messages.SUCCESS, "Training wurde gestartet."
                )
                return redirect("dashboard:index")
