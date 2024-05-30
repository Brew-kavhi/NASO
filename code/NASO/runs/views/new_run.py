import ast
import json
import os
import random

from django.contrib import messages
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView
from loguru import logger

from naso.models.page import PageSetup
from neural_architecture.autokeras import run_autokeras
from neural_architecture.models.architecture import NetworkConfiguration, NetworkLayer
from neural_architecture.models.autokeras import (
    AutoKerasModel,
    AutoKerasNode,
    AutoKerasRun,
    AutoKerasTuner,
)
from neural_architecture.models.dataset import Dataset
from neural_architecture.models.model_optimization import (
    ClusterableNetwork,
    PruningMethod,
    PruningPolicy,
    PruningSchedule,
)
from neural_architecture.models.templates import (
    AutoKerasNetworkTemplate,
    KerasNetworkTemplate,
)
from neural_architecture.neural_net import run_neural_net
from runs.forms.new_run_form import NewAutoKerasRunForm, NewRunForm
from runs.models.training import (
    CallbackFunction,
    EvaluationParameters,
    FitParameters,
    LossFunction,
    Metric,
    NetworkHyperparameters,
    NetworkTraining,
    Optimizer,
    TensorFlowModel,
    TrainingMetric,
)
from workers.helper_scripts.celery import get_all_workers


def get_model_options(request_params):
    """
    Extracts and organizes the arguments from the request parameters.

    Args:
        request_params (dict): The request parameters.

    Returns:
        array: Array containing the arguments for building the tensorflow_model
    """
    arguments = []
    for key, value in request_params:
        if key.startswith("tensorflow_model_argument_"):
            argument_name = key[len("tensorflow_model_argument_") :]
            argument = {"name": argument_name, "value": value}
            arguments.append(argument)
    return arguments


class NewRun(TemplateView):
    """
    View class for creating a new run/experiment.
    """

    template_name = "runs/new_tensorflow.html"
    page = PageSetup(title="Experimente", description="Neu")
    page.add_pageaction(reverse_lazy("runs:list"), "Alle Experimente")
    context = {"page": page.get_context()}

    def get_typewise_arguments(self, request_params):
        """
        Extracts and organizes the typewise arguments from the request parameters.

        Args:
            request_params (dict): The request parameters.

        Returns:
            tuple: A tuple containing optimizer arguments, loss arguments, metrics arguments, and callbacks arguments.
        """
        optimizer_arguments = []
        loss_arguments = []
        metrics_arguments = {}
        callbacks_arguments = {}
        for key, value in request_params:
            if key.startswith("optimizer_argument_"):
                argument_name = key[len("optimizer_argument_") :]
                optimizer_argument = {"name": argument_name, "value": value}
                optimizer_arguments.append(optimizer_argument)
            elif key.startswith("loss_argument_"):
                argument_name = key[len("loss_argument_") :]
                loss_argument = {"name": argument_name, "value": value}
                loss_arguments.append(loss_argument)
            elif key.startswith("callback_argument_"):
                # this is metric, get the metric key and check if there isa already a metric definition
                callback_id = int(key.split("_")[2])
                argument_name = "_".join(key.split("_")[3:])
                callbacks_argument = {"name": argument_name, "value": value}
                if callback_id not in callbacks_arguments:
                    callbacks_arguments[callback_id] = [callbacks_argument]
                else:
                    callbacks_arguments[callback_id].append(callbacks_argument)

            elif key.startswith("metric_argument_"):
                # this is metric, get the metric key and check if there isa already a metric definition
                metric_id = int(key.split("_")[2])
                argument_name = "_".join(key.split("_")[3:])
                metrics_argument = {"name": argument_name, "value": value}
                if metric_id not in metrics_arguments:
                    metrics_arguments[metric_id] = [metrics_argument]
                else:
                    metrics_arguments[metric_id].append(metrics_argument)
        return (
            optimizer_arguments,
            loss_arguments,
            metrics_arguments,
            callbacks_arguments,
        )

    def get(self, request, *args, **kwargs):
        """
        Handles the GET request for creating a new run/experiment.

        Args:
            request (HttpRequest): The HTTP request object.

        Returns:
            HttpResponse: The HTTP response object.
        """
        get_all_workers()
        # there is no training going on right now, so show form to start a new one
        form = NewRunForm(
            initial={
                "optimizer": None,  # Assuming 'optimizer' is the field name
                "loss": None,  # Assuming 'loss' is the field name
                "metrics": [],  # Assuming 'metrics' is the field name
            }
        )

        if "rerun" in request.GET:
            # this run should be a rerun from an older config, so load the config
            training_id = request.GET.get("rerun")
            training = NetworkTraining.objects.get(pk=training_id)
            form.initial["name"] = training.model_name
            form.initial["loss"] = training.hyper_parameters.loss.instance_type
            form.initial[
                "optimizer"
            ] = training.hyper_parameters.optimizer.instance_type
            if training.fit_parameters.callbacks:
                form.initial["callbacks"] = [
                    callback.instance_type
                    for callback in training.fit_parameters.callbacks.all()
                ]
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
            form.initial["steps_per_epoch"] = training.fit_parameters.steps_per_epoch
            form.initial["workers"] = training.fit_parameters.workers
            form.initial["dataset_loaders"] = training.dataset.dataset_loader
            form.initial["description"] = training.description
            form.initial[
                "use_multiprocessing"
            ] = training.fit_parameters.use_multiprocessing
            if training.network_config:
                nodes = [
                    {
                        "id": layer.name,
                        "label": f"{layer.name} ({layer.id})",
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
            elif training.tensorflow_model:
                form.initial["use_model_definition"] = True
                form.initial[
                    "tensorflow_model"
                ] = training.tensorflow_model.instance_type
                form.load_tensorflow_model_config(
                    training.tensorflow_model.additional_arguments
                )
            if training.network_model.pruning_method:
                form.initial["enable_pruning"] = True
                form.load_pruning_config(
                    training.network_model.pruning_method,
                    training.network_model.pruning_schedule,
                    training.network_model.pruning_policy,
                )
            form.load_optimizer_config(
                training.hyper_parameters.optimizer.additional_arguments
            )
            form.load_loss_config(training.hyper_parameters.loss.additional_arguments)
            form.load_metric_configs(
                [
                    {
                        "id": metric.instance_type.id,
                        "arguments": metric.additional_arguments,
                    }
                    for metric in training.hyper_parameters.metrics.all()
                ]
            )
            form.load_callbacks_configs(
                [
                    {
                        "id": callback.instance_type.id,
                        "arguments": callback.additional_arguments,
                    }
                    for callback in training.fit_parameters.callbacks.all()
                ]
            )

            if training.network_model.save_model:
                form.rerun_saved_model()

            # load dataset:
            form.initial["dataset"] = training.dataset.name
            form.initial["dataset_is_supervised"] = training.dataset.as_supervised
            self.context.update(form.get_extra_context())
        else:
            self.context = {"page": self.page.get_context()}

        self.context["form"] = form

        return self.render_to_response(self.context)

    def build_config(self, form_data, layers, connections):
        """
        Builds a network configuration based on the provided form data, layers, and connections.

        Args:
            form_data (dict): A dictionary containing the form data.
            layers (list): A list of dictionaries representing the network layers.
            connections (list): A list of dictionaries representing the network connections.

        Returns:
            NetworkConfiguration: The built network configuration.

        """
        network_config = NetworkConfiguration(name=form_data["name"])
        network_config.save()
        template = form_data["network_template"]
        if template:
            network_config.node_to_layer_id = template.node_to_layer_id
            network_config.layers.set(template.layers.all())
            network_config.connections = template.connections
        else:
            node_to_layers = {}
            for layer in layers:
                if layer["id"] == "input_node":
                    continue
                naso_layer = NetworkLayer(
                    layer_type_id=layer["naso_type"],
                    name=layer["id"],
                    additional_arguments=layer["additional_arguments"],
                )
                naso_layer.save()
                node_to_layers[layer["id"]] = naso_layer.id
                network_config.layers.add(naso_layer)

            network_config.connections = connections
            network_config.node_to_layer_id = node_to_layers
        if form_data["save_model"]:
            network_config.save_model = True
        network_config.save()
        return network_config

    def post(self, request, *args, **kwargs):
        """
        Handle the HTTP POST request for creating a new run.

        Args:
            request (HttpRequest): The HTTP request object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            HttpResponse: The HTTP response object.

        """
        if "epochs" in request.POST:
            form = NewRunForm(request.POST)
            if form.is_valid():
                # create the arrays for additional_arguments
                (
                    optimizer_arguments,
                    loss_arguments,
                    metrics_arguments,
                    callbacks_arguments,
                ) = self.get_typewise_arguments(request.POST.items())

                # Create Optimizer object
                optimizer, _ = Optimizer.objects.get_or_create(
                    instance_type=form.cleaned_data["optimizer"],
                    additional_arguments=optimizer_arguments,
                )

                hyper_parameters = NetworkHyperparameters(
                    run_eagerly=form.cleaned_data["run_eagerly"],
                    steps_per_execution=form.cleaned_data["steps_per_execution"],
                    jit_compile=form.cleaned_data["jit_compile"],
                )
                hyper_parameters.optimizer = optimizer
                hyper_parameters.loss = build_loss_function(
                    form.cleaned_data, loss_arguments
                )

                hyper_parameters.save()

                hyper_parameters.metrics.set(
                    build_metrics(form.cleaned_data, metrics_arguments)
                )

                eval_parameters = EvaluationParameters.objects.filter(
                    steps=form.cleaned_data["steps_per_execution"],
                    batch_size=form.cleaned_data["batch_size"],
                )
                if eval_parameters.exists():
                    eval_parameters = eval_parameters.first()
                else:
                    eval_parameters = EvaluationParameters.objects.create(
                        steps=form.cleaned_data["steps_per_execution"],
                        batch_size=form.cleaned_data["batch_size"],
                    )
                    eval_parameters.save()
                eval_parameters.callbacks.set(
                    build_callbacks(form.cleaned_data, callbacks_arguments)
                )

                fit_parameters = FitParameters.objects.create(
                    epochs=form.cleaned_data["epochs"],
                    batch_size=form.cleaned_data["batch_size"],
                    shuffle=form.cleaned_data["shuffle"],
                    steps_per_epoch=form.cleaned_data["steps_per_epoch"],
                    workers=form.cleaned_data["workers"],
                    use_multiprocessing=form.cleaned_data["use_multiprocessing"],
                )
                fit_parameters.save()
                fit_parameters.callbacks.set(
                    build_callbacks(form.cleaned_data, callbacks_arguments)
                )

                if form.cleaned_data["use_model_definition"]:
                    model_options = get_model_options(request.POST.items())
                    network_model = TensorFlowModel.objects.create(
                        instance_type=form.cleaned_data["tensorflow_model"],
                        name=form.cleaned_data["name"],
                        additional_arguments=model_options,
                    )
                    network_model.save()
                    if form.cleaned_data["save_model"]:
                        network_model.save_model = True
                        network_model.save()
                else:
                    network_model = self.build_config(
                        form_data=form.cleaned_data,
                        layers=json.loads(request.POST.get("nodes")),
                        connections=json.loads(request.POST.get("edges")),
                    )
                    if form.cleaned_data["save_network_as_template"]:
                        create_network_template(
                            form.cleaned_data["network_template_name"],
                            False,
                            network_model.layers.all(),
                            network_model.connections,
                            network_model.node_to_layer_id,
                        )

                if (
                    "rerun" in request.GET
                    and form.cleaned_data["fine_tune_saved_model"]
                ):
                    network_model.load_model = True
                    network_model.model_file = form.cleaned_data["load_model"]
                    id = os.path.splitext(
                        network_model.model_file[
                            network_model.model_file.rfind("_") + 1 :
                        ]
                    )[0]
                    try:
                        if network_model.model_file.rfind("/tensorflow/") > 0:
                            old_training = NetworkTraining.objects.filter(
                                network_config__id=id
                            ).first()
                        else:
                            old_training = NetworkTraining.objects.filter(
                                tensorflow_model__id=id
                            ).first()
                        fit_parameters.initial_epoch = (
                            old_training.fit_parameters.epochs
                        )
                        fit_parameters.save()
                    except Exception:
                        logger.info("Model could not be loaded")

                if form.cleaned_data["enable_clustering"]:
                    number_of_clusters = form.cleaned_data["number_of_clusters"]
                    centroids_init = form.cleaned_data["centroids_init"]
                    clustering_options = ClusterableNetwork(
                        use_clustering=True,
                        number_of_cluster=number_of_clusters,
                        cluster_centroids_init=centroids_init,
                    )
                    clustering_options.save()
                    network_model.clustering_options = clustering_options
                if form.cleaned_data["enable_pruning"]:
                    (
                        method_arguments,
                        scheduler_arguments,
                        policy_arguments,
                    ) = get_pruning_parameters(request.POST)
                    method, _ = PruningMethod.objects.get_or_create(
                        instance_type=form.cleaned_data["pruning_method"],
                        additional_arguments=method_arguments,
                    )
                    network_model.pruning_method = method
                    if form.cleaned_data["pruning_scheduler"]:
                        scheduler, _ = PruningSchedule.objects.get_or_create(
                            instance_type=form.cleaned_data["pruning_scheduler"],
                            additional_arguments=scheduler_arguments,
                        )
                        network_model.pruning_schedule = scheduler

                    if form.cleaned_data["pruning_policy"]:
                        policy, _ = PruningPolicy.objects.get_or_create(
                            instance_type=form.cleaned_data["pruning_policy"],
                            additional_arguments=policy_arguments,
                        )
                        network_model.pruning_policy = policy
                network_model.save()

                old_trainingmetric_queryset = None
                if network_model.load_model:
                    # copy all the existing metrics to this training as well
                    # but first get id of run from model:
                    id = os.path.splitext(
                        network_model.model_file[
                            network_model.model_file.rfind("_") + 1 :
                        ]
                    )[0]

                    if network_model.model_file.rfind("/tensorflow/") > 0:
                        old_trainingmetric_queryset = TrainingMetric.objects.filter(
                            neural_network__network_config__id=id
                        )
                    else:
                        old_trainingmetric_queryset = TrainingMetric.objects.filter(
                            neural_network__tensorflow_model__id=id
                        )

                workers = ast.literal_eval(form.cleaned_data["gpu"])
                inference_workers = ast.literal_eval(form.cleaned_data["inference_gpu"])
                for idx, worker in enumerate(workers):
                    queue, gpu = worker.split("|")
                    training = NetworkTraining()
                    training.hyper_parameters = hyper_parameters
                    training.dataset = build_dataset(form.cleaned_data)
                    if form.cleaned_data["use_model_definition"]:
                        training.tensorflow_model = network_model
                    else:
                        training.network_config = network_model
                    training.fit_parameters = fit_parameters
                    training.evaluation_parameters = eval_parameters
                    training.description = (
                        form.cleaned_data["description"] + f"(@{queue})"
                    )

                    training.gpu = gpu
                    training.worker = queue

                    training.save()
                    if old_trainingmetric_queryset:
                        for source_instance in old_trainingmetric_queryset:
                            new_instance = TrainingMetric.objects.create(
                                neural_network=training,
                                epoch=source_instance.epoch,
                                metrics=source_instance.metrics,
                            )
                            new_instance.save()

                    passed_inference_workers = []
                    if idx == len(workers) - 1:
                        passed_inference_workers = inference_workers
                    elif worker in inference_workers:
                        passed_inference_workers = [worker]
                        inference_workers.pop(inference_workers.index(worker))
                    run_neural_net.apply_async(
                        args=(
                            training.id,
                            passed_inference_workers,
                        ),
                        queue=queue,
                    )
                    messages.add_message(
                        request, messages.SUCCESS, "Training wurde gestartet."
                    )

                return redirect("dashboard:index")
        return redirect(request.path)


def get_pruning_parameters(request_params):
    """
    Extracts pruning parameters from the given request parameters.

    Args:
        request_params (dict): The request parameters containing the pruning arguments.

    Returns:
        tuple: A tuple containing three lists - method_arguments, scheduler_arguments, and policy_arguments.
               Each list contains dictionaries with 'name' and 'value' keys representing the argument name and value.

    Example:
        >>> request_params = {
        ...     'pruning-method_argument_threshold': '0.5',
        ...     'pruning-scheduler_argument_epochs': '10',
        ...     'pruning-policy_argument_decay': '0.1'
        ... }
        >>> get_pruning_parameters(request_params)
        ([
            {'name': 'threshold', 'value': '0.5'}
        ], [
            {'name': 'epochs', 'value': '10'}
        ], [
            {'name': 'decay', 'value': '0.1'}
        ])
    """
    method_arguments = []
    scheduler_arguments = []
    policy_arguments = []
    for key, value in request_params.items():
        if key.startswith("pruning-method_argument_"):
            argument_name = key[len("pruning-method_argument_") :]
            method_argument = {"name": argument_name, "value": value}
            method_arguments.append(method_argument)
        elif key.startswith("pruning-scheduler_argument_"):
            argument_name = key[len("pruning-scheduler_argument_") :]
            scheduler_argument = {"name": argument_name, "value": value}
            scheduler_arguments.append(scheduler_argument)
        elif key.startswith("pruning-policy_argument_"):
            argument_name = key[len("pruning-policy_argument_") :]
            policy_argument = {"name": argument_name, "value": value}
            policy_arguments.append(policy_argument)
    return (method_arguments, scheduler_arguments, policy_arguments)


def build_dataset(form_data):
    """
    Build a dataset based on the provided form data.

    Args:
        form_data (dict): A dictionary containing the form data.

    Returns:
        Dataset: The built dataset object.
    """
    data, _ = Dataset.objects.get_or_create(
        name=form_data["dataset"],
        as_supervised=form_data["dataset_is_supervised"],
        dataset_loader=form_data["dataset_loaders"],
    )
    return data


def build_loss_function(form_data, loss_arguments):
    """
    Builds a loss function based on the provided form data and loss arguments.

    Args:
        form_data (dict): A dictionary containing the form data.
        loss_arguments (str): Additional arguments for the loss function.

    Returns:
        LossFunction: The built loss function.

    """
    loss_function, _ = LossFunction.objects.get_or_create(
        instance_type=form_data["loss"],
        additional_arguments=loss_arguments,
    )
    return loss_function


def build_metrics(form_data, metrics_arguments):
    """
    Build a list of metrics based on the given form data and metrics arguments.

    Args:
        form_data (dict): The form data containing the selected metrics.
        metrics_arguments (dict): The arguments for each metric type.

    Returns:
        list: A list of Metric objects.

    """
    metrics = []

    for metric_type in form_data["metrics"]:
        metric_arguments = metrics_arguments.get(metric_type.id, [])
        metric, _ = Metric.objects.get_or_create(
            instance_type=metric_type, additional_arguments=metric_arguments
        )
        metrics.append(metric)
    return metrics


def build_callbacks(form_data, callbacks_arguments):
    """
    Build a list of callbacks based on the form data and callback arguments.

    Args:
        form_data (dict): The form data containing the selected callback types.
        callbacks_arguments (dict): A dictionary mapping callback type IDs to their arguments.

    Returns:
        list: A list of callback objects.

    """
    callbacks = []

    for callback_type in form_data["callbacks"]:
        callback_arguments = callbacks_arguments.get(callback_type.id, [])
        callback, _ = CallbackFunction.objects.get_or_create(
            instance_type=callback_type, additional_arguments=callback_arguments
        )
        callbacks.append(callback)
    return callbacks


def create_network_template(
    template_name, is_autokeras, layers, connections, node_to_layers
):
    """
    Create a network template based on the given parameters.

    Args:
        template_name (str): The name of the template.
        is_autokeras (bool): Indicates whether the template is for AutoKeras or Keras.
        layers (list): The list of layers to be associated with the template.
        connections (list): The list of connections between layers.
        node_to_layers (dict): A dictionary mapping nodes to their corresponding layers.

    Returns:
        None
    """
    if is_autokeras:
        template = AutoKerasNetworkTemplate.objects.create(
            name=template_name,
            connections=connections,
            node_to_layer_id=node_to_layers,
        )
        template.blocks.set(layers)
        template.save()
    else:
        template = KerasNetworkTemplate.objects.create(
            name=template_name,
            connections=connections,
            node_to_layer_id=node_to_layers,
        )
        template.layers.set(layers)
        template.save()


class NewAutoKerasRun(TemplateView):
    """
    View class for creating a new AutoKeras run.

    This class handles the creation of a new AutoKeras run
    by providing the necessary form and data loading functionality.
    """

    template_name = "runs/new_autokeras.html"
    page = PageSetup(title="Autokeras", description="Neu")
    context = {"page": page.get_context()}

    def get_typewise_arguments(self, request_dict):
        """
        Extracts and organizes the arguments from the given request dictionary based on their types.

        Args:
            request_dict (dict): A dictionary containing the request data.

        Returns:
            tuple: A tuple containing the following:
                - tuner_arguments (list): A list of dictionaries representing tuner arguments.
                - loss_arguments (list): A list of dictionaries representing loss arguments.
                - metrics_arguments (dict): A dictionary where the keys are metric IDs
                and the values are lists of dictionaries representing metric arguments.
                - callbacks_arguments (dict): A dictionary where the keys are callback IDs
                and the values are lists of dictionaries representing callback arguments.
                - metric_weights (dict): A dictionary where the keys are metric names
                and the values are their corresponding weights.
        """
        tuner_arguments = []
        loss_arguments = []
        metrics_arguments = {}
        callbacks_arguments = {}
        metric_weights = {}
        for key, value in request_dict.items():
            if key.startswith("tuner_argument_"):
                argument_name = key[len("tuner_argument_") :]
                tuner_arguments.append({"name": argument_name, "value": value})
            elif key.startswith("loss_argument_"):
                argument_name = key[len("loss_argument_") :]
                loss_arguments.append({"name": argument_name, "value": value})
            elif key.startswith("metric_argument_"):
                # this is metric, get the metric key and check if there isa already a metric definition
                metric_id = int(key.split("_")[2])
                argument_name = "_".join(key.split("_")[3:])
                metrics_argument = {"name": argument_name, "value": value}
                if metric_id not in metrics_arguments:
                    metrics_arguments[metric_id] = [metrics_argument]
                else:
                    metrics_arguments[metric_id].append(metrics_argument)

            elif key.startswith("callback_argument_"):
                # this is metric, get the metric key and check if there isa already a metric definition
                callback_id = int(key.split("_")[2])
                argument_name = "_".join(key.split("_")[3:])
                callbacks_argument = {"name": argument_name, "value": value}
                if callback_id not in callbacks_arguments:
                    callbacks_arguments[callback_id] = [callbacks_argument]
                else:
                    callbacks_arguments[callback_id].append(callbacks_argument)

            elif key.startswith("metric_weight_"):
                metric_name = key[len("metric_weight_") :]
                metric_weights[metric_name] = float(value)

        return (
            tuner_arguments,
            loss_arguments,
            metrics_arguments,
            callbacks_arguments,
            metric_weights,
        )

    def get(self, request, *args, **kwargs):
        get_all_workers()
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

            form.initial["loss"] = autokeras_run.model.loss.instance_type
            form.initial["metrics"] = [
                metric.instance_type for metric in autokeras_run.model.metrics.all()
            ]
            form.initial["callbacks"] = [
                callback.instance_type
                for callback in autokeras_run.model.callbacks.all()
            ]
            form.initial["tuner"] = autokeras_run.model.tuner.tuner_type
            form.initial["metric_weights"] = autokeras_run.model.metric_weights
            form.initial["max_epochs"] = autokeras_run.model.epochs
            form.initial["description"] = autokeras_run.description
            nodes = [
                {
                    "id": layer.name,
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
            if autokeras_run.model.pruning_method:
                form.initial["enable_pruning"] = True
                form.load_pruning_config(
                    autokeras_run.model.pruning_method,
                    autokeras_run.model.pruning_schedule,
                    autokeras_run.model.pruning_policy,
                )
            form.load_graph(nodes, autokeras_run.model.connections)
            form.load_tuner_config(autokeras_run.model.tuner.additional_arguments)
            form.load_loss_config(autokeras_run.model.loss.additional_arguments)
            form.load_metric_configs(
                [
                    {
                        "id": metric.instance_type.id,
                        "arguments": metric.additional_arguments,
                    }
                    for metric in autokeras_run.model.metrics.all()
                ]
            )
            form.load_callbacks_configs(
                [
                    {
                        "id": callback.instance_type.id,
                        "arguments": callback.additional_arguments,
                    }
                    for callback in autokeras_run.model.callbacks.all()
                ]
            )

            # load dataset:
            form.initial["dataset_loaders"] = autokeras_run.dataset.dataset_loader
            form.initial["dataset"] = autokeras_run.dataset.name
            form.initial["dataset_is_supervised"] = autokeras_run.dataset.as_supervised
            self.context.update(form.get_extra_context())
            self.context["metric_weights"] = autokeras_run.model.metric_weights
        else:
            self.context = {"page": self.page.get_context()}

        self.context["form"] = form

        return self.render_to_response(self.context)

    def build_model(self, form_data, tuner, loss, weights):
        """
        Builds and saves an AutoKerasModel based on the provided form data.

        Args:
            form_data (dict): A dictionary containing the form data for creating the model.
            tuner (str): The tuner used for the model.
            loss (str): The loss function used for training the model.
            weights (dict): A dictionary containing the weights for different metrics.

        Returns:
            AutoKerasModel: The created AutoKerasModel object.
        """
        model = AutoKerasModel.objects.create(
            project_name=form_data["name"],
            max_trials=form_data["max_trials"],
            directory=form_data["directory"],
            tuner=tuner,
            objective=form_data["objective"],
            max_model_size=form_data["max_model_size"],
            loss=loss,
            epochs=form_data["max_epochs"],
        )
        model.metric_weights = weights
        template = form_data["network_template"]
        if template:
            model.node_to_layer_id = template.node_to_layer_id
            model.connections = template.connections
            model.blocks.set(template.blocks.all())
        else:
            node_to_layers = {}
            for layer in form_data["nodes"]:
                autokeras_layer, _ = AutoKerasNode.objects.get_or_create(
                    node_type_id=layer["naso_type"],
                    name=layer["id"],
                    additional_arguments=layer["additional_arguments"],
                )
                autokeras_layer.save()
                node_to_layers[layer["id"]] = autokeras_layer.id
                model.blocks.add(autokeras_layer)

            model.node_to_layer_id = node_to_layers
            model.connections = form_data["edges"]
        model.save()
        return model

    def post(self, request, *args, **kwargs):
        form = NewAutoKerasRunForm(request.POST)
        if form.is_valid():
            # create the arrays for additional_arguments
            (
                tuner_arguments,
                loss_arguments,
                metrics_arguments,
                callbacks_arguments,
                weights,
            ) = self.get_typewise_arguments(dict(request.POST.items()))

            # Create Optimizer object
            tuner, _ = AutoKerasTuner.objects.get_or_create(
                tuner_type=form.cleaned_data["tuner"],
                additional_arguments=tuner_arguments,
            )

            form.cleaned_data["nodes"] = json.loads(request.POST.get("nodes"))
            form.cleaned_data["edges"] = json.loads(request.POST.get("edges"))

            model = self.build_model(
                form_data=form.cleaned_data,
                tuner=tuner,
                loss=build_loss_function(form.cleaned_data, loss_arguments),
                weights=weights,
            )

            if form.cleaned_data["enable_clustering"]:
                number_of_clusters = form.cleaned_data["number_of_clusters"]
                centroids_init = form.cleaned_data["centroids_init"]
                clustering_options = ClusterableNetwork(
                    use_clustering=True,
                    number_of_cluster=number_of_clusters,
                    cluster_centroids_init=centroids_init,
                )
                clustering_options.save()
                model.clustering_options = clustering_options

            if form.cleaned_data["enable_pruning"]:
                (
                    method_arguments,
                    scheduler_arguments,
                    policy_arguments,
                ) = get_pruning_parameters(request.POST)
                method, _ = PruningMethod.objects.get_or_create(
                    instance_type=form.cleaned_data["pruning_method"],
                    additional_arguments=method_arguments,
                )
                model.pruning_method = method
                if form.cleaned_data["pruning_scheduler"]:
                    scheduler, _ = PruningSchedule.objects.get_or_create(
                        instance_type=form.cleaned_data["pruning_scheduler"],
                        additional_arguments=scheduler_arguments,
                    )
                    model.pruning_schedule = scheduler

                if form.cleaned_data["pruning_policy"]:
                    policy, _ = PruningPolicy.objects.get_or_create(
                        instance_type=form.cleaned_data["pruning_policy"],
                        additional_arguments=policy_arguments,
                    )
                    model.pruning_policy = policy
            model.save()

            if form.cleaned_data["save_network_as_template"]:
                create_network_template(
                    form.cleaned_data["network_template_name"],
                    True,
                    model.blocks.all(),
                    model.connections,
                    model.node_to_layer_id,
                )

            model.metrics.set(build_metrics(form.cleaned_data, metrics_arguments))
            model.callbacks.set(build_callbacks(form.cleaned_data, callbacks_arguments))
            queue, gpu = form.cleaned_data["gpu"].split("|")

            run = AutoKerasRun.objects.create(
                dataset=build_dataset(form.cleaned_data),
                model=model,
                gpu=gpu,
                worker=queue,
                description=form.cleaned_data["description"],
            )

            run_autokeras.apply_async(args=(run.id,), queue=queue)
            messages.add_message(request, messages.SUCCESS, "Training wurde gestartet.")
            return redirect("dashboard:index")
        return redirect(request.path)
