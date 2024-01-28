from django.contrib import messages
from django.shortcuts import redirect
from django.views.generic.base import TemplateView

from api.views.autokeras import get_trial_details
from naso.models.page import PageSetup
from neural_architecture.models.autokeras import AutoKerasRun
from neural_architecture.models.model_optimization import (
    PruningMethod,
    PruningPolicy,
    PruningSchedule,
)
from neural_architecture.models.types import OptimizerType
from neural_architecture.neural_net import run_neural_net
from runs.forms.trial import RerunTrialForm
from runs.models.training import (
    EvaluationParameters,
    FitParameters,
    NetworkHyperparameters,
    NetworkTraining,
    Optimizer,
)
from runs.views.helper_scripts import build_additional_arguments, build_network_config
from runs.views.new_run import (
    build_callbacks,
    build_dataset,
    build_metrics,
    get_pruning_parameters,
)


class TrialView(TemplateView):
    """
    View for displaying and handling the details of a trial in an AutoKeras run.
    """

    template_name = "runs/autokeras_trial.html"
    page = PageSetup(title="Autokeras Trial", description="Details")
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
                - metrics_arguments (dict): A dictionary where the keys are metric IDs and the values are lists of dictionaries representing metric arguments.
                - callbacks_arguments (dict): A dictionary where the keys are callback IDs and the values are lists of dictionaries representing callback arguments.
                - metric_weights (dict): A dictionary where the keys are metric names and the values are their corresponding weights.
        """
        metrics_arguments = {}
        callbacks_arguments = {}
        for key, value in request_dict:
            if key.startswith("metric_argument_"):
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

        return (
            metrics_arguments,
            callbacks_arguments,
        )

    def get(self, request, run_id, trial_id, *args, **kwargs):
        """
        Handles the HTTP GET request for the trial view.

        Args:
            request (HttpRequest): The HTTP request object.
            run_id (int): The ID of the AutoKeras run.
            trial_id (int): The ID of the trial.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            HttpResponse: The HTTP response object.
        """
        run = AutoKerasRun.objects.get(pk=run_id)

        try:
            hp = get_trial_details(run, trial_id)
        except Exception:
            hp = {}

        self.context["hp"] = hp
        form = RerunTrialForm()
        self.context["form"] = form
        form.initial["metrics"] = [
            metric.instance_type for metric in run.model.metrics.all()
        ]
        form.initial["callbacks"] = [
            callback.instance_type for callback in run.model.callbacks.all()
        ]

        form.load_metric_configs(
            [
                {
                    "id": metric.instance_type.id,
                    "arguments": metric.additional_arguments,
                }
                for metric in run.model.metrics.all()
            ]
        )
        form.load_callbacks_configs(
            [
                {
                    "id": callback.instance_type.id,
                    "arguments": callback.additional_arguments,
                }
                for callback in run.model.callbacks.all()
            ]
        )
        self.context.update(form.get_extra_context())

        return self.render_to_response(self.context)

    def post(self, request, run_id, trial_id, *args, **kwargs):
        """
        Handles the HTTP POST request for the trial view.

        Args:
            request (HttpRequest): The HTTP request object.
            run_id (int): The ID of the AutoKeras run.
            trial_id (int): The ID of the trial.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            HttpResponse: The HTTP response object.
        """
        form = RerunTrialForm(request.POST)
        if form.is_valid():
            (
                metrics_arguments,
                callbacks_arguments,
            ) = self.get_typewise_arguments(request.POST.items())

            autokeras_run = AutoKerasRun.objects.get(pk=run_id)
            model = autokeras_run.model.load_trial(autokeras_run, trial_id)

            eval_params = EvaluationParameters.objects.create()
            eval_params.save()
            eval_params.callbacks.set(
                build_callbacks(form.cleaned_data, callbacks_arguments)
            )
            fit_params = FitParameters.objects.create(
                batch_size=form.cleaned_data["batch_size"],
                epochs=form.cleaned_data["epochs"],
            )
            fit_params.save()
            fit_params.callbacks.set(
                build_callbacks(form.cleaned_data, callbacks_arguments)
            )

            # create the optimizer config
            optimizer = get_optimizer_instance(model.optimizer)
            loss = autokeras_run.model.loss
            hyper_parameters = NetworkHyperparameters()
            hyper_parameters.optimizer = optimizer
            hyper_parameters.loss = loss
            hyper_parameters.save()
            hyper_parameters.metrics.set(
                build_metrics(form.cleaned_data, metrics_arguments)
            )
            name = f"{autokeras_run.model.project_name}_trial_{str(trial_id)}"

            training = NetworkTraining(
                description=f"""Fine tuning of trial {str(trial_id)} from Autokeras run 
                {autokeras_run.model.project_name}""",
            )

            training.hyper_parameters = hyper_parameters

            network_config = build_network_config(name, model)
            network_config.save_model = True
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
                network_config.pruning_method = method
                if form.cleaned_data["pruning_scheduler"]:
                    scheduler, _ = PruningSchedule.objects.get_or_create(
                        instance_type=form.cleaned_data["pruning_scheduler"],
                        additional_arguments=scheduler_arguments,
                    )
                    network_config.pruning_scheduler = scheduler
                if form.cleaned_data["pruning_policy"]:
                    policy, _ = PruningPolicy.objecs.get_or_create(
                        instance_type=form.cleaned_data["pruning_policy"],
                        additional_arguments=policy_arguments,
                    )
                    network_config.pruning_policy = policy
            network_config.save()

            training.dataset = build_dataset(form.cleaned_data)
            training.network_config = network_config
            training.fit_parameters = fit_params
            training.evaluation_parameters = eval_params
            training.gpu = form.cleaned_data["gpu"]

            training.save()

            run_neural_net.delay(training.id)
            messages.add_message(request, messages.SUCCESS, "Training wurde gestartet.")
            return redirect("dashboard:index")
        self.context["form"] = form
        self.context["hp"] = {}
        return self.render_to_response(self.context)


def get_optimizer_instance(optimizer):
    """
    this should eturn the build arguments for a NASO optimizer read from ana actual optimizer instance
    """
    optimizer_class = type(optimizer).__name__
    optimizer_module = type(optimizer).__module__
    if optimizer_module.startswith("keras.src.optimizers"):
        optimizer_module = "tensorflow.keras.optimizers"
    optimizer_type, _ = OptimizerType.objects.get_or_create(
        module_name=optimizer_module, name=optimizer_class
    )
    config = optimizer.get_config()
    arguments = build_additional_arguments(config)
    print(arguments)
    naso_optimizer, _ = Optimizer.objects.get_or_create(
        instance_type=optimizer_type, additional_arguments=arguments
    )
    return naso_optimizer
