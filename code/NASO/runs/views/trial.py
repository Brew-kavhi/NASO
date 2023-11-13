from django.shortcuts import redirect
from django.views.generic.base import TemplateView

from api.views.autokeras import get_trial_details
from naso.models.page import PageSetup
from neural_architecture.autokeras import run_autokeras_trial
from neural_architecture.models.autokeras import AutoKerasRun
from neural_architecture.models.model_optimization import (
    PruningMethod,
    PruningPolicy,
    PruningSchedule,
)
from neural_architecture.models.model_runs import KerasModel, KerasModelRun
from runs.forms.trial import RerunTrialForm
from runs.models.training import EvaluationParameters, FitParameters
from runs.views.new_run import build_dataset, get_pruning_parameters


class TrialView(TemplateView):
    template_name = "runs/autokeras_trial.html"
    page = PageSetup(title="Autokeras Trial", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, run_id, trial_id, *args, **kwargs):
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
        form = RerunTrialForm(request.POST)
        if form.is_valid():
            autokeras_run = AutoKerasRun.objects.get(pk=run_id)

            eval_params = EvaluationParameters.objects.create(callbacks=[])
            fit_params = FitParameters.objects.create(
                epochs=form.cleaned_data["epochs"], callbacks=[]
            )

            keras_model = KerasModel.objects.create(
                name=f"{autokeras_run.model.project_name}_trial_{str(trial_id)}",
                description=f"""Fine tuning of trial {str(trial_id)} from Autokeras run 
                {autokeras_run.model.project_name}""",
                evaluation_parameters=eval_params,
                fit_parameters=fit_params,
            )
            if form.cleaned_data["enable_pruning"]:
                (
                    method_arguments,
                    scheduler_arguments,
                    policy_arguments,
                ) = get_pruning_parameters(request.POST.items())
                method, _ = PruningMethod.objects.get_or_create(
                    instance_type=form.cleaned_data["pruning_method"],
                    additional_arguments=method_arguments,
                )
                keras_model.pruning_method = method
                if form.cleaned_data["pruning_scheduler"]:
                    scheduler, _ = PruningSchedule.objects.get_or_create(
                        instance_type=form.cleaned_data["pruning_scheduler"],
                        additional_arguments=scheduler_arguments,
                    )
                    keras_model.pruning_scheduler = scheduler
                if form.cleaned_data["pruning_policy"]:
                    policy, _ = PruningPolicy.objecs.get_or_create(
                        instance_type=form.cleaned_data["pruning_policy"],
                        additional_arguments=policy_arguments,
                    )
                    keras_model.pruning_policy = policy
            keras_model.save()

            keras_model_run = KerasModelRun.objects.create(
                dataset=build_dataset(form.cleaned_data),
                model=keras_model,
                gpu=form.cleaned_data["gpu"],
            )

            # load the model in the worker function to be sure, enough memory is available
            run_autokeras_trial.delay(run_id, trial_id, keras_model_run.id)
            return redirect("dashboard:index")
        self.context["form"] = form
        self.context["hp"] = {}
        return self.render_to_response(self.context)
