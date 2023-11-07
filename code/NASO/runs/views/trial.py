from django.shortcuts import redirect
from django.views.generic.base import TemplateView

from naso.models.page import PageSetup
from neural_architecture.autokeras import run_autokeras_trial
from neural_architecture.models.AutoKeras import AutoKerasRun
from neural_architecture.models.Dataset import Dataset
from runs.forms.Trial import RerunTrialForm


class TrialView(TemplateView):
    template_name = "runs/autokeras_trial.html"
    page = PageSetup(title="Autokeras Trial", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, run_id, trial_id, *args, **kwargs):
        run = AutoKerasRun.objects.get(pk=run_id)
        # TODO sometimes this leads to memory overflow and then the whole app gets stucked
        try:
            loaded_model = run.model.load_model(run)
            trial = loaded_model.tuner.oracle.trials[str(trial_id)]
            hp = trial.hyperparameters.values
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
            run = AutoKerasRun.objects.get(pk=run_id)

            dataset = form.cleaned_data["dataset"]
            dataset_is_supervised = form.cleaned_data["dataset_is_supervised"]
            data, _ = Dataset.objects.get_or_create(
                name=dataset,
                as_supervised=dataset_is_supervised,
                dataset_loader=form.cleaned_data["dataset_loaders"],
            )
            new_run = AutoKerasRun.objects.create(dataset=data, model=run.model)
            run_autokeras_trial.delay(new_run.id, trial_id, form.cleaned_data["epochs"])
            return redirect("dashboard:index")
        self.context["form"] = form
        self.context['hp'] = {}
        return self.render_to_response(self.context)
