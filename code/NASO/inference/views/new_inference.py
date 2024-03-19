from django.shortcuts import redirect
from django.views.generic.base import TemplateView

from inference.celery.run_inference import run_inference
from inference.forms.new import NewInferenceForm
from inference.models.inference import Inference
from naso.models.page import PageSetup
from runs.views.new_run import build_callbacks, build_dataset, build_metrics
from workers.helper_scripts.celery import get_all_workers


class NewInference(TemplateView):
    template_name = "inference/new_inference.html"
    page = PageSetup(title="Inference", description="New")
    context = {"page": page.get_context()}

    def get_typewise_arguments(self, request_params):
        """
        Extracts and organizes the typewise arguments from the request parameters.

        Args:
            request_params (dict): The request parameters.

        Returns:
            tuple: A tuple containing optimizer arguments, loss arguments, metrics arguments, and callbacks arguments.
        """
        metrics_arguments = {}
        callbacks_arguments = {}
        for key, value in request_params:
            if key.startswith("callback_argument_"):
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
            metrics_arguments,
            callbacks_arguments,
        )

    def get(self, request, *args, **kwargs):
        get_all_workers()
        self.page.actions = []
        self.context["page"] = self.page.get_context()
        form = NewInferenceForm()
        self.context["form"] = form
        if "rerun" in request.GET:
            old_inference_id = request.GET.get("rerun")
            old_inference = Inference.objects.get(pk=old_inference_id)
            form.initial["name"] = old_inference.name
            form.initial["batch_size"] = old_inference.batch_size
            form.initial["load_model"] = old_inference.model_file
            form.initial["metrics"] = [
                metric.instance_type for metric in old_inference.metrics.all()
            ]
            form.initial["callbacks"] = [
                callback.instance_type for callback in old_inference.callbacks.all()
            ]
            form.initial["description"] = old_inference.description
            form.initial["dataset_loaders"] = old_inference.dataset.dataset_loader
            form.initial["dataset"] = old_inference.dataset.name
        return self.render_to_response(self.context)

    def post(self, request, *args, **kwargs):
        form = NewInferenceForm(request.POST)
        if form.is_valid():
            (
                metrics_arguments,
                callbacks_arguments,
            ) = self.get_typewise_arguments(request.POST.items())
            queue, gpu = form.cleaned_data["gpu"].split("|")
            inference = Inference(
                name=form.cleaned_data["name"],
                description=form.cleaned_data["description"],
                model_file=form.cleaned_data["load_model"],
                gpu=gpu,
                worker=queue,
                batch_size=form.cleaned_data["batch_size"],
            )
            inference.save()
            inference.metrics.set(build_metrics(form.cleaned_data, metrics_arguments))
            inference.callbacks.set(
                build_callbacks(form.cleaned_data, callbacks_arguments)
            )
            inference.dataset = build_dataset(form.cleaned_data)
            inference.save()
            # now start the inference run
            run_inference.apply_async(args=(inference.id,), queue=queue)

            return redirect("inference:list")
        print(form.errors)

        self.page.actions = []
        self.context["page"] = self.page.get_context()
        self.context["form"] = NewInferenceForm()
        return self.render_to_response(self.context)
