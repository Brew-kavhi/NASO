from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Column, Field, Layout, Row
from django import forms

from neural_architecture.models.dataset import DatasetLoader
from neural_architecture.models.model_optimization import (
    PruningMethodTypes,
    PruningPolicyTypes,
    PruningScheduleTypes,
)
from neural_architecture.models.types import CallbackType, LossType, MetricType
from workers.models.celery_workers import CeleryWorker


class BaseRun(forms.Form):
    name = forms.CharField(label="Network Name", max_length=100)

    loss = forms.ModelChoiceField(
        label="Loss Function",
        queryset=LossType.objects.all(),
    )
    metrics = forms.ModelMultipleChoiceField(
        label="Metrics",
        queryset=MetricType.objects.all(),
        required=False,
        widget=forms.SelectMultiple(attrs={"class": "select2 w-100"}),
    )
    dataset_loaders = forms.ModelChoiceField(
        label="Dataset Loaders",
        queryset=DatasetLoader.objects.all(),
        widget=forms.Select(attrs={"class": "select2 w-100"}),
        required=True,
    )

    description = forms.CharField(
        widget=forms.Textarea, label=("Beschreibung"), required=False
    )

    save_network_as_template = forms.BooleanField(
        label="Als Vorlage speichern", required=False
    )
    network_template_name = forms.CharField(label="Vorlagename", required=False)

    dataset = forms.CharField(label="Dataset", required=True)
    dataset_is_supervised = forms.BooleanField(initial=True, required=False)
    gpu = forms.CharField(
        label="Device",
        widget=forms.SelectMultiple(attrs={"class": "select2 w-100"}),
        required=False,
    )

    extra_context = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_id = "start_new_run"
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-lg-2"
        self.helper.field_class = "col-lg-10"

        self.fields["metrics"].widget.attrs["onchange"] = "handleMetricChange(this)"
        self.fields["metrics"].widget.choices = self.get_metric_choices()

        self.fields["dataset_loaders"].widget.attrs[
            "onchange"
        ] = "handleDatasetLoaderChange(this)"

        self.fields["loss"].widget.attrs["onchange"] = "handleLossChange(this)"
        self.fields["loss"].widget.choices = self.get_loss_choices()
        self.fields["gpu"].widget.choices = get_gpu_choices()

    def load_metric_configs(self, arguments):
        self.extra_context["metric_configs"] = arguments

    def load_loss_config(self, arguments):
        self.extra_context["loss_config"] = arguments

    def get_loss_choices(self):
        loss_choices = []
        modules = LossType.objects.values_list("module_name", flat=True).distinct()

        for module in modules:
            losses = LossType.objects.filter(module_name=module)
            loss_choices.append((module, [(loss.id, loss.name) for loss in losses]))

        return loss_choices

    def load_graph(self, nodes, edges):
        self.extra_context["nodes"] = nodes
        self.extra_context["edges"] = edges

    def get_metric_choices(self):
        metric_choices = []
        modules = MetricType.objects.values_list("module_name", flat=True).distinct()

        for module in modules:
            metrics = MetricType.objects.filter(module_name=module)
            metric_choices.append(
                (module, [(metric.id, metric.name) for metric in metrics])
            )

        return metric_choices

    def get_extra_context(self):
        return self.extra_context

    def metric_html(self):
        return Layout(
            Field(
                "metrics",
                css_class="chosen-select select2 w-100",
                data_placeholder="Select Metrics",
                multiple="multiple",
            ),
            HTML(
                """
                <div id='metrics-arguments' class='card rounded-3 d-flex flex-row flex-wrap'></div>
                """
            ),
        )

    def gpu_field(self, multiple: bool = True):
        if not multiple:
            self.fields["gpu"].widget = forms.Select(attrs={"class": "select2 w-100"})
            self.fields["gpu"].widget.choices = get_gpu_choices()
        return Layout(
            Row(
                HTML(
                    """
                    <h2>Hardware</h2>
                    """
                ),
                css_class="border-top pt-3",
            ),
            Field(
                "gpu",
                css_class="chosen-select select2 w-100",
                data_placeholder="Select GPU",
            ),
        )

    def loss_html(self):
        return Layout(
            Field(
                "loss",
                css_class="select2 w-100 mt-3",
                data_placeholder="Select Loss function",
            ),
            HTML("<div id='loss-arguments' class='card rounded-3'></div>"),
        )

    def dataloader_html(self):
        return Layout(
            Row(
                HTML(
                    """
                    <h2>Dataset</h2>
                    """
                ),
                css_class="border-top pt-3",
            ),
            Row(
                Column(
                    Field(
                        "dataset_loaders",
                        css_class="chosen-select select2 w-100",
                        wrapper_class="align-items-center",
                    )
                ),
                Column(
                    Field(
                        "dataset",
                        css_class="autocomplete w-100",
                    )
                ),
                Column(Field("dataset_is_supervised"), css_class="col-3"),
                css_class="align-items-center",
            ),
        )


class ClusterableForm(forms.Form):
    enable_clustering = forms.BooleanField(required=False, initial=False)
    number_of_clusters = forms.IntegerField(required=False, initial=3)
    centroids_init = forms.ChoiceField(
        required=False,
        choices=[
            ("linear", "Linear"),
            ("random", "Random"),
            ("kmeans", "Kmeans++"),
            ("density", "Density Based"),
        ],
        initial="linear",
    )

    def get_clustering_fields(self):
        self.fields["enable_clustering"].widget.attrs[
            "onchange"
        ] = "toggleClustering(this)"
        return Layout(
            Row(
                HTML(
                    """
                    <h2>Clustering</h2>
                    """
                ),
                css_class="border-top pt-3",
            ),
            Row(
                Column(
                    Field(
                        "enable_clustering",
                        css_class="form-check-input",
                        wrapper_class="form-check offset-0",
                    ),
                    css_class="col-3",
                ),
                Column(
                    Field(
                        "number_of_clusters",
                    ),
                    css_class="col-4 d-none",
                ),
                Column(
                    Field("centroids_init", css_class="select2"),
                    css_class="col-5 d-none",
                ),
            ),
        )


class PrunableForm(forms.Form):
    enable_pruning = forms.BooleanField(required=False, initial=False)
    pruning_method = forms.ModelChoiceField(
        label="Pruning function",
        queryset=PruningMethodTypes.objects.all(),
        required=False,
    )
    pruning_scheduler = forms.ModelChoiceField(
        label="Pruning scheduler",
        queryset=PruningScheduleTypes.objects.all(),
        required=False,
    )
    pruning_policy = forms.ModelChoiceField(
        label="Pruning policy",
        queryset=PruningPolicyTypes.objects.all(),
        required=False,
    )

    def get_pruning_fields(self):
        self.fields["enable_pruning"].widget.attrs["onchange"] = "togglePruning(this)"
        self.fields["pruning_method"].widget.attrs[
            "onchange"
        ] = "handlePruningMethodChange(this)"
        self.fields["pruning_scheduler"].widget.attrs[
            "onchange"
        ] = "handlePruningSchedulerChange(this)"
        self.fields["pruning_policy"].widget.attrs[
            "onchange"
        ] = "handlePruningPolicyChange(this)"
        return Layout(
            Row(
                HTML(
                    """
                    <h2>Pruning</h2>
                    """
                ),
                css_class="border-top pt-3",
            ),
            Row(
                Column(
                    Field(
                        "enable_pruning",
                        css_class="form-check-input",
                        wrapper_class="form-check offset-0",
                    ),
                ),
                Column(
                    Field(
                        "pruning_method",
                        css_class="select2 w-100 mt-3",
                        data_placeholder="Select Pruning Method",
                    ),
                    HTML(
                        """
                        <div id='pruning-methods-arguments' class='card rounded-3 d-flex flex-row flex-wrap'></div>
                        """
                    ),
                    css_class="d-none",
                ),
            ),
            Row(
                Column(
                    Field(
                        "pruning_scheduler",
                        css_class="select2 w-100 mt-3",
                        data_placeholder="Select Pruning Scheduler",
                    ),
                    HTML(
                        """
                        <div id='pruning-scheduler-arguments' class='card rounded-3 d-flex flex-row flex-wrap'></div>
                        """
                    ),
                    css_class="d-none",
                ),
                Column(
                    Field(
                        "pruning_policy",
                        css_class="select2 w-100 mt-3",
                        data_placeholder="Select Pruning policy",
                    ),
                    HTML(
                        """
                        <div id='pruning-policy-arguments' class='card rounded-3 d-flex flex-row flex-wrap'></div>
                        """
                    ),
                    css_class="d-none",
                ),
            ),
        )

    def load_pruning_config(self, pruning_method, pruning_scheduler, pruning_policy):
        self.extra_context["pruning_method_config"] = [
            {
                "id": pruning_method.instance_type.id,
                "arguments": pruning_method.additional_arguments,
            }
        ]
        self.initial["pruning_method"] = pruning_method.instance_type
        if pruning_scheduler:
            self.extra_context["pruning_scheduler_config"] = [
                {
                    "id": pruning_scheduler.instance_type.id,
                    "arguments": pruning_scheduler.additional_arguments,
                }
            ]
            self.initial["pruning_scheduler"] = pruning_scheduler.instance_type
        if pruning_policy:
            self.extra_context["pruning_policy_config"] = [
                {
                    "id": pruning_policy.instance_type.id,
                    "arguments": pruning_policy.additional_arguments,
                }
            ]
            self.initial["pruning_policy"] = pruning_policy.instance_type


class BaseRunWithCallback(BaseRun):
    callbacks = forms.ModelMultipleChoiceField(
        label="Callbacks",
        queryset=CallbackType.objects.all(),
        required=False,
        widget=forms.SelectMultiple(attrs={"class": "select2 w-100"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["callbacks"].widget.attrs["onchange"] = "handleCallbackChange(this)"
        self.fields["callbacks"].widget.choices = self.get_callbacks_choices()

    def get_callbacks_choices(self):
        callback_choices = []
        modules = CallbackType.objects.values_list("module_name", flat=True).distinct()

        for module in modules:
            callbacks = CallbackType.objects.filter(module_name=module)
            callback_choices.append(
                (module, [(callback.id, callback.name) for callback in callbacks])
            )

        return callback_choices

    def load_callbacks_configs(self, arguments):
        self.extra_context["callbacks_configs"] = arguments

    def callback_html(self):
        return Layout(
            Field(
                "callbacks",
                css_class="select2 w-100 mt-3",
                data_placeholder="Select Callbacks",
                multiple="multiple",
            ),
            HTML("<div id='callbacks-arguments' class='card rounded-3'></div>"),
        )


def get_gpu_choices():
    celery_workers = CeleryWorker.objects.filter(active=True)
    return [
        (
            celery_worker.hostname,
            [
                (
                    f"{celery_worker.queue_name}|{list(device.keys())[0]}",
                    f"{list(device.keys())[0]} ({list(device.values())[0]})",
                )
                for device in celery_worker.devices
            ],
        )
        for celery_worker in celery_workers
    ]
