from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Column, Field, Layout, Row
from django import forms

from neural_architecture.models.dataset import DatasetLoader
from neural_architecture.models.types import CallbackType, LossType, MetricType


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
        required=False,
    )

    save_network_as_template = forms.BooleanField(
        label="Als Vorlage speichern", required=False
    )
    network_template_name = forms.CharField(label="Vorlagename", required=False)

    dataset = forms.CharField(label="Dataset", required=False)
    dataset_is_supervised = forms.BooleanField(initial=True, required=False)

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
