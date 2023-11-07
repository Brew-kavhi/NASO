from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Column, Field, Layout, Row, Submit
from django import forms

from neural_architecture.models.Dataset import DatasetLoader
from neural_architecture.models.Types import CallbackType, MetricType


class RerunTrialForm(forms.Form):
    epochs = forms.IntegerField(
        label="Epochen",
        initial=10,
        widget=forms.TextInput(attrs={"type": "number", "min": 0}),
    )

    metrics = forms.ModelMultipleChoiceField(
        label="Metrics",
        queryset=MetricType.objects.all(),
        required=False,
        widget=forms.SelectMultiple(attrs={"class": "select2 w-100"}),
    )
    callbacks = forms.ModelMultipleChoiceField(
        label="Callbacks",
        queryset=CallbackType.objects.all(),
        required=False,
        widget=forms.SelectMultiple(attrs={"class": "select2 w-100"}),
    )
    dataset_loaders = forms.ModelChoiceField(
        label="Dataset Loaders",
        queryset=DatasetLoader.objects.all(),
        widget=forms.Select(attrs={"class": "select2 w-100"}),
        required=False,
    )

    dataset = forms.CharField(label="Dataset", required=False)
    dataset_is_supervised = forms.BooleanField(initial=True, required=False)

    extra_context = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_id = "rerun_trial"
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-lg-2"
        self.helper.field_class = "col-lg-10"
        self.helper.layout = Layout(
            HTML('<div class="row mb-3"><h2>Fine tune</h2></div>'),
            Field("epochs"),
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
            Field(
                "callbacks",
                css_class="select2 w-100 mt-3",
                data_placeholder="Select Callbacks",
                multiple="multiple",
            ),
            HTML("<div id='callbacks-arguments' class='card rounded-3'></div>"),
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
            Submit("customer-general-edit", "Training starten"),
        )

        self.fields["metrics"].widget.attrs["onchange"] = "handleMetricChange(this)"
        self.fields["callbacks"].widget.attrs["onchange"] = "handleCallbackChange(this)"
        self.fields["dataset_loaders"].widget.attrs[
            "onchange"
        ] = "handleDatasetLoaderChange(this)"

        self.fields["metrics"].widget.choices = self.get_metric_choices()
        self.fields["callbacks"].widget.choices = self.get_callbacks_choices()

    def get_metric_choices(self):
        metric_choices = []
        modules = MetricType.objects.values_list("module_name", flat=True).distinct()

        for module in modules:
            metrics = MetricType.objects.filter(module_name=module)
            metric_choices.append(
                (module, [(metric.id, metric.name) for metric in metrics])
            )

        return metric_choices

    def get_callbacks_choices(self):
        callback_choices = []
        modules = CallbackType.objects.values_list("module_name", flat=True).distinct()

        for module in modules:
            callbacks = CallbackType.objects.filter(module_name=module)
            callback_choices.append(
                (module, [(callback.id, callback.name) for callback in callbacks])
            )

        return callback_choices

    def load_metric_configs(self, arguments):
        self.extra_context["metric_configs"] = arguments

    def load_callbacks_configs(self, arguments):
        self.extra_context["callbacks_configs"] = arguments

    def get_extra_context(self):
        return self.extra_context
