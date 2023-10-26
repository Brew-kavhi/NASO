import tensorflow_datasets as tfds
from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Column, Field, Layout, Row, Submit
from django import forms
from django.urls import reverse_lazy

from neural_architecture.models.AutoKeras import AutoKerasNodeType, AutoKerasTunerType
from neural_architecture.models.Templates import (
    AutoKerasNetworkTemplate,
    KerasNetworkTemplate,
)
from neural_architecture.models.Types import (
    CallbackType,
    LossType,
    MetricType,
    NetworkLayerType,
    OptimizerType,
)


class NewRunForm(forms.Form):
    name = forms.CharField(label="Network Name", max_length=100)
    optimizer = forms.ModelChoiceField(
        label="Optimizer",
        queryset=OptimizerType.objects.all(),
    )
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
    layers = forms.ModelChoiceField(
        label="Layer",
        queryset=NetworkLayerType.objects.all(),
        required=False,
    )
    run_eagerly = forms.BooleanField(label="Run eagerly", required=False)
    steps_per_execution = forms.IntegerField(
        label="Steps per execution", required=False, initial=1
    )
    jit_compile = forms.BooleanField(label="JIT Compile", required=False)
    save_network_as_template = forms.BooleanField(
        label="Als Vorlage speichern", required=False
    )
    network_template_name = forms.CharField(label="Vorlagename", required=False)
    network_template = forms.ModelChoiceField(
        label="Vorlage",
        queryset=KerasNetworkTemplate.objects.all(),
        widget=forms.SelectMultiple(attrs={"class": "select2 w-100"}),
    )

    epochs = forms.IntegerField(
        label="Epochen",
        initial=10,
        widget=forms.TextInput(attrs={"type": "number", "min": 0}),
    )

    batch_size = forms.IntegerField(
        label="Batch size",
        initial=32,
        widget=forms.TextInput(attrs={"type": "number", "min": 0}),
    )

    shuffle = forms.BooleanField(required=False, initial=True, label="Shuffle data")
    steps_per_epoch = forms.IntegerField(
        required=False, initial=1, label="Steps per Epoch"
    )
    workers = forms.IntegerField(required=False, initial=1, label="Workers")
    use_multiprocessing = forms.BooleanField(
        required=False, initial=False, label="Use multiprocessing"
    )

    dataset = forms.ChoiceField(choices=(), required=False)
    dataset_is_supervised = forms.BooleanField(initial=True, required=False)

    extra_context = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_id = "start_new_run"
        self.helper.form_action = reverse_lazy("runs:new")
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-lg-2"
        self.helper.field_class = "col-lg-10"
        self.helper.layout = Layout(
            HTML('<div class="row mb-3"><h2>Training konfigurieren</h2></div>'),
            Field("name"),
            Field(
                "metrics",
                css_class="chosen-select select2 w-100",
                data_placeholder="Select Metrics",
                multiple="multiple",
            ),
            HTML(
                "<div id='metrics-arguments' class='card rounded-3 d-flex flex-row flex-wrap'></div>"
            ),
            Field("optimizer", css_class="select2 w-100 mt-3"),
            HTML("<div id='optimizer-arguments' class='card rounded-3'></div>"),
            # Field("loss", css_class="select2 w-100 mt-3"),
            # HTML("<div id='loss-arguments' class='card rounded-3'></div>"),
            HTML('<div class="clearfix"></div>'),
            Row(
                # Column("epochs", css_class="form-group col-6 mb-0"),
                Column(
                    Field("epochs", template="crispyForms/small_field.html"),
                    css_class="col-3",
                ),
                Column(
                    Field("batch_size", template="crispyForms/small_field.html"),
                    css_class="col-3",
                ),
                Column(Field("run_eagerly"), css_class="col-2"),
                Column(
                    Field(
                        "steps_per_execution", template="crispyForms/small_field.html"
                    ),
                    css_class="col-2",
                ),
                Column(Field("jit_compile"), css_class="col-2"),
                css_class="mt-5 pt-3 border-top",
            ),
            Row(
                # Column("epochs", css_class="form-group col-6 mb-0"),
                Column(Field("shuffle"), css_class="col-3"),
                Column(
                    Field("steps_per_epoch", template="crispyForms/small_field.html"),
                    css_class="col-3",
                ),
                Column(Field("use_multiprocessing"), css_class="col-2"),
                Column(
                    Field("workers", template="crispyForms/small_field.html"),
                    css_class="col-2",
                ),
                css_class="mt-5 pt-3",
            ),
            HTML(
                """<br>
            <div class='d-flex mb-5' id='networkgraph'>
            
                <div id="graph-container" style="height:30em; border-radius: 10px" class='bg-white mr-3 col-lg-8'>
                <h2 id='graph_header' class='m-2' >Graph</h2>
                </div>
                <div id="form-container" class='col-lg-4'>
                    <div class='row'>
                    <h2 id='node_header'>Details</h2>
                    </div>
                    <div class='row'>
                        <label class="col-form-label col-lg-2" for="existing_nodes">Select Nodes:</label>
                        <div class="col-lg-10"> 
                        <select id="existing_nodes" class='select2 w-100' multiple>
                            <!-- Populate with existing nodes -->
                            <option value="node1">Node 1</option>
                            <option value="node2">Node 2</option>
                            <!-- Add more options as needed -->
                        </select></div>
                    </div>"""
            ),
            Field("layers", css_class="select2 w-100 mt-3"),
            HTML(
                """
                    <button type="button" name='addnode' class='btn btn-primary mb-3' onclick="addNode()">
                        Ebene hinzufugen
                    </button>
                    <button type="button" name='updatenode' class='btn btn-primary mb-3 d-none' onclick="updateNode()">
                        aktualisieren
                    </button>
                    <button type="button" name='deletenode' class='btn btn-danger mb-3 d-none' onclick="deleteNode()">
                        Loschen
                    </button>
                    <div id='layer-arguments' class='card rounded-3'></div>
                    <input type='hidden' name='nodes' id='architecture_nodes'>
                    <input type='hidden' name='edges' id='architecture_edges'>
                </div>
            </div>
            """
            ),
            Row(Field("save_network_as_template")),
            Row(Field("network_template_name")),
            Row(Field("network_template")),
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
                        "dataset",
                        css_class="chosen-select select2 w-100",
                    )
                ),
                Column(Field("dataset_is_supervised"), css_class="col-3"),
                css_class="align-items-center",
            ),
            Submit("customer-general-edit", "Training starten"),
        )

        tensorflow_datasets = []
        for dataset in tfds.list_builders():
            tensorflow_datasets.append((dataset, dataset))
        self.fields["dataset"].choices = tensorflow_datasets

        self.fields["optimizer"].widget.attrs[
            "onchange"
        ] = "handleOptimizerChange(this)"
        self.fields["loss"].widget.attrs["onchange"] = "handleLossChange(this)"
        self.fields["metrics"].widget.attrs["onchange"] = "handleMetricChange(this)"
        self.fields["layers"].widget.attrs["onchange"] = "handleLayerChange(this)"

        self.fields["metrics"].widget.choices = self.get_metric_choices()
        self.fields["optimizer"].widget.choices = self.get_optimizer_choices()
        self.fields["loss"].widget.choices = self.get_loss_choices()
        self.fields["layers"].widget.choices = self.get_layer_choices()

    def load_graph(self, nodes, edges):
        self.extra_context["nodes"] = nodes
        self.extra_context["edges"] = edges

    def load_optimizer_config(self, arguments):
        self.extra_context["optimizer_config"] = arguments

    def load_loss_config(self, arguments):
        self.extra_context["loss_config"] = arguments

    def load_metric_configs(self, arguments):
        self.extra_context["metric_configs"] = arguments

    def get_extra_context(self):
        return self.extra_context

    def get_metric_choices(self):
        metric_choices = []
        modules = MetricType.objects.values_list("module_name", flat=True).distinct()

        for module in modules:
            metrics = MetricType.objects.filter(module_name=module)
            metric_choices.append(
                (module, [(metric.id, metric.name) for metric in metrics])
            )

        return metric_choices

    def get_loss_choices(self):
        loss_choices = []
        modules = LossType.objects.values_list("module_name", flat=True).distinct()

        for module in modules:
            losses = LossType.objects.filter(module_name=module)
            loss_choices.append((module, [(loss.id, loss.name) for loss in losses]))

        return loss_choices

    def get_layer_choices(self):
        layer_choices = []
        modules = NetworkLayerType.objects.values_list(
            "module_name", flat=True
        ).distinct()

        for module in modules:
            layers = NetworkLayerType.objects.filter(module_name=module)
            layer_choices.append((module, [(layer.id, layer.name) for layer in layers]))

        return layer_choices

    def get_optimizer_choices(self):
        optimizer_choices = []
        modules = OptimizerType.objects.values_list("module_name", flat=True).distinct()

        for module in modules:
            optimizers = OptimizerType.objects.filter(module_name=module)
            optimizer_choices.append(
                (module, [(optimizer.id, optimizer.name) for optimizer in optimizers])
            )

        return optimizer_choices


class NewAutoKerasRunForm(forms.Form):
    name = forms.CharField(label="Network Name", max_length=100)
    tuner = forms.ModelChoiceField(
        label="Tuner",
        required=False,
        queryset=AutoKerasTunerType.objects.all(),
        widget=forms.Select(attrs={"class": "select"}),
    )
    layers = forms.ModelChoiceField(
        label="Blocks",
        queryset=AutoKerasNodeType.objects.all(),
        required=False,
        widget=forms.Select(
            attrs={"class": "select2", "style": "width: -webkit-fill-available"}
        ),
    )
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
    callbacks = forms.ModelMultipleChoiceField(
        label="Callbacks",
        queryset=CallbackType.objects.all(),
        required=False,
        widget=forms.SelectMultiple(attrs={"class": "select2 w-100"}),
    )
    max_model_size = forms.IntegerField(
        label="Max Model size",
        required=False,
        widget=forms.TextInput(attrs={"type": "number", "min": 0}),
    )

    objective = forms.CharField(label="Objective", required=False, initial="val_loss")

    max_trials = forms.IntegerField(
        label="Max Durchlaufe",
        initial=100,
        widget=forms.TextInput(attrs={"type": "number", "min": 0}),
    )
    metric_weights = forms.CharField(
        label="Metric weights",
        required=False,
        initial="{}",
        widget=forms.Textarea(attrs={"type": "text"}),
    )

    save_network_as_template = forms.BooleanField(
        label="Als Vorlage speichern", required=False
    )
    network_template_name = forms.CharField(label="Vorlagename", required=False)
    network_template = forms.ModelChoiceField(
        label="Vorlage",
        queryset=AutoKerasNetworkTemplate.objects.all(),
        widget=forms.SelectMultiple(attrs={"class": "select2 w-100"}),
    )

    directory = forms.CharField(label="Directory", required=False)

    dataset = forms.ChoiceField(choices=(), required=False)
    dataset_is_supervised = forms.BooleanField(initial=True, required=False)

    extra_context = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_id = "start_new_run"
        self.helper.form_action = reverse_lazy("runs:new_autokeras")
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-lg-2"
        self.helper.field_class = "col-lg-10"
        self.helper.layout = Layout(
            HTML('<div class="row mb-3"><h2>Autokeras konfigurieren</h2></div>'),
            Field("name"),
            Field(
                "metrics",
                css_class="chosen-select select2 w-100",
                data_placeholder="Select Metrics",
                multiple="multiple",
            ),
            HTML(
                """
                <div id='metrics-arguments' class='card rounded-3 d-flex flex-row flex-wrap'></div>
                <div id='metric_weights_arguments' class='card rounded-3'></div>
                """
            ),
            Field(
                "loss",
                css_class="select2 w-100 mt-3",
                data_placeholder="Select Loss function",
            ),
            HTML("<div id='loss-arguments' class='card rounded-3'></div>"),
            Field(
                "callbacks",
                css_class="select2 w-100 mt-3",
                data_placeholder="Select Callbacks",
                multiple="multiple",
            ),
            HTML("<div id='callbacks-arguments' class='card rounded-3'></div>"),
            Field("tuner", css_class="select2 w-100 mt-3"),
            HTML("<div id='tuner-arguments' class='card rounded-3'></div>"),
            HTML('<div class="clearfix"></div>'),
            Row(
                # Column("epochs", css_class="form-group col-6 mb-0"),
                Column(
                    Field("max_model_size", template="crispyForms/small_field.html"),
                    css_class="col-3",
                ),
                Column(
                    Field("max_trials", template="crispyForms/small_field.html"),
                    css_class="col-3",
                ),
                Column(
                    Field("objective"),
                    css_class="col-6",
                    data_tooltip="valid options: 'val_loss', 'metrics' for weighted sum of metrics, 'model_size'  or metric_name for single metric",
                ),
                Column(
                    Field("directory"),
                    css_class="col-12",
                ),
                css_class="mt-5 pt-3 border-top",
            ),
            HTML(
                """<br>
            <div class='d-flex mb-5' id='networkgraph'>
            
                <div id="graph-container" style="height:30em; border-radius: 10px" class='bg-white mr-3 col-lg-8'>
                <h2 id='autokeras-graph_header' class='m-2' >Graph</h2>
                </div>
                <div id="form-container" class='col-lg-4'>
                    <div class='row'>
                    <h2 id='node_header'>Details</h2>
                    </div>
                    <div class='row'>
                        <label class="col-form-label col-lg-2" for="existing_nodes">Select Nodes:</label>
                        <div class="col-lg-10"> 
                        <select id="existing_nodes" class='select2 w-100' multiple>
                        </select></div>
                    </div>"""
            ),
            Field("layers", css_class="select2 mt-3"),
            HTML(
                """
                    <button type="button" name='addnode' class='btn btn-primary mb-3' onclick="addNode()">
                        Ebene hinzufugen
                    </button>
                    <button type="button" name='updatenode' class='btn btn-primary mb-3 d-none' onclick="updateNode()">
                        aktualisieren
                    </button>
                    <button type="button" name='deletenode' class='btn btn-danger mb-3 d-none' onclick="deleteNode()">
                        Loschen
                    </button>
                    <div id='layer-arguments' class='card rounded-3'></div>
                    <input type='hidden' name='nodes' id='architecture_nodes'>
                    <input type='hidden' name='edges' id='architecture_edges'>
                </div>
            </div>
            """
            ),
            Row(Field("save_network_as_template")),
            Row(Field("network_template_name")),
            Row(Field("network_template")),
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
                        "dataset",
                        css_class="chosen-select select2 w-100",
                    )
                ),
                Column(Field("dataset_is_supervised"), css_class="col-3"),
                css_class="align-items-center",
            ),
            Submit("customer-general-edit", "Training starten"),
        )

        tensorflow_datasets = []
        for dataset in tfds.list_builders():
            tensorflow_datasets.append((dataset, dataset))
        self.fields["dataset"].choices = tensorflow_datasets

        self.fields["tuner"].widget.attrs["onchange"] = "handleKerasTunerChange(this)"
        self.fields["layers"].widget.attrs["onchange"] = "handleKerasBlockChange(this)"
        self.fields["metrics"].widget.attrs["onchange"] = "handleMetricChange(this)"
        self.fields["loss"].widget.attrs["onchange"] = "handleLossChange(this)"
        self.fields["callbacks"].widget.attrs["onchange"] = "handleCallbackChange(this)"

        self.fields["loss"].widget.choices = self.get_loss_choices()
        self.fields["metrics"].widget.choices = self.get_metric_choices()
        self.fields["tuner"].widget.choices = self.get_tuner_choices()
        self.fields["layers"].widget.choices = self.get_layer_choices()
        self.fields["callbacks"].widget.choices = self.get_callbacks_choices()

    def load_graph(self, nodes, edges):
        self.extra_context["nodes"] = nodes
        self.extra_context["edges"] = edges

    def load_tuner_config(self, arguments):
        self.extra_context["tuner_config"] = arguments

    def load_metric_configs(self, arguments):
        self.extra_context["metric_configs"] = arguments

    def load_callbacks_configs(self, arguments):
        self.extra_context["callbacks_configs"] = arguments

    def load_loss_config(self, arguments):
        self.extra_context["loss_config"] = arguments

    def get_extra_context(self):
        return self.extra_context

    def get_layer_choices(self):
        layer_choices = []
        modules = AutoKerasNodeType.objects.values_list(
            "module_name", flat=True
        ).distinct()

        for module in modules:
            layers = AutoKerasNodeType.objects.filter(module_name=module)
            layer_choices.append((module, [(layer.id, layer.name) for layer in layers]))

        return layer_choices

    def get_tuner_choices(self):
        tuner_choices = []
        modules = AutoKerasTunerType.objects.values_list(
            "module_name", flat=True
        ).distinct()

        for module in modules:
            tuners = AutoKerasTunerType.objects.filter(module_name=module)
            tuner_choices.append((module, [(tuner.id, tuner.name) for tuner in tuners]))

        return tuner_choices

    def get_loss_choices(self):
        loss_choices = []
        modules = LossType.objects.values_list("module_name", flat=True).distinct()

        for module in modules:
            losses = LossType.objects.filter(module_name=module)
            loss_choices.append((module, [(loss.id, loss.name) for loss in losses]))

        return loss_choices

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
