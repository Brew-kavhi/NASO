from os import listdir
from os.path import isfile, join

from crispy_forms.layout import HTML, Column, Field, Layout, Row, Submit
from django import forms
from django.urls import reverse_lazy

from neural_architecture.models.autokeras import AutoKerasNodeType, AutoKerasTunerType
from neural_architecture.models.templates import (
    AutoKerasNetworkTemplate,
    KerasNetworkTemplate,
)
from neural_architecture.models.types import NetworkLayerType, OptimizerType
from runs.forms.base import BaseRunWithCallback, PrunableForm


class NewRunForm(BaseRunWithCallback, PrunableForm):
    optimizer = forms.ModelChoiceField(
        label="Optimizer",
        queryset=OptimizerType.objects.all(),
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
    network_template = forms.ModelChoiceField(
        label="Vorlage",
        queryset=KerasNetworkTemplate.objects.all(),
        widget=forms.Select(attrs={"class": "select2 w-100"}),
        required=False,
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
    steps_per_epoch = forms.IntegerField(required=False, label="Steps per Epoch")
    workers = forms.IntegerField(required=False, initial=1, label="Workers")
    use_multiprocessing = forms.BooleanField(
        required=False, initial=False, label="Use multiprocessing"
    )

    save_model = forms.BooleanField(required=False, initial=False, label="Save model")
    fine_tune_saved_model = forms.BooleanField(
        required=False, initial=False, label="Load saved model from disk"
    )
    load_model = forms.ChoiceField(
        required=False,
        widget=forms.Select(attrs={"class": "select2 w-100"}),
        label="Model auswählen",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.helper.layout = Layout(
            HTML('<div class="row mb-3"><h2>Training konfigurieren</h2></div>'),
            Field("name"),
            Field("description"),
            self.metric_html(),
            self.callback_html(),
            Field("optimizer", css_class="select2 w-100 mt-3"),
            HTML("<div id='optimizer-arguments' class='card rounded-3'></div>"),
            self.loss_html(),
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
            Row(
                Column(Field("save_network_as_template")),
                Column(Field("network_template_name")),
                Column(Field("network_template")),
            ),
            self.dataloader_html(),
            self.gpu_field(),
            self.get_pruning_fields(),
            Row(
                Column(Field("save_model"), css_class="col-2"),
                Column(
                    Field("fine_tune_saved_model"), css_class="col-3", id="fine_tune"
                ),
                Column(Field("load_model"), css_class="d-none col-7", id="load_model"),
            ),
            Submit("customer-general-edit", "Training starten"),
        )

        self.fields["optimizer"].widget.attrs[
            "onchange"
        ] = "handleOptimizerChange(this)"

        self.fields["layers"].widget.attrs["onchange"] = "handleLayerChange(this)"

        self.fields["optimizer"].widget.choices = self.get_optimizer_choices()
        self.fields["layers"].widget.choices = self.get_layer_choices()
        self.fields["load_model"].choices = self.get_saved_models()

    def rerun_saved_model(self):
        """
        This function needs to be called, when the rerun parameter is set to true and
        the model to be rerun has been saved.
        Then we want to provide a checkbox to the user, which allows him to load the
        model from file.
        """
        self.helper.layout[-2][1].css_class = "d-block"

    def load_optimizer_config(self, arguments):
        self.extra_context["optimizer_config"] = arguments

    def get_saved_models(self):
        models_path = "keras_models/tensorflow"
        models = [
            (join(models_path, f), f)
            for f in listdir(models_path)
            if isfile(join(models_path, f)) and f.endswith(".h5")
        ]
        return models

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


class NewAutoKerasRunForm(BaseRunWithCallback, PrunableForm):
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
    max_model_size = forms.IntegerField(
        label="Max Model size",
        required=False,
        widget=forms.TextInput(attrs={"type": "number", "min": 0}),
    )
    max_epochs = forms.IntegerField(
        label="Max Epochs",
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

    network_template = forms.ModelChoiceField(
        label="Vorlage",
        queryset=AutoKerasNetworkTemplate.objects.all(),
        widget=forms.Select(attrs={"class": "select2 w-100"}),
        required=False,
    )

    directory = forms.CharField(label="Directory", required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper.form_action = reverse_lazy("runs:autokeras:new")

        self.helper.layout = Layout(
            HTML('<div class="row mb-3"><h2>Autokeras konfigurieren</h2></div>'),
            Field("name"),
            Field("description"),
            self.metric_html(),
            self.loss_html(),
            self.callback_html(),
            Field(
                "tuner", data_placeholder="Select Tuner", css_class="select2 w-100 mt-3"
            ),
            HTML("<div id='tuner-arguments' class='card rounded-3'></div>"),
            HTML('<div class="clearfix"></div>'),
            Row(
                # Column("epochs", css_class="form-group col-6 mb-0"),
                Column(
                    Field("max_model_size", template="crispyForms/small_field.html"),
                    css_class="col-2",
                ),
                Column(
                    Field("max_trials", template="crispyForms/small_field.html"),
                    css_class="col-2",
                ),
                Column(
                    Field("max_epochs", template="crispyForms/small_field.html"),
                    css_class="col-2",
                ),
                Column(
                    Field("objective"),
                    css_class="col-6",
                ),
                Column(
                    Field("directory"),
                    css_class="col-12",
                ),
                css_class="mt-5 pt-3 border-top",
            ),
            HTML(
                """<div id='metric_weights_arguments' class='card rounded-3 d-none'></div>
                <br>
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
            Row(
                Column(Field("save_network_as_template")),
                Column(Field("network_template_name")),
                Column(Field("network_template")),
            ),
            self.dataloader_html(),
            self.get_pruning_fields(),
            self.gpu_field(),
            Submit("customer-general-edit", "Training starten"),
        )

        self.fields["tuner"].widget.attrs["onchange"] = "handleKerasTunerChange(this)"
        self.fields["layers"].widget.attrs["onchange"] = "handleKerasBlockChange(this)"

        self.fields["tuner"].widget.choices = self.get_tuner_choices()
        self.fields["layers"].widget.choices = self.get_layer_choices()

    def load_tuner_config(self, arguments):
        self.extra_context["tuner_config"] = arguments

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
