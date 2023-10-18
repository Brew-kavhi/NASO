from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Column, Field, Layout, Row, Submit
from crispy_forms.utils import TEMPLATE_PACK
from django import forms
from django.core.exceptions import ValidationError
from django.urls import reverse_lazy

from neural_architecture.models.Types import (LossType, MetricType,
                                              NetworkLayerType, OptimizerType)


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
        widget=forms.SelectMultiple(attrs={"class": "select2"}),
    )
    layers = forms.ModelChoiceField(
        label="Layer",
        queryset=NetworkLayerType.objects.all(),
        required=False,
        widget=forms.Select(attrs={"class": "select2"}),
    )
    run_eagerly = forms.BooleanField(label="Run eagerly", required=False)
    steps_per_execution = forms.IntegerField(
        label="Steps per execution", required=False, initial=1
    )
    jit_compile = forms.BooleanField(label="JIT Compile", required=False)

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
            Field("loss", css_class="select2 w-100 mt-3"),
            HTML("<div id='loss-arguments' class='card rounded-3'></div>"),
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
                    <button type="button" name='addnode' class='btn btn-primary mb-3' onclick="addNode()">Ebene hinzufugen</button>
                    <button type="button" name='updatenode' class='btn btn-primary mb-3 d-none' onclick="updateNode()">aktualisieren</button>
                    <button type="button" name='deletenode' class='btn btn-danger mb-3 d-none' onclick="deleteNode()">Loschen</button>
                    <div id='layer-arguments' class='card rounded-3'></div>
                    <input type='hidden' name='nodes' id='architecture_nodes'>
                    <input type='hidden' name='edges' id='architecture_edges'>
                </div>
            </div>
            """
            ),
            Submit("customer-general-edit", "Training starten"),
        )

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
        self.extra_context['nodes'] = nodes
        self.extra_context['edges'] = edges

    def load_optimizer_config(self, arguments):
        self.extra_context['optimizer_config'] = arguments

    def load_loss_config(self, arguments):
        self.extra_context['loss_config'] = arguments

    def load_metric_configs(self, arguments):
        print(arguments)
        self.extra_context['metric_configs'] = arguments

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
