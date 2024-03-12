from crispy_forms.layout import HTML, Column, Field, Layout, Row, Submit
from django import forms

from runs.forms.base import BaseRunWithCallback, ClusterableForm, PrunableForm


class RerunTrialForm(BaseRunWithCallback, PrunableForm, ClusterableForm):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper.form_id = "rerun_trial"
        self.helper.layout = Layout(
            HTML('<div class="row mb-3"><h2>Fine tune</h2></div>'),
            Field("name"),
            Field("description"),
            Row(
                Column(
                    Field("epochs"),
                ),
                Column(
                    Field("batch_size", template="crispyForms/small_field.html"),
                    css_class="col-3",
                ),
            ),
            self.metric_html(),
            self.callback_html(),
            self.dataloader_html(),
            self.get_pruning_fields(),
            self.get_clustering_fields(),
            self.gpu_field(),
            Submit("customer-general-edit", "Training starten"),
        )
        self.fields["name"].required = False
        self.fields["loss"].required = False
