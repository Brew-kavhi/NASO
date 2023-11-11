from crispy_forms.layout import HTML, Field, Layout, Submit
from django import forms

from runs.forms.base import BaseRunWithCallback, PrunableForm


class RerunTrialForm(BaseRunWithCallback, PrunableForm):
    epochs = forms.IntegerField(
        label="Epochen",
        initial=10,
        widget=forms.TextInput(attrs={"type": "number", "min": 0}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper.form_id = "rerun_trial"
        self.helper.layout = Layout(
            HTML('<div class="row mb-3"><h2>Fine tune</h2></div>'),
            Field("epochs"),
            self.metric_html(),
            self.callback_html(),
            self.dataloader_html(),
            # TODO self.get_pruning_fields(), not working yet
            Submit("customer-general-edit", "Training starten"),
        )
        self.fields["name"].required = False
        self.fields["loss"].required = False
