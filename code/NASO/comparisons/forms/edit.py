from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Field, Layout, Submit
from django import forms

from runs.models.training import NetworkTraining
from inference.models.inference import Inference


class AddRunForm(forms.Form):
    run = forms.ModelMultipleChoiceField(
        required=False,
        label="Netzwerk",
        widget=forms.SelectMultiple(attrs={"class": "select2 form-control w-100"}),
        queryset=NetworkTraining.objects.all(),
    )

    inference = forms.ModelChoiceField(
        required=False,
        label="inference",
        widget=forms.Select(attrs={"class": "select-2 form-control w-100"}),
        queryset=Inference.objects.all(),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_id = "save_comparison"
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-lg-2"
        self.helper.field_class = "col-lg-10"
        self.helper.layout = Layout(
            HTML('<div class="row mb-3"><h2>Vergleich hinzufügen</h2></div>'),
            Field("run"),
            Field("inference"),
            Submit("Save", "speichern"),
        )
        self.fields["run"].widget.choices = self.get_runs()

    def get_runs(self):
        runs_choices = []
        runs = NetworkTraining.objects.all()

        for run in runs:
            runs_choices.append((run.id, str(run)))

        return runs_choices
