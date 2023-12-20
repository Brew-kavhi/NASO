from crispy_forms.helper import FormHelper
from crispy_forms.layout import Field, Layout, Row, Column, Submit
from django import forms
from runs.models.training import NetworkTraining
from neural_architecture.models.autokeras import AutoKerasRun


class UpdateRun(forms.ModelForm):
    name = forms.CharField(label=("Bezeichnung"), required=False)

    description = forms.CharField(
        widget=forms.Textarea, label=("Beschreibung"), required=False
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_id = "update_run"
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-lg-2"
        self.helper.field_class = "col-lg-10"
        self.helper.layout = Layout(
            Row(
                Column(
                    Field("description"),
                ),
            ),
            Row(
                Column(Field("name"), css_class="col-8"),
                Column(
                    Submit("submit", "Speichern", css_class="btn-primary"),
                    css_class="col",
                ),
                css_class="justify-content-between",
            ),
        )


class UpdateNetworkTrainingRun(UpdateRun):
    class Meta:
        model = NetworkTraining
        fields = ["description"]

    def __init__(self, *args, **kwargs):
        run = kwargs.get("instance")
        super().__init__(*args, **kwargs)
        if len(args) == 1:
            self.fields["name"].initial = args[0]["name"]
            self.fields["description"].initial = args[0]["description"]
        else:
            self.fields["name"].initial = run.network_config.name
            self.fields["description"].initial = run.description

    def save(self, *args, **kwargs):
        run = super().save(*args, **kwargs)
        run.network_config.name = self.cleaned_data["name"]
        run.description = self.cleaned_data["description"]
        run.save()
        run.network_config.save()
        return run


class UpdateAutokerasRun(UpdateRun):
    class Meta:
        model = AutoKerasRun
        fields = ["description"]

    def __init__(self, *args, **kwargs):
        run = kwargs.get("instance")
        super().__init__(*args, **kwargs)
        if len(args) == 1:
            self.fields["name"].initial = args[0]["name"]
            self.fields["description"].initial = args[0]["description"]
        else:
            self.fields["name"].initial = run.model.project_name
            self.fields["description"].initial = run.description

    def save(self, commit=True):
        run = super().save(commit=False)
        run.model.project_name = self.cleaned_data["name"]
        run.description = self.cleaned_data["description"]
        run.model.save()
        run.save()
        return run
