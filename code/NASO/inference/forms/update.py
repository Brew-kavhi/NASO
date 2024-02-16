from django import forms

from inference.models.inference import Inference


class UpdateInference(forms.ModelForm):
    name = forms.CharField(label=("Bezeichnung"), required=False)

    description = forms.CharField(
        widget=forms.Textarea, label=("Beschreibung"), required=False
    )

    class Meta:
        model = Inference
        fields = ["description", "name"]

    def __init__(self, *args, **kwargs):
        inference = kwargs.get("instance")
        super().__init__(*args, **kwargs)
        if len(args) == 1:
            self.fields["name"].initial = args[0]["name"]
            self.fields["description"].initial = args[0]["description"]
        else:
            self.fields["name"].initial = inference.name
            self.fields["description"].initial = inference.description
