# forms.py
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit
from django import forms

from plugins.models.plugins import Plugin


class PluginForm(forms.Form):
    file = forms.FileField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            "file",
            Submit("submit", "Upload Plugin", css_class="btn-primary"),
        )

    def clean_file(self):
        uploaded_file = self.cleaned_data.get("file")
        if uploaded_file:
            # Check if the uploaded file is a zip file
            if not uploaded_file.name.endswith(".zip"):
                raise forms.ValidationError("Only ZIP files are allowed.")
        return uploaded_file
