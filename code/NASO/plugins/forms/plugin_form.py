# forms.py
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit
from django import forms

from plugins.models.plugins import Plugin


class PluginForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            "python_file",
            "config_file",
            Submit("submit", "Upload Plugin", css_class="btn-primary"),
        )

    class Meta:
        model = Plugin
        fields = [
            "python_file",
            "config_file",
        ]
