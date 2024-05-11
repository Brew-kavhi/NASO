from os import listdir
from os.path import isfile, join

from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Column, Field, Layout, Row, Submit
from django import forms

from runs.forms.base import BaseRunWithCallback
from decouple import config


class NewInferenceForm(BaseRunWithCallback):
    load_model = forms.ChoiceField(
        widget=forms.Select(attrs={"class": "select2 w-100"}),
        label="Model auswählen",
    )
    batch_size = forms.IntegerField(
        label="Batch size",
        initial=1,
        widget=forms.TextInput(attrs={"type": "number", "min": 0}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["loss"].required = False
        self.fields["metrics"].required = False

        self.helper.layout = Layout(
            HTML('<div class="row mb-3"><h2>Inference konfigurieren</h2></div>'),
            Field("name"),
            Field("description"),
            Row(
                Column(
                    Field("load_model"),
                    css_class="col-9",
                ),
                Column(
                    Field("batch_size", template="crispyForms/small_field.html"),
                    css_class="col-3",
                ),
            ),
            self.callback_html(),
            self.dataloader_html(),
            self.gpu_field(),
            Submit("customer-general-edit", "Inference starten"),
        )
        self.fields["load_model"].choices = self.get_saved_models()

    def get_saved_models(self):
        models_path = config("TENSORFLOW_MODEL_PATH") + "tensorflow"
        keras_models_path = config("TENSORFLOW_MODEL_PATH") + "kerasModel"
        models = [
            (
                "Manual Designs",
                [
                    (join(models_path, f), f)
                    for f in listdir(models_path)
                    if isfile(join(models_path, f))
                    and (f.endswith(".h5") or f.endswith(".keras"))
                ],
            ),
            (
                "Keras Models",
                [
                    (join(keras_models_path, f), f)
                    for f in listdir(keras_models_path)
                    if isfile(join(keras_models_path, f))
                    and (f.endswith(".h5") or f.endswith(".keras"))
                ],
            ),
        ]
        return models
