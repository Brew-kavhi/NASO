from django.core.exceptions import ValidationError
from django.db import models
from .Types import BaseType, TypeInstance
import autokeras


# This handles all python classses.
# that is in these types i just want to save what optimizers are availabel and how to call these classes
# instantiation with all arguments is done by the actual models, that jsut have this type assigned
class AutoKerasNodeType(BaseType):
    keras_type = models.CharField(max_length = 100)
    

class AutoKerasNode(TypeInstance):
    node_type = models.ForeignKey(AutoKerasNodeType, on_delete=models.deletion.CASCADE)

class AutoKerasTuner(models.Model):
    name = models.CharField(max_length = 100)
    module_name = models.CharField(max_length = 200, default = None, null = True)
    native_tuner = models.BooleanField(default = True)

    def save(self, *args, **kwargs):
        if self.name not in ['greedy', 'bayesian', 'hyperband', 'random']:
            self.native_tuner = False
            if not self.module_name or len(self.module_name):
                raise ValidationError(
                    "If the tuner is not a native AutoKeras Tuner we need a class to import the tuner from, module_name cannot be empty."
                )
        else:
            self.module_name = None
            self.native_tuner = True
        super(AutoKerasTuner, self).save(*args, **kwargs)


class AutoKerasModel(models.Model):
    project_name = models.CharField(max_length = 100, default='auto_model')
    inputs = models.ManyToManyField(AutoKerasNode, related_name = 'Inputs')
    outputs = models.ManyToManyField(AutoKerasNode, related_name = 'Outputs')
    max_trials = models.IntegerField(default=100)
    directory = models.CharField(max_length = 100, null = True, default = None)
    objective = models.CharField(max_length = 100, default='val_loss')
    tuner = models.ForeignKey(AutoKerasTuner, null = True, on_delete = models.deletion.SET_NULL)
    max_model_size = models.IntegerField(null = True)

    auto_model: autokeras.AutoModel = None

    def build_model(self):
        # build the model here:
        self.auto_model = autokeras.AutoModel()

    # calls the fit method of the autokeras model
    def fit(self, *args, **kwargs):
        if not self.auto_model:
            self.build_model(self)
        self.auto_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        if not self.auto_model:
            self.build_model(self)
        self.auto_model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if not self.auto_model:
            self.build_model(self)
        self.auto_model.evaluate(*args, **kwargs)