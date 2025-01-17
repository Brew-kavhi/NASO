# Register your models here.
from django.contrib import admin

from runs.models.training import (
    CallbackFunction,
    FitParameters,
    NetworkTraining,
    TrainingMetric,
)

admin.site.register(NetworkTraining)
admin.site.register(TrainingMetric)
admin.site.register(CallbackFunction)
admin.site.register(FitParameters)
