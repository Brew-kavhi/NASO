# Register your models here.
from django.contrib import admin

from runs.models.Training import CallbackFunction, NetworkTraining, TrainingMetric

admin.site.register(NetworkTraining)
admin.site.register(TrainingMetric)
admin.site.register(CallbackFunction)
