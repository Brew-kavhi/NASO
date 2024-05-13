# Register your models here.
from django.contrib import admin

from neural_architecture.models.architecture import NetworkConfiguration
from neural_architecture.models.autokeras import (
    AutoKerasModel,
    AutoKerasNode,
    AutoKerasRun,
)
from neural_architecture.models.dataset import DatasetLoader
from neural_architecture.models.templates import (
    AutoKerasNetworkTemplate,
    KerasNetworkTemplate,
)
from neural_architecture.models.types import (
    AutoKerasNodeType,
    AutoKerasTunerType,
    CallbackType,
    LossType,
    MetricType,
    NetworkLayerType,
    OptimizerType,
)

admin.site.register(OptimizerType)
admin.site.register(LossType)
admin.site.register(MetricType)
admin.site.register(NetworkLayerType)
admin.site.register(AutoKerasNodeType)
admin.site.register(AutoKerasNode)
admin.site.register(CallbackType)
admin.site.register(AutoKerasRun)
admin.site.register(AutoKerasNetworkTemplate)
admin.site.register(AutoKerasModel)
admin.site.register(KerasNetworkTemplate)
admin.site.register(AutoKerasTunerType)
admin.site.register(NetworkConfiguration)
admin.site.register(DatasetLoader)
