# Register your models here.
from django.contrib import admin

from neural_architecture.models.AutoKeras import (AutoKerasNodeType,
                                                  AutoKerasRun)
from neural_architecture.models.Types import (LossType, MetricType,
                                              NetworkLayerType, OptimizerType)

admin.site.register(OptimizerType)
admin.site.register(LossType)
admin.site.register(MetricType)
admin.site.register(NetworkLayerType)
admin.site.register(AutoKerasNodeType)
admin.site.register(AutoKerasRun)
