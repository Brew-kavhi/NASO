# Register your models here.
from django.contrib import admin

from neural_architecture.models.AutoKeras import (AutoKerasModel,
                                                  AutoKerasNodeType,
                                                  AutoKerasRun)
from neural_architecture.models.Templates import (AutoKerasNetworkTemplate,
                                                  KerasNetworkTemplate)
from neural_architecture.models.Types import (CallbackType, LossType,
                                              MetricType, NetworkLayerType,
                                              OptimizerType)

admin.site.register(OptimizerType)
admin.site.register(LossType)
admin.site.register(MetricType)
admin.site.register(NetworkLayerType)
admin.site.register(AutoKerasNodeType)
admin.site.register(CallbackType)
admin.site.register(AutoKerasRun)
admin.site.register(AutoKerasNetworkTemplate)
admin.site.register(AutoKerasModel)
