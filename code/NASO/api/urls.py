from django.urls import path
from .views import autokeras
from .views import tensorflow


urlpatterns = [
    path('metrics/<int:pk>/', tensorflow.get_metrics, name="get_metrics")
]

# autokeras api:
urlpatterns += [
    path('autokeras/metrics/<int:pk>/<int:trial_id>/', autokeras.get_metrics, name="autokeras:get_metrics")
]
