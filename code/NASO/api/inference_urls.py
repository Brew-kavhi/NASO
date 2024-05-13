from django.urls import path

from .views import inference
from .views.metrics import MetricsAPIView

urlpatterns = [
    path("run_from_id", inference.run_from_id, name="run_from_id"),
    path("<int:pk>/metrics/", MetricsAPIView.as_view(), name="metrics"),
]
