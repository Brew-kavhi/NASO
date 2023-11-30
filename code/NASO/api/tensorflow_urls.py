from django.urls import path

from .views import tensorflow

urlpatterns = [
    path(
        "<int:pk>/metrics/", tensorflow.get_metrics_for_run, name="get_metrics_for_run"
    ),
    path("<int:pk>/rate/", tensorflow.rate_run, name="rate_run"),
]
