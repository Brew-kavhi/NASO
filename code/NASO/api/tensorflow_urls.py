from django.urls import path

from .views import tensorflow
from .views.metrics import TensorflowMetricAPIView

urlpatterns = [
    path("<int:pk>/rate/", tensorflow.rate_run, name="rate_run"),
    path("<int:pk>/hardelete/", tensorflow.hard_delete, name="delete_run"),
    path("<int:pk>/undelete/", tensorflow.undelete, name="undelete_run"),
    path(
        "<int:pk>/metrics/",
        TensorflowMetricAPIView.as_view(),
        kwargs={"is_prediction": 0},
        name="metrics",
    ),
    path(
        "<int:pk>/metrics/<int:is_prediction>/",
        TensorflowMetricAPIView.as_view(),
        name="metrics",
    ),
]
