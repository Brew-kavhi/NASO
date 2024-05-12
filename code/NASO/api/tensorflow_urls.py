from django.urls import path

from .views import tensorflow

urlpatterns = [
    path("<int:pk>/rate/", tensorflow.rate_run, name="rate_run"),
    path("<int:pk>/hardelete/", tensorflow.hard_delete, name="delete_run"),
    path("<int:pk>/undelete/", tensorflow.undelete, name="undelete_run"),
    path("<int:pk>/metrics/", tensorflow.MetricAPIView.as_view(), name="metrics"),
]
