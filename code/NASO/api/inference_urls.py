from django.urls import path

from .views import inference

urlpatterns = [
    path("run_from_id", inference.run_from_id, name="run_from_id"),
]
