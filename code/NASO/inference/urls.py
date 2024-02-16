from django.urls import path

from inference.views.detail import InferenceDetail
from inference.views.list import ListInferences, delete_inference
from inference.views.new_inference import NewInference

urlpatterns = [
    path("new", NewInference.as_view(), name="new"),
    path("all", ListInferences.as_view(), name="list"),
    path("delete/<int:pk>", delete_inference, name="delete"),
    path("detail/<int:pk>", InferenceDetail.as_view(), name="details"),
]
