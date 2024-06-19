from django.urls import path

from datasets.views.list import ListDatasets

urlpatterns = [
    path("", ListDatasets.as_view(), name="list"),
]
