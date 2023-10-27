from django.urls import path

from neural_architecture.models.Dataset import get_datasets
from runs.views.ListRuns import ListRuns, delete_autokeras_run, delete_run
from runs.views.NewRun import NewAutoKerasRun, NewRun
from runs.views.RunDetails import (AutoKerasRunDetails, RunDetails,
                                   TrainingProgress)

urlpatterns = [
    path("", ListRuns.as_view(), name="list"),
    path("<int:pk>/details/", RunDetails.as_view(), name="details"),
    path(
        "autokeras/<int:pk>/details/",
        AutoKerasRunDetails.as_view(),
        name="autokeras_details",
    ),
    path(
        "training_progress/<task_id>",
        TrainingProgress.as_view(),
        name="training_progress",
    ),
    path("new/tensorflow", NewRun.as_view(), name="new"),
    path("new/autokeras", NewAutoKerasRun.as_view(), name="new_autokeras"),
    path("delete/<int:pk>/", delete_run, name="delete"),
    path("delete/autokeras/<int:pk>/", delete_autokeras_run, name="delete_autokeras"),
    path("get_dataset/<int:pk>", get_datasets, name="get_dataset"),
]
