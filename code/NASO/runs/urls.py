from django.urls import include, path

from neural_architecture.models.Dataset import get_datasets
from runs.views.ListRuns import ListRuns, delete_run
from runs.views.NewRun import NewRun
from runs.views.RunDetails import RunDetails, TrainingProgress

urlpatterns = [
    path("", ListRuns.as_view(), name="list"),
    path("<int:pk>/details/", RunDetails.as_view(), name="details"),
    path(
        "training_progress/<task_id>",
        TrainingProgress.as_view(),
        name="training_progress",
    ),
    path("new/tensorflow", NewRun.as_view(), name="new"),
    path("delete/<int:pk>/", delete_run, name="delete"),
    path("get_dataset/<int:pk>", get_datasets, name="get_dataset"),
    path(
        "autokeras/",
        include(("runs.autokeras_urls", "autokeras"), namespace="autokeras"),
    ),
]
