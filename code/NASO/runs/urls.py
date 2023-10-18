from django.urls import path

from runs.views.ListRuns import ListRuns, delete_run
from runs.views.NewRun import NewRun
from runs.views.RunDetails import RunDetails, TrainingProgress

urlpatterns = [
    path("", ListRuns.as_view(), name="list"),
    path("<int:pk>/details", RunDetails.as_view(), name="details"),
    path(
        "training_progress/<task_id>",
        TrainingProgress.as_view(),
        name="training_progress",
    ),
    path("new/", NewRun.as_view(), name="new"),
    path("delete/<int:pk>/", delete_run, name="delete"),
]
