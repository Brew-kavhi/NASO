from django.urls import include, path

from neural_architecture.models.dataset import get_datasets
from runs.views.comparison import ComparisonView
from runs.views.list_runs import ListRuns, delete_run
from runs.views.new_run import NewRun
from runs.views.run_details import RunDetails, TrainingProgress
from runs.views.softdelete import DeletedRuns

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
    path("comparison/", ComparisonView.as_view(), name="comparison"),
    path("deletedruns/", DeletedRuns.as_view(), name="deleted_runs"),
    path(
        "autokeras/",
        include(("runs.autokeras_urls", "autokeras"), namespace="autokeras"),
    ),
    path(
        "templates/",
        include(("runs.template_urls", "templates"), namespace="templates"),
    ),
]
