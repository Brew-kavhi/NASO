from django.urls import path

from runs.views.list_runs import delete_autokeras_run
from runs.views.new_run import NewAutoKerasRun
from runs.views.run_details import AutoKerasRunDetails
from runs.views.trial import TrialView

urlpatterns = [
    path(
        "<int:pk>/details/",
        AutoKerasRunDetails.as_view(),
        name="details",
    ),
    path("new/", NewAutoKerasRun.as_view(), name="new"),
    path("delete/<int:pk>/", delete_autokeras_run, name="delete"),
    path("<int:run_id>/trial/<str:trial_id>", TrialView.as_view(), name="trial"),
]
