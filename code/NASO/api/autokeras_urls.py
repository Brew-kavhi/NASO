from django.urls import path

from .views import autokeras

# autokeras api:
urlpatterns = [
    path(
        "<int:pk>/metrics/<str:trial_id>/",
        autokeras.get_metrics_json,
        name="get_metrics",
    ),
    path(
        "<int:pk>/metrics/<str:trial_id>/download",
        autokeras.download_metrics,
        name="download_metrics",
    ),
    path("<int:pk>/metrics/", autokeras.get_all_metrics, name="all_metrics"),
    path("<int:pk>/trials/", autokeras.get_all_trial_details, name="trial_details"),
    path(
        "<int:pk>/metricdata/",
        autokeras.get_metrics_for_run,
        name="get_metrics_for_run",
    ),
    path("<int:pk>/hardelete/", autokeras.hard_delete, name="delete_run"),
    path("<int:pk>/undelete/", autokeras.undelete, name="undelete_run"),
    path("<int:pk>/rate/", autokeras.rate_run, name="rate_run"),
]
