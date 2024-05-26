from django.urls import path

from .views import autokeras

# autokeras api:
urlpatterns = [
    path(
        "<int:pk>/metrics/<str:trial_id>/",
        autokeras.MetricAPIView.as_view(),
        kwargs={"is_prediction": 0},
        name="get_metrics",
    ),
    path(
        "<int:pk>/metrics/<str:trial_id>/<int:is_prediction>/",
        autokeras.MetricAPIView.as_view(),
        name="get_metrics",
    ),
    # same as above, just as downloadable CSV
    path(
        "<int:pk>/metrics/<str:trial_id>/download",
        autokeras.download_metrics,
        name="download_metrics",
    ),
    # all metrics of run including the final ones. sorted by trial and epoch
    path("<int:pk>/metrics/", autokeras.get_all_metrics, name="all_metrics"),
    # just the inference metrics of all trials
    path(
        "<int:pk>/inferencemetrics/",
        autokeras.get_inference_metrics,
        name="inference_metrics",
    ),
    # just the final metrics of all trials
    path("<int:pk>/finalmetrics/", autokeras.get_final_metrics, name="final_metrics"),
    # only gets extremum values for all trials
    path(
        "<int:pk>/metrics_short/",
        autokeras.get_trial_details_short,
        name="metrics_short",
    ),
    # gets the configuration details of all trials
    path("<int:pk>/trials/", autokeras.get_all_trial_details, name="trial_details"),
    # should be deprecated, all_metrics
    path(
        "<int:pk>/metricdata/",
        autokeras.get_metrics_for_run,
        name="get_metrics_for_run",
    ),
    path("<int:pk>/hardelete/", autokeras.hard_delete, name="delete_run"),
    path("<int:pk>/undelete/", autokeras.undelete, name="undelete_run"),
    path("<int:pk>/rate/", autokeras.rate_autokeras_run, name="rate_run"),
    path(
        "<int:pk>/configuration/", autokeras.get_configuration, name="get_configuration"
    ),
]
