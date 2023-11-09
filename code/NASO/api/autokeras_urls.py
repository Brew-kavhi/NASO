from django.urls import path

from .views import autokeras

# autokeras api:
urlpatterns = [
    path(
        "<int:pk>/metrics/<str:trial_id>/",
        autokeras.get_metrics,
        name="get_metrics",
    ),
    path(
        "<int:pk>/metrics/<str:trial_id>/download",
        autokeras.download_metrics,
        name="download_metrics",
    ),
    path("<int:pk>/metrics/", autokeras.get_all_metrics, name="all_metrics"),
    path("<int:pk>/trials/", autokeras.get_trial_details, name="trial_details"),
]