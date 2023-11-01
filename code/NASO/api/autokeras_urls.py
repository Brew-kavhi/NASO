from django.urls import path

from .views import autokeras

# autokeras api:
urlpatterns = [
    path(
        "autokeras/<int:pk>/metrics/<int:trial_id>/",
        autokeras.get_metrics,
        name="get_metrics",
    ),
    path("autokeras/<int:pk>/metrics/", autokeras.get_all_metrics, name="all_metrics"),
    path(
        "autokeras/<int:pk>/trials/", autokeras.get_trial_details, name="trial_details"
    ),
]
