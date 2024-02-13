from django.urls import path

from .views import comparisons

urlpatterns = [
    path("clear_session/", comparisons.clear_session, name="clear_session"),
    path(
        "remove_run/<int:comparison_id>/<str:run_id>",
        comparisons.remove_run,
        name="remove_run",
    ),
]
