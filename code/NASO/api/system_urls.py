from django.urls import path

from .views import system

urlpatterns = [
    path("get_logs/", system.get_logs, name="get_logs"),
    path("update_session/", system.UpdateSessionView.as_view(), name="update_session"),
]
