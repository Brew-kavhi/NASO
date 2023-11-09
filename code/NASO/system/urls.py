from django.urls import path

from system.views.log_view import LogFileView

urlpatterns = [
    path("logs/", LogFileView.as_view(), name="logs"),
]
