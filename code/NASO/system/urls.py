from django.urls import path

from system.views.LogView import LogFileView

urlpatterns = [
    path("logs/", LogFileView.as_view(), name="logs"),
]
