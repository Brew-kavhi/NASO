from django.urls import path

from dashboard.views.Dashboard import Dashboard

urlpatterns = [
    path("", Dashboard.as_view(), name="index"),
]
