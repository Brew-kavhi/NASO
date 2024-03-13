from django.urls import path

from workers.views.list import ListWorkers

urlpatterns = [
    path("all", ListWorkers.as_view(), name="list"),
]
