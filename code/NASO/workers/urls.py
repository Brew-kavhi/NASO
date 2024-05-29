from django.urls import path

from workers.views.actions import delete_worker
from workers.views.list import ListWorkers

urlpatterns = [
    path("all", ListWorkers.as_view(), name="list"),
    path("delete/<int:pk>/", delete_worker, name="delete"),
]
