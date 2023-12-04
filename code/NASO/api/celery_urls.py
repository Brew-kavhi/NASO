from django.urls import path

from .views import celery

urlpatterns = [
    path("get_workers/", celery.get_workers, name="get_worker"),
    path("task_details/<str:task_id>", celery.get_task_details, name="task_details"),
    path("restart_worker/<str:worker>", celery.restart_worker, name="restart_worker"),
]
