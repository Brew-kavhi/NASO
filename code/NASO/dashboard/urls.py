from django.urls import path

from dashboard.views.dashboard import Dashboard, kill_task

urlpatterns = [
    path("", Dashboard.as_view(), name="index"),
    path("kill_task/<str:task_id>", kill_task, name="kill_celery_task"),
]
