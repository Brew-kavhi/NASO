from django.urls import path

from .views import comparisons

urlpatterns = [
    path("clear_session/", comparisons.clear_session, name="clear_session"),
]
