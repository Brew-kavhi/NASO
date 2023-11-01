from django.urls import path

from .views import tensorflow

urlpatterns = [path("metrics/<int:pk>/", tensorflow.get_metrics, name="get_metrics")]
