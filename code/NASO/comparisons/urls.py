from django.urls import include, path

from comparisons.views.details import ComparisonDetailView, SessionComparisonView
from comparisons.views.list import ComparisonListView, delete_comparison

urlpatterns = [
    path("", ComparisonListView.as_view(), name="list"),
    path("<int:pk>/", ComparisonDetailView.as_view(), name="details"),
    path("delete/<int:pk>/", delete_comparison, name="delete"),
    path("session/", SessionComparisonView.as_view(), name="session"),
]
