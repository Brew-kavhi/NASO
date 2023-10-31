"""naso URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from django.views.generic.base import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("dashboard/", include(("dashboard.urls", "dashboard"), namespace="dashboard")),
    path("", RedirectView.as_view(pattern_name="dashboard:index", permanent=True)),
    path(
        "settings/plugins/", include(("plugins.urls", "plugins"), namespace="plugins")
    ),
    path("runs/", include(("runs.urls", "runs"), namespace="runs")),
    path("system/", include(("system.urls", "system"), namespace="system")),
    path("api/", include(("api.urls", "api"), namespace="api")),
]
