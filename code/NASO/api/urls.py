from django.urls import include, path
from rest_framework.authtoken import views

urlpatterns = [
    path(
        "autokeras/",
        include(("api.autokeras_urls", "autokeras"), namespace="autokeras"),
    ),
    path(
        "tensorflow/",
        include(("api.tensorflow_urls", "tensorflow"), namespace="tensorflow"),
    ),
    path(
        "system/",
        include(("api.system_urls", "system"), namespace="system"),
    ),
    path("api-token-auth/", views.obtain_auth_token),
]
