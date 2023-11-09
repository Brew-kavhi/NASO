from django.urls import include, path

urlpatterns = [
    path(
        "autokeras/",
        include(("api.autokeras_urls", "autokeras"), namespace="autokeras"),
    ),
    path(
        "tensorflow/",
        include(("api.tensorflow_urls", "tensorflow"), namespace="tensorflow"),
    ),
]