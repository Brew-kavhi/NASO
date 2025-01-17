from django.urls import path

from runs.views.templates import (
    AutoKerasTemplateDetails,
    AutoKerasTemplateNew,
    TemplateDetails,
    TemplateList,
    TemplateNew,
    delete_template,
)

urlpatterns = [
    path("keras/<int:pk>", TemplateDetails.as_view(), name="details"),
    path(
        "autokeras/<int:pk>",
        AutoKerasTemplateDetails.as_view(),
        name="autokeras_details",
    ),
    path("<int:pk>/delete", delete_template, name="delete"),
    path("", TemplateList.as_view(), name="list"),
    path("new", TemplateNew.as_view(), name="new"),
    path("new/autokeras", AutoKerasTemplateNew.as_view(), name="new_autokeras"),
]
