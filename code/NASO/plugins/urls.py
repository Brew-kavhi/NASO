from django.urls import path

from plugins.views.NewPlugin import NewPlugin
from plugins.views.PluginDetails import PluginDetails
from plugins.views.PluginList import PluginList

urlpatterns = [
    path("", PluginList.as_view(), name="list"),
    path("<int:pk>/details", PluginDetails.as_view(), name="details"),
    path("new/", NewPlugin.as_view(), name="new"),
]
