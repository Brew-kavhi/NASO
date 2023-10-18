from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils import timezone as tz
from django.views.generic.base import TemplateView
from django.views.generic.edit import CreateView, DeleteView

from naso.models.page import PageSetup


class PluginDetails(TemplateView):
    template_name = "plugins/plugin_details.html"

    page = PageSetup(title="Plugins", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        return self.render_to_response(self.context)
