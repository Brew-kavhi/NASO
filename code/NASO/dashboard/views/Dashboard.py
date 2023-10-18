from django.contrib import messages
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils import timezone as tz
from django.views.generic.base import TemplateView
from django.views.generic.edit import CreateView, DeleteView

from naso.celery import get_tasks
from naso.models.page import PageSetup


class Dashboard(TemplateView):
    template_name = "dashboard/dashboard.html"

    page = PageSetup(title="Dashboard")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        self.context["celery"] = get_tasks()
        return self.render_to_response(self.context)
