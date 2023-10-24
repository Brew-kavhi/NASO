from django.views.generic.base import TemplateView

from naso.celery import get_tasks
from naso.models.page import PageSetup


class Dashboard(TemplateView):
    template_name = "dashboard/dashboard.html"

    page = PageSetup(title="Dashboard")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        self.context["celery"] = get_tasks()
        return self.render_to_response(self.context)
