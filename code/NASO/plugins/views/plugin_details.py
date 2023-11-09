from django.views.generic.base import TemplateView

from naso.models.page import PageSetup


class PluginDetails(TemplateView):
    template_name = "plugins/plugin_details.html"

    page = PageSetup(title="Plugins", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        return self.render_to_response(self.context)
