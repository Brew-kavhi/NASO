from django.views.generic.base import TemplateView

from naso.models.page import PageSetup
from plugins.models.plugins import Plugin


class PluginList(TemplateView):
    template_name = "plugins/plugin_list.html"

    page = PageSetup(title="Plugins", description="Liste")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        plugins = Plugin.objects.all()
        self.context["plugins"] = plugins
        return self.render_to_response(self.context)
