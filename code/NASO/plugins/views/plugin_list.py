from django.views.generic.base import TemplateView

from naso.models.page import PageSetup
from plugins.models.plugins import Plugin


class PluginList(TemplateView):
    """
    View class for displaying a list of plugins.

    Attributes:
        template_name (str): The name of the template to be rendered.
        page (PageSetup): An instance of the PageSetup class for setting up the page.
        context (dict): The context data to be passed to the template.

    Methods:
        get(self, request, *args, **kwargs): Handles GET requests and returns the rendered template.
    """

    template_name = "plugins/plugin_list.html"

    page = PageSetup(title="Plugins", description="Liste")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        """
        Handles GET requests and returns the rendered template.

        Args:
            request (HttpRequest): The HTTP request object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            HttpResponse: The rendered template with the context data.
        """
        plugins = Plugin.objects.all()
        self.context["plugins"] = plugins
        return self.render_to_response(self.context)
