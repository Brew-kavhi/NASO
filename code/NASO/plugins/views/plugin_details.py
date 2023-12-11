from django.views.generic.base import TemplateView

from naso.models.page import PageSetup


class PluginDetails(TemplateView):
    """
    A view class for displaying plugin details.

    Inherits from TemplateView and renders the "plugins/plugin_details.html" template.
    """

    template_name = "plugins/plugin_details.html"

    page = PageSetup(title="Plugins", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        """
        Handles GET requests and returns the rendered template with the context.

        Args:
            request: The HTTP request object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The rendered template with the context.
        """
        return self.render_to_response(self.context)
