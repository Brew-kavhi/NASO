from django.views.generic import TemplateView


class LogFileView(TemplateView):
    """
    A view that displays the content of a log file.

    Attributes:
        template_name (str): The name of the template to be rendered.

    Methods:
        get_context_data(**kwargs): Retrieves the context data for rendering the template.
    """

    template_name = "system/logfile.html"

    def get_context_data(self, **kwargs):
        """
        Retrieves the context data for rendering the template.

        Returns:
            dict: The context data containing the log content or an error message.
        """
        context = super().get_context_data(**kwargs)
        log_file_path = "net.log"  # Adjust the path to your log file
        try:
            with open(log_file_path, "r", encoding="utf-8") as log_file:
                log_content = log_file.read()

            context["log_content"] = log_content
        except FileNotFoundError:
            context["log_content"] = "Log file not found."
        return context
