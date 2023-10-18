from django.views.generic import TemplateView


class LogFileView(TemplateView):
    template_name = "system/logfile.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        log_file_path = "net.log"  # Adjust the path to your log file
        try:
            with open(log_file_path, "r") as log_file:
                log_content = log_file.read()

            context["log_content"] = log_content
        except FileNotFoundError:
            context["log_content"] = "Log file not found."
        return context
