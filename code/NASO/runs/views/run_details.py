from django.http import JsonResponse
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView, View

from naso.celery import get_celery_task_state
from naso.models.page import PageSetup
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import NetworkTraining


class TrainingProgress(View):
    """
    A view class for retrieving the training progress of a task.

    This class handles the GET request and returns the training progress
    of a task identified by its task_id. The progress is returned as a
    JSON response.

    Attributes:
        None

    Methods:
        get: Retrieves the training progress of a task and returns it as
             a JSON response.

    Usage:
        To use this class, create an instance of it and map it to a URL
        pattern in your Django project's URL configuration.

        Example:
        ```
        from django.urls import path
        from .views import TrainingProgress

        urlpatterns = [
            path('progress/<str:task_id>/', TrainingProgress.as_view(), name='training_progress'),
        ]
        ```
    """

    def get(self, request, task_id):
        context = get_celery_task_state(task_id)
        return JsonResponse(context, safe=False)


class RunDetails(TemplateView):
    """
    View class for displaying details of a network training run.
    """

    template_name = "runs/run_details.html"

    page = PageSetup(title="Experimente", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        """
        Handles the GET request for the run details view.

        Retrieves the network training run based on the provided primary key (pk),
        updates the context with the run details, and renders the response.

        Args:
            request (HttpRequest): The HTTP request object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            HttpResponse: The rendered response with the run details.
        """
        run = NetworkTraining.objects.get(pk=kwargs["pk"])
        self.context["run"] = run
        self.page.title = run.network_config.name
        self.page.actions = []
        self.page.add_pageaction(
            reverse_lazy("runs:new") + "?rerun=" + str(run.pk),
            "Run again",
            color="primary",
        )
        nodes = [
            {
                "id": layer.name,
                "label": f"{layer.name} ({layer.id})",
                "x": 0.0,
                "y": 1,
                "size": 3,
                "color": "#008cc2",
                "naso_type": layer.layer_type.id,
                "type": "image",
                "additional_arguments": layer.additional_arguments,
            }
            for layer in run.network_config.layers.all()
        ]
        nodes.append(
            {
                "id": "input_node",
                "label": "Input",
                "x": 0,
                "y": nodes[0]["y"] - 1,
                "size": 3,
                "color": "#008cc2",
                "type": "image",
            }
        )

        layers = [
            {
                "id": layer.layer_type.id,
                "name": layer.layer_type.name,
                "required_arguments": layer.layer_type.required_arguments,
            }
            for layer in run.network_config.layers.all()
        ]
        self.context["layers"] = layers
        self.context["nodes"] = nodes
        self.context["page"] = self.page.get_context()
        return self.render_to_response(self.context)


class AutoKerasRunDetails(TemplateView):
    """
    A view class for displaying details of an AutoKeras run.

    This class extends the `TemplateView` class and provides the necessary
    methods to render the details of an AutoKeras run to a template.

    Attributes:
        template_name (str): The name of the template to be used for rendering.
        page (PageSetup): An instance of the `PageSetup` class for setting up the page.
        context (dict): A dictionary containing the context data for rendering the template.
    """

    template_name = "runs/autokeras_run_details.html"

    page = PageSetup(title="Experimente", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        run = AutoKerasRun.objects.get(pk=kwargs["pk"])
        self.context["run"] = run
        nodes = [
            {
                "id": layer.name,
                "label": f"{layer.name} ({layer.id})",
                "x": 0.0,
                "y": 1,
                "size": 3,
                "color": "#008cc2",
                "naso_type": layer.node_type.id,
                "type": "image",
                "additional_arguments": layer.additional_arguments,
            }
            for layer in run.model.blocks.all()
        ]
        layers = [
            {
                "id": layer.node_type.id,
                "name": layer.node_type.name,
                "required_arguments": layer.node_type.required_arguments,
            }
            for layer in run.model.blocks.all()
        ]
        self.context["layers"] = layers
        self.context["nodes"] = nodes
        self.page.title = run.model.project_name
        self.page.actions = []
        self.page.add_pageaction(
            reverse_lazy("runs:autokeras:new") + "?rerun=" + str(run.pk),
            "Run again",
            color="primary",
        )
        self.context["page"] = self.page.get_context()
        return self.render_to_response(self.context)
