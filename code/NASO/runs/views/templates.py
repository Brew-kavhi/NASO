import json
import abc
from django.views.generic.base import TemplateView
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect

from naso.models.page import PageSetup
from neural_architecture.models.architecture import NetworkLayerType, NetworkLayer
from neural_architecture.models.autokeras import AutoKerasNodeType, AutoKerasNode
from neural_architecture.models.templates import (
    AutoKerasNetworkTemplate,
    KerasNetworkTemplate,
)


class TemplateDetails(TemplateView):
    """
    A view for displaying and managing details of a network template.

    This view provides methods for retrieving a template, building nodes and layers,
    and handling GET and POST requests for displaying and saving template details.

    Attributes:
        template_name (str): The name of the template's HTML file.
        page (PageSetup): An instance of the PageSetup class for setting up the page.
        context (dict): A dictionary containing the context data for rendering the template.

    Methods:
        get_template(**kwargs): Abstract method for retrieving the template.
        build_nodes_and_layers(template): Abstract method for building nodes and layers.
        node_to_layers(template, layers): Abstract method for mapping nodes to layers.
        get(request, *args, **kwargs): Handler for GET requests.
        post(request, *args, **kwargs): Handler for POST requests.
    """

    template_name = "templates/details.html"

    page = PageSetup(title="NetworkTemplate", description="Details")
    context = {"page": page.get_context(), "editable": True}

    @abc.abstractmethod
    def get_template(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def build_nodes_and_layers(self, template):
        raise NotImplementedError

    @abc.abstractmethod
    def node_to_layers(self, template, layers):
        raise NotImplementedError

    def get(self, request, *args, **kwargs):
        """
        Handle GET requests for displaying template details.

        This method retrieves the template, builds nodes and layers, and renders the template
        with the context data.

        Args:
            request (HttpRequest): The HTTP request object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            HttpResponse: The rendered template response.
        """
        template = self.get_template(**kwargs)
        (nodes, layers) = self.build_nodes_and_layers(template)
        self.context["layers"] = layers
        self.context["nodes"] = nodes
        self.context["template"] = template
        return self.render_to_response(self.context)

    def post(self, request, *args, **kwargs):
        """
        Handle POST requests for saving template details.

        This method retrieves the template, updates its name and connections based on the
        POST data, saves the template, and renders the template with the updated context data.

        Args:
            request (HttpRequest): The HTTP request object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            HttpResponse: The rendered template response.
        """
        template = self.get_template(**kwargs)
        template.name = request.POST.get("name")
        layers = json.loads(request.POST.get("nodes"))
        edges = json.loads(request.POST.get("edges"))
        template.connections = edges

        self.node_to_layers(template, layers)
        messages.add_message(
            request, messages.SUCCESS, "Template wurde erfolgreich gespeichert"
        )
        template.save()
        (nodes, layer_options) = self.build_nodes_and_layers(template)
        self.context["nodes"] = nodes
        self.context["layers"] = layer_options
        self.context["template"] = template
        return self.render_to_response(self.context)


class AutoKerasTemplateDetails(TemplateDetails):
    """
    A class representing the details of an AutoKeras template.

    This class provides methods to retrieve the template, build nodes and layers,
    and map nodes to layers.

    Attributes:
        None

    Methods:
        get_template: Retrieves the AutoKerasNetworkTemplate object based on the provided primary key.
        build_nodes_and_layers: Builds the nodes and layers based on the template.
        node_to_layers: Maps the nodes to layers based on the provided layers.

    """

    def get_template(eslf, **kwargs):
        return AutoKerasNetworkTemplate.objects.get(pk=kwargs["pk"])

    def build_nodes_and_layers(self, template):
        layers = [
            {
                "id": layer.id,
                "name": layer.name,
                "required_arguments": layer.required_arguments,
            }
            for layer in AutoKerasNodeType.objects.all()
        ]
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
            for layer in template.blocks.all()
        ]
        return (nodes, layers)

    def node_to_layers(self, template, layers):
        node_to_layers = {}
        template.blocks.clear()
        for layer in layers:
            if layer["id"] == "input_node":
                continue
            naso_layer = AutoKerasNode(
                node_type_id=layer["naso_type"],
                name=layer["id"],
                additional_arguments=layer["additional_arguments"],
            )
            naso_layer.save()
            node_to_layers[layer["id"]] = naso_layer.id
            template.blocks.add(naso_layer)

        template.node_to_layer_id = node_to_layers


class TemplateDetails(TemplateDetails):
    """
    Provides details and operations for a template.

    Attributes:
        - build_nodes_and_layers: Builds the nodes and layers for the template.
        - get_template: Retrieves the template based on the provided keyword arguments.
        - node_to_layers: Maps nodes to layers and updates the template accordingly.
    """

    def build_nodes_and_layers(self, template):
        """
        Builds the nodes and layers for the template.

        Args:
            template: The template for which to build the nodes and layers.

        Returns:
            A tuple containing the nodes and layers.
        """
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
            for layer in template.layers.all()
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
                "id": layer.id,
                "name": layer.name,
                "required_arguments": layer.required_arguments,
            }
            for layer in NetworkLayerType.objects.all()
        ]
        return (nodes, layers)

    def get_template(self, **kwargs):
        """
        Retrieves the template based on the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments used to filter the template.

        Returns:
            The matching template.
        """
        return KerasNetworkTemplate.objects.get(pk=kwargs["pk"])

    def node_to_layers(self, template, layers):
        """
        Maps nodes to layers and updates the template accordingly.

        Args:
            template: The template to update.
            layers: The layers to map to nodes.

        Returns:
            A dictionary mapping node IDs to layer IDs.
        """
        node_to_layers = {}
        template.layers.clear()
        for layer in layers:
            if layer["id"] == "input_node":
                continue
            naso_layer = NetworkLayer(
                layer_type_id=layer["naso_type"],
                name=layer["id"],
                additional_arguments=layer["additional_arguments"],
            )
            naso_layer.save()
            node_to_layers[layer["id"]] = naso_layer.id
            template.layers.add(naso_layer)

        template.node_to_layer_id = node_to_layers


class TemplateList(TemplateView):
    """
    A view for displaying a list of network templates.

    Attributes:
        template_name (str): The name of the template to be used for rendering the view.
        page (PageSetup): An instance of the PageSetup class for setting up the page.
        context (dict): A dictionary containing the context data for rendering the template.

    Methods:
        get(self, request): Handles the GET request and returns the rendered template.
    """

    template_name = "templates/list.html"
    page = PageSetup(title="NetworkTemplate", description="Alle")
    context = {"page": page.get_context()}

    def get(self, request):
        keras_templates = KerasNetworkTemplate.objects.all()
        autokeras_templates = AutoKerasNetworkTemplate.objects.all()
        self.context["templates"] = keras_templates
        self.context["autokeras_templates"] = autokeras_templates
        self.context["hide_layer_details"] = True
        return self.render_to_response(self.context)


class AutoKerasTemplateNew(TemplateView):
    """
    A class-based view for creating a new AutoKeras template.

    Attributes:
        template_name (str): The name of the template file.
        page (PageSetup): An instance of the PageSetup class for setting up the page.
        context (dict): A dictionary containing the context data for rendering the template.

    Methods:
        get(self, request): Handles the GET request and renders the template with the context data.
        post(self, request): Handles the POST request and saves the template data to the database.
    """

    template_name = "templates/new.html"
    page = PageSetup(title="NetworkTemplate", description="Details")
    context = {"page": page.get_context(), "editable": True}

    def get(self, request):
        """
        Handles the GET request and renders the template with the context data.

        Args:
            request (HttpRequest): The HTTP request object.

        Returns:
            HttpResponse: The HTTP response object containing the rendered template.
        """
        layers = [
            {
                "id": layer.id,
                "name": layer.name,
                "required_arguments": layer.required_arguments,
            }
            for layer in AutoKerasNodeType.objects.all()
        ]
        self.context["layers"] = layers
        return self.render_to_response(self.context)

    def post(self, request):
        """
        Handles the POST request and saves the template data to the database.

        Args:
            request (HttpRequest): The HTTP request object.

        Returns:
            HttpResponseRedirect: A redirect response to the template list page.
        """
        template_name = request.POST.get("name")
        layers = json.loads(request.POST.get("nodes"))
        edges = json.loads(request.POST.get("edges"))
        template = AutoKerasNetworkTemplate.objects.create(
            name=template_name, connections=edges
        )
        node_to_layers = {}
        for layer in layers:
            if layer["id"] == "input_node":
                continue
            naso_layer = AutoKerasNode.objects.create(
                node_type_id=layer["naso_type"],
                name=layer["id"],
                additional_arguments=layer["additional_arguments"],
            )
            naso_layer.save()
            node_to_layers[layer["id"]] = naso_layer.id
            template.blocks.add(naso_layer)

        template.node_to_layer_id = node_to_layers
        messages.add_message(
            request, messages.SUCCESS, "Template wurde erfolgreich gespeichert"
        )
        template.save()
        return redirect("runs:templates:list")


class TemplateNew(TemplateView):
    """
    A class-based view for creating a new template.

    Attributes:
        template_name (str): The name of the template file.
        page (PageSetup): An instance of the PageSetup class for page setup.
        context (dict): The context data for rendering the template.

    Methods:
        get(self, request): Handles the GET request for creating a new template.
        post(self, request): Handles the POST request for creating a new template.
    """

    template_name = "templates/new.html"
    page = PageSetup(title="NetworkTemplate", description="Details")
    context = {"page": page.get_context(), "editable": True}

    def get(self, request):
        """
        Handles the GET request for creating a new template.

        Retrieves the available network layers and renders the template with the context data.

        Args:
            request (HttpRequest): The HTTP request object.

        Returns:
            HttpResponse: The rendered template response.
        """
        layers = [
            {
                "id": layer.id,
                "name": layer.name,
                "required_arguments": layer.required_arguments,
            }
            for layer in NetworkLayerType.objects.all()
        ]
        self.context["layers"] = layers
        return self.render_to_response(self.context)

    def post(self, request):
        """
        Handles the POST request for creating a new template.

        Creates a new KerasNetworkTemplate object with the provided template name and connections.
        Saves the network layers associated with the template.
        Displays a success message and redirects to the template list page.

        Args:
            request (HttpRequest): The HTTP request object.

        Returns:
            HttpResponseRedirect: The redirect response to the template list page.
        """
        template_name = request.POST.get("name")
        layers = json.loads(request.POST.get("nodes"))
        edges = json.loads(request.POST.get("edges"))
        template = KerasNetworkTemplate.objects.create(
            name=template_name, connections=edges
        )
        node_to_layers = {}
        for layer in layers:
            if layer["id"] == "input_node":
                continue
            naso_layer = NetworkLayer(
                layer_type_id=layer["naso_type"],
                name=layer["id"],
                additional_arguments=layer["additional_arguments"],
            )
            naso_layer.save()
            node_to_layers[layer["id"]] = naso_layer.id
            template.layers.add(naso_layer)

        template.node_to_layer_id = node_to_layers
        messages.add_message(
            request, messages.SUCCESS, "Template wurde erfolgreich gespeichert"
        )
        template.save()
        return redirect("runs:templates:list")


def delete_template(request, pk):
    """
    Deletes a KerasNetworkTemplate object with the given primary key.

    Args:
        request (HttpRequest): The HTTP request object.
        pk (int): The primary key of the KerasNetworkTemplate object to delete.

    Returns:
        JsonResponse: A JSON response indicating the successful deletion of the object.
            The response contains a "message" field with the success message and an "id" field with the deleted object's primary key.
    """
    # Fetch the object or return a 404 response if it doesn't exist
    obj = get_object_or_404(KerasNetworkTemplate, pk=pk)

    # Delete the object
    obj.delete()

    # Return a JSON response to indicate successful deletion
    return JsonResponse({"message": "Object deleted successfully", "id": pk})
