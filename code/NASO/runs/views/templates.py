import json
from django.views.generic.base import TemplateView
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy

from naso.models.page import PageSetup
from neural_architecture.models.architecture import NetworkLayerType, NetworkLayer
from neural_architecture.models.autokeras import AutoKerasNodeType, AutoKerasNode
from neural_architecture.models.templates import (
    AutoKerasNetworkTemplate,
    KerasNetworkTemplate,
)


class AutoKerasTemplateDetails(TemplateView):
    template_name = "templates/details.html"

    page = PageSetup(title="NetworkTemplate", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        # get all templates
        template = AutoKerasNetworkTemplate.objects.get(pk=kwargs["pk"])
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
        layers = [
            {
                "id": layer.id,
                "name": layer.name,
                "required_arguments": layer.required_arguments,
            }
            for layer in AutoKerasNodeType.objects.all()
        ]
        self.context["layers"] = layers
        self.context["editable"] = True
        self.context["nodes"] = nodes
        self.context["template"] = template
        return self.render_to_response(self.context)


class TemplateDetails(TemplateView):
    template_name = "templates/details.html"

    page = PageSetup(title="NetworkTemplate", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        # get all templates
        template = KerasNetworkTemplate.objects.get(pk=kwargs["pk"])
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
        self.context["layers"] = layers
        self.context["editable"] = True
        self.context["nodes"] = nodes
        self.context["template"] = template
        return self.render_to_response(self.context)


class TemplateList(TemplateView):
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
    template_name = "templates/new.html"
    page = PageSetup(title="NetworkTemplate", description="Details")
    context = {"page": page.get_context(), "editable": True}

    def get(self, request):
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
                layer_type_id=layer["naso_type"],
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
    template_name = "templates/new.html"
    page = PageSetup(title="NetworkTemplate", description="Details")
    context = {"page": page.get_context(), "editable": True}

    def get(self, request):
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
    # Fetch the object or return a 404 response if it doesn't exist
    obj = get_object_or_404(KerasNetworkTemplate, pk=pk)

    # Delete the object
    obj.delete()

    # Return a JSON response to indicate successful deletion
    return JsonResponse({"message": "Object deleted successfully", "id": pk})
