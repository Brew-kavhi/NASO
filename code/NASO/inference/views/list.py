from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView

from inference.models.inference import Inference
from naso.models.page import PageSetup


class ListInferences(TemplateView):
    template_name = "inference/list.html"
    page = PageSetup(title="Inference", description="List")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        inferences = reversed(Inference.objects.all())
        self.page.actions = []
        self.page.add_pageaction(reverse_lazy("inference:new"), "New Inference")
        self.context["page"] = self.page.get_context()
        self.context["inferences"] = inferences
        return self.render_to_response(self.context)


def delete_inference(request, pk):
    obj = get_object_or_404(Inference, pk=pk)
    obj.delete()
    return JsonResponse({"message": "object deleted succesfully", "id": pk})
