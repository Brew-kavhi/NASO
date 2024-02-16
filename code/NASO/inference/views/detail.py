from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView

from inference.forms.update import UpdateInference
from inference.models.inference import Inference
from naso.models.page import PageSetup


class InferenceDetail(TemplateView):
    template_name = "inference/detail.html"
    page = PageSetup(title="Inference", description="Detail")

    context = {"page": page.get_context()}

    def get(self, request, pk):
        inference = Inference.objects.get(pk=pk)
        self.page.title = inference.name
        self.page.actions = []
        self.page.add_pageaction(
            reverse_lazy("inference:new") + "?rerun=" + str(inference.pk),
            "Run again",
            color="primary",
        )
        self.context = {"page": self.page.get_context()}
        self.context["object"] = inference
        self.context["update_form"] = UpdateInference(instance=inference)
        return self.render_to_response(self.context)

    def post(self, request, *args, **kwargs):
        inference = Inference.objects.get(pk=kwargs.get("pk"))
        form = UpdateInference(request.POST, instance=inference)

        if form.is_valid():
            form.save()

        return redirect(request.path)
