from django.views.generic.base import TemplateView
from django.shortcuts import redirect

from api.views.comparisons import clear_session
from comparisons.forms.create import SaveSession
from comparisons.forms.edit import AddRunForm
from comparisons.models.comparison import Comparison, get_comparison_details
from naso.models.page import PageSetup
from runs.forms.base import get_gpu_choices


class ComparisonDetailView(TemplateView):
    template_name = "comparison/comparison.html"
    page = PageSetup(
        title="Vergleich", description="Metriken zwischen Laufen vergleichen"
    )
    context = {"page": page.get_context()}

    def get(self, request, pk):
        comparison = Comparison.objects.get(id=pk)
        # get the ComparisonDetailsView
        # then the the etails and append to the runs
        self.context["runs"] = comparison.get_details()
        self.context["comparison"] = comparison
        self.context["add_form"] = AddRunForm()
        self.context["gpus"] = get_gpu_choices()
        return self.render_to_response(self.context)

    def post(self, request, pk):
        comparison = Comparison.objects.get(id=pk)
        add_form = AddRunForm(request.POST)
        if add_form.is_valid():
            # ad the run here
            if "run" in add_form.cleaned_data and add_form.cleaned_data["run"]:
                comparison.runs[
                    "tensorflow:" + str(add_form.cleaned_data["run"].id)
                ] = "tensorflow"
            if (
                "inference" in add_form.cleaned_data
                and add_form.cleaned_data["inference"]
            ):
                comparison.runs[
                    "inference:" + str(add_form.cleaned_data["inference"].id)
                ] = "inference"
            comparison.save()
            self.context["runs"] = comparison.get_details()
            self.context["add_form"] = AddRunForm()
            self.context["comparison_id"] = comparison.id
            self.context["gpus"] = get_gpu_choices()
            return self.render_to_response(self.context)


class SessionComparisonView(TemplateView):
    template_name = "comparison/comparison.html"
    page = PageSetup(
        title="Vergleich", description="Metriken zwischen Laufen vergleichen"
    )
    context = {"page": page.get_context()}

    def get(self, request):
        details = []
        if "comparison" in request.session:
            details = get_comparison_details(request.session["comparison"])
        # get the ComparisonDetailsView
        # then the the etails and append to the runs
        save_form = SaveSession()
        self.context["session"] = save_form
        self.context["runs"] = details
        self.context["gpus"] = get_gpu_choices()
        return self.render_to_response(self.context)

    def post(self, request):
        save_form = SaveSession(request.POST)
        if save_form.is_valid():
            comparison = Comparison(
                name=save_form.cleaned_data["name"],
                description=save_form.cleaned_data["description"],
            )
            comparison.runs = request.session["comparison"]
            comparison.save()
            clear_session(request)
            return redirect("comparisons:details", pk=comparison.id)
        self.context["session"] = save_form
        self.context["runs"] = get_comparison_details(request.session["comparison"])
        return self.render_to_response(self.context)
