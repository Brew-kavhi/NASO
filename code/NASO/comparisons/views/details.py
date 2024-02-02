from django.views.generic.base import TemplateView

from comparisons.forms.create import SaveSession
from comparisons.forms.edit import AddRunForm
from comparisons.models.comparison import Comparison, get_comparison_details
from naso.models.page import PageSetup
from runs.models.training import NetworkTraining


class ComparisonDetailView(TemplateView):
    template_name = "comparison/comparison.html"
    page = PageSetup(
        title="Vergleich", description="Metriken zwischen Laufen vergleichen"
    )
    context = {"page": page.get_context()}

    def get(self, request, pk):
        comparison = Comparison.objects.get(id=pk)
        all_runs = NetworkTraining.objects.all()
        # get the ComparisonDetailsView
        # then the the etails and append to the runs
        self.context["runs"] = comparison.get_details()
        self.context["comparison_id"] = comparison.id
        self.context["add_form"] = AddRunForm()
        return self.render_to_response(self.context)

    def post(self, request, pk):
        comparison = Comparison.objects.get(id=pk)
        add_form = AddRunForm(request.POST)
        if add_form.is_valid():
            # ad the run here
            comparison.runs[
                "tensorflow:" + str(add_form.cleaned_data["run"].id)
            ] = "tensorflow"
            comparison.save()
            self.context["runs"] = comparison.get_details()
            self.context["add_form"] = AddRunForm()
            self.context["comparison_id"] = comparison.id
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
        self.context["session"] = save_form
        self.context["runs"] = get_comparison_details(request.session["comparison"])
        return self.render_to_response(self.context)
