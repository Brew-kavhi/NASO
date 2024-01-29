from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.generic.base import TemplateView

from comparisons.models.comparison import Comparison
from naso.models.page import PageSetup


class ComparisonListView(TemplateView):
    template_name = "comparison/comparison_list.html"
    page = PageSetup(
        title="Vergleich", description="Metriken zwischen Laufen vergleichen"
    )
    context = {"page": page.get_context()}

    def get(self, request):
        comparisons = Comparison.objects.all()
        self.context["comparisons"] = comparisons
        return self.render_to_response(self.context)


def delete_comparison(request, pk):
    obj = get_object_or_404(Comparison, pk=pk)

    # Delete the object
    obj.delete()

    # Return a JSON response to indicate successful deletion
    return JsonResponse({"message": "Object deleted successfully", "id": pk})
