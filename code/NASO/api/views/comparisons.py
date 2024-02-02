from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from comparisons.models.comparison import Comparison


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def clear_session(request):
    if "comparison" in request.session:
        request.session["comparison"] = {}
        return Response({"success": True})

    return Response({"success": False})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def remove_run(request, comparison_id, run_id):
    comparison = Comparison.objects.get(id=comparison_id)
    if run_id in comparison.runs:
        comparison.runs.pop(run_id)
        comparison.save()
        return Response({"success": True})
    return Response({"success": False})
