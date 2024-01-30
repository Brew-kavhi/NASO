from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def clear_session(request):
    if "comparison" in request.session:
        request.session["comparison"] = {}
        return Response({"success": True})

    return Response({"success": False})
