from django.http import JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from api.serializers.session import UpdateSessionSerializer
from system.templatetags.log_filters import colorize_log


class UpdateSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = UpdateSessionSerializer(data=request.data)

        if serializer.is_valid():
            comparison_key = serializer.validated_data["comparison"]
            values_list = serializer.validated_data["values"]
            delete_flag = serializer.validated_data["delete"]
            run_type = serializer.validated_data["run_type"]

            # Update session
            if not request.session[comparison_key]:
                request.session[comparison_key] = {}
            if not delete_flag:
                for value in values_list:
                    if str(value) not in request.session[comparison_key]:
                        request.session[comparison_key][str(value)] = run_type
            else:
                for value in values_list:
                    if str(value) in request.session[comparison_key]:
                        request.session[comparison_key].pop(str(value))
            request.session.save()

            print(request.session["comparison"])
            return Response(
                {"success": True, "session": request.session[comparison_key]},
                status=status.HTTP_200_OK,
            )
        return Response(
            {"success": False, "error": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )


def get_logs(request):
    """
    This view returns the logs of the net.log file.
    it also returns the new position of the file if the file was updated.

    Args:
        request (Request): The request object.

    Returns:
        JsonResponse: The logs of the net.log file.
    """
    log_file_path = "net.log"  # Adjust the path to your log file
    try:
        last_position = int(request.GET.get("last_position", 0))

        with open(log_file_path, "r", encoding="UTF-8") as log_file:
            log_file.seek(last_position)
            log_content = log_file.read()
            new_position = log_file.tell()
            log_content = colorize_log(log_content)

        return JsonResponse(
            {"log_content": log_content, "new_position": new_position}, safe=False
        )
    except FileNotFoundError:
        return JsonResponse({"log_content": "Log file not found."})
