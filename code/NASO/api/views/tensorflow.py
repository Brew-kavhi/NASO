from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.helper_scripts.run import rate_run
from api.serializers.tensorflow import NetworkTrainingSerializer
from runs.models.training import NetworkTraining
from runs.views.softdelete import harddelete_run, undelete_run


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def rate_tensorflow_run(request, pk):
    """
    This view rates a run.

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.

    Returns:
        Response: The response object with parameter 'success'
    """
    run = NetworkTraining.objects.get(pk=pk)
    return rate_run(request, run)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_configuration(request, pk):
    run = NetworkTraining.objects.get(pk=pk)
    serial_model = NetworkTrainingSerializer(run)
    return Response(serial_model.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hard_delete(request, pk):
    """
    This view finally deletes a run.

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.

    Returns:
        Response: The response object with parameter 'success'
    """
    success = harddelete_run(pk, "tensorflow")
    return Response({"success": success, "id": pk})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def undelete(request, pk):
    """
    This view undoes the deletion of a run

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.

    Returns:
        Response: The response object with parameter 'success'
    """
    success = undelete_run(pk, "tensorflow")
    return Response({"success": success, "id": pk})
