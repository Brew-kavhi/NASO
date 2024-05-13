from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from inference.celery.run_inference import run_inference
from inference.models.inference import Inference
from runs.models.training import NetworkTraining


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def run_from_id(request):
    """
    This view rates a run.

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.

    Returns:
        Response: The response object with parameter 'success'
    """
    comparison_id = request.data.get("run_id")

    # so far only allow to run tensroflow nets
    if comparison_id.startswith("tensorflow:"):
        run_id = comparison_id.split("tensorflow:")[1]
        run = NetworkTraining.objects.get(pk=run_id)
        queue, gpu = request.data.get("gpu").split("|")
        batch_size = request.data.get("batch_size")
        inference = Inference(
            description=run.description,
            name=run.model_name + "_" + request.data.get("gpu_name"),
            dataset=run.dataset,
            gpu=gpu,
            worker=queue,
            network_training=run,
            batch_size=batch_size,
            model_file=run.model_file,
        )
        inference.save()
        inference.callbacks.set(run.evaluation_parameters.callbacks.all())
        inference.save()

        run_inference.apply_async(args=(inference.id,), queue=queue)
        return Response({"success": True})
    print("fehler not tensrofow")
    return Response({"success": False})
