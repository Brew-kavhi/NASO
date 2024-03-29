import csv
import json

from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from neural_architecture.models.autokeras import AutoKerasRun
from runs.views.softdelete import harddelete_run, undelete_run


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def rate_run(request, pk):
    """
    This view rates a run.

    Args:
        request (Request): The request object. Its body must contain rate parameter.
        pk (int): The primary key of the run.

    Returns:
        Response: The response object with parameter 'success'
    """

    run = AutoKerasRun.objects.get(pk=pk)
    rate = request.data.get("rate")
    if rate != "":
        run.rate = rate
    else:
        run.rate = 0
    run.save()
    return Response({"success": True})


def get_metrics(pk, trial_id):
    """
    This view returns the metrics for a specific trial of the run.

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.
        trial_id (str): The id of the trial.

    Returns:
        epochal_metriocs: Dict of metrics given by epoch

    """
    autokeras_run = AutoKerasRun.objects.get(pk=pk)
    metrics = autokeras_run.metrics.all()

    epochal_metrics = {}
    for metric in metrics:
        epoch = metric.epoch
        for measure in metric.metrics:
            # filter by trial_id in the metric:
            if "trial_id" in measure and trial_id == measure["trial_id"]:
                # add it to the array
                if epoch in epochal_metrics:
                    epochal_metrics[epoch] = measure["metrics"]
                else:
                    epochal_metrics[epoch] = measure["metrics"]

    return epochal_metrics


def get_metrics_json(request, pk, trial_id):
    return JsonResponse(get_metrics(pk, trial_id), safe=True)


def download_metrics(request, pk, trial_id):
    """
    This view offers a downloadable CSV file containiong the metrics for a specific trial of the run.

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.
        trial_id (str): The id of the trial.

    Returns:
        response: CSV file containing the metrics for a specific trial of the run.
    """
    json_data = get_metrics(pk, trial_id)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="data_' + trial_id + '.csv"'

    # Convert JSON to CSV
    writer = csv.writer(response)
    rows = []
    header = ["epochs"]
    for key, value in json_data.items():
        data = [key]
        for metric in value:
            data.append(value[metric])
            if metric not in header:
                header.append(metric)
        rows.append(data)
    writer.writerow(header)
    writer.writerows(rows)

    return response


def get_final_metrics(request, pk):
    """
    This view returns the last metrics for all trials of the run.

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.

    Returns:
        JsonResponse: Dict of metrics given by trial_id and epoch
    """
    autokeras_run = AutoKerasRun.objects.get(pk=pk)
    metrics = autokeras_run.metrics.all()

    trial_metrics = {}
    trial_id = ""
    for metric in metrics:
        for measure in metric.metrics:
            if "trial_id" in measure:
                trial_id = measure["trial_id"]
            if "final_metric" not in measure:
                trial_metrics[trial_id] = measure["metrics"]

    return JsonResponse(trial_metrics, safe=True)


def get_all_metrics(request, pk):
    """
    This view returns the metrics for all trials of the run.

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.

    Returns:
        JsonResponse: Dict of metrics given by trial_id and epoch
    """
    autokeras_run = AutoKerasRun.objects.get(pk=pk)
    metrics = autokeras_run.metrics.all()

    trial_metrics = {}
    trial_id = ""
    for metric in metrics:
        epoch = metric.epoch
        for measure in metric.metrics:
            if "trial_id" in measure:
                trial_id = measure["trial_id"]
            if "final_metric" not in measure:
                if trial_id in trial_metrics:
                    # add it to the array
                    if epoch in trial_metrics[trial_id]:
                        trial_metrics[trial_id][epoch] = measure["metrics"]
                    else:
                        trial_metrics[trial_id][epoch] = measure["metrics"]
                else:
                    trial_metrics[trial_id] = {}
                    trial_metrics[trial_id][epoch] = measure["metrics"]
            else:
                trial_metrics[trial_id]["final"] = measure["metrics"]

    return JsonResponse(trial_metrics, safe=True)


def get_trial_details_short(request, pk):
    autokeras_run = AutoKerasRun.objects.get(pk=pk)
    metrics = autokeras_run.metrics.all()

    trial_json = {}
    trial_id = ""
    distinct_metrics = []

    for metric in metrics:
        epoch = metric.epoch
        for measure in metric.metrics:
            if "trial_id" in measure:
                trial_id = measure["trial_id"]
            if "final_metric" not in measure:
                # only get the max metrics.
                if trial_id in trial_json:
                    # add it to the array
                    for metric_name in measure["metrics"]:
                        if (
                            metric_name in trial_json[trial_id]["min"]
                            and measure["metrics"][metric_name]
                            and measure["metrics"][metric_name]
                            < trial_json[trial_id]["min"][metric_name]
                        ):
                            trial_json[trial_id]["min"][metric_name] = measure[
                                "metrics"
                            ][metric_name]
                        if (
                            metric_name in trial_json[trial_id]["max"]
                            and measure["metrics"][metric_name]
                            and measure["metrics"][metric_name]
                            > trial_json[trial_id]["max"][metric_name]
                        ):
                            trial_json[trial_id]["max"][metric_name] = measure[
                                "metrics"
                            ][metric_name]
                else:
                    trial_json[trial_id] = {"min": {}, "max": {}, "final": {}}
                    trial_json[trial_id]["min"] = measure["metrics"]
                    trial_json[trial_id]["max"] = measure["metrics"]
                    for metric_name in measure["metrics"]:
                        if not metric_name in distinct_metrics:
                            distinct_metrics.append(metric_name)
                    trial_json[trial_id][epoch] = measure["metrics"]
            else:
                trial_json[trial_id]["final"] = measure["metrics"]

    trial_json["metrics"] = distinct_metrics
    return JsonResponse(trial_json, safe=True)


def get_all_trial_details(request, pk):
    """
    Retrieve details of all trials for a given AutoKerasRun.

    Args:
        request: The HTTP request object.
        pk: The primary key of the AutoKerasRun.

    Returns:
        A JSON response containing the trial details for the AutoKerasRun.

    Raises:
        AutoKerasRun.DoesNotExist: If the AutoKerasRun with the given primary key does not exist.
    """
    autokeras_run = AutoKerasRun.objects.get(pk=pk)
    metrics = autokeras_run.metrics.all()

    trial_json = {}
    for metric in metrics:
        for measure in metric.metrics:
            if "final_metric" not in measure and measure["trial_id"] not in trial_json:
                trial_json[measure["trial_id"]] = get_trial_details(
                    autokeras_run, measure["trial_id"]
                )

    return JsonResponse(trial_json, safe=True)


def get_trial_details(autokeras_run: AutoKerasRun, trial_id):
    """
    Retrieve the hyperparameters values for a specific trial in an AutoKeras run.

    Args:
        autokeras_run (AutoKerasRun): The AutoKeras run object.
        trial_id (int): The ID of the trial.

    Returns:
        dict: A dictionary containing the hyperparameter values for the specified trial.
    """
    path = autokeras_run.model.get_trial_hyperparameters_path(trial_id)
    # read json file from path and convert to dict
    with open(path, "r", encoding="UTF-8") as file:
        trial_dict = json.load(file)
        return trial_dict["hyperparameters"]["values"]


def get_metrics_for_run(request, pk):
    """
    Retrieve the metrics for a specific AutoKeras run.

    Args:
        request (HttpRequest): The HTTP request object.
        pk (int): The primary key of the AutoKeras run.

    Returns:
        JsonResponse: A JSON response containing the metrics data.

    Raises:
        AutoKerasRun.DoesNotExist: If the AutoKeras run with the given primary key does not exist.
    """
    run = AutoKerasRun.objects.get(pk=pk)
    metrics = run.metrics.all()
    data = []
    for metric in metrics:
        data.append(metric.metrics[0])
    return JsonResponse(data, safe=False)


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
    success = harddelete_run(pk, "autokeras")
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
    success = undelete_run(pk, "autokeras")
    return Response({"success": success, "id": pk})
