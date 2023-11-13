import csv
import json

from django.http import HttpResponse, JsonResponse

from neural_architecture.models.autokeras import AutoKerasRun


def get_metrics(request, pk, trial_id):
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


def download_metrics(request, pk, trial_id):
    json_data = get_metrics(request, pk, trial_id)
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


def get_all_metrics(request, pk):
    autokeras_run = AutoKerasRun.objects.get(pk=pk)
    metrics = autokeras_run.metrics.all()

    trial_metrics = {}
    for metric in metrics:
        epoch = metric.epoch
        for measure in metric.metrics:
            if "final_metric" not in measure:
                if measure["trial_id"] in trial_metrics:
                    # add it to the array
                    if epoch in trial_metrics[measure["trial_id"]]:
                        trial_metrics[measure["trial_id"]][epoch] = measure["metrics"]
                    else:
                        trial_metrics[measure["trial_id"]][epoch] = measure["metrics"]
                else:
                    trial_metrics[measure["trial_id"]] = {}
                    trial_metrics[measure["trial_id"]][epoch] = measure["metrics"]

    return JsonResponse(trial_metrics, safe=True)


def get_all_trial_details(request, pk):
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
    path = autokeras_run.model.get_trial_hyperparameters_path(trial_id)
    # read json file from path and convert to dict
    with open(path, "r", encoding="UTF-8") as file:
        trial_dict = json.load(file)
        return trial_dict["hyperparameters"]["values"]


def get_metrics_for_run(request, pk):
    run = AutoKerasRun.objects.get(pk=pk)
    metrics = run.metrics.all()
    data = []
    for metric in metrics:
        data.append(metric.metrics[0])
    return JsonResponse(data, safe=False)
