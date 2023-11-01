from django.http import JsonResponse
from keras_tuner.engine.trial import TrialStatus

from neural_architecture.models.AutoKeras import AutoKerasRun


def get_metrics(request, pk, trial_id):
    autokeras_run = AutoKerasRun.objects.get(pk=pk)
    metrics = autokeras_run.metrics.all()

    epochal_metrics = dict()
    # filter by trial_id in the metric:
    for metric in metrics:
        epoch = metric.epoch
        for measure in metric.metrics:
            if trial_id == measure.trial_id:
                # add it to the array
                if epoch in epochal_metrics:
                    epochal_metrics[epoch].append(measure)
                else:
                    epochal_metrics[epoch] = measure

            return epochal_metrics


def get_all_metrics(request, pk):
    autoekras_run = AutoKerasRun.objects.get(pk=pk)
    metrics = autoekras_run.metrics.all()

    trial_metrics = dict()
    # filter by trial_id in the metric:
    for metric in metrics:
        epoch = metric.epoch
        for measure in metric.metrics:
            if "final_metric" not in measure:
                if measure["trial_id"] in trial_metrics:
                    # add it to the array
                    if epoch in trial_metrics[measure["trial_id"]]:
                        trial_metrics[measure["trial_id"]][epoch].append(
                            measure["metrics"]
                        )
                    else:
                        trial_metrics[measure["trial_id"]][epoch] = measure["metrics"]
                else:
                    trial_metrics[measure["trial_id"]] = dict()
                    trial_metrics[measure["trial_id"]][epoch] = measure["metrics"]

    return JsonResponse(trial_metrics, safe=True)


def get_trial_details(request, pk):
    autokeras_run = AutoKerasRun.objects.get(pk=pk)
    loaded_model = autokeras_run.model.load_model(autokeras_run)
    trials = [
        t
        for t in loaded_model.tuner.oracle.trials.values()
        if t.status == TrialStatus.COMPLETED
    ]
    trial_json = {}
    for trial in trials:
        trial_json[trial.trial_id] = trial.hyperparameters.values

    return JsonResponse(trial_json, safe=True)
