def get_metrics(request, pk, trial_id):
    autokeras_run = AutoKerasRun.objects.get(pk = pk)
    metrics = autokeras_run.metrics.all()

    epochal_metrics = dict()
    # filter by trial_id in the metric:
    for(metric in metrics):
        epoch = metric.epoch
        for measure in metric.metrics:
            if trial_id == measure.trial_id:
                # add it to the array
                if epoch in epochal_metrics:
                    epochal_metrics[epoch].append(measure)
                else:
                    epochal_metrics[epoch] = measure

    return epochal_metrics