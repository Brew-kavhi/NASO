from decouple import config
from django.conf import settings
from django.urls import reverse_lazy

from api.views.celery import get_workers_information
from inference.models.inference import Inference
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import NetworkTraining


def get_celery_workers(request):
    """
    This function returns the workers in the celery cluster with its information.
    """
    return {"WORKERS": get_workers_information()}


def api_token(request):
    """
    This function returns the API_TOKEN for the NASO project to be used in javascript fetching apis.
    """
    return {"API_TOKEN": config("API_TOKEN", default="")}


def app_name(request):
    """
    This function returns the APP_NAME for the NASO project.
    """
    return {"APP_TITLE": settings.APP_TITLE}


def app_version(request):
    """
    This function returns the APP_VERSION for the NASO project.
    """
    return {"APP_VERSION": settings.APP_VERSION}


def get_comparison_runs(request):
    """
    This function returns all runs that currently selected for comparison
    """
    if "comparison" not in request.session:
        return {"COMPARISON": []}
    comparison = request.session["comparison"]
    runs = {}
    for comparison_id in comparison:
        run_id = comparison_id
        if ":" in comparison_id:
            run_id = comparison_id.split(":")[1]
        if comparison[comparison_id] == "tensorflow":
            run = NetworkTraining.objects.filter(pk=run_id).first()
            if run:
                runs[comparison_id] = {
                    "link": reverse_lazy("runs:details", kwargs={"pk": run.id}),
                    "model": run,
                }
        elif comparison[comparison_id] == "inference":
            run = Inference.objects.filter(pk=run_id).first()
            if run:
                runs[comparison_id] = {
                    "link": reverse_lazy("inference:details", kwargs={"pk": run.id}),
                    "model": run,
                }
        elif comparison[comparison_id] == "autokeras":
            run = AutoKerasRun.objects.filter(pk=run_id).first()
            if run:
                runs[comparison_id] = {
                    "link": reverse_lazy(
                        "runs:autokeras:details", kwargs={"pk": run.id}
                    ),
                    "model": run,
                }
        elif comparison[comparison_id] == "autokeras_trial":
            if "_" not in run_id:
                continue
            [autokeras_run, trial_id] = run_id.split("_")
            run = AutoKerasRun.objects.filter(pk=autokeras_run).first()
            if run:
                runs[run_id] = {
                    "link": reverse_lazy(
                        "runs:autokeras:details", kwargs={"pk": run.id}
                    )
                    + "#canvas_trial_"
                    + trial_id,
                    "model": run,
                }

    return {"COMPARISON": runs}
