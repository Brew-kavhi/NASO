from decouple import config
from django.conf import settings
from api.views.celery import get_workers_information


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
