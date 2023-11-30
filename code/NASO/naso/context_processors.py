from decouple import config
from django.conf import settings


def api_token(request):
    return {"API_TOKEN": config("API_TOKEN", default="")}


def app_name(request):
    return {"APP_TITLE": settings.APP_TITLE}


def app_version(request):
    return {"APP_VERSION": settings.APP_VERSION}
