from django.conf import settings


def app_name(request):
    return {"APP_TITLE": settings.APP_TITLE}


def app_version(request):
    return {"APP_VERSION": settings.APP_VERSION}
