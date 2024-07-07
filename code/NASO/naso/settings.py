"""
Django settings for naso project.

Generated by 'django-admin startproject' using Django 3.2.19.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.2/ref/settings/
"""

import os
import re
from pathlib import Path

import toml
from decouple import config

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config("SECRET_KEY")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config("DEBUG", default=True)

APP_TITLE = config("APP_TITLE", default="NASO")
with open(
    os.path.join(Path(__file__).resolve().parent.parent.parent.parent, ".cz.toml"),
    encoding="utf-8",
) as f:
    toml_config = toml.load(f)
APP_VERSION = toml_config["tool"]["commitizen"]["version"]

CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_STORE_ERRORS_EVEN_IF_IGNORED = True
CELERY_BROKER_URL = config("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = "django-db"

ALLOWED_HOSTS = [
    config("DEFAULT_HOST", default="localhost"),
    "naso.mariusgoehring.de",
    "localhost",
]
TRUSTED_ORIGINS = ["https://naso.mariusgoehring.de"]

MEDIA_ROOT = PROJECT_ROOT + "/../media/"  # noqa
MEDIA_URL = "/media/"

# Kaggle interaction:
KAGGLE_USERNAME = config("KAGGLE_USERRNAME", default="")
KAGGLE_API_KEY = config("KAGGLE_KEY", default="")
ENERGY_MEASUREMENT_FREQUENCY = config(
    "ENERGY_MEASUREMENT_FREQUENCY", default=0.5, cast=float
)

# Application definition

INSTALLED_APPS = [
    # Optional: Django admin theme (must be before django.contrib.admin)
    # General use templates & template tags (should appear first)
    "adminlte3",
    "adminlte3_theme",
    "django.contrib.admin",
    "rest_framework",
    "rest_framework.authtoken",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "simple_history",
    "dashboard",
    "plugins",
    "runs",
    "datasets",
    "inference",
    "comparisons",
    "system",
    "workers",
    "api",
    "neural_architecture",
    "crispy_forms",
    "crispy_bootstrap5",
    "django_celery_results",
    "flake8",
    "safedelete",
]


MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "simple_history.middleware.HistoryRequestMiddleware",
]

ROOT_URLCONF = "naso.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "naso.context_processors.app_name",
                "naso.context_processors.get_celery_workers",
                "naso.context_processors.app_version",
                "naso.context_processors.api_token",
                "naso.context_processors.get_comparison_runs",
            ],
        },
    },
]
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"
CRISPY_CLASS_CONVERTERS = {"form-control": "form-control-sm"}

WSGI_APPLICATION = "naso.wsgi.application"


# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases


OPEN_URLS = [
    re.compile("^/api/"),
    re.compile("^/accounts/password_reset/"),
    re.compile("^/accounts/reset/"),
]

# Password validation
# https://docs.djangoproject.com/en/3.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
    ],
}

# Internationalization
# https://docs.djangoproject.com/en/3.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.2/howto/static-files/

STATIC_URL = "/static/"
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static"), os.path.join(BASE_DIR, "docs")]
STATIC_ROOT = "naso/static"

# Default primary key field type
# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"

EMAIL_HOST = config("EMAIL_HOST")
EMAIL_PORT = config("EMAIL_PORT")
EMAIL_HOST_USER = config("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD")
EMAIL_USE_TLS = config("EMAIL_USE_TLS")
SERVER_EMAIL = config("EMAIL_FROM_ADDRESS")
DEFAULT_FROM_EMAIL = config("EMAIL_FROM_ADDRESS")

ALLOWED_HOSTS = [
    "127.0.0.1",
    config("DEFAULT_HOST", default="localhost"),
    "code.mariusgoehring.de",
]
TRUSTED_ORIGINS = ["https://code.mariusgoehring.de"]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}