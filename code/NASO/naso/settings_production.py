"""
Django settings for naso project.

Generated by 'django-admin startproject' using Django 3.2.19.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.2/ref/settings/
"""
from decouple import config
from dj_database_url import parse as db_url

from .settings import *

DATABASES = {"default": config("DATABASE_URL", cast=db_url)}
DATABASES["default"]["ENGINE"] = "django.contrib.gis.db.backends.mysql"
DATABASES["default"]["OPTIONS"] = {"init_command": "SET default_storage_engine=INNODB"}
