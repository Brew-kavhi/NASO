# models.py
import os

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.db import models


class OverwriteStorage(FileSystemStorage):
    """
    Muda o comportamento padrão do Django e o faz sobrescrever arquivos de
    mesmo nome que foram carregados pelo usuário ao invés de renomeá-los.
    """

    def get_available_name(self, name, max_length=None):
        if self.exists(name):
            os.remove(os.path.join(settings.MEDIA_ROOT, name))
        return name


class Plugin(models.Model):
    """
    Represents a plugin in the system.

    Attributes:
        name (str): The name of the plugin.
        description (str): The description of the plugin.
        author (str): The author of the plugin.
        version (str): The version of the plugin.
        python_file (FileField): The Python file associated with the plugin.
        config_file (FileField): The configuration file associated with the plugin.
    """

    name = models.CharField(max_length=100)
    description = models.TextField()
    author = models.CharField(max_length=100)
    version = models.CharField(max_length=11)
    folder_name = models.CharField(max_length=200)

    class Meta:
        unique_together = ("name", "version")

    def __str__(self):
        return f"{self.name} ({self.version})"
