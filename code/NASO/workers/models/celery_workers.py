from django.db import models


class CeleryWorker(models.Model):
    queue_name = models.CharField(max_length=100)
    hostname = models.CharField(max_length=100, unique=True)
    last_ping = models.DateTimeField(auto_now=True)
    devices = models.JSONField(default=dict)
    active = models.BooleanField()
    last_active = models.DateTimeField()
    concurrency = models.IntegerField()

    def __str__(self):
        return f"{self.hostname}: {self.queue_name}, Devices:({self.devices})"
