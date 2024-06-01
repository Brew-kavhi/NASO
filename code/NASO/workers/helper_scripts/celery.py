import datetime

import tensorflow as tf
from cpuinfo import get_cpu_info

from celery import shared_task
from naso.celery import app
from workers.models.celery_workers import CeleryWorker


def get_all_workers():
    # here get all the active workers and update the worker models
    try:
        alive_workers = app.control.ping(timeout=0.5)
        registered_workers = CeleryWorker.objects.all().values_list(
            "hostname", "queue_name"
        )
        known_worker_ids = [
            f"{hostname}-{queue_name}" for hostname, queue_name in registered_workers
        ]

        for worker in alive_workers:
            hostname = list(worker.keys())[0]
            celery_worker = CeleryWorker.objects.filter(hostname=hostname).first()

            details = app.control.inspect([hostname]).active_queues()
            configuration = app.control.inspect([hostname]).stats()
            concurrency = 1
            if configuration[hostname]["pool"]["max-concurrency"]:
                concurrency = configuration[hostname]["pool"]["max-concurrency"]
            queue = details[hostname][0]["name"]
            if celery_worker:
                if f"{hostname}-{queue}" in known_worker_ids:
                    known_worker_ids.remove(f"{hostname}-{queue}")
                celery_worker.concurrency = concurrency
                celery_worker.last_active = datetime.datetime.now()
                celery_worker.last_ping = datetime.datetime.now()
                celery_worker.active = True
                celery_worker.queue_name = queue
                celery_worker.save()
                if celery_worker.devices == {}:
                    collect_devices.apply_async(args=(celery_worker.id,), queue=queue)

            else:
                # create a new one:
                celery_worker = CeleryWorker(
                    hostname=hostname,
                    queue_name=queue,
                    active=True,
                    concurrency=concurrency,
                    last_active=datetime.datetime.now(),
                )
                celery_worker.last_ping = datetime.datetime.now()
                celery_worker.save()
                collect_devices.apply_async(args=(celery_worker.id,), queue=queue)

        for worker in known_worker_ids:
            # these are all inactive.
            hostname, queue = worker.split("-")
            worker = CeleryWorker.objects.filter(
                hostname=hostname, queue_name=queue
            ).first()
            if worker:
                worker.active = False
                worker.save()
    except Exception as e:
        print(f"Failed to fetch workers: {str(e)}")


@shared_task
def collect_devices(celery_worker_id):
    devices = [
        {device.name.split("physical_device:")[1]: get_compute_device_name(device)}
        for device in tf.config.list_physical_devices()
    ]
    celery_worker = CeleryWorker.objects.get(pk=celery_worker_id)
    celery_worker.devices = devices
    celery_worker.save()


def get_compute_device_name(device):
    details = tf.config.experimental.get_device_details(device)
    if "device_name" in details:
        return details["device_name"]
    # it is a CPU so fin another way to get the CPU
    return get_cpu_info()["brand_raw"]
