import traceback

import tensorflow as tf
from loguru import logger

from celery import shared_task
from inference.models.inference import Inference
from naso.celery import restart_all_workers
from neural_architecture.NetworkCallbacks.evaluation_base_callback import (
    EvaluationBaseCallback,
)
from workers.helper_scripts.celery import get_compute_device_name


@shared_task(bind=True)
def run_inference(self, inference_id):
    restart_all_workers()
    self.update_state(
        state="PROGRESS",
        meta={"run_id": inference_id, "inference": True, "current": 1, "total": 1},
    )
    inference = Inference.objects.get(pk=inference_id)

    base_callback = EvaluationBaseCallback(inference)
    gpus = tf.config.experimental.list_physical_devices()
    for gpu in gpus:
        if gpu.name.split("physical_device:")[1] == inference.gpu:
            inference.compute_device = get_compute_device_name(gpu)
            inference.save()
            if inference.gpu.startswith("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)

    try:
        with tf.device(inference.gpu):
            inference.measure_inference(callbacks=[base_callback])
            self.update_state(state="SUCCESS")
    except Exception:
        logger.error(
            "Failure while executing the autokeras model: " + traceback.format_exc()
        )
        self.update_state(state="FAILED")
    finally:
        tf.compat.v1.reset_default_graph()
