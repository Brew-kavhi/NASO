import traceback
import tensorflow as tf
from loguru import logger

from celery import shared_task
from inference.models.inference import Inference
from neural_architecture.NetworkCallbacks.evaluation_base_callback import (
    EvaluationBaseCallback,
)

from naso.celery import restart_all_workers


@shared_task(bind=True)
def run_inference(self, inference_id):
    restart_all_workers()
    self.update_state(
        state="PROGRESS",
        meta={"run_id": inference_id, "inference": True, "current": 1, "total": 1},
    )
    inference = Inference.objects.get(pk=inference_id)

    base_callback = EvaluationBaseCallback(inference)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        if gpu.name.split("physical_device:")[1] == inference.gpu:
            details = tf.config.experimental.get_device_details(gpu)
            if "device_name" in details:
                inference.compute_device = details["device_name"]
                inference.save()
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
