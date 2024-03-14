import threading
import traceback
from contextlib import redirect_stdout

import tensorflow as tf
from loguru import logger

from celery import shared_task
from helper_scripts.extensions import start_async_measuring
from naso.celery import restart_all_workers
from neural_architecture.models.autokeras import AutoKerasRun
from neural_architecture.NetworkCallbacks.autokeras_callback import AutoKerasCallback
from neural_architecture.NetworkCallbacks.base_callback import BaseCallback
from neural_architecture.NetworkCallbacks.timing_callback import TimingCallback

logger.add("net.log", backtrace=True, diagnose=True)


@shared_task(bind=True)
def run_autokeras(self, run_id):
    """
    Runs the AutoKeras model training and evaluation.

    Args:
        run_id (int): The ID of the AutoKeras run.

    Returns:
        None
    """
    restart_all_workers()
    self.update_state(state="PROGRESS", meta={"autokeras_id": run_id})

    run = AutoKerasRun.objects.get(pk=run_id)
    autokeras_model = run.model

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Place the model on the specified GPU
    with tf.device(run.gpu):
        (train_dataset, test_dataset, eval_dataset) = run.dataset.get_data()

        callback = AutoKerasCallback(self, run)
        timing_callback = TimingCallback()
        base_callback = BaseCallback(self, run, epochs=run.model.epochs, batch_size=32)

        stop_event = threading.Event()
        database_lock = threading.Lock()

        try:
            with open("net.log", "w", encoding="UTF-8") as _f, redirect_stdout(_f):
                autokeras_model.build_model(run)
                threading.Thread(
                    target=start_async_measuring,
                    args=(stop_event, run, database_lock),
                    daemon=True,
                ).start()
                if run.gpu.startswith("GPU"):
                    run.memory_usage = tf.config.experimental.get_memory_info(run.gpu)[
                        "current"
                    ]
                else:
                    run.memory_usage = -1
                run.save()
                autokeras_model.fit(
                    train_dataset,
                    callbacks=[timing_callback]
                    + autokeras_model.get_callbacks(run)
                    + [callback]
                    + [base_callback],
                    verbose=2,
                    epochs=autokeras_model.epochs,
                )

            autokeras_model.predict(test_dataset.take(200).batch(1), run)
            self.update_state(state="SUCCESS")
        except Exception:
            logger.error(
                "Failure while executing the autokeras model: " + traceback.format_exc()
            )
            self.update_state(state="FAILED")
        finally:
            stop_event.set()
            tf.compat.v1.reset_default_graph()
