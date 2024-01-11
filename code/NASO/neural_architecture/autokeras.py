import threading
import traceback
from contextlib import redirect_stdout

import tensorflow as tf
from loguru import logger

from celery import shared_task
from helper_scripts.extensions import start_async_measuring
from neural_architecture.models.autokeras import AutoKerasRun
from neural_architecture.models.model_runs import KerasModelRun
from neural_architecture.NetworkCallbacks.autokeras_callback import AutoKerasCallback
from neural_architecture.NetworkCallbacks.base_callback import BaseCallback

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
    self.update_state(state="PROGRESS", meta={"autokeras_id": run_id})

    run = AutoKerasRun.objects.get(pk=run_id)
    autokeras_model = run.model

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Place the model on the specified GPU
    with tf.device(run.gpu):
        (train_dataset, test_dataset) = run.dataset.get_data()

        callback = AutoKerasCallback(self, run)
        base_callback = BaseCallback(self, run, epochs=run.model.epochs)

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
                run.memory_usage = tf.config.experimental.get_memory_info(run.gpu)[
                    "current"
                ]
                run.save()
                autokeras_model.fit(
                    train_dataset,
                    callbacks=autokeras_model.get_callbacks(run)
                    + [callback]
                    + [base_callback],
                    verbose=2,
                    epochs=autokeras_model.epochs,
                )

            # Evaluate the best model with testing data.
            print(autokeras_model.evaluate(test_dataset))
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


@shared_task(bind=True)
def run_autokeras_trial(self, run_id, trial_id, keras_model_run_id):
    """
    Runs a trial of AutoKeras model training.

    Args:
        run_id (int): The ID of the AutoKerasRun object.
        trial_id (int): The ID of the trial.
        keras_model_run_id (int): The ID of the KerasModelRun object.

    Returns:
        None
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    run = AutoKerasRun.objects.get(pk=run_id)
    keras_model_run = KerasModelRun.objects.get(pk=keras_model_run_id)
    (train_dataset, validation_dataset) = run.model.save_trial_as_model(
        run, keras_model_run, trial_id
    )
    model = keras_model_run.model

    log_callback = BaseCallback(
        self, keras_model_run, epochs=keras_model_run.model.fit_parameters.epochs
    )

    stop_event = threading.Event()
    database_lock = threading.Lock()

    try:
        with tf.device(run.gpu), open(
            "net.log", "w", encoding="UTF-8"
        ) as _f, redirect_stdout(_f):
            threading.Thread(
                target=start_async_measuring,
                args=(stop_event, run, database_lock),
                daemon=True,
            ).start()
            run.memory_usage = tf.config.experimental.get_memory_info(run.gpu)[
                "current"
            ]
            run.save()
            model.fit(
                train_dataset,
                verbose=2,
                callbacks=model.get_pruning_callbacks()
                + model.evaluation_parameters.get_callbacks(run)
                + [log_callback],
            )

            # Evaluate the best model with testing data.
            model.predict(validation_dataset.take(200).batch(1), keras_model_run)
            self.update_state(state="SUCCESS")
    except Exception:
        logger.error(
            "Failure while executing the autokeras model: " + traceback.format_exc()
        )
        self.update_state(state="FAILED")
    finally:
        stop_event.set()
        tf.compat.v1.reset_default_graph()
