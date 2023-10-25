import traceback
from contextlib import redirect_stdout

import tensorflow as tf
import tensorflow_datasets as tfds
from loguru import logger

from celery import shared_task
from neural_architecture.models.AutoKeras import AutoKerasRun
from neural_architecture.NetworkCallbacks.AutoKerasCallback import \
    AutoKerasCallback

logger.add("net.log", backtrace=True, diagnose=True)


@shared_task(bind=True)
def run_autokeras(self, run_id):
    self.update_state(state="PROGRESS", meta={"autokeras_id": run_id})

    run = AutoKerasRun.objects.get(pk=run_id)
    autokeras_model = run.model

    (train_dataset, test_dataset) = tfds.load(
        run.dataset.name,
        split=["train", "test"],
        as_supervised=run.dataset.as_supervised,
    )
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    callback = AutoKerasCallback(self, run)

    try:
        # load the datasets from the documentation in here
        with open("net.log", "w") as f:
            with redirect_stdout(f):
                autokeras_model.fit(
                    train_dataset,
                    callbacks=[callback] + autokeras_model.get_callbacks(),
                    verbose=2,
                )

        # Evaluate the best model with testing data.
        print(autokeras_model.evaluate(test_dataset))
    except Exception:
        logger.error(
            "Failure while executing the autokeras model: " + traceback.format_exc()
        )
        self.update_state(state="FAILED")

    self.update_state(state="SUCCESS")
