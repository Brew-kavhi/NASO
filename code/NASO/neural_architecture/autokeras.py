import traceback
from contextlib import redirect_stdout

from loguru import logger

from celery import shared_task
from neural_architecture.models.autokeras import AutoKerasRun
from neural_architecture.models.model_runs import KerasModelRun
from neural_architecture.NetworkCallbacks.autokeras_callback import AutoKerasCallback
from neural_architecture.NetworkCallbacks.keras_model_callback import KerasModelCallback

logger.add("net.log", backtrace=True, diagnose=True)


@shared_task(bind=True)
def run_autokeras(self, run_id):
    self.update_state(state="PROGRESS", meta={"autokeras_id": run_id})

    run = AutoKerasRun.objects.get(pk=run_id)
    autokeras_model = run.model

    (train_dataset, test_dataset) = run.dataset.get_data()

    callback = AutoKerasCallback(self, run)

    try:
        # load the datasets from the documentation in here
        with open("net.log", "w", encoding="UTF-8") as _f, redirect_stdout(_f):
            autokeras_model.build_model(run)
            autokeras_model.fit(
                train_dataset,
                callbacks=autokeras_model.get_callbacks(run) + [callback],
                verbose=2,
                epochs=autokeras_model.epochs,
            )

        # Evaluate the best model with testing data.
        print(autokeras_model.evaluate(test_dataset))
    except Exception:
        logger.error(
            "Failure while executing the autokeras model: " + traceback.format_exc()
        )
        self.update_state(state="FAILED")

    self.update_state(state="SUCCESS")


@shared_task(bind=True)
def run_autokeras_trial(self, run_id, trial_id, keras_model_run_id):
    run = AutoKerasRun.objects.get(pk=run_id)
    keras_model_run = KerasModelRun.objects.get(pk=keras_model_run_id)
    (train_dataset, validation_dataset) = run.model.save_trial_as_model(
        run, keras_model_run, trial_id
    )
    model = keras_model_run.model

    log_callback = KerasModelCallback(self, keras_model_run)

    try:
        # load the datasets from the documentation in here
        with open("net.log", "w", encoding="UTF-8") as _f, redirect_stdout(_f):
            model.fit(
                train_dataset,
                verbose=2,
                callbacks=[log_callback],
            )

            # Evaluate the best model with testing data.
            print(model.evaluate(validation_dataset))
    except Exception:
        logger.error(
            "Failure while executing the autokeras model: " + traceback.format_exc()
        )
        self.update_state(state="FAILED")

    self.update_state(state="SUCCESS")
