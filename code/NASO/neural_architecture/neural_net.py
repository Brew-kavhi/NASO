import tensorflow as tf
from loguru import logger

from celery import shared_task
from neural_architecture.models.architecture import NetworkConfiguration
from neural_architecture.NetworkCallbacks.celery_update_callback import (
    CeleryUpdateCallback,
)
from neural_architecture.NetworkCallbacks.evaluation_base_callback import (
    EvaluationBaseCallback,
)
from runs.models.training import NetworkTraining, TrainingMetric

logger.add("net.log", backtrace=True, diagnose=True)


@shared_task(bind=True)
def run_neural_net(self, training_id):
    self.update_state(state="PROGRESS", meta={"run_id": training_id})
    training = NetworkTraining.objects.get(pk=training_id)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    update_call = CeleryUpdateCallback(self, run=training)
    try:
        with tf.device(training.gpu):
            _nn = NeuralNetwork(training)
            _nn.run_from_config(training, update_call)
    except Exception as _e:
        logger.error("Failure while training the network: " + str(_e))
        self.update_state(state="FAILED")

    self.update_state(state="SUCCESS")


class NeuralNetwork:
    celery_callback = None
    train_dataset = None
    test_dataset = None

    def __init__(self, training_config: NetworkTraining = None):
        if training_config:
            self.training_config = training_config
            self.load_data()
            self.build_model_from_config(training_config.network_config)

    def run_from_config(self, config: NetworkTraining, celery_callback):
        self.celery_callback = celery_callback

        if config:
            self.training_config = config

        # then load the dataset:

        self.train()
        self.validate()

    def build_model_from_config(self, config: NetworkConfiguration = None) -> None:
        if not self.training_config or not self.training_config.hyper_parameters:
            raise AttributeError
        try:
            input_shape = self.training_config.dataset.get_element_size()
            logger.info(f"Using input shape {input_shape}")
        except Exception:
            input_shape = (28, 28)
            logger.warning("Using default input shape")

        model = config.build_model(input_shape)

        model = config.build_pruning_model(model)
        model.compile(**self.training_config.hyper_parameters.get_as_dict())
        logger.success("Model is initiated.")
        model.summary()
        config.size = model.count_params()
        config.save()

        self.model = model

    def load_data(self):
        (
            self.train_dataset,
            self.test_dataset,
        ) = self.training_config.dataset.get_data()

        logger.success("Data is loaded.")

    def train(self, training_config: NetworkTraining = None):
        if training_config:
            self.training_config = training_config
        fit_parameters = self.training_config.fit_parameters
        epochs = fit_parameters.epochs
        batch_size = fit_parameters.batch_size

        logger.success("Started training of the network...")

        callbacks = self.training_config.fit_parameters.get_callbacks(
            self.training_config
        ) + (
            [self.celery_callback]
            + self.training_config.network_config.get_pruning_callbacks()
        )

        self.model.fit(
            self.train_dataset.shuffle(60000).batch(batch_size),
            epochs=epochs,
            validation_data=self.test_dataset.batch(batch_size),
            callbacks=callbacks,
            shuffle=fit_parameters.shuffle,
            class_weight=fit_parameters.class_weight,
            sample_weight=fit_parameters.sample_weight,
            initial_epoch=fit_parameters.initial_epoch,
            steps_per_epoch=fit_parameters.steps_per_epoch,
            max_queue_size=fit_parameters.max_queue_size,
            workers=fit_parameters.workers,
            use_multiprocessing=fit_parameters.use_multiprocessing,
        )
        self.training_config.network_config.save_model_on_disk(self.model)

        logger.success("Finished training of neural network.")

    def validate(self):
        batch_size = self.training_config.evaluation_parameters.batch_size
        steps = self.training_config.evaluation_parameters.steps

        logger.info("Started evaluation of the network...")
        metrics = self.model.evaluate(
            self.test_dataset.batch(64),
            batch_size=batch_size,
            steps=steps,
            verbose=2,
            return_dict=True,
            callbacks=self.training_config.evaluation_parameters.get_callbacks(
                self.training_config
            )
            + [EvaluationBaseCallback(self.training_config)],
        )
        logger.info("evaluation of the network done...")

        time = "Fehler"
        if self.celery_callback:
            time = self.celery_callback.get_total_time()

        eval_metric = TrainingMetric(
            neural_network=self.training_config,
            epoch=self.training_config.fit_parameters.epochs + 1,
            metrics=[{"metrics": metrics, "time": time}],
        )
        eval_metric.save()
        self.training_config.final_metrics = eval_metric
        self.training_config.save()

        self.predict(self.test_dataset.take(200).batch(1))

    def predict(self, dataset):
        batch_size = 1
        return self.model.predict(
            dataset,
            batch_size,
            verbose=2,
            steps=None,
            callbacks=self.training_config.evaluation_parameters.get_callbacks(
                self.training_config
            )
            + [EvaluationBaseCallback(self.training_config)],
        )
