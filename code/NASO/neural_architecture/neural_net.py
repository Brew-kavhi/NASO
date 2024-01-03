import tensorflow as tf
import threading
from helper_scripts.extensions import start_async_measuring
from loguru import logger
import traceback

from celery import shared_task
from neural_architecture.models.architecture import NetworkConfiguration
from neural_architecture.NetworkCallbacks.base_callback import BaseCallback
from neural_architecture.NetworkCallbacks.evaluation_base_callback import (
    EvaluationBaseCallback,
)
from runs.models.training import NetworkTraining, TrainingMetric

logger.add("net.log", backtrace=True, diagnose=True)


@shared_task(bind=True)
def run_neural_net(self, training_id):
    """
    Runs the neural network training process.

    Args:
        training_id (int): The ID of the network training.

    Returns:
        None

    Raises:
        Exception: If there is a failure while training the network.

    """
    self.update_state(state="PROGRESS", meta={"run_id": training_id})
    training = NetworkTraining.objects.get(pk=training_id)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    update_call = BaseCallback(
        self, run=training, epochs=training.fit_parameters.epochs
    )
    stop_event = threading.Event()
    database_lock = threading.Lock()

    try:
        with tf.device(training.gpu):
            threading.Thread(
                target=start_async_measuring,
                args=(stop_event, training, database_lock),
                daemon=True,
            ).start()
            _nn = NeuralNetwork(training)
            _nn.run_from_config(training, update_call)
            self.update_state(state="SUCCESS")
    except Exception as _e:
        logger.error(
            "Failure while executing the autokeras model: " + traceback.format_exc()
        )
        self.update_state(state="FAILED")
    finally:
        stop_event.set()
        tf.reset_default_graph()


class NeuralNetwork:
    celery_callback = None
    train_dataset = None
    test_dataset = None

    def __init__(self, training_config: NetworkTraining = None):
        """
        Initializes a NeuralNetwork object.

        Args:
            training_config (NetworkTraining, optional): The configuration object for network training. Defaults to None.

        Returns:
            None
        """
        if training_config:
            self.training_config = training_config
            self.load_data()
            self.build_model_from_config(training_config.network_config)

    def run_from_config(self, config: NetworkTraining, celery_callback):
        """
        Runs the neural network training and validation process based on the provided configuration.

        Args:
            config (NetworkTraining): The configuration object containing the network training settings.
            celery_callback: The callback function to be executed during the training process.

        Returns:
            None
        """
        self.celery_callback = celery_callback

        if config:
            self.training_config = config

        # then load the dataset:

        self.train()
        self.validate()

    def build_model_from_config(self, config: NetworkConfiguration = None) -> None:
        """
        Builds a neural network model based on the provided configuration.

        Args:
            config (NetworkConfiguration): The configuration object specifying the network architecture.

        Raises:
            AttributeError: If the training configuration or hyperparameters are not set.

        Returns:
            None
        """
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
        """
        Loads the training and test datasets using the specified dataset configuration.

        Returns:
            None
        """
        (
            self.train_dataset,
            self.test_dataset,
        ) = self.training_config.dataset.get_data()

        logger.success("Data is loaded.")

    def train(self, training_config: NetworkTraining = None):
        """
        Trains the neural network model.

        Args:
            training_config (NetworkTraining, optional): Configuration for training the network. If not provided,
                the default training configuration of the network will be used.

        Returns:
            None
        """
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
        self.training_config.memory_usage = tf.config.experimental.get_memory_info(
            self.training_config.gpu
        )["current"]
        self.training_config.save()

        logger.success("Finished training of neural network.")

    def validate(self):
        """
        Perform network validation using the test dataset.

        Returns:
            None
        """
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
        """
        Predicts the output for the given dataset using the trained model.

        Args:
            dataset: The input dataset for prediction.

        Returns:
            The predicted output for the dataset.

        Raises:
            None.
        """
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
