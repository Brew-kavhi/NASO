import threading
import time
import traceback

import keras
import numpy as np
import tensorflow as tf
from loguru import logger

from celery import shared_task
from helper_scripts.extensions import start_async_measuring
from naso.celery import restart_all_workers
from neural_architecture.NetworkCallbacks.base_callback import BaseCallback
from neural_architecture.NetworkCallbacks.evaluation_base_callback import (
    EvaluationBaseCallback,
)
from neural_architecture.NetworkCallbacks.timing_callback import TimingCallback
from runs.models.training import NetworkTraining, TrainingMetric

logger.add("net.log", backtrace=True, diagnose=True)
K = keras.backend


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
    restart_all_workers()
    self.update_state(state="PROGRESS", meta={"run_id": training_id})
    training = NetworkTraining.objects.get(pk=training_id)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    update_call = BaseCallback(
        self,
        run=training,
        epochs=training.fit_parameters.epochs,
        batch_size=training.fit_parameters.batch_size,
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
    except Exception:
        logger.error(
            "Failure while executing the keras model: " + traceback.format_exc()
        )
        self.update_state(state="FAILED")
    finally:
        stop_event.set()
        tf.compat.v1.reset_default_graph()


class NeuralNetwork:
    celery_callback = None
    train_dataset = None
    test_dataset = None
    eval_dataset = None

    def __init__(self, training_config: NetworkTraining = None):
        """
        Initializes a NeuralNetwork object.

        Args:
            training_config (NetworkTraining, optional): The configuration object for network
            training. Defaults to None.

        Returns:
            None
        """
        if training_config:
            self.training_config = training_config
            self.load_data()
            self.build_model_from_config()

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
        config.size_on_disk = config.get_gzipped_model_size()
        config.save()

    def build_model_from_config(self) -> None:
        """
        Builds a neural network model based on the provided configuration.

        Args:

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

        model = self.training_config.build_model(input_shape)

        model = self.training_config.network_model.build_pruning_model(model)
        if self.training_config.network_model.clustering_options:
            model = self.training_config.network_model.clustering_options.build_clustered_model(
                model
            )
        model.compile(**self.training_config.hyper_parameters.get_as_dict())
        logger.success("Model is initiated.")
        model.summary()
        self.training_config.model_size = int(
            np.sum([K.count_params(w) for w in model.trainable_weights])
        )

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
            self.eval_dataset,
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

        timing_callback = TimingCallback()
        callbacks = (
            [timing_callback]
            + self.training_config.fit_parameters.get_callbacks(self.training_config)
            + (
                self.training_config.network_model.get_pruning_callbacks()
                + [self.celery_callback]
            )
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
        self.training_config.save_model_on_disk(self.model)
        if self.training_config.gpu.startswith("GPU"):
            self.training_config.memory_usage = tf.config.experimental.get_memory_info(
                self.training_config.gpu
            )["current"]
        else:
            self.training_config.memory_usage = -1
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
        timing_callback = TimingCallback()
        metrics = self.model.evaluate(
            self.eval_dataset.batch(64),
            batch_size=batch_size,
            steps=steps,
            verbose=2,
            return_dict=True,
            callbacks=[timing_callback]
            + self.training_config.evaluation_parameters.get_callbacks(
                self.training_config
            )
            + [EvaluationBaseCallback(self.training_config)],
        )
        logger.info("evaluation of the network done...")

        eval_metric = TrainingMetric(
            neural_network=self.training_config,
            epoch=self.training_config.fit_parameters.epochs + 1,
            metrics=[{"metrics": metrics}],
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
        # sleep for one second to cool donw the gpu
        # energy measurement returns the average over the last second,
        # so make sure the training does not affect this metric
        time.sleep(1)
        timing_callback = TimingCallback()
        batch_size = 1
        predict_model = self.training_config.network_model.get_export_model(self.model)
        if self.training_config.network_model.clustering_options:
            predict_model = self.training_config.network_model.clustering_options.get_cluster_export_model(
                predict_model
            )
        return predict_model.predict(
            dataset,
            batch_size,
            verbose=2,
            steps=None,
            callbacks=[timing_callback]
            + self.training_config.evaluation_parameters.get_callbacks(
                self.training_config
            )
            + [EvaluationBaseCallback(self.training_config)],
        )
