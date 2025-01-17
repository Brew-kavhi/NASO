import os

import autokeras
import keras_tuner
from decouple import config
from django.db import models
from loguru import logger

from helper_scripts.extensions import (
    custom_hypermodel_build,
    custom_on_epoch_begin_decorator,
    custom_on_epoch_end_decorator,
    custom_on_trial_begin_decorator,
    custom_on_trial_end_decorator,
)
from helper_scripts.importing import get_arguments_as_dict, get_class, get_object
from neural_architecture.models.model_optimization import (
    ClusterableNetwork,
    PrunableNetwork,
)
from neural_architecture.NetworkCallbacks.evaluation_base_callback import (
    EvaluationBaseCallback,
)
from neural_architecture.NetworkCallbacks.timing_callback import TimingCallback
from runs.models.training import (
    CallbackFunction,
    LossFunction,
    Metric,
    Run,
    TrainingMetric,
)

from .types import (
    AutoKerasNodeType,
    AutoKerasTunerType,
    BuildModelFromGraph,
    TypeInstance,
)


# This handles all python classses.
# that is in these types i just want to save what optimizers are availabel and how to call these classes
# instantiation with all arguments is done by the actual models, that jsut have this type assigned
class AutoKerasNode(TypeInstance):
    """
    Represents a node in the AutoKeras architecture.

    Attributes:
        name (str): The name of the node.
        node_type (AutoKerasNodeType): The type of the node.
    """

    name = models.CharField(max_length=50)
    node_type = models.ForeignKey(AutoKerasNodeType, on_delete=models.deletion.CASCADE)

    def __str__(self):
        return self.name


class AutoKerasTuner(TypeInstance):
    """
    A class representing an AutoKeras tuner.

    Attributes:
        tuner_type (ForeignKey): The type of AutoKeras tuner.
    """

    tuner_type = models.ForeignKey(
        AutoKerasTunerType, on_delete=models.deletion.CASCADE
    )


class AutoKerasModel(BuildModelFromGraph, PrunableNetwork):
    """
    A class representing an AutoKeras model.

    Attributes:
        project_name (str): The name of the project.
        blocks (ManyToManyField): The blocks used in the model.
        max_trials (int): The maximum number of trials for the model.
        directory (str): The directory where the model is saved.
        objective (str): The objective used for model optimization.
        tuner (ForeignKey): The tuner used for model tuning.
        max_model_size (int): The maximum size of the model.
        node_to_layer_id (JSONField): A mapping of nodes to layer IDs.
        metrics (ManyToManyField): The metrics used for model evaluation.
        callbacks (ManyToManyField): The callbacks used during training.
        loss (ForeignKey): The loss function used for model training.
        metric_weights (JSONField): The weights for each metric.
        epochs (int): The number of epochs for model training.
        auto_model (AutoModel): The AutoKeras model.
        loaded_model (AutoModel): The loaded AutoKeras model.
        inputs (dict): The input nodes of the model.
        tuner_object: The tuner object used for model tuning.

    Methods:
        build_tuner(run): Builds the tuner object for the model.
        build_model(run): Builds the AutoKeras model.
        load_model(run): Loads the AutoKeras model.
        load_trial(run, trial_id): Loads a trial of the AutoKeras model.

    """

    project_name = models.CharField(max_length=100, default="auto_model")
    blocks = models.ManyToManyField(AutoKerasNode, related_name="Blocks")
    max_trials = models.IntegerField(default=100)
    directory = models.CharField(max_length=100, default=None)
    objective = models.CharField(max_length=100, default="val_loss")
    trial_folder = models.CharField(max_length=256, default="")
    tuner = models.ForeignKey(
        AutoKerasTuner, null=True, on_delete=models.deletion.SET_NULL
    )
    max_model_size = models.IntegerField(null=True)
    node_to_layer_id = models.JSONField(default=dict)

    metrics = models.ManyToManyField(Metric, related_name="autokeras_metrics")
    callbacks = models.ManyToManyField(
        CallbackFunction, related_name="autokeras_callbacks"
    )
    loss = models.ForeignKey(
        LossFunction, on_delete=models.deletion.SET_NULL, null=True
    )
    metric_weights = models.JSONField(null=True)
    epochs = models.IntegerField(default=1000)
    clustering_options = models.ForeignKey(
        ClusterableNetwork, null=True, on_delete=models.deletion.CASCADE
    )

    auto_model: autokeras.AutoModel = None
    loaded_model: autokeras.AutoModel = None
    inputs: dict = {}
    tuner_object = None
    trial_model = {"model": None, "trial_id": 0, "metrics": []}

    def build_tuner(self, run: "AutoKerasRun"):
        """
        Builds the tuner object for the model.

        Args:
            run (AutoKerasRun): The AutoKerasRun object.

        """
        if not self.tuner_object:
            self.tuner_object = get_class(
                self.tuner.tuner_type.module_name, self.tuner.tuner_type.name
            )
            self.tuner_object.on_epoch_end = custom_on_epoch_end_decorator(
                self.tuner_object.on_epoch_end, run
            )
            self.tuner_object.on_epoch_begin = custom_on_epoch_begin_decorator(
                self.tuner_object.on_epoch_begin
            )
            (train_data, _, dataset) = run.dataset.get_data()
            self.tuner_object.on_trial_end = custom_on_trial_end_decorator(
                self.tuner_object.on_trial_end, run, train_data, dataset
            )
            self.tuner_object.on_trial_begin = custom_on_trial_begin_decorator(
                self.tuner_object.on_trial_begin
            )

    def build_model(self, run: "AutoKerasRun"):
        """
        Builds the AutoKeras model.

        Args:
            run (AutoKerasRun): The AutoKerasRun object.

        """
        # build the model here:
        # first build the layers:
        for input_node in self.get_input_nodes():
            self.layer_outputs[input_node] = self.inputs[input_node]
            self.build_connected_layers(input_node)
        # inputs are those nodes who are only source and never target
        # and ouputs is the other way around
        if len(self.directory) == 0 or not self.directory:
            self.directory = f"{self.project_name}_{self.id}"
            self.trial_folder = f"{self.project_name}_{self.id}/{self.project_name}"
            self.save()
        self.build_tuner(run)

        additional_kwargs = {}
        tuner_arguments = get_arguments_as_dict(
            self.tuner.additional_arguments, self.tuner.tuner_type.required_arguments
        )
        if tuner_arguments["max_consecutive_failed_trials"]:
            additional_kwargs["max_consecutive_failed_trials"] = tuner_arguments[
                "max_consecutive_failed_trials"
            ]
        self.auto_model = autokeras.AutoModel(
            inputs=self.inputs,
            outputs=self.outputs,
            overwrite=True,
            max_trials=self.max_trials,
            project_name=self.project_name,
            directory=config("NAS_MODEL_PATH") + self.directory,
            tuner=self.tuner_object,
            metrics=self.get_metrics(),
            objective=keras_tuner.Objective(self.objective, direction="min"),
            max_model_size=self.max_model_size,
            **additional_kwargs,
        )
        self.auto_model.tuner.max_epochs = self.epochs
        self.auto_model.tuner.hypermodel.build = custom_hypermodel_build(
            self.auto_model.tuner.hypermodel.build, run
        )

    def load_model(self, run: "AutoKerasRun"):
        """
        Loads the AutoKeras model.

        Args:
            run (AutoKerasRun): The AutoKerasRun object.

        Returns:
            loaded_model (AutoModel): The loaded AutoKeras model.

        """
        self.inputs = {}
        self.outputs = {}
        self.layer_outputs = {}
        self.tuner_object = None

        for input_node in self.get_input_nodes():
            self.layer_outputs[input_node] = self.inputs[input_node]
            self.build_connected_layers(input_node)
        if len(self.directory) == 0 or not self.directory:
            self.directory = f"{self.project_name}_{self.id}"
            self.trial_folder = f"{self.project_name}_{self.id}/{self.project_name}"
            self.save()

        self.build_tuner(run)
        additional_kwargs = {}
        tuner_arguments = get_arguments_as_dict(
            self.tuner.additional_arguments, self.tuner.tuner_type.required_arguments
        )
        if tuner_arguments["max_consecutive_failed_trials"]:
            additional_kwargs["max_consecutive_failed_trials"] = tuner_arguments[
                "max_consecutive_failed_trials"
            ]

        self.loaded_model = autokeras.AutoModel(
            inputs=self.inputs,
            outputs=self.outputs,
            overwrite=False,
            max_trials=self.max_trials,
            project_name=self.project_name,
            directory=config("NAS_MODEL_PATH") + self.directory,
            tuner=self.tuner_object,
            metrics=self.get_metrics(),
            objective=keras_tuner.Objective(self.objective, direction="min"),
            max_model_size=self.max_model_size,
            **additional_kwargs,
        )

        return self.loaded_model

    def load_trial(self, run: "AutoKerasRun", trial_id: str):
        """
        Loads a trial of the AutoKeras model.

        Args:
            run (AutoKerasRun): The AutoKerasRun object.
            trial_id (str): The ID of the trial.

        Returns:
            trial_model (AutoModel): The loaded trial model.

        """
        if not self.loaded_model:
            self.loaded_model = self.load_model(run)
        weights_path = self.get_trial_checkpoint_path(trial_id)

        # build a dataset to set the inputs size and everything.
        (train_dataset, test_dataset, _) = run.dataset.get_data()

        # need this for the input shapes and so on
        _, hyper_parameters, _, _ = self.prepare_data_for_trial(
            train_dataset, test_dataset, trial_id
        )
        trial_model = self.loaded_model.tuner._try_build(hyper_parameters)
        try:
            # loading weights here is okay as this is just for building the model and the necessary database items.
            # the training model is then build as trensorflow model from the databse
            trial_model.load_weights(weights_path)
        except Exception as exc:
            logger.info(f"{weights_path} weights was not successful, Fehler: {exc}")
        trial_model.compile(
            optimizer=trial_model.optimizer,
            loss=trial_model.loss,
            metrics=self.get_metrics(),
        )
        trial_model.summary()
        return trial_model

    def get_best_model(self) -> str:
        """
        Returns the path to the best model found in this search.
        If it is not finished, then throw an Exception
        """
        best_path = f"{config('NAS_MODEL_PATH')}{self.trial_folder}/best_model"

        if os.path.exists(best_path):
            return best_path
        raise RuntimeError("search is not finished yet and best model cannot be found.")

    def get_trial_checkpoint_path(self, trial_id) -> str:
        """
        Returns the checkpoint path for a specific trial.

        Args:
            trial_id (int): The ID of the trial.

        Returns:
            str: The checkpoint path for the specified trial.
        """
        if os.path.exists(
            f"{config('NAS_MODEL_PATH')}{self.trial_folder}/trial_{trial_id}/checkpoint"
        ):
            return f"{config('NAS_MODEL_PATH')}{self.trial_folder}/trial_{trial_id}/checkpoint"

        file_path = f"{config('NAS_MODEL_PATH')}{self.directory}/{self.project_name}/trial_{trial_id}/checkpoint"
        if os.path.exists(file_path):
            return file_path
        file_path = f"{config('NAS_MODEL_PATH')}{self.project_name}_{self.id}/{self.project_name}/trial_{trial_id}/checkpoint"
        return file_path

    def get_trial_hyperparameters_path(self, trial_id) -> str:
        """
        Returns the path to the trial hyperparameters file.

        Args:
            trial_id (int): The ID of the trial.

        Returns:
            str: The path to the trial hyperparameters file.
        """
        if os.path.exists(
            f"{config('NAS_MODEL_PATH')}{self.directory}/trial_{trial_id}/trial.json"
        ):
            return f"{config('NAS_MODEL_PATH')}{self.directory}/trial_{trial_id}/trial.json"

        file_path = f"{config('NAS_MODEL_PATH')}{self.directory}/{self.project_name}/trial_{trial_id}/trial.json"
        if os.path.exists(file_path):
            return file_path
        file_path = f"{config('NAS_MODEL_PATH')}{self.project_name}_{self.id}/{self.project_name}/trial_{trial_id}/trial.json"
        return file_path

    def prepare_data_for_trial(
        self, train_dataset, test_dataset, trial_id: str
    ) -> tuple:
        """
        Prepares the data for a trial in AutoKeras.

        Args:
            train_dataset: The training dataset.
            test_dataset: The test dataset.
            trial_id: The ID of the trial.

        Returns:
            A tuple containing the pipeline, hyperparameters, trial data, and validation data.
        """
        if not self.loaded_model:
            raise ValueError("load model first")
        if not trial_id:
            raise ValueError("Supply a Trial id")
        trial = self.get_trial(trial_id)

        # convert dataset to a shape , that autokeras can use.
        dataset, validation_data = self.loaded_model._convert_to_dataset(
            train_dataset, y=None, validation_data=test_dataset, batch_size=32
        )
        self.loaded_model._analyze_data(dataset)
        self.loaded_model._build_hyper_pipeline(dataset)

        # finally prepare model and dataset again for the structure of the model. this also reformats the labels
        (
            pipeline,
            trial_data,
            validation_data,
        ) = self.loaded_model.tuner._prepare_model_build(
            trial.hyperparameters, x=dataset, validation_data=validation_data
        )
        self.loaded_model.tuner.hypermodel.set_fit_args(0)
        return (pipeline, trial.hyperparameters, trial_data, validation_data)

    def get_trial(self, trial_id):
        """
        Retrieve a trial object based on the given trial ID.

        Args:
            trial_id (int): The ID of the trial to retrieve.

        Returns:
            Trial: The trial object corresponding to the given trial ID.

        Raises:
            ValueError: If the model has not been loaded yet.
        """
        if not self.loaded_model:
            raise ValueError("load model first")
        trial = self.loaded_model.tuner.oracle.trials[trial_id]
        return trial

    def get_metrics(self):
        """
        Retrieve the metrics used for evaluating the model.

        Returns:
            A list of metric objects used for evaluating the model.
        """
        metrics = []
        for metric in self.metrics.all():
            metrics.append(
                get_object(
                    metric.instance_type.module_name,
                    metric.instance_type.name,
                    metric.additional_arguments,
                    metric.instance_type.required_arguments,
                )
            )
        return metrics

    def get_callbacks(self, run: "AutoKerasRun"):
        callbacks = []
        for callback in self.callbacks.all():
            callbacks.append(
                get_object(
                    callback.instance_type.module_name,
                    callback.instance_type.name,
                    callback.additional_arguments + [{"name": "run", "value": run}],
                    callback.instance_type.required_arguments,
                )
            )
        callbacks += self.get_pruning_callbacks()
        return callbacks

    def get_input_nodes(self):
        """
        Returns a dictionary of input nodes in the neural architecture.

        Input nodes are nodes that are not targets but have at least one source.

        Returns:
            dict: A dictionary where the keys are input nodes and the values are the corresponding blocks.
        """
        for node in self.node_to_layer_id:
            incoming_edges = self.edges_to_target(node)
            if len(incoming_edges) == 0:
                # check if this node is a source somewhere:
                outgoing_nodes = self.edges_from_source(node)
                if len(outgoing_nodes) > 0:
                    self.inputs[node] = self.get_block_for_node(node)
        return self.inputs

    def get_block_for_node(self, node_id):
        """
        Retrieve the block associated with a given node ID.

        Parameters:
            node_id (int): The ID of the node.

        Returns:
            block: The block associated with the given node ID.
        """
        autokeras_node_id = self.node_to_layer_id[node_id]
        autokeras_node = self.blocks.get(pk=autokeras_node_id)
        block = get_object(
            autokeras_node.node_type.module_name,
            autokeras_node.node_type.name,
            autokeras_node.additional_arguments,
            autokeras_node.node_type.required_arguments,
        )
        return block

    # calls the fit method of the autokeras model
    def fit(self, *args, **kwargs):
        """
        Fits the AutoKeras model to the training data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If the model has not been built yet.
        """
        if not self.auto_model:
            raise ValueError("Model has not been built yet.")
        self.auto_model.fit(*args, **kwargs)

    def predict(self, dataset, run: "AutoKerasRun"):
        """
        Predicts the output for the given dataset using the built model.

        Args:
            dataset: The dataset to make predictions on.
            run: An instance of AutoKerasRun.

        Returns:
            The predicted output for the dataset.

        Raises:
            ValueError: If the model has not been built yet.
        """
        if not self.auto_model:
            raise ValueError("Model has not been built yet.")
        batch_size = 1
        timing_callback = TimingCallback()
        export_model = self.get_export_model(self.auto_model.export_model())
        if self.clustering_options:
            export_model = self.clustering_options.get_cluster_export_model(
                export_model
            )
        return export_model.predict(
            dataset,
            batch_size,
            verbose=2,
            steps=None,
            callbacks=[timing_callback]
            + self.get_callbacks(run)
            + [EvaluationBaseCallback(run)],
        )

    def evaluate(self, *args, **kwargs):
        """
        Evaluates the model on a given dataset.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If the model has not been built yet.
        """
        if not self.auto_model:
            raise ValueError("Model has not been built yet.")
        self.auto_model.evaluate(*args, **kwargs)


class AutoKerasRun(Run):
    """
    Represents a run of the AutoKeras model.

    Attributes:
        model (AutoKerasModel): The AutoKeras model associated with this run.
        metrics (ManyToManyField): The training metrics associated with this run.
        prediction_metrics (ManyToManyField): The prediction metrics associated with this run.
    """

    model = models.ForeignKey(AutoKerasModel, on_delete=models.deletion.CASCADE)
    metrics = models.ManyToManyField(TrainingMetric, related_name="autokeras_metrics")
    prediction_metrics = models.ManyToManyField(
        TrainingMetric,
        related_name="autokeras_prediction_metrics",
    )

    inference_metrics = models.ManyToManyField(
        TrainingMetric, related_name="autokeras_inference_metrics"
    )

    def __str__(self):
        return self.model.project_name

    def get_trial_metric(self, trial_id):
        model_size = 0
        avg_power = "NaN"
        measured_power = []
        for metric in self.metrics.all():
            if "trial_id" not in metric.metrics[0]:
                continue
            if metric.metrics[0]["trial_id"] == trial_id:
                model_size = metric.metrics[0]["metrics"]["model_size"]
                if "power_consumption" not in metric.metrics[0]["metrics"]:
                    continue
                measured_power.append(metric.metrics[0]["metrics"]["power_consumption"])
        if len(measured_power) > 0:
            avg_power = sum(measured_power) / len(measured_power)
        return {"model_size": model_size, "average_power": avg_power}

    def get_power_measurements(self):
        if self.power_measurements == "":
            return [
                metric.metrics[0]["metrics"]["power_consumption"]
                for metric in self.metrics.all()
                if "power_consumption" in metric.metrics[0]["metrics"]
            ]
        return super().get_power_measurements()

    def get_energy_consumption(self):
        energy = 0
        for metric in self.metrics.all():
            energy += metric.get_energy_consumption()

        return energy

    def get_times(self):
        times = []
        last_time = 0
        for metric in self.metrics.all():
            if "execution_time" in metric.metrics[0]["metrics"]:
                last_time += metric.metrics[0]["metrics"]["execution_time"]
                times.append(last_time)
        return times
