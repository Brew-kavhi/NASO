import autokeras
import keras_tuner
import tensorflow as tf
from django.core.exceptions import ValidationError
from django.db import models
from keras import backend as K

from helper_scripts.extensions import (
    custom_on_epoch_begin_decorator,
    custom_on_epoch_end_decorator,
    custom_on_trial_begin_decorator,
    custom_on_trial_end_decorator,
)
from helper_scripts.importing import get_class, get_object
from neural_architecture.models.model_runs import KerasModelRun
from runs.models.training import (
    CallbackFunction,
    LossFunction,
    Metric,
    Run,
    TrainingMetric,
)

from .types import BaseType, TypeInstance


# This handles all python classses.
# that is in these types i just want to save what optimizers are availabel and how to call these classes
# instantiation with all arguments is done by the actual models, that jsut have this type assigned
class AutoKerasNodeType(BaseType):
    autokeras_type = models.CharField(max_length=100)


class AutoKerasNode(TypeInstance):
    name = models.CharField(max_length=50)
    node_type = models.ForeignKey(AutoKerasNodeType, on_delete=models.deletion.CASCADE)


class AutoKerasTunerType(BaseType):
    native_tuner = models.BooleanField(default=True)

    def save(self, *args, **kwargs):
        if self.name not in ["greedy", "bayesian", "hyperband", "random"]:
            self.native_tuner = False
            if not self.module_name or len(self.module_name) == 0:
                raise ValidationError(
                    """If the tuner is not a native AutoKeras Tuner we need a class to import the tuner
                    from, module_name cannot be empty."""
                )
        else:
            self.module_name = None
            self.native_tuner = True
        super().save(*args, **kwargs)


class AutoKerasTuner(TypeInstance):
    tuner_type = models.ForeignKey(
        AutoKerasTunerType, on_delete=models.deletion.CASCADE
    )


class AutoKerasModel(models.Model):
    project_name = models.CharField(max_length=100, default="auto_model")
    blocks = models.ManyToManyField(AutoKerasNode, related_name="Blocks")
    max_trials = models.IntegerField(default=100)
    directory = models.CharField(max_length=100, default=None)
    objective = models.CharField(max_length=100, default="val_loss")
    tuner = models.ForeignKey(
        AutoKerasTuner, null=True, on_delete=models.deletion.SET_NULL
    )
    max_model_size = models.IntegerField(null=True)
    connections = models.JSONField(default=dict)
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

    auto_model: autokeras.AutoModel = None
    loaded_model: autokeras.AutoModel = None
    layer_outputs: dict = {}
    inputs: dict = {}
    outputs: dict = {}
    tuner_object = None

    def build_tuner(self, run: "AutoKerasRun"):
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
            self.tuner_object.on_trial_end = custom_on_trial_end_decorator(
                self.tuner_object.on_trial_end
            )
            self.tuner_object.on_trial_begin = custom_on_trial_begin_decorator(
                self.tuner_object.on_trial_begin
            )

    def build_model(self, run: "AutoKerasRun"):
        # build the model here:
        # first build the layers:
        for input_node in self.get_input_nodes():
            self.layer_outputs[input_node] = self.inputs[input_node]
            self.build_connected_layers(input_node)
        # inputs are those nodes who are only source and never target
        # and ouputs is the other way around
        if len(self.directory) == 0 or not self.directory:
            self.directory = f"{self.project_name}_{self.id}"
            self.save()
        self.build_tuner(run)

        self.auto_model = autokeras.AutoModel(
            inputs=self.inputs,
            outputs=self.outputs,
            overwrite=True,
            max_trials=self.max_trials,
            project_name=self.project_name,
            directory="auto_model/" + self.directory,
            tuner=self.tuner_object,
            metrics=self.get_metrics(),
            objective=keras_tuner.Objective(self.objective, direction="min"),
        )

    def load_model(self, run: "AutoKerasRun"):
        if len(self.inputs) == 0 or len(self.outputs) == 0:
            for input_node in self.get_input_nodes():
                self.layer_outputs[input_node] = self.inputs[input_node]
                self.build_connected_layers(input_node)
        if len(self.directory) == 0 or not self.directory:
            self.directory = f"{self.project_name}_{self.id}"
            self.save()

        self.build_tuner(run)

        self.loaded_model = autokeras.AutoModel(
            inputs=self.inputs,
            outputs=self.outputs,
            overwrite=False,
            max_trials=self.max_trials,
            project_name=self.project_name,
            directory="auto_model/" + self.directory,
            tuner=self.tuner_object,
            metrics=self.get_metrics(),
            objective=keras_tuner.Objective(self.objective, direction="min"),
        )

        return self.loaded_model

    def load_trial(self, run: "AutoKerasRun", trial_id: str):
        if not self.loaded_model:
            self.loaded_model = self.load_model(run)

        # build a dataset to set the inputs size and everything.
        data = run.dataset.get_data()
        train_dataset = data[0]
        if len(data) > 1:
            test_dataset = data[1]
        else:
            test_dataset = train_dataset

        # need this for the input shapes and so on
        _, hp, _, _ = self.prepare_data_for_trial(train_dataset, test_dataset, trial_id)
        trial_model = self.loaded_model.tuner._try_build(hp)
        weights_path = self.get_trial_checkpoint_path(trial_id)
        trial_model.load_weights(weights_path)
        trial_model.compile(
            optimizer=trial_model.optimizer,
            loss=trial_model.loss,
            metrics=self.get_metrics(),
        )
        print(trial_model.summary())
        return trial_model

    def save_trial_as_model(
        self, run: "AutoKerasRun", keras_model_run: KerasModelRun, trial_id: str
    ) -> (tf.data.Dataset, tf.data.Dataset):
        """
        This method saves a trial as a KerasModel and returns the train and the validation dataset
        """
        keras_model_run.model.metrics.set(self.metrics.all())

        trial_model = self.load_trial(run, trial_id)
        keras_model_run.model.set_model(trial_model)
        (train_data, validation_data) = keras_model_run.dataset.get_data()
        (_, _, train_dataset, validation_dataset) = self.prepare_data_for_trial(
            train_data, validation_data, trial_id
        )

        # free up the memory
        K.clear_session()
        return (train_dataset, validation_dataset)

    def get_trial_checkpoint_path(self, trial_id) -> str:
        return f"auto_model/{self.directory}/{self.project_name}/trial_{trial_id}/checkpoint"

    def prepare_data_for_trial(
        self, train_dataset, test_dataset, trial_id: str
    ) -> tuple:
        if not self.loaded_model:
            raise ValueError("load model first")
        if not trial_id:
            raise Exception("Supply a Trial id")
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
        return (pipeline, trial.hyperparameters, trial_data, validation_data)

    def get_trial(self, trial_id):
        if not self.loaded_model:
            raise Exception("load model first")
        trial = self.loaded_model.tuner.oracle.trials[trial_id]
        return trial

    def get_metrics(self):
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
        return callbacks

    def edges_from_source(self, node_id):
        return [d for d in self.connections if d["source"] == node_id]

    def edges_to_target(self, node_id):
        return [d for d in self.connections if d["target"] == node_id]

    def is_merge_node(self, node_id):
        # merge node if this node is target opf multiple edges
        return len(self.edges_to_target(node_id)) > 1

    def is_head_node(self, node_id):
        # it is a head node, if this node is not a source:
        return not len(self.edges_from_source(node_id))

    def get_input_nodes(self):
        # input nodes are nodes that are no target. but at list one source:
        for node in self.node_to_layer_id:
            incoming_edges = self.edges_to_target(node)
            if len(incoming_edges) == 0:
                # check if this node is a source somewhere:
                outgoing_nodes = self.edges_from_source(node)
                if len(outgoing_nodes) > 0:
                    self.inputs[node] = self.get_block_for_node(node)
        return self.inputs

    def get_block_for_node(self, node_id):
        autokeras_node_id = self.node_to_layer_id[node_id]
        autokeras_node = self.blocks.get(pk=autokeras_node_id)
        block = get_object(
            autokeras_node.node_type.module_name,
            autokeras_node.node_type.name,
            autokeras_node.additional_arguments,
            autokeras_node.node_type.required_arguments,
        )
        return block

    def build_connected_layers(self, node_id):
        for edge in self.edges_from_source(node_id):
            if self.is_merge_node(edge["target"]):
                can_merge = False
                merge_sources = []
                for merge_edge in self.edges_to_target(edge["target"]):
                    if merge_edge["source"] in self.layer_outputs:
                        can_merge = True
                        merge_sources.append(self.layer_outputs[merge_edge["source"]])
                    else:
                        # as soon
                        can_merge = False
                        break
                if can_merge:
                    self.layer_outputs[edge["target"]] = self.get_block_for_node(
                        edge["target"]
                    )(merge_sources)
            else:
                self.layer_outputs[edge["target"]] = self.get_block_for_node(
                    edge["target"]
                )(self.layer_outputs[edge["source"]])
            if not self.is_head_node(edge["target"]):
                # call this exact same loop again for the target node this time if its not a head
                self.build_connected_layers(edge["target"])
            else:
                # this  node is s head, so add it to outputs:
                self.outputs[edge["target"]] = self.layer_outputs[edge["target"]]

    # calls the fit method of the autokeras model
    def fit(self, *args, **kwargs):
        if not self.auto_model:
            raise ValueError("Model has not been built yet.")
        self.auto_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        if not self.auto_model:
            raise ValueError("Model has not been built yet.")
        self.auto_model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if not self.auto_model:
            raise ValueError("Model has not been built yet.")
        self.auto_model.evaluate(*args, **kwargs)


class AutoKerasRun(Run):
    model = models.ForeignKey(AutoKerasModel, on_delete=models.deletion.CASCADE)
    metrics = models.ManyToManyField(TrainingMetric, related_name="autokeras_metrics")
