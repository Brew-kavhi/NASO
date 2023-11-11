import tensorflow as tf

from runs.models.training import NetworkTraining


class EvaluationBaseCallback(tf.keras.callbacks.Callback):
    def __init__(self, run: NetworkTraining):
        super().__init__()
        self.run = run
        # i need the epochs

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"Starting training; got log keys: {keys}")

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print(f"Stop training; got log keys: {keys}")

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.additional_callbacks:
            callback.on_epoch_begin(epoch, logs)
        keys = list(logs.keys())
        print(f"Start epoch {epoch} of training; got log keys: {keys}")

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.additional_callbacks:
            callback.on_epoch_end(epoch, logs)
        keys = list(logs.keys())
        print(f"End epoch {epoch} of training; got log keys: {keys}")

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"Start testing; got log keys: {keys}")

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print(f"Stop testing; got log keys: {keys}")

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"Start predicting; got log keys: {keys}")

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print(f"Stop predicting; got log keys: {keys}")

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Training: start of batch {batch}; got log keys: {keys}")

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Training: end of batch {batch}; got log keys: {keys}")

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Evaluating: start of batch {batch}; got log keys: {keys}")

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Evaluating: end of batch {batch}; got log keys: {keys}")

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Predicting: start of batch {batch}; got log keys: {keys}")

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Predicting: end of batch {batch}; got log keys: {keys}")
