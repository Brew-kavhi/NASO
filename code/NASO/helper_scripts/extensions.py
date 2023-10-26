# Define a decorator function that wraps the on_epoch_end method
def custom_on_epoch_end_decorator(original_on_epoch_end):
    def on_epoch_end(self, trial, model, epoch, logs=None):
        logs["model_size"] = model.count_params()
        if original_on_epoch_end:
            original_on_epoch_end(self, trial, model, epoch, logs)

    return on_epoch_end
