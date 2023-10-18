def register_activation_function():
    # This should be used by the plugin to register an activation function
    # it morfe or less just adds an object of ActivationFunction to the database
    raise NotImplementedError


def register_lr_strategy():
    # This should be used by the plugin to register an learning rate strategy function
    raise NotImplementedError


def register_nn_layer():
    # This should be used by the plugin to register a new type of nerual network layer
    raise NotImplementedError
