Pruning
===
The way this works in tensorflow is, that a pruning method is wrapped around the model or a layer for finer granularity.
Therefor we provide again PruningmetyhodType model that allows for registering different methods for pruning. They need to follow a specific signature though. The function or class it refers to is passed the model or layer to and it needs to return the model or the layer with the pruning function enabled.
For details see [Tensorflow guide on pruning](https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide)

# Training with pruning
We now offer the option to set a model pruning method. (layers are not yet possible, but planned) The whole model is then prunnable and with training it gets automatically prunned. With the option of runing a saved model (via rerun parameter) we are able to fine tune the model with pruning instead of just training with pruning, which is recommended.
During training we need to call the ```UpdatePruningStepcallback```, so the pryuing is working. With ```PrunningSummaries``` we get useful dbuggin help.

## pruning schedule
Different schdulers allow for different sparsity per step based on step and frequency. This is extendable with plugins. 

## Pruning policies
The policy decides whether a layer should be pruned or not.

## Model or layer
Principally, the method works for both, but there are slight differences. When pruning the whole model, we just apply the funhction to the model with or without previsouly loading the weights, that doesnt matter. But i we want to only prune layers, we first have to load the model and the weights and then clone the model and in the clone function apply the pruning method layer-wise.

# Tipps for plugins
The methhod should have a specific signature: ```method(model)->model``` where model refers to either a model or a layer. Furthermore, layers can only be pruned if it implements ```tfmot.sparsity.keras.PrunableLayer```. So if you decide to create custom layers, that you want to be prunable, you need to inherit from this class.