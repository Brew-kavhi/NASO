RUNS
===
This is a file explaiuning what 'run' refers to in the context of this app. A run is basically just the training process of a neural network, or so to say the execution of one neural network task. That is a run does not only mean the training of a tensorflow netowrk, but also the execution of autokeras neural architecture search.

# Tensorflow / Single network
This runs a tensorflow network, training and evaluation. In the run object we need to store the fit parameters, the evaluation parameters, as well as teh hyperparameters of the network. This is, so we know what configuration the network was run with. The accodring model is named NEtworkTraining.

# Autokeras / multiple trials
This is a neural architecture search, that produces multiple trials, which are all part of that one run. This run stores metrics for each trial, we can get the architecture information as dict for each trial, realod trials and execute them. Furthermore, this run also stores some hyperparameters about the search, Like the hypermodel, callbacks, metrics and so on. 

# Trial
A trial is one particular execution/training of a hypermodel built by the neural architecture search in Autokeras. It contains some metrics from the autokeras run and can be rerun. This is all implemened yet. We store these runs in a ModelRun that has KerasModel attribute, where we store the file that defines the model, so we can load the model and then run it. 