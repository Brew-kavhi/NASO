Experiments
===
This file contains ideas and information about the experiments we want to conduct. Feel free to just edit.

# Types
We willl restrict ourselves to simpler tasks, that is we are not interested in large language models weith lots of transformers or very complex architectures with many skip conections and so on. So we decided to do the experiments in classification and regression. We nned to keep the task fairly difficult though, so that there is actually something to improve upon.

## Classification
DNn for classification tasks. So basically image datasets are always a good idea. We have mnist dataset, which might be a bit small, but might be a good strating point. This can be trained with a simple feed-forward neural net, which i dont have a name for. But i guess, just flattening and a few dense layers should give a pretty good performance.

> Paper: A Survey of Model Compression and Acceleration
for Deep Neural Networks
> They used mainly image classification with ILSVC-2012 dataset and AlexNet, VGG-16 and GoogleNet models. Advantage: we have examples for prunning efficiency. This paper also investigated VGG-16 net performance on Cifar-100 and Cifar-10 dataset. Further this paper proposes LeNet, All-CNN-nets and the Lenet-300-100 or LeNet-5
Another paper (the survey) compares different NAS algorithms.
> In this paper, we get metrics, for how accurat and big models are that the differet NAS akgorithms have found. They were trained on ImageNet 

Then of course we have the bigger datasets, like cifar-10, that gets more complex and probably already needs a convolutional layer.
Bigger datasets like Cifar-100 and imagenet probably need advanced nets with more convolational layers and recurrent networks. So these are out i guess.

### Sentiment analysis
I dont know if this is suited. Best perforkance is given by transformer models or LSTM and i dont know if we want that. But we can also use Conv networks. We can build models with convulational layers and max pooling tha tare followec by fully connected layers. We also need weord embeddings here. I guess this could be suitable.

### Spam analysis

## Regression
A lot of housding prices tasks. Might be good, we  can use a few linear layers and find a good performing netowrk. If too easy maybe introduce more features. We can also do regression tasks on temporal data like stock market or blood sugar, but this almost always involves LSTMs or RNNs and i donw know if we want that.
I guess for the purpose of regression there are a few datasets on kaggle we can use:
- [Housing prices]()
- [Crab age](https://www.kaggle.com/competitions/playground-series-s3e16/data)


# Collection of datasets
|Dataset | Model | Performance| Size | Type | Comment
|-------|------|----------|-------- | --- | --
|Mnist | Lenet-300-100|good|small | Classification | maybe too small
|Cifar-10 | Resnet-18|good |medium | Classification | good
|Imagenet| Resnet-50 | good | big | Classification | probably too big
|**ILSVC-2012**| GoogleNet, AlexNet, VGG-16 | good(90%) | very big | Classification | was used in a paper
|Boston or California Housing | a frew linear layers| good| small| Regression | not suited
| Diabetes | few linear layers | good| small | Regression| too simple, probably not suited, however the exact task
| Housing prices | MLP | good | medium |Regression | good, i guess, [See on kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)
|NYC taxi fare | Dense layers, sequence | good | small-medium | Regression | okay i guess, [See on kaggle](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/code)
| IMDB reviews | CNN | good enough i guess | medium |Classification | might be suited.
|PascalVOC | resnet | | medium | Classification | might not work
|Stock market | LSTM | good | medium| Regression | Well it is a LSTM
|MIT scene parsing | Resnet | good | medium | Classification | [Online](http://sceneparsing.csail.mit.edu/index_challenge.html)
|Brain tumor | vgg16 | | | Classification |[Brain tumor detection](https://www.kaggle.com/code/nhutdang/brain-tumor-detection-group-7#5) with vgg16
|Cifar-10 | PyramidNet, Shake-Shake | good | big | Classification | see survey on NAS paper, page 38
|ILSVRC | everything | good | everythinh | classification | See Bianco, S., Cadene, R., Celona, L., Napoletano, P., 2018. Benchmark
analysis of representative deep neural network architectures. IEEE
Access 6, 64270–64277. doi:10.1109/ACCESS.2018.2877890

===

# Update from 20.12.2023:
## Types
We have two cases for which we want to investigate the pruning methods and NAS. The first one is classification and the other one is regression. 

### Classification
We have three datsets basically, all of the m are multi-class:
- [ ] mnist: small image classification dataset, can be solved using a linear DNN
- [ ] Cifar10: medium image classification, needs a CNN 
- [ ] DeepSat (Sat-6): Small image classification problem with 6 classes. Donw know yet how to solve it. Eithert by a big linear DNN or by a CNN with lots of linear layers.

### Regression
One dataset for regression, that we use with a linear DNN:
- [ ] californa Housing

### MNIST
Found a suitable net for this task: SoftmaxBig2 is its name in NASO. Performs reasonably well. Loss is around 10% and accuracy is around 97%, which is good enough. It has 483.4882 parameters and three dense layers. Prediction metrics are 129W of energy consumption and 0.015 seconds of inference speed.

### Cifar10
We have a Cifar10 CNN t hat shoudol perfomr reasionably well. Its sparse cateogrical accuraxy is more than 90%. It is a CNN with 122.570 parameters and a total of 3 convolutional layers. 

### DeepSat
The deepsat DNN has a Conblock and a dense block. The first block extracts features and the dense block may classify the features. This model has 184.454 parameters and procudes a categorical accuracy of Over 97% and a loss of below 7%.

### California housing
Regreesion dataset and found a small net with only 11739 parameters. Performs pretty good with loss of 0.9%, validation loss  of around 2 - 3% and mean absolute error of around 6% on trianing data and 10% on unseen test data.

## Experiments
For all the specified datasets, we need a baseline. These are obtained with the above mentioned nets, they perform reasonably well, to be the baseline of our experiments. 
Then we need to run them with pruning and check the investigation goals. That is, we need to log some metrics. First of all the typical peroformance metrics, like loss and accuracy orcategorical accuracy. Furthermore we  need the insight metrics. These are model size, energy consumption and maybe the memory footprint

## Investigation goals/ insights
- [ ] We want to see, how much the network gets smaller when prunning. This is in terms of memory footprint, modelsize regarding the number of parameters and eneergy consumption. 
- [ ] We want to seem, if we can genenrate or find better networks using neural architecture search. In this case again better refeers to smaller model size, less energy consumption and maybe smaller memory footprint.

## Measuring
 we need top measure several hardware specific properties of an run.

### Memory usage
This is currently done using ```tf.config.experimental.get_memory_info(<GPU-name>)```. Gives the usage of the model in bytes.
Need to pay attention when to measure the memory usage. for normal runs it is okay to measure it before fit function, but for pruning runs, we need to measure it after pruning and compacting as pruning apperently only shows off when being compacted.
Problems: 
- maybe the dataset for trining is also included in the measuring which obviously falsify the results. 
Alternatives:
- measuring with ```nvidia-smi``` but this also includes the memory reserved by other appolication on the PC.

### Energy consumption
This is done using nvidia-smi in two ways, on each epoch begin and end a callback measures the current power draw. The second measuring takes place in a separate thread that measures energy consumption every 2 seconds. The thrad measuring gives the more accurate measurements as its just measuring all the time.
Problems with callback:
- may give wrong numbers on long epochs, as the gpu just starts up or cools down
Problems with thread:
- may not be assignable to a autokeras trial, as its just measuring all the time.

### model size
This can be done using tensorflows ```model.count_params``` function. This is accurate and there is no problem with that. This howver cannot replace the memory usage parameter as different parameters require different size in storage

## Pipeline:
1. Baseline: train a baseline model and enable save_model to store the model on the disk. Ensure is has metrics (loss, accuracy, model_size, energy, memory)
2. Pruning: take the pretrained model and enable pruning and retrain it. 
3. NAS: take the same dataset and let NAS run. Ensure again it has the same metrics and especially teh correct objective, to include model_size and energy.
4. NAS & Pruning: Take the best trial according to the metrics objective and retrain it with pruning enabled. Ensure the correct metrics are enabled. 
5. NAS with pruning: Have a look at simultaneously applying pruning to a neural architecure search.

# TODO:
- [ ] Ensure the purning run measures the memory usage and model-size correctly
- [ ] Ensure the model_size and memoy_usage are coretcly obtained in extensions.py for autokeras trials.
- [ ] Ensure the corectb model is choosen when running a trial from autokeras an pruning is applied corectly.
- [ ] For a autoekras trial run, also make sure the memory usage and model_size is measured correctly

