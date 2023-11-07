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

