Experiments
===
This file contains ideas and information about the experiments we want to conduct. Feel free to just edit.

# Types
We willl restrict ourselves to simpler tasks, that is we are not interested in large language models weith lots of transformers or very complex architectures with many skip conections and so on. So we decided to do the experiments in classification and regression. We nned to keep the task fairly difficult though, so that there is actually something to improve upon.

## Classification
DNn for classification tasks. So basically image datasets are always a good idea. We have mnist dataset, which might be a bit small, but might be a good strating point. This can be trained with a simple feed-forward neural net, which i dont have a name for. But i guess, just flattening and a few dense layers should give a pretty good performance.

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
|Boston or California Housing | a frew linear layers| good| small| Regression | not suited
| Diabetes | few linear layers | good| small | Regression| too simple, probably not suited, however the exact task
| Housing prices | MLP | good | medium |Regression | good, i guess, [See on kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)
|NYC taxi fare | Dense layers, sequence | good | small-medium | Regression | okay i guess, [See on kaggle](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/code)
| IMDB reviews | CNN | good enough i guess | medium |Classification | might be suited.
|PascalVOC | resnet | | medium | Classification | might not work
|Stock market | LSTM | good | medium| Regression | Well it is a LSTM
|MIT scene parsing | Resnet | good | medium | Classification | [Online](http://sceneparsing.csail.mit.edu/index_challenge.html)
|Brain tumor | vgg16 | | | Classification |[Brain tumor detection](https://www.kaggle.com/code/nhutdang/brain-tumor-detection-group-7#5) with vgg16