# Session 5: Deep (Conv)Nets

Meeting date: November 10th, 2016

## Recommended Preparatory Work

* [Ch. 6 of Michael Nielsen's text (the final chapter)](http://neuralnetworksanddeeplearning.com/chap6.html)
* [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html)

## Summary

#### Three Key Properties of Convolutional Neural Networks

1. local receptive fields
2. shared weights and biases (within a given _kernel_ or _filter_)
3. pooling layers

#### Architecture Changes That Can Improve Classification Accuracy

See [this Jupyter notebook](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week5/network3.ipynb) for a Theano-focused script (based on Nielsen's code and text) that incrementally improves MNIST digit classification accuracy by: 

1. increasing the number of convolutional-pooling layers
2. using ReLU units in place of the sigmoid or _tanh_ variety
3. algorithmically expanding the training data
4. adding fully-connected layers (modest improvement)
5. using an ensemble of networks

#### Why Does ConvNet Training Work (Despite Unstable, e.g., Vanishing, Gradients)?

1. convolutional layers have fewer parameters because of weight- and bias-sharing
2. "powerful" regularization techniques (e.g., dropout) to reduce overfitting
3. ReLU units (quicker training relative to sigmoid/_tanh_)
4. using GPUs if we're training for many epochs
5. sufficiently large set of training data (including algorithmic expansion if possible)
6. appropriate cost function choice
7. sensible weight initialization

#### Other Classes of Deep Neural Nets We Touched on Briefly

1. _recurrent neural networks_ (RNNs), with special discussion of _long short-term memory units_ (LSTMs)
2. _deep belief networks_ (DBNs)

#### TensorFlow for Poets

* makes it trivial to leverage the powerful neural net image-classification architecture of _Inception v3_
* study group member Thomas Balestri quickly trained it into an impressive image-classification tool for consumer products

## Up Next

[CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/) notes and lectures
