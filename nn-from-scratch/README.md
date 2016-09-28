# NN from scratch

The purpose here was to  write a neural network "from scratch", which is to say without using any of the available libraries. The advantage being deeper understanding of the principles and how they work, the disadvantages being performance, versatility and effort.

This nn incorporates most of the features we've dealt with so far in the course (that is, up to somewhere in week 3): cross entropy, L2 regularization, and improved weight initialization.

Note: everything is done in Python 3.X so if you ahven't updated yet, expect some things to break (most obviously, print()). Also, if you're on Python 2.X you'll likely want to look at MNIST-loader.ipynb and pickle your own data.

Lastly, the MNIST-loader notebook throws warnings about converting uint8 data into float64 during the scaling process. This didn't seem unusual to me. I'm sure I could suppress the warnings, or do the conversion in the array before passing to the scaler.

The to do list:
- Create more versatility in terms of number of layers, number of neurons per layer
- Implement some form on minibatching? Is this practical / logical given the optimizer is doing the learning?
- Speed. Right now, with 50000 images, it takes quite a while on my core i7 to train the model. SGD and mini-batching could speed this up...
- Switch to a SGD based model and remove the scipy.optimize function altogether