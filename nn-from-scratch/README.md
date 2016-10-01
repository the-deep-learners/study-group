# NN from scratch

Update: I wrote a simple SGD version of the original scipy.optimize script, and then I re-wrote that to incoporate a flexible architecture. Also, I found an error in the weigh update part of the code. It is fixed here.

The purpose here was to  write a neural network "from scratch", which is to say without using any of the available libraries. The advantage being deeper understanding of the principles and how they work, the disadvantages being performance, versatility and effort.

This nn incorporates most of the features we've dealt with so far in the course (that is, up to somewhere in week 3): cross entropy, L2 regularization, and improved weight initialization.

Note: everything is done in Python 3.X so if you ahven't updated yet, expect some things to break (most obviously, print()). Also, if you're on Python 2.X you'll likely want to look at MNIST-loader.ipynb and pickle your own data.

MNIST-nn-scipy.ipynb uses the scipy.optimize L_BFGS optimizer to minimize the cost. This is the kind of method that was deployed in the Coursera course I referenced in the top of the file.

MNIST-nn-SGD.ipynb removes the optimizer in exchange for standard stochastic gradient descent. This more closely matches what we have been studying thus far in the Nielsen textbook and as such it will be where I develop this script further.

MNIST-nn-flex_arch.ipynb is the above SGD-based algorithm but with modifications for a more flexible architecture. This makes the individual steps of forward and backpropogation slightly more opaque, so if you're looking for ease-of-understanding, look elsewhere.

Lastly, the MNIST-loader notebook throws warnings about converting uint8 data into float64 during the scaling process. This didn't seem unusual to me. I'm sure I could suppress the warnings, or do the conversion in the array before passing to the scaler.

The to do list:
- <del>Incoporate gradient descent</del>
- <del>Create more versatility in terms of number of layers, number of neurons per layer</del>
- Incoporate early stopping
- Incoporate a learning rate schedule
- Make use of the validation data (it's sort of ignored in these notebooks for now)