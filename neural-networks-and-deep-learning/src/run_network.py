
# coding: utf-8

# # Network from Nielsen's Chapter 1
# http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits

# ## Load MNIST Data

# In[5]:

import mnist_loader


# In[6]:

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


# ## Set up Network

# In[9]:

import network


# In[10]:

# 784 (28 x 28 pixel images) input neurons; 30 hidden neurons; 10 output neurons
net = network.Network([784, 30, 10])


# ## Train Network

# In[12]:

# Use stochastic gradient descent over 30 epochs, with mini-batch size of 10, learning rate of 3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


# ## Exercise: Create network with just two layers

# In[13]:

two_layer_net = network.Network([784, 10])


# In[14]:

two_layer_net.SGD(training_data, 10, 10, 1.0, test_data=test_data)


# In[15]:

two_layer_net.SGD(training_data, 10, 10, 2.0, test_data=test_data)


# In[16]:

two_layer_net.SGD(training_data, 10, 10, 3.0, test_data=test_data)


# In[17]:

two_layer_net.SGD(training_data, 10, 10, 4.0, test_data=test_data)


# In[18]:

two_layer_net.SGD(training_data, 20, 10, 3.0, test_data=test_data)


# In[ ]:



