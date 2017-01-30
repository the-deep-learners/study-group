
# coding: utf-8

# # Simple Deep Neural Net for MNIST Classification

# #### based on https://github.com/tflearn/tflearn/blob/master/examples/images/dnn.py

# In[1]:

from __future__ import division, print_function, absolute_import


# In[2]:

import tflearn


# #### Data Loading and Preprocessing

# In[3]:

import tflearn.datasets.mnist as mnist


# In[4]:

X, Y, testX, testY = mnist.load_data(one_hot=True)


# #### Building Deep Neural Network

# In[5]:

input_layer = tflearn.input_data(shape=[None, 784])


# In[6]:

dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh', regularizer='L2', weight_decay=0.001)


# In[7]:

dropout1 = tflearn.dropout(dense1, 0.8)


# In[8]:

dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh', regularizer='L2', weight_decay=0.001)


# In[9]:

dropout2 = tflearn.dropout(dense2, 0.8)


# In[10]:

softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')


# #### Regression using SGD with learning rate decay and Top-3 accuracy

# In[11]:

sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)


# In[12]:

top_k = tflearn.metrics.Top_k(3)


# In[13]:

net = tflearn.regression(softmax, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')


# #### Training

# In[14]:

model = tflearn.DNN(net, tensorboard_verbose=0)


# In[16]:

model.fit(X, Y, n_epoch=10, validation_set=(testX, testY), show_metric=True, run_id='dense_model')


# In[ ]:



