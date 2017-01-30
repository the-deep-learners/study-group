
# coding: utf-8

# # AlexNet

# #### for Oxford's 17 Category Flower Dataset Classification

# #### Based on https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py

# In[1]:

from __future__ import division, print_function, absolute_import


# In[2]:

import tflearn


# In[3]:

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


# #### Import Data

# In[4]:

import tflearn.datasets.oxflower17 as oxflower17


# In[ ]:



