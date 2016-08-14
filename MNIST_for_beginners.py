
# coding: utf-8

# ### MNIST for Beginners
# ### from https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html

# ### The MNIST Data

# In[1]:

# The MNIST Data are hosted on Yann LeCun's website, but made available directly by the TensorFlow team.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# ### Implementing Softmax Regression

# In[2]:

import tensorflow as tf


# In[3]:

# Assign placeholder to x that will be filled during computation. 
# We'll be flattening MNIST images into a 784-dimensional vector, 
# represented as a 2-D tensor of floating-point numbers. 
x = tf.placeholder(tf.float32, [None, 784])


# In[4]:

# Assign the model parameters to Variables, which are modifiable tensors
# within a graph of interacting operations. 
# Initialize as zeros. 
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# In[5]:

# Implementation proper takes only one line. 
y = tf.nn.softmax(tf.matmul(x, W) + b)


# ### Training

# In[6]:

# Assign a placeholder into which we'll be inputting correct answers:
y_ = tf.placeholder(tf.float32, [None, 10])


# In[7]:

# Implement cross-entropy, which we'll use as the cost function: 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[8]:

# Use gradient descent to minimize cost with learning rate of 0.5. 
# The beauty of TensorFlow is that we're effortlessly using backpropagation. 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[11]:

# Initialize all variables: 
init = tf.initialize_all_variables()


# In[12]:

# Launch the model within a session: 
sess = tf.Session()
sess.run(init)


# In[15]:

# Train with one thousand iterations. 
# Batches of one hundred random data points are used for stochastic training (i.e., SGD)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# 
# ### Model Evaluation

# In[16]:

# Use argmax to examine whether the most likely predicted label matches reality: 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


# In[17]:

# Cast Booleans to floating point numbers and take mean to assess overall accuracy: 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[18]:

# Run and output to screen: 
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[ ]:



