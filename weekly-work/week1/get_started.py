
# coding: utf-8

# # TensorFlow Getting Started Tutorial

# #### from https://www.tensorflow.org/versions/r0.10/get_started/basic_usage.html#interactive-usage

# In[1]:

import tensorflow as tf


# In[2]:

sess = tf.InteractiveSession()


# In[3]:

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])


# In[4]:

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()


# In[5]:

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())


# In[ ]:




# #### from https://www.tensorflow.org/versions/r0.9/get_started/index.html

# In[6]:

import numpy as np


# In[7]:

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


# In[8]:

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b


# In[9]:

# Minimize the mean squared errors
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# In[10]:

# Before starting, initialize the variables. We will 'run' this first. 
init = tf.initialize_all_variables()


# In[11]:

# Launch the graph.
sess = tf.Session()
sess.run(init)


# In[12]:

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))


# In[ ]:



