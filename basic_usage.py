
# coding: utf-8

# ### from https://www.tensorflow.org/versions/r0.9/get_started/basic_usage.html

# ## Building the graph

# In[1]:

import tensorflow as tf


# In[2]:

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])


# In[3]:

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])


# In[4]:

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)


# ## Launching the graph in a session

# In[7]:

# Launch the default graph.
sess = tf.Session()


# In[8]:

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print(result)


# In[9]:

# Close the Session when we're done.
sess.close()


# ### To be continued...

# In[ ]:



