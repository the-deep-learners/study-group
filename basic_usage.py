
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

# In[5]:

# Launch the default graph.
sess = tf.Session()


# In[6]:

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


# In[7]:

# Close the Session when we're done to release resources. 
sess.close()


# ## Alternative session launch with "with"

# In[8]:

with tf.Session() as sess:
    result = sess.run([product])
    print(result)


# In[9]:

# If you want to use more than GPU, you need to specify this explicitly,
# for which "with" comes in handy: 
#with tf.Session() as sess:
#    with tf.device("/gpu:1"): # zero-indexed, so this is the second GPU
#        matrix1 = tf.constant([[3., 3.]])
#        matrix2 = tf.constant([[2.],[2.]])
#        product = tf.matmul(matrix1, matrix2)
#        #etc.


# In[10]:

# "with" also comes in handy for launching the graph in a distributed session, e.g.:
#with tf.Session("http://example.org:2222") as sess:


# ## Interactive Usage

# In[11]:

# Great for use within IPython notebooks like this one :)
import tensorflow as tf
sess = tf.InteractiveSession()


# In[12]:

x = tf.Variable([1., 2.])
a = tf.constant([3., 3.])


# In[13]:

# Initialize x with run() method of initializer op. 
x.initializer.run()


# In[14]:

# Add an op to subtract 'a' from 'x'. 
sub = tf.sub(x, a)


# In[15]:

# Print result.
print(sub.eval())


# In[16]:

sess.close()


# ## Variables

# In[18]:

# Create a Variable, which will be initialized to the scalar zero.
state = tf.Variable(0, name="counter")


# In[20]:

# Create an Op to add one to 'state'.
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)


# In[21]:

# Initialize variables. 
init_op = tf.initialize_all_variables()


# In[22]:

# Launch the grph and run the ops. 
with tf.Session() as sess:
    # Run the 'init' op.
    sess.run(init_op)
    # Print the initial value of 'state'.
    print(sess.run(state))
    # Run the op that updates 'state' and print 'state'.
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


# ## Fetches

# In[23]:

# To fetch op outputs, execute the graph with a run() call on the Session object
# and pass in the tensors to retrieve. 
input1 = tf.constant([3.])
input2 = tf.constant([2.])
input3 = tf.constant([5.])
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)


# ## Feeds

# In[24]:

# TensorFlow provides a feed mechanism for patching a tensor directly 
# into any operation in the graph. 
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))


# In[ ]:



