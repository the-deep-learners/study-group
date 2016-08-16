
# coding: utf-8

# # Deep MNIST

# #### Construct a deep convolutional MNIST classifier

# #### from https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html

# ## Load MNIST Data

# In[2]:

from tensorflow.examples.tutorials.mnist import input_data


# In[3]:

# Load training, validation, and testing sets as NumPy arrays. 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# ## Start TensorFlow InteractiveSession

# In[4]:

# The InteractiveSession class is ideal for IPython notebooks like this one. 
# It facilitates flexibility in how you structure your code, 
# and you can alternate between operations that build the computation graph
# with those that run that graph. 
import tensorflow as tf
sess = tf.InteractiveSession()


# ## Build a Softmax Regression Model

# In[ ]:

# Build a softmax regression model with a single linear layer. 


# In[5]:

# Create placeholder nodes for the input images and target output classes. 
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# In[6]:

# Define the weights and biases for the model as Variables.
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# In[8]:

# Initialize variables for use in session.
sess.run(tf.initialize_all_variables())


# In[9]:

# Implement as a softmax regression model. 
y = tf.nn.softmax(tf.matmul(x,W) + b)


# In[10]:

# Specify the model's cost function as cross-entropy. 
# Use reduce_sum to sum across all classes; reduce_mean to take sum of averages. 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# #### Train the Model

# In[11]:

# Select steepest gradient descent, with step length of 0.5, to descend the cross entropy. 
# TensorFlow automatically adds operations to: 
# - compute gradients
# - compute parameter update steps
# - apply update steps to the parameters
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[12]:

# Run train_step to repeatedly apply gradient descent updates to the parameters. 
# Each training iteration (batch) loads fifty training examples, 
# which feed_dict replaces placeholder tensors x and y_ with. 
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# #### Evaluate the Model

# In[13]:

# Use arg_max to identify the label that the model thinks is most likely for each input. 
correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))


# In[14]:

# Convert booleans to floating point numbers. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[15]:

# Evaluate and print to screen. 
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[ ]:

# 90.92% classification accuracy. We can do better. 


# # Build a Multilayer Convolutional Network

# #### Weight Initialization

# In[16]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# In[17]:

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# #### Convolution and Pooling

# In[18]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# In[19]:

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# #### First Convolutional Layer

# In[20]:

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# In[21]:

x_image = tf.reshape(x, [-1,28,28,1])


# In[22]:

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# #### Second Convolutional Layer

# In[23]:

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])


# In[24]:

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# #### Densely Connected Layer

# In[25]:

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])


# In[26]:

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[28]:

# Apply dropout before readout layer to reduce overfitting. 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# #### Readout Layer

# In[29]:

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


# In[30]:

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# #### Train and Evaluate the Model

# In[34]:

# Use ADAM optimizer instead of steepest gradient descent. 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# In[35]:

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# In[ ]:



