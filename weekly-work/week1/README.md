# Week 1
meeting date: *08-17-2016*

### Covered
- [Chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html) from Nielsen's Ebook
- [Tensorflow Setup](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)
- [Tensoflow intro tutorial](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html)


### Nielsen Chapter 1
- 2 important artificial neuron
	- perceptron
	- sigmoid
- Standard learning algo for neural networks:  Stochastic Gradient Descent (SGD)

#### Perceptrons

- developed in 1950s and 60s
- takes N binary inputs and produces a single binary output
	- output is 1 if weighted sum of inputs is > some threshold
	- output = `{ 0 if w⋅x + b    0; 1 if w⋅b  > 0`
- `bias (b)` is a measure of how easily the neuron "fires"
- Proof
	- Perceptrons can simulate circuit of many NAND gates
	- NAND gates are universal for computation (we can build any computation out of them)
	- => perceptions are universial for computation
- We can devise learning algos that automatically tune the weights and biases of a network of artificial neurons
	- this tuning happens in response to external stimuli
- Instead of laying NAND gates out explicitly, neural networks can simply learn to solve problems

#### Sigmoid neurons

- Perceptrons are touchy. A small change in weight/bias can lead to drastic changes in outputs
- Segmoid's small changes in weight/bias lead to small output changes
- Inputs are not binary (any real between 0 and 1)
- `w*x + b` is now input into the Sigmoid/Logistic function to get the final output
- Sigmoid is a smoothed out step function (step function correlates to perceptron)
  
#### The architecture of neural networks

- input and output layers with hidden layers in between
- multilayer networks can be called multilayer perceptrons (MLPs), despite containing Sigmoid neurons
- Feedfoward
 	- output from one layer is input for next (no loops)
- Recurrent
 	- can have loops
 	- neurons fire for a limited duration
 	- less influential so far
 	- closer in spirit to how the human brain works
  
#### Simple network to classify digits

- 2 parts: digit segmentation and individual digit recognition
- Having good idividual digit recognition allows you to validate segmentation algo, so we'll focus on digit recogintion first
- Digit recognition
 	- input neurons 784 = 28 x 28 grayscale image pixels
 	- output neurons 10 and highest activation value corresponds to the digit estimate
  
#### Learning with Gradient Descent

- MNIST dataset
 	- training: 60,000 handwritten 28 X 28 images from 250 people
 	- test: 10,000 handwritten 28 X 28 images from 250 other people
 	- `y = y(x) = (0,0,0,0,0,0,1,0,0,0)T`
   	- x: 28 X 28 = 784-dime vector of pixel greyvalues
   	- y: 10-dim vector of digit estimates
- Cost function
 	- measures network accurracy
 	- `C(w,b)` closer to 0 => better
- \#images classified correctly is not a smooth function of the weights and biases in the network
- Smooth funciton like quadratic cost is smoother and hence easier to detect improvement of small changes
- We could use calculus to minimize cost function, but that doesn't scale well (could have billions of weights and biases in a NN)
- `Δv=−η∇C`
 	- η is the learning rate or increment SGD algorithm uses to "descend" and minimize C
 	- η too big => approximation could not hold and lead to increase in C
 	- η too small => algorithm is slow
- In practice computing the gradient requires computing individual gradients for each training input
 	- this is very slow for many inputs
 	- SGD approximates gradient by averaging indvidual gradients for a samll sample of inputs
   		- this is called a mini-batch

#### Implementing our Network
- backpropagation
 	- fast way of computing gradient of cost function
