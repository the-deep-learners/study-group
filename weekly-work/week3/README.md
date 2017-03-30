# Week 3
meeting date: *9-28-2016*

## Covered
- [Chapter 3](http://neuralnetworksanddeeplearning.com/chap3.html) from Nielsen's Ebook
- "Part I: Introduction" of Peleg and Maggio's [Keras tutorial](https://github.com/leriomaggio/deep-learning-keras-euroscipy2016) from EuroSciPy in August

## Nielsen Chapter 3: Improving the way Neural Networks Learn

#### to avoid learning slowdown

- choose cost functions that learn more quickly when the predicted output is far from the desired one, e.g.:
	- if you’d like to consider outputs independently, select sigmoid neurons paired with cross-entropy cost
	- if you’d like to consider outputs simultaneously and as probability distributions, select a softmax layer of neurons with log-likelihood cost
	
#### to avoid overfitting 

- **stop training early**, i.e., when classification accuracy on test data flattens
- use the popular [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) methodology
- artificially expand your data set, e.g., by rotating MNIST digits slightly or adding noise to audio recordings
- regularize: we covered [L1 and L2 regularization](https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization) in detail, with nuclear physicist [Thomas Balestri](https://www.linkedin.com/in/thomasbalestri) leading elucidation 

#### to initialize weights and biases

- to avoid initial saturation of neurons, sample randomly from a normal         distribution with mean of zero and a standard deviation of 1/√(n inputs)

#### Nielsen’s suggested sequence for choosing hyper-parameters

1. Broad Strategy 
	- first, achieve any level of learning that is better than chance
	- this may require simplifying the problem the network is trying to solve (e.g., distinguishing the digits 0 and 1 instead of attempting to classify all ten digits)
	- this may require simplifying the network architecture or reducing the size of the training data by orders of magnitude
	- speed up experimentation by maximizing the frequency with which you can monitor your network, thereby getting instantaneous feedback on performance (and, in my opinion, reducing the opportunity to be distracted by other tasks)
2. Learning Rate 𝜼
	- monitor cost to tune 𝜼 but monitor accuracy for the other hyper-parameters covered here
	- initially adjust 𝜼 by orders of magnitude to find a relatively smooth cost curve, i.e., with minimal oscillation
	- fine-tune 𝜼 to the smooth cost further
	- last, consider a variable learning rate schedule that begins fast (large 𝜼) and slows down (smaller 𝜼), perhaps repeatedly
3. Number of Epochs
	- as mentioned above, early stopping (when classification accuracy on test data flattens out) prevents overfitting
	- having a no-accuracy-improvement-in-n rule (e.g., n = 10 epochs) introduces another hyper-parameter that you could potentially fit as networks can plateau for a while before improving again, but try not to obsess over it
4. Regularization Parameter ƛ
	- initially start with no regularization (i.e., ƛ = 0) while determining the above hyper-parameters
	- use the validation data to select a better ƛ starting with ƛ = 1.0
	- increase or decrease ƛ by orders of magnitude, then fine tune
	- re-visit and re-optimize 𝜼
5. Mini-Batch Size
	- optimal mini-batch size varies as a function of the available memory on your machine, the dimensionality of your data, and the complexity of your neural network architecture
	- if too large, model weights aren’t updated enough; if too small, hardware and software resources are wasted
	- after tuning 𝜼 and ƛ, plot validation accuracy versus real elapsed time to close in on a mini-batch size that maximizes training speed
	- re-visit and re-optimize both 𝜼 and ƛ
6. Automated Techniques
	- you can use a grid search, including open-source software, to optimize hyper-parameters automatically (e.g., [Spearmint](https://github.com/JasperSnoek/spearmint))

#### Variations on Stochastic Gradient Descent

- **Hessian optimization**
	- incorporates the gradient descent analogue of momentum (second-order changes) into weight and bias optimization
	- demonstrably converges on a minimum in fewer steps than standard gradient descent
	- requires considerably more memory than standard gradient descent because of the enormity of the Hessian matrix
- **Momentum-based gradient descent**
	- inspired by Hessian optimization but avoids excessively large matrices
	- to balance between speed and avoiding overshooting a minimum, involves tuning the momentum coefficient μ between zero and one on validation data
- BFGS, limited-memory BFGS, Nesterov’s accelerated gradient
	- these are further popular alternative methods, but we didn’t cover them in any detail

#### Alternative Artificial Neurons

- **tanh**
	- bizarrely, apparently pronounced *tanch*
	- shape approximates the sigmoid function, but ranges from -1 to 1 instead of zero to one, thereby facilitating both positive and negative activations
	- [some evidence](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) suggests it outperforms sigmoid neurons
- **ReLU**
	- rectified linear unit or rectified linear neuron
	- linear, so computationally simpler relative to sigmoid or tanh, but in a network can approximate their performance and nevertheless compute any function


## Applications

In addition to the theoretical work above, we applied our knowledge to software applications:

- untapt’s lead engineer Gabe Rives-Corbett demonstrated the high-level deep-learning library Keras with some of our in-house models as well as Peleg and Maggio’s (above-mentioned) tutorial
- virologist [Grant Beyleveld](https://grantbeyleveld.wordpress.com/) unveiled the neural network he built from scratch in Python and committed into the study group repo [here](https://github.com/the-deep-learners/study-group/tree/master/nn-from-scratch)
