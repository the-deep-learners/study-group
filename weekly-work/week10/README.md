# Session X: word2vec Mania + GANs

*Meeting date: March 27th, 2017*

We continued coverage Stanford's [CS224d](https://cs224d.stanford.edu/) course, which is taught by [Richard Socher](http://www.socher.org/) and focuses on Deep Learning applied to Natural Language Processing. 

In addition, we enjoyed perfectly topical, highly-interactive presentations from: 

1. **[VT Rajan](https://www.linkedin.com/in/vtrajanphd/)**, who whiteboarded through derivations of the word2vec algorithm
2. **Karl Habermas** (slides [here](https://github.com/the-deep-learners/study-group/blob/master/slides/2017-03-27__karl_habermas__CS224d_assignment1.pdf)) on *implementing* the word2vec algorithm (i.e., CS224d 2017 Assignment 1) 
3. **David Epstein** with an introduction to Generative Adversarial Networks (focused on [Goodfellow NIPS 2014](https://arxiv.org/pdf/1701.00160.pdf) and [Radford et al. ICLR 2016](https://arxiv.org/pdf/1511.06434.pdf))

A summary blog post, replete with photos of the session, can be found [here](https://medium.com/@jjpkrohn/deep-learning-study-group-10-word2vec-mania-generative-adversarial-networks-80922e962d1). 


## Recommended Preparatory Work

The recommended preparatory work for Session X was lectures four through six of CS224d (2016), each of which is 75 to 80 minutes long: 

1. [Word Window Classification and Neural Networks](https://www.youtube.com/watch?v=ghYajIXvzOI)
1. [Neural Networks and Backpropagation](https://www.youtube.com/watch?v=Tkt54rYmRdI&t=4s), and
1. [Neural Net Tips/Tricks and Recurrent Neural Networks](https://www.youtube.com/watch?v=MeIrQCZvlkE&t=1754s)

## Summary


Topic highlights of the session included: 


#### From Lecture 4 (Word Window Classification and Neural Networks)

##### Classification setup and notation

* training data sets typically consist of samples *x* and *y*
	* *x_i*: inputs, e.g.:
		* words (either indices or vectors)
		* context windows
		* sentences
		* documents
	* *y_i*: labels we are trying to predict, e.g.:
		* other words
		* class (sentiment, named entities, buy/sell decision)
		* multi-word sequences (this is advanced)
		
##### Classification Intuition

* play around with Andrej Karpathy's [ConvNetJS](http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)

##### Classification: Regularization

* probably obviously, regularization is essential for preventing overfitting and applying a model to unseen data
* create three data sets: 
	1. **training** set: fit the classifier's parameters (i.e., weights)
	1. **validation** set: 
		* tune the model's hyperparameters and/or architecture
		* Socher calls it a development set
	1. **test** set: used only rarely to assess model performance / generalisation (Socher: "use only once per week or once per month")	
	
##### Classification Difference with Word Vectors

* when applying Deep Learning to natural language problems, it's common to simultaneously learn both models weights *W* as well as the word vectors *x*

##### Words Outside the Training Data

* when we train word vectors:
	* words in the training data move around 
	* words in testing data *only* 
* this means words in the test data only can end up on the wrong side of the classifier
* take-home message:
	* with a small training data set, don't train the word vectors
	* with a very large data set, it may work better to train word vectors to the task

##### GloVe

* Global Vectors for Word Representation
* [Pennington, Socher & Manning (2014)](https://nlp.stanford.edu/projects/glove/)
* state-of-the-art unsupervised learning algorithm for obtaining vector representations of words

##### Window Classification

* classifying single words is rare; classifying windows of words is common
* with context under consideration, ambiguity can be resolved
* Richard's examples:
	* auto-antonyms:
		* "to sanction" can mean "to permit" or "to punish"
		* "to seed" can mean "to place seeds" or "to remove seeds"
	* ambiguous named entities:
		* *Paris* can represent the capital of France or a celebrity
		* *Hathaway* can represent a conglomerate or a celebrity (resolving this ambiguity is critical for NLP-based trading signals; see [here](https://www.theatlantic.com/technology/archive/2011/03/does-anne-hathaway-news-drive-berkshire-hathaways-stock/72661/))
* train softmax classifier by assigning a label to a center word and concatenating all of the word vectors surrounding it
	* e.g., classifying *Paris* in the context of a sentence with window length of two:
		* "... museums in Paris are amazing ..."
		* *x_window* = [*x_museums* *x_in* *x_Paris* *x_are* *x_amazing*]^T
		
##### Softmax (~= logistic regression for multi-class problems)

* is not very powerful
* only gives linear decision boundaries in the original space
* with little data, it can regularize well
* with more data, it is limiting

##### Neural Networks

* relative to logistic/softmax, can learn:
	* much more complex functions 
	* nonlinear decision boundaries
	
##### Non-Linearities (e.g., sigmoid function): Why they're needed

* without non-linearities, deep neural networks can't do anything more than a linear transform
* extra layers could be complied down into a single linear transform
* more layers enable approximations of more complex functions




#### From Lecture 5 (Neural Networks and Backpropagation)

##### Building a Neural Net Project

1. define task: 
	* e.g.: summarisation
2. define data set
	1. search for academic data sets
		* conveniently, these already have baselines
		* e.g.: Document Understanding Conference (DUC)
	2. define your own 
		* this is harder as there are no existing baselines
3. define your metric
	* search oneline for well-established metrics on task
	* e.g., for summarisation, use [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))
4. split your dataset
	* train / validation ("dev") / test
	* academic data sets are often pre-split
	* don't use the test set more than once a week (ideally once a month)
5. establish a baseline
	* implement the simplest model (often logistic regression on unigrams and bigrams) first
	* compute metrics on train *and* validation sets
	* analyse errors
	* if the results of at this stage are very good, the problem may be too easy for NN
6. implement existing neural net model
	* compute metric on train *and* validation sets again
	* analyse output and errors
7. always be close to your data
	* visualise the data set
	* collect summary statistics
	* look at errors
	* analyse how varying hyperparameters affect performance
8. try out various model variants, e.g.:
	* word vector averaging model (neural bag of words) 
	* fixed window neural model
	* recurrent neural network
	* recursive neural network
	* convolutional neural network
	
##### Project Ideas

* "default project": [sentiment on movie reviews](https://nlp.stanford.edu/sentiment/)
* summarisation
* [named entity recognition](https://arxiv.org/abs/1103.0398)
* [simple question answering](https://cs.umd.edu/~miyyer/pubs/2014_qb_rnn.pdf)
* [image-to-text mapping or generation](https://nlp.stanford.edu/~socherr/SocherKarpathyLeManningNg_TACL2013.pdf)
* entity-level sentiment
* use deep learning to solve an NLP challenge on Kaggle, like [here](https://www.kaggle.com/c/asap-sas)

##### Max-margin objective function

* objective for a single window is *J* = max(0,1 - *s* + *s_c*)
* if our positive labels are locations, then each window with a location at its centre should have a score +1 higher than any window without a location at its centre
* for full objective function:
	* sample several corrupt windows per true one
	* sum over all training windows



#### From Lecture 6 (Neural Net Tips and Tricks)

##### Unsupervised Word Vector Pre-Training on a Large Text Collection

* ...is sometimes the secret sauce in achieving high model performance 
* Socher provided examples in Part-of-Speech tagging and Named Entity Recogntion

##### General Strategy of Successful Neural Nets

1. select a network structure appropriate for the problem
	1. structure could be:
		* single words, fixed windows, sentence-based, or document-level
		* bag of words, recurrent, recursive, ConvNet
	2. non-linearity (sigmoid, *tanh*, ReLU, etc.)
1. use gradient checks to weed out possible implementation bugs
2. parameter initialisation
3. optimisation tricks
4. check if the model is powerful enough to overfit
	5. if not, change the model structure or make the model "larger"
	5. if you can overfit, regularise

##### Non-Linearities Used

* logistic ("sigmoid")
* tanh
	* rescaled a shifted sigmoid
	* similar to sigmoid, has a nice derivative
	* for many models is superior to sigmoid:
		* backpropagates more efficiently
		* values closer to zero at initialisation
		* faster convergence in practice
* hard tanh
	* similar but computationally cheaper than tanh and saturates hard

##### Gradient Checks

* "are awesome"
* allow you to know that there are no bugs in your neural network implementation
* steps:
	1. implement your gradient
	1. implement a finite difference computation by:
		* looping through the parameters of your network, adding and subtracting a small epsilon (~10^-4) and estimate derivatives
	1. compare the two and make sure they are almost the same
* if gradient fails and you don't know why:
	* create a very tiny synthetic model and data set
	* simply your model until you have no bug, e.g.:
		* only softmax on fixed input
		* backprop into word vectors and softmax
		* add single-unit single hidden layer
		* add more units to single hidden layer
		* add bias
		* add second hidden layer with single unit
			* add multiple units
			* add bias
		* add one softmax on top
		* add second softmax on top
		
##### Parameter Initialization

* from [Glorot and Bengio, 2010](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.2059&rep=rep1&type=pdf)
* initialise hidden layer biases to zero and output (or reconstruct) biases to optimal value if weights were zero (e.g., mean target or inverse sigmoid of mean target)
* initialise weights to ~Uniform(-*r*, *r*), *r* inversely proportional to fan-in (previous layer size) and fan-out (next layer size): (6/(fan-in + fan-out))^1/2
	* ...for *tanh* units and 4 times bigger for sigmoid units

##### Stochastic Gradient Descent

* **gradient descent** uses total gradient over all examples per update, while *SGD* updates after only one or a few examples
* ordinary gradient descent as a batch method is very slow and **should never be used**
	* use second-order batch methods such as L-BFGS
* on large datasets, SGD usually wins over all batch methods
* on smaller datasets, L-BFGS or Conjugate Gradients win
* large-batch L-BFGS extends the reach of L-BFGS ([Quoc Le et al. (2011)](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Le_210.pdf))

##### Mini-Batch SGD

* gradient descent uses total gradient over all examples per update; SGD after only one example
* most commonly used now is **mini-batches** (ranging from 20 to 1000 samples)
* helps parallelise any model by computing gradients for multiple elements of the batch in parallel

##### Momentum

* improvement over SGD
* concept: add a fraction of the previous update to the current one
* when the gradient keeps pointing in the same direction, this will increase the size of the steps taken towards the minimum
* reduce global learning rate *alpha* when using a lot of momentum
* *v* is initialised at 0
* common: *mu* = 0.9
* momentum often increased after some epochs from (0.5 --> )
* Jon Krohn loves the interactive exposition by *distill.pub* [here](http://distill.pub/2017/momentum/)

##### Learning Rates

* simplest recipe: keep it fixed and use the same for all parameters
* results are better when allowing learning rates to decrease over epochs:
	* reduction by 0.5 when validation error stops improving
	* reduction by *O*(1/*t*) because of theoretical convergence guarantees (see slide 30 for formula details)
* even better: no manually-set learning rates by using AdaGrad

##### Adagrad

* **ada**ptive learning rates for *each parameter*
* [Duchi et al., 2011](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
* learning rate adapts for each individual parameter
* rare parameters get larger updates than frequently occurring parameters
* leverages word vectors

##### Preventing Overfitting: Model Size and Regularisation

* simple first step: reduce model size by lowering number of units and layers and other parameters
* standard L1 or L2 regularisation on weights
* **early stopping**: use the parameters that give the lowest validation error

##### Prevent Feature Co-Adaptation

* **dropout** ([Hinton et al., 2012](https://arxiv.org/pdf/1207.0580.pdf); [Hinton et al., 2014](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf))
	* at training time: at each instance of evaluation (in online SGD-training), randomly set 50% of the inputs to each neuron to zero
	* at test time: halve the model weights (there are now twice as many)
	* prevents **feature co-adaptation**: A feature cannot only be useful in the presence of particular other features
	* in a single layer: acts as a kind of middle-ground between Naïve Bayes (where all features weights are set independently) and logistic regression modles (where weights are set in the context of all others)
	* can be thought of as a form of model bagging
	* also acts as a strong regulariser

##### Deep Learning Tricks of the Trade

* from [Bengio (2012)](https://arxiv.org/abs/1206.5533)
	* unsupervised pre-training
	* stochastic gradient descent and setting learning rates
	* main hyperparameters
		* learning rate schedule & early stopping
		* mini-batches
		* parameter initialisation
		* number of hidden units
		* regularisation (i.e., weight decay)
	* how to efficiently search for hyperparameter configurations
		* *random* hyperparameter search


## Up Next

1. the next three lectures of the course, which cover TensorFlow, RNNs, Gated Recurrent Units (GRUs), and Long-Short Term Memory Units (LSTMs)
