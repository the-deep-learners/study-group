# Session XI: Recurrent Neural Networks, including GRUs and LSTMs

*Meeting date: April 19th, 2017*

For our third consecutive session, we focused on [CS224d](https://cs224d.stanford.edu/), which is taught by [Richard Socher](http://www.socher.org/) and covers *Natural Language Processing with Deep Learning*. The Stanford University School of Engineering released the [Winter 2017 lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) on April 3rd, so we began working from that collection. 

In addition, we were treated to relevant talks by two heavy-hitters from the field of data science: 

1. **[Claudia Perlich](https://sites.google.com/site/claudiaperlich/home)** on predictability and how it creates biases when your target is created by mixtures (slides [here](https://github.com/the-deep-learners/study-group/blob/master/slides/2017-04-19__claudia_perlich__predictability.pdf))
2. **[Brian Dalessandro](https://www.linkedin.com/in/briandalessandro/)** on [generating text with Keras LSTM models](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)

A summary blog post, replete with photos of the session, can be found [here](https://insights.untapt.com/deep-learning-study-group-xi-recurrent-neural-networks-including-grus-and-lstms-22c17fa36deb). 


## Recommended Preparatory Work

The recommended preparatory work for Session XI was lectures seven through nine of CS224d (2017), each of which is 75 to 80 minutes long: 

1. [Introduction to TensorFlow](https://www.youtube.com/watch?v=PicxU81owCs&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=7)
1. [Recurrent Neural Networks and Language Models](https://www.youtube.com/watch?v=Keqep_PKrY8&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=8&t=170s), and
1. [Machine Translation and Advanced Recurrent LSTMs and GRUs](https://www.youtube.com/watch?v=QuELiw8tbx8&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=9)

## Summary


Topic highlights of the session included: 


#### From Lecture 7 (Introduction to TensorFlow)

##### Programming Model 

* "the **big idea**": express a numeric computation as a **graph**
	* graph *nodes*:
		* **operations**
		* have any number of inputs and outputs
	* graph *edges*:
		* **tensors**
		* flow between nodes
* **variables**:
	* "stateful" nodes
	* output their current value
	* their state is retained across multiple executions of a graph
	* primarily used for model *parameters*
* **placeholders**:
	* nodes whose values are fed in at execution time
	* used for, e.g., model *inputs*, *labels*
* **mathematical operations**, e.g.:
	* **MatMul**: multiply two matrix values
	* **Add**: add elementwise (with broadcasting)
	* **ReLU**: activate with elementwise rectified linear function

##### Getting Output

* use e.g. `sess.run(fetches, feeds)`
* **fetches**: 
	* list of graph nodes
	* return the outputs of these nodes
* **feeds**:
	* dictionary-mapping from graph nodes to concrete values
	* specifies the value of each graph node given in the dictionary


#### From Lecture 8 (Recurrent Neural Networks and Language Models)

##### Language Models 

* **language model**
	* computes a probability for a sequence of words
	* e.g., `P(w_1, ..., w_T)`
	* useful for machine translation, e.g.:
		* word ordering: `p(the cat is small) > p(small the is cat)`
		* word choice: `p(walking home after school) > p(walking house after school)`

##### Traditional Language Models

* probability is usually conditioned on window of *n* previous words
	* an incorrect, but necessary, Markov assumption
* to estimate probabilities, compute the probability of:
	* unigrams, bigrams
	* ...conditioned on one, two previous word(s)
* even with a small-ish corpus (e.g., 100k words), this quickly becomes a *lot* of probabilities
	* i.e., an exponential increase in n-grams with *n* words

##### Recurrent Neural Networks (to the rescue!)

* RNNs tie the weights at each time step
* condition the neural network on all previous words
* RAM requirement only scales linearly with the number of words
* use the cross-entropy loss function, but predict words instead of classes

##### The Unstable Gradient Problem

* gradients can vanish or explode
* multiplying the same matrix at each step during backpropagation makes training RNNs hard
* typically gradients vanish, and in the case of language modelling or question-answering, words from time steps far away are not taken into consideration when training to predict the next word
* an example where this is a problem: 
	* *Jane walked into the room. John walked in too. It was late in the day. Jane said hi to __.*
* clipping trick for exploding gradients: 
	* introduced by Tomas Mikolov
	* makes a big difference for RNNs
	* clip gradients to a maximum value
	* e.g.: clip large value (say, 100) to some maximum (say, 5) 
	* doesn't work for vanishing gradients because multiplying small numbers would cause jumps over local minimum
* trick for vanishing gradients: 
	* initialise weights to identity matrix I and `f(z) = rect(z) = max(z,0)`
	* makes a "huge difference" (Socher)
	* idea first introduced in [Socher et al. (2013)](https://nlp.stanford.edu/pubs/SocherBauerManningNg_ACL2013.pdf)
	* new experiment with RNNs in [Le et al. (2015)](https://arxiv.org/abs/1504.00941)

##### Sequence Modelling for Other Tasks

* classify each word into:
	* Named Entity Recognition
	* entity-level sentiment in context
	* opinionated expressions
* example application and slides in [Irsoy and Cardie (2014)](https://www.cs.cornell.edu/~oirsoy/files/emnlp14drnt.pdf)

##### Evaluation

* [F1 score](https://en.wikipedia.org/wiki/F1_score) is common
* more hidden layers does *not* always improve network performance



#### From Lecture 9 (Machine Translation and Advanced Recurrent LSTMs and GRUs)

##### Recap of most important concepts (see slides four through six for formulae)

* word2vec
* GloVe
* neural net & max-margin error
* multi-layer neural net & backpropagation
* recurrent neural networks
* cross-entropy error
* mini-batched stochastic gradient descent

##### Machine Translation

* methods are statistical
* use large-scale parallel corpora, e.g., those produced by European Parliament
* the first parallel corpus was the **Rosetta Stone** 
* the systems in traditional approaches are very complex

##### Deep Learning (to the rescue, again!)

* traditional Machine Translation systems required hundreds of curated features and decades of research, leading to many specialised companies
* with short sentences, an RNN encoder (e.g., encoding German)-decoder (outputting English) pair works

##### RNN Translation Model Extensions

1. train different RNN weights for encoding and decoding
2. compute every hidden state in decoder form
3. train deep RNNs, i.e., with multiple layers
4. potentially train bidirectional encoder
5. train input sequence in reverse order for simpler optimisation problem
	* i.e., instead of `ABC --> XY`, train with `CBA --> XY` so that equivalent words tend to be closer
6. **better units**:
	* the "main improvement" (Socher)
	* **Gated Recurrent Units**
		* introduced by [Cho et al. (2014)](https://arxiv.org/abs/1409.1259)
		* keep around memories to capture long-distance dependencies
		* allow error messages to flo at different strengths depending on inputs
		* contain:
			* update gate
			* reset gate
			* *new memory content*: if reset gate unit is ~0, then previous memory is ignored and only the new word's information is stored
			* final memory at time step *t* combines current and previous time steps
		* **take-home message**: essentially, RNNs weight each word equally; GRUs, meanwhile, ignore unimportant words in a sequence while retaining important words in memory
	* **Long Short-Term Memory Units (LSTMs)**
		* introduced by [Hochreiter & Schmidhuber (1997)](http://dl.acm.org/citation.cfm?id=1246450)
		* relative to GRUs, these units are even more complex 
		* at each time step, LSTMs are able to modify:
			* **input gate**: "current cell *matters*"
			* **forget**: gate 0, i.e., forget paste
			* **output**: how much cell is exposed
			* **new memory cell**
		* the final memory cell and final hidden state have separate equations (see slide 41 for all)
		* "very hip" (Socher)
			* *en vogue* default model for most sequence-labelling tasks
			* "very powerful", especially when stacked and made even deeper (each hidden layer is already computed by a deep internal network)
			* most useful if you have "lots and lots" of data
		* in 2015, Deep LSTMs were slightly behind the performance of traditional methods
		* by 2016, Deep LSTMs were unquestionably better (e.g., at [WMT 16](http://www.statmt.org/wmt16/) competition; *metamind ensemble* finished second, but all top performers with respect to [BLEU score](https://en.wikipedia.org/wiki/BLEU) for evaluating machine translation were Deep LSTMs)
		* PCA of vectors from last time-step hidden layer in e.g., in [Sutskever et al. (2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), while they should be interpreted with caution of selection bias, suggest meaning -- not simply word order -- is captured by LSTM approach

![Sutskever_et_al_2014__PCA](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week11/sutskever_et_al_2014__PCA.png)

# Up Next

We are taking a break until early June while I (Jon Krohn) work on an *Introduction to Deep Learning with TensorFlow* project. When we return, we'll cover the next six lectures of the course, which cover: 

1. Neural Machine Translation and Models with Attention
2. GRUs and Further Topics in NMT
3. End-to-End Models for Speech Processing
4. Convolutional Neural Networks
5. Tree Recursive Neural Networks and Constituency Pairing
6. Coreference resolution
