# Session XII: Models with Attention

*Meeting date: June XX, 2017*

For our fourth consecutive session, we continued to delve into *Natural Language Processing with Deep Learning* by following along with the Stanford [Winter 2017 CS224N lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6). 

A summary blog post, replete with photos of the session, can be found [here](ADD LINK). 


## Recommended Preparatory Work

Because of the extended gap between sessions to create space for recording [Deep Learning with TensorFlow LiveLessons](https://github.com/the-deep-learners/TensorFlow-LiveLessons) for the O'Reilly Safari, the recommended preparatory work for Session XII was double the usual, i.e., lectures ten through fifteen: 

1. [Neural Machine Translation and Models with Attention](https://www.youtube.com/watch?v=IxQtK2SjWWM&index=11&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
1. [Gated Recurrent Units and Further Topics in NMT](https://www.youtube.com/watch?v=6_MO12fPC-0&index=12&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6), and
1. [End-to-End Models for Speech Processing](https://www.youtube.com/watch?v=3MjIkWxXigM&index=13&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
2. [Convolutional Neural Networks](https://www.youtube.com/watch?v=Lg6MZw_OOLI&index=14&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
3. [Tree Recursive Neural Networks and Constituency Parsing](https://www.youtube.com/watch?v=RfwgqPkWZ1w&index=15&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
4. [Coreference Resolution](https://www.youtube.com/watch?v=rpwEWLaueRk&index=16&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)

## Summary


Topic highlights of the session included: 


#### From Lecture 10 (Neural Machine Translation and Models with Attention)

##### NMT: Neural Machine Translation

* NMT is the approach of modelling the *entire* MT process via *one* big ANN
	* in practice, there are often practical comprises on this objective
* e.g.: run input text through `encoder` to create numeric representation (e.g., VSE), then run vectors through `decoder` to generated translated text
* Christopher goes through a few slides on the history of Neural MT (back to Allen, 1987 in IEEE)
* "Workshop on Machine Translation" assess performance of international research groups' models annually
	* from 2013 onward, phrase- and syntax-based SMT had slowly-progressing performance
	* meanwhile, NMT approaches -- first tested only in 2015 -- in 2016 outperformed the traditional approaches
* four "big wins" for NMT relative to other approaches:
	1. end-to-end training: *all* parameters simultaneously optimized to minimize a loss function on network's output
	2. distributed representations share strength: similar words and similar phrases have weights distributed amongst them
	3. better exploitation of context: NMT can use much bigger context (made possible by the distributed, as opposed to one-hot, representations (2.)) -- both source and partial target text -- to translate more accurately 
	4. more fluent text generation: deep learning text generation is much higher quality (as a result of 1., 2., and 3.); indeed, the deep learning fluency can outperform the human one it was trained on
* what did traditional approaches have that NMT doesn't?
	1. black box component models for translation subtasks, e.g., reordering, transliteration
	2. explicit use of syntatic or semantic structures
	3. explicit use of discourse structure, e.g., anaphora
* commercial roll-out of NMT:
	1. 2016-02: Microsoft launched NMT running *offline* on *Android/iOS* (!); traditional codebase would've been too large
	2. 2016-08: Systran launches purely NMT model
	3. 2016-09: Google launches NMT (with overblown hype claiming to equal human translation quality)
	
##### Attention

* the problem: vanilla NMT works well on short sentences, but not long ones
* its effect: it aligns equivalent sections of text in two different languages
	* analogous in effect to phrase-based SMT's alignment of words, which is a preprocessing step for SMT, but a part of the model-fit for NMT
* mechanism of attention: 
	* originated in computer vision (Larochelle & Hinton, 2010; Denil, Bazzani, Larochelle & Freitas, 2012)
	* uses *pool of source states* 
		* similar to Random Access Memory (i.e., "retrieve as needed")
		* if translating English (source) to French (target):
* compare target and source states (from source-state pool) to generate a score, passing through softmax for relative probability...
	
![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/scoring_attention.png)

* ...then build context vector, weighting words by attention scores, to predict next word (hidden state) in French sentence

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/attn_hidden_state.png)

* to calculate score, compare similarity of target and source states
	* Christopher's approach is the **bilinear form** introduced in EMNLP paper with first author Luong:

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/bilinear_form.png)

* focusing on (i.e., backprop-ing through) all source states (i.e., a global approach) is likely to be computationally "unpleasant" for long sequences
* instead, is more efficient (though not higher-performing) to use a subset of all states (i.e., local)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/global_vs_local.png)

* an LSTM without attention can remember up to about 30 words in Christopher's example before it drops off
* attention models required for performance beyond that
* they also perform better than the LSTM with shorter sentences

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/attention_for_long_sentences_plot.png)

* NMTs, particularly those without attention, create elegant sentences, but tend to insert words unrelated to the translation that are semantically sensible

###### Doubly attention

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/doubly_attention.png)

###### Sequence Model Decoders

1. simple and exact decoding algorithm
2. score each and every possible translation
3. pick the best one

This is approach would be intractable in practice. So instead:

* **ancestral sampling**
	* pros: 
		* efficient and unbiased
		* well-suited to academic uses
	* cons:
		* high variance
		* inefficient
		* not well-suited to practical uses
* **greedy search**
	* pro: super efficient (both for computation and memory)
	* con: heavily suboptimal
* **beam search**
	* de facto standard in NMT
	* maintain *K* hypotheses at a given position; pick from top hypotheses
	* "pretty" but not particularly efficient
	* computationally expensive
	* not easy to parallelise
	* much bettery quality than greedy search
	* small beam works well (*K* = 5 or 10); see plot below (higher BLEU score is better)
	* larger beams are more expensive and don't improve BLEU much
	
![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/decoding.png)


#### From Lecture 11 (Gated Recurrent Units and Further Topics in NMT)

##### How Gated Units Fix Things -- Backpropagation through time

* RNNs quantify importance of something at n_t-x on n_t
	* is the gradient vanishing because n_t-x is unimportant or simply because of vanishing gradient? 
		* "like landing aircraft on an aircraft carrier" -- small window to get hyperparameters just right in

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/RNN_visualisation.png)


###### Gated Recurrent Units

* overcome the above issue by "shortcutting" connections -- circumnavigating all of the intermediate backprop 
* key literature: 
	* Cho et al., EMNLP 2014
	* Chung, Gulcehre, Cho, Bengio, DLUFL 2014

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/GRU_shortcut.png)

* **candidate update** equivalent to RNN update
* **update gate** adaptively allows information from far past timesteps to directly interact with the current timestep
* **reset gate** enables parts of the hidden state to be forgotten, otherwise some past information would be around forever

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/GRU_gates.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/GRU_visualisation.png)

* the reset gate decides what portion is "readable"

###### Long Short-Term Memory (Units)

* a specialised type of GRU; widely-used
* more complex than vanilla GRU above as LSTMs as they:
	1. have an additional gate
	2. has "extra *h_t*" output gate: makes "cell" "less exposed" 
* enable memory of previous states to reach back 100 as opposed to 10 with gate-free RNN
* key literature:
	* Hochreiter & Schmidhuber, NC 1999
	* Gers (thesis) 2001
* "cell" (*c*) of LSTM behaves like "hidden state" (*h*) of Cho's GRU
	* *h* in LSTM is different, exposed
* LSTM can both keep all information from the past as well as from the current step
	* Cho's GRUs, on the other hand, have a trade-off between past and present
* LSTM's candidate update and GRU's candidate update are analogous
* the LSTM's "forget gate" is usually written as a "don't forget gate"

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/LSTM_secret.png)

* the first timestep is 127 and we count back to zero here:

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/lstm_vs_rnn_127.png)

* at ~100 timesteps, LSTMs tap out too: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/lstm_vs_rnn_32.png)


##### Steps for Training a (gated) RNN

From the following key literature:

* Saxe et al., ICLR 2014
* Ba, Kingma, ICLR 2015
* Zeiler, arXiv 2012
* Pascanu et al., ICML 2013

...come the following steps:

1. use an LSTM or a GRU 
	* makes life "so much simpler"
2. initialise recurrent matrices to be orthogonal
3. initialise other matrices with a sensible ("small!") scale
4. initialise forget gate bias near one or two
	* i.e., default to "don't forget" / "remember"
	* typically we set things near zero; it's a mistake here because we want to bias toward remembering from the past timesteps
5. use adaptive learning-rate algorithms
	* Adam
	* Adadelta
6. clip the norm of the gradient 
	* 1-5 "seem to be a reasonable threshold when used together with Adam or Adadelta"
7. either only dropout vertically (this is trivial)
	* if you want to do it horizontally, learn how to do it "right" (e.g., Bayesian dropout) as it's tricky
8. "Be patient!"
	* the network may simply need to train longer
	* GPU is essential for any decent-sized data set
	

##### Ensembles

* train 8-10 nets
* average their predictions
* get an extra 2%
* some approaches:
	1. majority voting scheme (OR) 
	2. consensus building scheme (AND)
* key paper: Jung, Cho & Bengio, ACL 2016


##### Machine Translation Evaluation

* manual
	* may be best quality
	* correct vs incorrect
	* adequacy and fluency (e.g., on 5- or 7-point Likert scales)
	* error categorisation (highly subjective)
	* comparative ranking of translations
	* slow
	* expensive
* testing within an application that uses MT as one sub-component
	* e.g., question-answering from foreign language documents
		* may not test many aspects of the transation
* automatic metric
	* ideally fast, cheap to apply
	* WER
		* word error rate
		* problematic because
	* BLEU
		* developed at IBM (Papineni et al., ACL 2002)
		* n-gram precision
		* penalty for brevity
		* "gaming"
			* was thought to be difficult to "game" the metric (i.e., if BLUE goes up, quality does too)
			* should be run with multiple reference translations (recently, many people use one)
			* initial results correlated very well human judgments (e.g., adequacy, fluency)
			* today, there is a perversion: 
				* MT BLEU scores are as high as human translation BLEU scores (e.g., in Google Translate announcements)
				* however, human translations are of much higher quality 

##### The Word Generation Problem: Dealing with a Large Output Vocab

* because languages have so many words
	* the number of softmax parameters is very large
		* e.g., there's a google MT example where half of the computational power is for the softmax layer
* if vocabs are modest (e.g., 50k-word vocab), the target language loses its elegance
* first idea: scale the softmax
	* e.g.:
		* "Hierarchical models": tree-structured vocabulary (Bengio group, 2005, 2009)
		* "Noise-contrastive estimation": binary classification
	* neither of the above approaches are GPU-friendly
	* better: Large-Vocab NMT
		* GPU-friendly
		* fast for training and testing
		* training: 
			* each time train on a smaller subset of the vocab
			* segment the data into categories (e.g., sports, business, etc.)
			* this enables 500k vocab to be covered in ~50k chunks
		* testing:
			* select candidate words
* second idea: scaling the softmax is insufficient
	* new names, new numbers, etc. show up at test time (i.e., in any new piece of text)
	* to be covered in the next lecture! 


#### From Lecture 12 (End-to-End Models for Speech Processing)

##### SECTION TITLE

* bullet





#### From Lecture 13 (Convolutional Neural Networks)

##### SECTION TITLE

* bullet





#### From Lecture 14 (Tree Recursive Neural Networks and Constituency Parsing)

##### SECTION TITLE

* bullet





#### From Lecture 15 (Coreference Resolution)

##### SECTION TITLE

* bullet





# Up Next

For our next session, the recommended preparatory work is: 

1. lecture
2. lecture
3. lecture
