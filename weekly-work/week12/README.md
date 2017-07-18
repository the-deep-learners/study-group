# Session XII: Models with Attention

*Meeting date: July 1st, 2017* (Happy 150th Birthday, Canada!)

For our fourth consecutive session, we delved into *Natural Language Processing with Deep Learning* by following along with the Stanford [Winter 2017 CS224N lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6). 

A summary blog post, replete with photos of the session, can be found [here](https://insights.untapt.com/how-to-understand-how-lstms-work-a5934e9d602d). In addition to discussing the content from four of the CS224N lectures (detailed notes below), we:

* pored over the [details of LSTM units](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) -- thanks to Thomas Balestri for leading this
* were delighted by Grant Beyleveld's [Quick Draw](https://quickdraw.withgoogle.com/) octopus-drawing Generative Adversarial Network: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/octopus-gan.gif)



---
## Recommended Preparatory Work

Because of the extended gap between sessions to create space for recording [Deep Learning with TensorFlow LiveLessons](https://github.com/the-deep-learners/TensorFlow-LiveLessons) for the O'Reilly Safari, the recommended preparatory work for Session XII was double the usual, i.e., lectures ten through fifteen: 

1. [Neural Machine Translation and Models with Attention](https://www.youtube.com/watch?v=IxQtK2SjWWM&index=11&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
1. [Gated Recurrent Units and Further Topics in NMT](https://www.youtube.com/watch?v=6_MO12fPC-0&index=12&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6), and
1. [End-to-End Models for Speech Processing](https://www.youtube.com/watch?v=3MjIkWxXigM&index=13&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
2. [Convolutional Neural Networks](https://www.youtube.com/watch?v=Lg6MZw_OOLI&index=14&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
3. [Tree Recursive Neural Networks and Constituency Parsing](https://www.youtube.com/watch?v=RfwgqPkWZ1w&index=15&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)


---
## Lecture Notes

### Lecture 10: Neural Machine Translation and Models with Attention

#### NMT: Neural Machine Translation

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
	
#### Attention

* the problem: vanilla NMT works well on short sentences, but not long ones
* its effect: it aligns equivalent sections of text in two different languages
	* analogous in effect to phrase-based SMT's alignment of words, which is a preprocessing step for SMT, but a part of the model-fit for NMT
* mechanism of attention: 
	* originated in computer vision (Larochelle & Hinton, 2010; Denil, Bazzani, Larochelle & Freitas, 2012)
	* uses *pool of source states* 
		* similar to Random Access Memory (i.e., "retrieve as needed")
		* if translating English (source) to French (target):
* compare target and source states (from source-state pool) to generate a score, passing through softmax for relative probability...
	
![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/scoring_attention.png)

* ...then build context vector, weighting words by attention scores, to predict next word (hidden state) in French sentence

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/attn_hidden_state.png)

* to calculate score, compare similarity of target and source states
	* Christopher's approach is the **bilinear form** introduced in EMNLP paper with first author Luong:

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/bilinear_form.png)

* focusing on (i.e., backprop-ing through) all source states (i.e., a global approach) is likely to be computationally "unpleasant" for long sequences
* instead, is more efficient (though not higher-performing) to use a subset of all states (i.e., local)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/global_vs_local.png)

* an LSTM without attention can remember up to about 30 words in Christopher's example before it drops off
* attention models required for performance beyond that
* they also perform better than the LSTM with shorter sentences

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/attention_for_long_sentences_plot.png)

* NMTs, particularly those without attention, create elegant sentences, but tend to insert words unrelated to the translation that are semantically sensible

##### Doubly attention

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/doubly_attention.png)

##### Sequence Model Decoders

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
	
![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/decoding.png)




---
### Lecture 11 :Gated Recurrent Units and Further Topics in NMT

#### How Gated Units Fix Things -- Backpropagation through time

* RNNs quantify importance of something at n_t-x on n_t
	* is the gradient vanishing because n_t-x is unimportant or simply because of vanishing gradient? 
		* "like landing aircraft on an aircraft carrier" -- small window to get hyperparameters just right in

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/RNN_visualisation.png)


##### Gated Recurrent Units

* overcome the above issue by "shortcutting" connections -- circumnavigating all of the intermediate backprop 
* key literature: 
	* Cho et al., EMNLP 2014
	* Chung, Gulcehre, Cho, Bengio, DLUFL 2014

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/GRU_shortcut.png)

* **candidate update** equivalent to RNN update
* **update gate** adaptively allows information from far past timesteps to directly interact with the current timestep
* **reset gate** enables parts of the hidden state to be forgotten, otherwise some past information would be around forever

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/GRU_gates.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/GRU_visualisation.png)

* the reset gate decides what portion is "readable"

##### Long Short-Term Memory (Units)

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

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/LSTM_secret.png)

* the first timestep is 127 and we count back to zero here:

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/lstm_vs_rnn_127.png)

* at ~100 timesteps, LSTMs tap out too: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/lstm_vs_rnn_32.png)


#### Steps for Training a (gated) RNN

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
	

#### Ensembles

* train 8-10 nets
* average their predictions
* get an extra 2%
* some approaches:
	1. majority voting scheme (OR) 
	2. consensus building scheme (AND)
* key paper: Jung, Cho & Bengio, ACL 2016


#### Machine Translation Evaluation

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

#### The Word Generation Problem: Dealing with a Large Output Vocab

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


---

### Lecture 12: End-to-End Models for Speech Processing

* guest lecturer: Navdeep Jaitly
	* from U Toronto Hinton Lab
	* most of work discussed in his lecture done while on Google Brain team
	* Socher says: "name is on all of the exciting speech papers in the past few years"
	* now at NVIDIA

#### Automatic Speech Recognition (ASR)

* converts speech to text
* a natural interface for human communication
	* hands-free
	* no need to learn any new skills to use it
* applications are endless
	* controlling devices
		* cars
		* homes
		* handhelds
	* interacting with intelligent devices
		* chatbots
		* call-centre help desks
		* our machine overlords
		
#### Traditional Speech Recognition

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/traditional_ASR.png)

#### Neural Network Approach to Speech Recognition

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/nn_ASR.png)

* each component is trained independently, with different objective functions
* errors in one component may not behave well with errors in another component
* end-to-end models would be better, e.g.:
	* Connectionist Temporal Classification (CTC)
		* used in production systems at Baidu and Google
		* requires a lot of training
	* Sequence to Sequence
		* trend is in the direction of this approach
		* e.g., "Listen Attend and Spell", which is the focus of Navdeep's lecture

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/end_to_end_ASR_as_model.png)

* works with raw audio
* performs better with minimal preprocessing, i.e., log spectrogram
	* emulates human tendency to hear well in narrow middle range only 
	* humans require logarithmic differences above or below that range

#### Connectionist Temporal Classification

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/CTC_peaks.png)

* produces correct sounds but lacks correct spelling and grammar
	* by using language model to rescore or during training (e.g., with "OK, Google"), this can be fixed
* "no big data" (or "big enough") available in this domain
	* even google trains on fairly small data set (81 hours -- Wall Street Journal data)

#### Listen Attend and Spell

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/seq2seq_ASR.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/seq2seq_ASR_attn.png)

* for attention: 
	* calculate similarity scores between the decoder "query" and encoder "state" timesteps
	* normalise with softmax to create attention vector
	* linear blend encoder states using attention vector
* hierarchical encoder (Chan, Jaitly et al., ICCSP 2015)
	* breaks up timesteps that need to be backpropagated through
* multimodal outputs
	* same input has multiple outputs with various high-probability guesses
* retains "causality"
* limitations
	* not an online model (all input must be received before transcripts can be produced)
	* attention is a computational bottleneck (every output token pays attention to every input timestep)
	* word error rate goes up at very short lengths

#### Online Seq2Seq

##### Neural Transducer

* seq2seq models on local chunks of data
* outputs are produced as inputs are received
* maintains causality
* inference is done by using beam search to find highest probability output sequence for an input
* beam search fails easily at input-to-output alignment
	* "Approximate Dynamic Programming" fares better

##### Very Deep Convolutional Encoders

* "Very Deep Convolutional Encoders" (Pyramidal RNNs) reduce the time resolution of inputs
	* like image patches, acoustic signals occur within local windows that are well-suited to convolutiona

#### Choosing Output Targets

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/choosing_output_targets.png)

* even better:

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/choosing_better_targets.png)

* a problem with this approach is that during sequence generation (decoding), shorter, less accurate senteneces have the lowest cost, e.g., "" may have the lowest cost of all
	* the solution is to add a "coverage reward", which rewards model for generating longer sentences by lowering cost for this

#### ASR with Simultaneous Translation

* is possible
* attention corresponds to same sections of sentence as with those methods discussed in lecture 10 (i.e., MT alone -- no ASR)




---

### Lecture 13: Convolutional Neural Networks

#### From RNNs to CNNs

* RNNs are "awesome" but have some issues:
	* can't capture phrases without prefix context that feeds into later words (for classification, you may only be interested in later words but they're inextricable from earlier parts of phrase)
	* relatedly, RNNs often capture too much of last words in final vector (softmax is often only at the last step)
* CNNs
	* resolve some of these issues
	* main idea: what if we compute vectors for every possible (sub-)phrase within a phrase of words
		* regardless of whether sub-phrase is grammatical
		* often sub-phrase is not linguistically or cognitively plausible
		* afterward, group the sub-phrases

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/what_is_a_convolution.png)

* the equivalent to a pixel in NLP is a word vector

#### Single-Layer CNN

* simple variant using one convolutional layer and pooling
* key literature:
	* Collobert and Weston (2011)
	* Kim (2014) "CNNs for Sentence Classification"
* filter of size *k*=2 convolves over bigrams, *k*=3 trigrams

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/single_layer_CNN.png)

* add zero-padding to both the end of the sentence and (not shown in next slide!) the beginning of the sentence

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/CNN_feature_map.png)

* pooling layer
	* "max-over-time" pooling layer (= max-pooling layer)
	* captures the most important activations (i.e., the maximum over time)
	* in theory would work:
		* min-pooling *but not if you use ReLU units!*
		* average-pooling: there's literature on this, but this may wash strongest effects out
* multiple filters
	* ideally with different lengths
	* capture significant unigrams, bigrams, trigrams, 4-grams, etc.
* multi-channel idea
	* overcomes the "television" / "telly" problem covered in an earlier lecture
	* initialise with pre-trained word vectors (w2v or GloVe)
	* start with two copies 
	* backprop into only one set, keep other "static"
	* both channels are added to *c_i* before max-pooling
* top-level steps:
	1. one convolution
	2. one max-pooling
	3. simple final softmax layer
* use dropout
	* Kim (2014) had 2-4% improved accuracy and ability to use very large networks without overfitting
* hyperparamaeter tuning:
	* Richard follows Bengio's suggestion to set a range for each hyperparameter and sample randomly over the hyperparameter space, cross-validating on each separately
	* ideally running ~5 models at each best location and ensembling them
* model comparisons to historically significant models are often flawed, straw men
	* author isn't incentivised to spend time optimising existing leading approaches or incorporating state-of-the-art techniques (e.g., supplementing a pre-2014 model with the Dropout technique devised since)

#### CNN Alternatives

* narrow vs wide convolutions
* complex pooling schemeds (over sequences)
* deeper convolutional layers
* key literature: Kalchbrenner et al. (2014)
* Richard: "at some point, there is no more intuition on why" we should make one decision or another
* Kalchbrenner and Blunsom (2013):
	* one of the first successful machine translation efforts
	* CNN for encoding
	* RNN for decoding

#### Models for Comparison of NLP Performance

* **bag of vectors**
	* "surprisingly good baseline for simple classification problems"
	* ...especially if followed by a few ReLU layers
* **window model**
	* good for single-word classification for problems that don't need wide context
* **CNNs**
	* good for classification
	* can't incorporate phrase-level annotation (can only take a single label)
	* need zero padding for shorter phrases
	* hard to interpret
	* easy to parallelise on GPUs
* **RNNs**
	* cognitively plausible (i.e., reading from left to right)
	* not best for classification
	* slower than CNNs
	* can do sequence tagging and classification
	* very active research

#### Character-Level Encoding

* improves classification models by a few percent
* has fewer parameters (a couple dozen characters versus tens of thousands of word vectors)
* input sequences are much longer (because there are 5-10 characters per word)
	* net effect is that models tend to be much longer to train

#### Training with Development Data

* given training data split three ways:
	1. training
	2. dev(elopment)
	3. test
* we could, in theory:
	* repeatedly train with training data, validate with dev set, to select optimal hyperparameters
	* with optimal hyperparameters, combine training and dev set to more accurately identify global minimum
* in practice:
	* this approach works well with convex problems
	* Deep Learning models are rarely clear convex problems with a clear global minimum
	


---
### Lecture 14: Tree Recursive Neural Networks and Constituency Parsing

* Christopher theorises language has an inherent tree structure
* recursive is "some kind" of recursion but distinct from RNNs proper, involve tree structure

#### The Spectrum of Language in Computer Science

* **compositionality**
	* how can we know when larger units are similar in meaning?
	* e.g., (snowboarder) = (person on a snowboard)
	* language understanding and AI require being able to understand bigger things from knowing about smaller parts
	* tree recursive networks are one solution
	* here's a computer vision analogy: 
	
![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/heres_the-church_here_are_the_people.png)

* recursive structure:
	* used in CS algos
	* Chomsky posits that recursive language capabilities is what makes humans uniquely intelligent
	* as above, Christopher theorises language has an inherent tree structure
	
![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/are_languages_recursive.png)

##### Building on Word Vector Space Models

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/bldg_on_WVSMs.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/phrases_in_vector_space.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/learned_tree_structure.png)

##### RNNs vs RNNs

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/recursive_vs_recurrent_NN.png)

* recurrent networks capture prefixes (Richard mentioned above), but not whole sentence structure
* despite this -- the "good linguistic reasons" for liking recursive networks -- recursive networks haven't "swept the world" like recurrent ones have (the latter used 10x as much)
* the drawback of recursive models is that it requires categorical choices that don't backprop well and aren't GPU-friendly

##### RNNs vs CNNs

* recursive NNs: get compositional vectors for grammatical phrases only
* CNNs: computes vectors for every possible sub-phrase (discussed above)
	* many of the sub-phrases aren't grammatical and don't make sense
	* doesn't need a parser
	* maybe not linguistically or cognitively plausible

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/RNNs_vs_CNNs.png)

#### Recursive NNs for Structure Prediction

* it *is* possible to backpropagate through tree structure (Goller & Küchler, 1996)

#### Simple TreeRNNs 

* "decent results" with single weight matrix
* could capture some phenomena but not adequate for more complex, higher-order composition or parsing long sentences
* no real interaction between input words

#### TreeRNN v2: Syntatically-Untied RNN

* symbolic Context-Free Grammar (CFG) backbone is adequate for basic syntatic structure
* I don't really understand any of this *composition matrix* business, but it is the critical difference that made v2 the better performer
* **Compositional Vector Grammar** = PCFG + TreeRNN

#### TreeRNN v3: Compositionality through Recursive Matrix-Vector Spaces

* proposes a new composition function
* pretty fun sentiment distributions: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week12/img/sentiment_distributions.png)



---

## Up Next

For our next session, the recommended preparatory work is wrapping up viewing the CS224N lectures, i.e.:

1. [Coreference Resolution](https://www.youtube.com/watch?v=rpwEWLaueRk&index=16&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
2. [Dynamic Neural Networks for Question Answering](https://www.youtube.com/watch?v=T3octNTE7Is&index=17&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
3. [Issues in NLP and Possible Architectures for NLP](https://www.youtube.com/watch?v=B4v545V3Dq0&index=18&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
4. [Tackling the Limits of Deep Learning for NLP](https://www.youtube.com/watch?v=JYwNmSe4HqE&index=19&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
