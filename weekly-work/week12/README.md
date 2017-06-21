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

##### SECTION TITLE

* bullet
 




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
