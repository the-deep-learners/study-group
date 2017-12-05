# Session XIII: Model Architectures for Answering Questions and Overcoming NLP Limits

*Meeting date: August 5th, 2017* 

Lucky Session Numero 13 concludes our journey through the Stanford [Winter 2017 CS224N lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) on *Natural Language Processing with Deep Learning*. This session marks the first anniversary of the Deep Learning Study Group -- we held our first meeting on August 17th, 2016. Woot woot. 

A summary blog post, replete with photos of the session, can be found [here](https://insights.untapt.com/deep-reinforcement-learning-our-prescribed-study-path-52b959a61f76). In addition to wrapping up the CS224N lectures (detailed notes from the final four lectures below), we enjoyed delightful presentations and discussions with: 

* [Marianne Monteiro](https://www.linkedin.com/in/mariannelinharesm/) on TensorFlow Recurrent Neural Network implementations that leverage [Estimators](https://www.tensorflow.org/extend/estimators) (see code and resources from her talk [here](https://github.com/mari-linhares/tensorflow-workshop/tree/master/workshops/Deep%20Learning%20Study%20Group%20NYC))
* [Druce Vertes](https://www.linkedin.com/in/drucevertes/) on predicting which financial stories go viral on social media (code [here](https://github.com/druce/deeplearning20170805))




---
## Recommended Preparatory Work

1. [Coreference Resolution](https://www.youtube.com/watch?v=rpwEWLaueRk&index=16&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
2. [Dynamic Neural Networks for Question Answering](https://www.youtube.com/watch?v=T3octNTE7Is&index=17&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
3. [Issues in NLP and Possible Architectures for NLP](https://www.youtube.com/watch?v=B4v545V3Dq0&index=18&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
4. [Tackling the Limits of Deep Learning for NLP](https://www.youtube.com/watch?v=JYwNmSe4HqE&index=19&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)



---
## Lecture Notes

### Lecture 15: Coreference Resolution

#### What is Coreference Resolution? 

* identification of all noun phrases (i.e., mentions) that refer to the same real-world entity
* applications:
	* full text understanding, e.g., understanding an extended discourse
	* machine translation (especially if languages have different features of gender, number, etc.)
	* text summarisation, e.g., web snippets
	* information extraction / question-answering
	
#### Evaluation

* **B^3** algorithm for evaluation
* in essence, is precision and recall calculations for individual entity buckets: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/b_cubed.png)

#### Kinds of Reference

1. referring expressions
2. free variables
3. bound variables

#### Anaphora

* refers back to a word used earlier (e.g., "Jon typed *his* Markdown file.")
* contrasts with less-common, "never-modelled" **cataphora**, which reference a later word
* not all anaphoric relations are coreferential

#### Building Coreference Architectures

* fundamentally, these need to model nouns and pronouns (which are often anaphores) as distinct

##### Traditional Approach

* **Hobbs' Naïve Algorithm** (1976)
	* a complex sequence or rules to follow (nine steps, with "go to" elements)
	* used a feature as most NLP ML systems until a few years ago
	* what Hobbs was really interested in was **Knowledge-based Pronomial Coreference**
		* this requires an understanding of the world and a representation of likely relationships 
		* can be tested with Winograd (1972) challenges; resurrected in 2015

##### Types of Coreference Models

Each of these model types is introduced by Manning: 

1. Mention Pair models
2. Mention Ranking models
3. Entity-Mention models

##### Neural Coreference Models

* "not much has been done" (Manning)
* there are four relevant papers
	* by two author sets 
	* all since 2015
	* two of four Manning co-authored with Kevin Clark
		* one of the Clark & Manning papers involves deep reinforcement learning and is a Mention-Ranking model; Manning describes this approach in detail





---
### Lecture 16: Dynamic Neural Networks for Question Answering

#### Can *Any* NLP Task be Thought of as a Question-Answering Task?

* consider the following:

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/qa_examples.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/more_qa_examples.png)

##### Major Obstacles

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/obstacle_1.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/obstacle_2.png)

#### Dynamic Memory Networks

* an architecture that resolves the first major obstacle (but makes no in-roads on the second)
* multi-task learning
* if we have a problem to tackle like this...

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/harder_questions.png)

* ...a dynamic memory network is well-suited to the task: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/dynamic_memory_network.png)

* as a standard computer-science practice, modules should be independent; changes in one shouldn't affect interoperability with the others at their interfaces

##### Modules

* input module
	* computes hidden state for every word in a single continuous RNN (e.g., GRU) sequence
	* hidden state from last word in previous sentence forming starting point for first word in next sentence

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/input_module.png)

* question vector
	* also RNN, e.g., GRU
	* *can* share weights with input module
	* output hidden state *q* after going through hidden state of every word in question
	* *q*, e.g, incorporating hidden state for "football", triggers attentional mechanisms over input module

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/question_module.png)

* episodic module 
	* incorporates attention by having a global gate that turns off GRUs if deemed irrelevant to the question or memory
	* memory state *m* is final hidden state
	* *m* therefore stores any relevant facts, e.g., about "John" and "football"

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/episodic_module.png)

* *q* and *m* are considered jointly by answer module (softmax with cross-entropy error)
	* facilitates end-to-end training
* this architecture facilitates **transitive reasoning** (understanding relationship of A and D by A->B->C->D logic) by taking multiple passes over the inputs
	* in the example:
		1. first pass picks up significance of "John" to "football"
		2. second pass ties "John" to the hallway where he left the football
* **co-attention mechanisms** to revise context of question by inputs (e.g., "can I cut you?" means different things if input involves knife-wielding vs involving a queue of people)
* "surprisingly broad" set of "reasoning" capabilities as long as that type of reasoning is in the training set

#### Breadth of DMN Tasks with Single Architecture

Involves different hyperparameters between tasks (with number of a passes being a hyperparameter), but same architecture; tasks include: 

* **sentiment analysis** (two passes better than one as attention is sharper

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/sharper_attn.png)

* **part-of-speech tagging** (one pass sufficient)
* **visual question-answering** (state-of-the-art "out of the box")
	* doesn't, however, distinguish distinct objects
	* and can't count a large number of objects (anything more than a handful)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/diff_inputs.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/visual_attn.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/visual_attn_2.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/visual_attn_3.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/tennis_Qs.png)

##### Summary

* most NLP tasks can be reducd to QA
* DMN accurately solves a variety of tasks
* more advanced option: **Dynamic Coattention Networks** (next lecture)

---
### Lecture 17: Issues in NLP and Possible Architectures for NLP

#### People are out to "solve" language

* E.g., Yann LeCun
* Manning doesn't think there will be an IMAGENET/AlexNet single data set/model combo in NLP 
* "instead of just applying LSTMs to language, we could be reaching for the stars" 
* **what we have**: 
	1. BiLSTMs with attention are taking over NLP, improving our ability to do *everything*
	2. neural methods are leading to a *renaissance* for all language generation tasks, e.g., MT, dialog, QA, summarisation
	3. there's a real scientific question of where (and whether!) we need
		* explicit, localist language, and
		* knowledge representations / inferential mechanisms
* **what we still need**:
	1. our methods for building and access **memories** and **knowledge** are very primitive (LSTMs go back up to 100 cells)
	2. current models have almost nothing for developing and executing **goals** and **plans**
	3. we have inadequate abilities for understanding and using **inter-sentence relationships*
	4. we can't, at a large scale, do **elaborations** from a situation using **common sense knowledge**
	
#### Political Ideology Detection Using RNNs

* a paper by *Iyyer, Enns, Boyd-Graber & Reznik (2014)*
	* infers liberalism or conservatism of statements
	* involve TreeRNNs (*recursive* NNs) as discussed earlier
* TreeRNNs
	* pros: 
		* theoretically appealing
		* empirically competitive
	* cons:
		* prohibitively slow (not well-suited to batch computations on GPUs because of input-specific structure)
		* often used with an external parser
		* don't exploit complementary linear structure of language
	* a solution to slowness:
		* proposed by Bowman, Gauthier et al. (2016) as **Shift-reduce Parser-Interpreter NN (SPINN)**
		* has base model equivalent to TreeRNN 
		* supports batched computation on GPUs (25x speedup)
		* bonus: effective new hybrid that combines linear and tree-structured context
		* can stand alone without parser

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/SPINN.png)

* the SPINN model was evaluated on its inference capability with this corpus: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/inference_corpus.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/SNLI_results.png)

* situations where SPINN is better: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/where_SPINN_is_better.png)

#### Below the Word Level: Writing Systems

* the primary idea here is character- or morpheme-level 
* complicated by human-language writing having a fair bit of variability: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/writing_systems.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/ws_2.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/ws_3.png)

##### Morphology 

* in traditional NLP systems, the morpheme was the smallest semantic unit
* in deep learning, it's little studied
	* Luong, Socher & Manning (2013) applied TreeRNN to morphemes
	* could alternatively work with character n-grams (this may have many of the benefits of morphemes, modelled more easily)

#####  Character-Level Models

* word embeddings can be composed from character embeddings
	* generates embeddings for unknown words
	* similar spellings have similar embeddings
	* has proven to work very well
* **character-based LSTM**
	* *Ling et al. (2015, EMNLP)*
	* recurrent language model with BiLSTMs building word representations
	* used as Language Model and for Part-of-Speech tagging

##### Two Trends in Sub-Word NMT

1. **seq2seq** architecture
	* uses smaller units
	* e.g.:
		* Sennrich et al. (ACL 2016)
		* Chung, Cho & Bengio (ACL 2016)
2. **hybrid** architecture
	*  RNN for *words* + something else for *characters*
	* e.g.:
		* Costa-Jussà & Fonollosa (ACL 2016)
		* Luong & Manning (ACL 2016)


---
### Lecture 18: Tackling the Limits of Deep Learning for NLP

#### The Limits of Single-Task Learning

* great performance improvements
* projects start from random
* single unsupervised task can't fix it
* how can we express different tasks in the same framework, 
	* e.g.:
		* sequence tagging
		* sentence-level classification
		* seq2seq?
	* this is important because language involves many different types of inference or knowledge
* in this lecture, Richard tackles Obstacle 2 (**Joint Many-Task Learning**) from Lecture 16

#### Tackling Joint Training

* at this stage, "we should know everything we need to" to build this architecture: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/tackling_joint_training.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/semantic_relatedness.png)

* *chunk training* with delta parameter to prevent lower levels from changing too much (POS weights stay relatively constant): 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/chunking_training.png)

* this multi-task model achieves state-of-the-art results: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/state_of_the_art.png)

#### Obstacle 3: No Zero Shot Word Predictions

* it's natural to learn new words in an active conversation 
	* systems should be able to pick them up
* a solution may be: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/pointer_mixture.png)


#### Obstacle 4: Duplicate Word Representations

* different encodings for encoder (w2v/glove word vectors) and decoder (softmax classification for words)
* duplicate parameters/meaning exist
* a solution: tie word vectors and train single weights jointly
	* theoretically motivated
	* *Inan, Khosravi & Socher (ICLR 2017)*

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/tying_word_vectors.png)

#### Obstacle 5: Questions have input-independent representations

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/QA_independence.png)

#### Obstacle 6: RNNs are Slow

* solution: take the best and parallelisable parts of RNNs and CNNs
	* **Quasi-Recurrent Neural Network**
	* from Bradbury, ..., & Socher (ICLR 2017)
	
#### Obstacle 7: Architecture Search is Slow

* no shit: "manual process that requires a lot of expertise"
* solution: use "AI" to find an optimal architecture for any problem
	* *Neural architecture search with reinforcement learning* (Zoph & Lee, 2016)
	
##### 	Approach

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/architecture_search.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/arch_search_2.png)

##### Performance

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/arch_search_3.png)

##### The Neural Architecture's Solution

* it created a cell type much more complex than LSTM
	* likely has no intuitive underpinnings
	* it simply works well for the task

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week13/img/arch_search_4.png)

#### Lots of Limits for DeepNLP

* Q&A
* Multi-Task Learning
* combined Multimodal, Logical, and Memory-Based Reasoning
* learning from few or single examples
	* current approaches all dependent on lots of relevant training examples


#### Research Highlight: Neural Turing Machines

* references
	* Olah & Carter (2016) *Distill*
	* *deepmind.com/blog/differentiable-neural-computers*

---
## Up Next

For our next session, we will be studying [Reinforcement Learning](https://github.com/the-deep-learners/study-group/tree/master/weekly-work/week14).
