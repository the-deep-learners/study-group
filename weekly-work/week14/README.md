# Session XIV: Reinforcement Learning

*Meeting date: October 17th, 2017* 

We kicked off our coverage of the field of Reinforcement Learning with this session. In addition to discussing all of the Recommended Preparatory Work (see below), we also enjoyed theory and code demos from Thomas Balestri (slides [here](https://github.com/the-deep-learners/study-group/blob/master/slides/2017-10-17__thomas_balestri__reinforcement_learning.pdf)) based largely on the final chapter of Aurélien Géron's [Hands-On Machine Learning](http://shop.oreilly.com/product/0636920052289.do). A summary blog post, replete with photos of the session, can be found [here](https://insights.untapt.com/deep-reinforcement-learning-our-prescribed-study-path-52b959a61f76).


---
## Recommended Preparatory Work

1. [The introductory lecture](https://www.youtube.com/watch?v=8jQIKgTzQd4) from the University of California, Berkeley's (CS294-112) *Deep Reinforcement Learning* Course
2. Emma Brunskill's *Tutorial on Reinforcement Learning* (both parts [one](https://www.youtube.com/watch?v=fIKkhoI1kF4) and [two](https://www.youtube.com/watch?v=8hK0NnG_DhY))



---
## Lecture Notes

### UC Berkeley CS294-112 Lecture I (1/18/17) on Deep Reinforcement Learning

#### Three Course Instructors

In this lecture, each of the instructors provides an individual high-level overview of Deep RL. 

1. Sergey Levine (Berkeley Asst. Prof.)
2. John Schulman (OpenAI Research Scientist)
3. Chelsea Finn (Berkeley Ph.D. Student)

#### Broad Course Syllabus

1. From Supervised Learning to Decision Making
2. Basic RL: Q-Learning and Policy Gradients
3. Advanced Model Learning and Prediction, Distillation, Reward Learning
4. Advanced Deep RL
	* trust region policy gradients
	* actor-critic methods
	* exploration
5. Open Problems, Research Talks, Invited Lectures

#### Assignments

1. Imitation Learning (Control via Supervised Learning)
2. Basic (Shallow) RL
3. Deep Q Learning
4. Deep Policy Gradients
5. Research-Level Project in Groups of 2-3 students

#### Sergey Levine (Berkeley)

* ways to build AI (or, *what is the basis of intelligence?*)
	1. replicate the modules of the (e.g., human) brain in a qualitative way since we know that that is a proven method
	2. 	use the process of **learning** alone as the basis of intelligence
		* in this case, is there a single, flexible algorithm for learning (this may be the case) instead of needing separate algorithms for separate neural modules (e.g., V1 vs V4 vision corticies)
		* evidence for a single learning algorithm inclues:
			* humans seeing with their tongue
			* human echolocation
			* mapping vision to the auditory cortex in experimental animals
			* neocortex is uniform across brain 

##### What must a single algorithm be able to do?

1. interpret rich, extremely high-dimensional sensory inputs (like eyes, ears, noses, etc.)
2. choose complex actions (it would theoretically be possible for a small number of commands to control a large range of muscle movements; this may happen in some insects, but doesn't happen in most animals)

##### Why Deep RL?

The following two points line up with the two points in the preceding section. 

1. deep = can process complex sensory input
	* ...and can compute "really" complex functions
2. RL = can choose complex actions

##### Evidence in Favor of Deep Learning

* [Saxe et al. (2011) NIPS](https://papers.nips.cc/paper/4331-unsupervised-learning-models-of-primary-cortical-receptive-fields-and-receptive-field-plasticity.pdf)
	* demonstrates that in several senses -- vision, audition, and touch -- neural receptive field adapt in similar ways
	* suggests that "a qualitatively similar learning algorithm acts throughout primary sensory cortices"

##### Evidence in Favor of RL

* RL was an animal behaviour theory before it was "ported" over to Computer Science
* see, e.g., [Niv (2009)](https://www.princeton.edu/~yael/Publications/Niv2009.pdf)
	* percepts that anticipate reward become associated with similar firing patterns as the reward itself
	* basal ganglia appear to be related to the reward system
	* model-free RL-like adaptation is often (but not always) a good fit for experimental data of animal adaptation

##### What DL & RL can do well now

* acquire a high degree of proficiecy in domains governed by simple, known rules (e.g., Atari, Go)
* learn simple skills with raw sensory inputs, given enough experience (e.g., robot that learns to grasp objects of many shapes)
* learn from imitating enough human-provided expert behaviour (e.g., driving a car)

##### What has proven challenging so far?

* humans (and other animals, e.g., rats) can learn incredibly quickly
	* Deep RL methods are usually slow
* humans can reuse past knowledge
	* transfer learning in Deep RL is an open problem (in fact, it's "not even well-defined")
* not clear what the reward function should be
* not clear what the role of prediction should be

##### A Relevant Alan Turing Quote

*Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's? If this were then subjected to an appropriate course of education, one would obtain the adult brain*


#### John Schulman (OpenAI)

##### What is RL?

* Sergey introduced RL via analogy to neuroscience
* John formalises the definition of RL:
	* branch of ML concerned with taking sequences of actions
	* usually described in terms of an agent that:
		* interacts with a previously unknown environment
		* tries to maximize a *cumulative* reward
	* can be called a POMDP (Partially Observable Markov Decision Process)
* N.B.: the above formal definitions don't cover imitation learning

##### Applications of Decision-Making Problems

* business operations
	* inventory management
		* **observations**: current inventory levels
		* **actions**: number of units of each item to purchase
* as part of other ML problems
	* classification with Hard Attention [(Mnih et al., 2014 NIPS)](https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)
		* **observation**: current image window
		* **action**: where to look (crop the image window to a specific, likely-most-relevant, section of it)
		* **reward**: +1 for correct classification
	* sequential/structured prediction, e.g., in machine translation
		* **observations**: words in source language
		* **action**: emit word in target language
		* **reward**: sentence-level accuracy metric, e.g., BLEU score 

##### What is Deep Reinforcement Learning?

* *deep* as an adjective in ML tends to mean using NNs to approximate functions
* in RL you can be more creative with your NN positioning within your model architecture than in most other types of ML
	* three common possible places to include an NN:
		1. **policies**: select next action
		2. **value functions**: measure goodness of states or state-action pairs
		3. **dynamics models**: predict next state and rewards

##### How does RL relate to other ML problems

1. **supervised learning**
	* environment samples input-output pair
	* agent predicts *y_hat*
	* agent receives loss *l*
	* *environment asks agent a question, and then tells it the right answer*
	* application: Atari games (JK *thinks*)
2. **contextual bandits** (halfway between supervised and reinforcement learning)
	* environment samples input
	* agent takes action *y_hat*
	* agent receives cost *c*, which is related to an unknown probability distribution *P*
	* *environment asks agent a question, and gives agent a noisy score on its answer*
	* application: personalized recommendations, e.g., on Amazon
		* input is your shopping history
		* action is what to choose to advertise to you
		* reward is what you buy; but what you really want to buy is hidden so must be found through exploration of many customers
3. **reinforcement learning**
	* environment samples input
		* **environment is stateful!**: input depends on the agent's previous actions
	* like with contextual bandit, the agent receives cost *c*, which is related to an unknown probability distribution *P*

**In summary**, there are two big differences between RL and the other two approaches: 

1. in **RL**, the agent doesn't have full access to the function being optimized
	* the function must be queried through interaction
	* this property is shared with **contextual bandit**, but distinguishes both approaches from **supervised learning**
2. in **RL**, the agent is interacting with a *stateful* world: input in the current time step depends on actions taken in previous time steps
	* this is the key property that distinguishes **RL** from **contextual bandit**

##### History of Deep RL

Early influential references combining NNs with RL are:

1. [Narendra and Parthasarathy's (1990)](http://ieeexplore.ieee.org/document/80202/) paper *Identification and control of dynamical systems using neural networks* (Yale)
2. [Miller, Werbos and Sutton's (1991)](https://mitpress.mit.edu/books/neural-networks-control) book *Neural Networks for Control*, which they edited (Alberta)
3. [Lin's (1992)](https://dl.acm.org/citation.cfm?id=168871) CMU doctoral thesis *Reinforcement learning for robots using neural networks*, which has an abstract that reads like a contemporary state-of-the-art Deep RL paper abstract
4. [Tesauro's (1995)](https://dl.acm.org/citation.cfm?id=203343) paper, via which Backgammon experts were defeated

Recent success stories in Deep RL:

1. Atari-playing (all using techniques that will be covered in this course): 
	* **deep Q-learning** ([Mnih, Kavukcuoglu, Silver et al., 2013](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf))
	* **policy gradients** ([John Schulman, Sergey Levine, Moritz, Michael I. Jordan, and Abbeel, 2015](https://arxiv.org/abs/1502.05477); [Mnih et al., 2016](https://arxiv.org/abs/1602.01783))
	* **DAGGER** ([Guo et al., 2014](https://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning))
2. Robotic manipulation using **guided policy search**: 
	* [Sergey Levine et al.'s (2015)](https://arxiv.org/abs/1504.00702) *End-to-end training of deep visuomotor policies* (*NY Times* article with leading image of Chelsea Finn and Sergey Levine [here](https://www.nytimes.com/2015/05/22/science/robots-that-can-match-human-dexterity.html))
	* [John Schulman, Moritz, Sergey Levine, Michael I. Jordan, and Abbeel's (2015)](https://arxiv.org/abs/1506.02438) *High-Dimensional Continuous Control Using Generalized Advantage Estimation*
3. AlphaGo ([Silver et al., 2016](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html)), which combines many ML elements:
	* supervised learning
	* policy gradients
	* value functions
	* Monte-Carlo tree search


#### Chelsea Finn

* Chelsea is focusing on situations where the *reward function* is unclear
* in many popular examples (e.g., Tesauro's Backgammon, Mnih et al.'s Atari, Silver et al.'s AlphaGo), the reward function is clear
* in the real-world domains, the reward/cost function is difficult to specify, including in:
	* a child learning countless capabilities like pouring water into glass
	* robotic manipulation
	* autonomous driving
	* dialog systems
	* virtual assistants

##### What other forms of supervision?

The following are covered by Chelsea in sequence: 

1. demonstrated behaviour --> imitation, inferring intention
2. self-supervision, prediction --> model-based control
3. auxiliary objectives and additional sensing modalities

##### 1. demonstrated behaviour --> imitation, inferring intention

###### Human Learning via Imitation

* at 8 months, imitate simple actions and expressions
* at 18 months, imitate after a delay and multi-step actions (e.g., [Meltzoff (1988)](http://ilabs.washington.edu/meltzoff/pdf/88Meltzoff_DevPsy.pdf) where 14-month old infants imitate a novel action with a novel stimulus after a one-week delay)
* at 36 months, imitate multi-step actions after a delay

###### ML via Imitation

* in [Bojarski et al. (2016)](https://arxiv.org/abs/1604.07316), autonomous driving systems learn how to drive from humans in varying conditions, e.g., snow, rain

###### Human Learning via Inferring Intention

* [Warneken and Tomasello (2006)](https://www.ncbi.nlm.nih.gov/pubmed/16513986) demonstrate that 18-month-old infants infer intentions, i.e., the reward function, without even needing to imitate (this is called **Inverse Reinforcement Learning**: the reward function is inferred): 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/WnT1.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/WnT2.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/WnT3.png)

###### ML via Inferring Intention

* [Chelsea Finn, Ian Goodfellow and Sergey Levine (2016)](https://arxiv.org/abs/1605.07157) demonstrate comparable Inverse RL in robots: 
	* reward function for the dish-placing task involves both:
		1. be gentle with the plate / manipulate it slowly
		2. successfully place the dishes in the rack
	* reward function for the pouring task is how many almonds make it into the cup

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/finn1.png)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/finn2.png)

##### 2. self-supervision, prediction --> model-based control

* "the idea that we *predict the consequences of our motor commands* has emerged as an important theoretical concept in all aspects of sensorimotor control"
* key papers on human learning include:
	* [Sarah J. Blakemore, Susan Goodbody and Daniel Wolpert (1998, UCL)](http://www.jneurosci.org/content/18/18/7511) 
	* [Rao and Ballard (1999)](https://www.ncbi.nlm.nih.gov/pubmed/10195184)
	* [Flanagan et al. (2003)](https://www.ncbi.nlm.nih.gov/pubmed/12546789)
* given a perfect model and optimization, motor predictions and simulation can inform autonomous performance, e.g., in Tan et al. (2014; N.B. JK couldn't find this citation): 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/tan14.png)

* Oh et al. (2015; JK also couldn't find this citation) predict video (left frame) based on ground truth video (right frame): 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/oh15.png)

* Finn et al. (2014, 2015) demonstrate both:
	1. learning to predict real-world video (top row; second, fourth, and sixth frames), with blur shown as uncertainty
	2. use predicted motor movement to grasp appropriately (bottom row)

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/finn1617.png)

##### 3. auxiliary objectives and additional sensing modalities

* not covered much in this course
* can involve:
	* additional sensory modalties (e.g., touch, audio, depth) for a vision task ([Mirowski et al., 2017, DeepMind](https://arxiv.org/pdf/1611.03673.pdf))
	* learning multiple, related tasks at the same time
	* task-relevant properties of the world

---
### Emma Brunskill's Tutorial on Reinforcement Learning

#### What is Reinforcement Learning? 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/emmaRL.png)

* ML vs RL
	* both involve **optimization**
	* RL is also at the intersection of:
		* generalization (same algorithm may work across multiple use-cases)
		* exploration (task could be very complex and require significant exploration of possibilities)
		* delayed consequences (action at time step *t* may not impact until much later)

#### Outline

1. The Standard RL Setting
2. Exploration
3. Evaluating an RL Algo
4. Current and Future

#### 1. The Standard RL Setting

* often involve a **Markov Decision Process**

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/markovDP.png)

* MDP planning involves solving the *Bellman Equation* with a *Q* on the left-hand side
	* see Emma's talk for equations involving:
		* value iteration convergence
		* policy iteration convergence
* three ways that people tend to think about MDP planning:
	1. models (dynamics T, reward R)
		* **Model-Based RL** uses data efficiently, but is computationally expensive
	2. state-action value function Q/V
		* **Q-Learning** is a model-free alternative to the above point that is computationally cheap, but only propagates experience one step (**replay** enables further propagation, but this requires more computation)
	3. policy 𝛑
		* **Policy Search** directly search 𝛑 space, parameterize policy, and do SGD
* the above three camps are summarised visually by David Silver thusly: 

![](https://github.com/the-deep-learners/study-group/blob/master/weekly-work/week14/img/silverVenn.png)


#### 2. Exploration

##### Exploration vs Exploitation

* start off not knowing how (stochastic) world works
* given some experience (data), can estimate how world works / Q-values / policies
* but there's a chance that info so far has been misleading and/or that haven't seen everything yet
* so we may want to try things that aren't seen to be optimal yet (**exploration**)
* ...but eventually want to gather high reward (i.e., **exploitation**)
* there is, therefore, a tension in the algorithm between exploration and exploitation
* key ideas:
	1. mostly act greedily, sometimes randomly
	2. be optimistic in face of uncertainty
	3. maintain a distribution over worlds
		1. sample a world and act as if sample was representative, real world
		2. choose an action reasoning about the full distribution
* **multi-armed bandits**
	* simpler case, but highly related to RL
	* pulling arms has a cost
	* we want to sample the arms to find the most lucrative one(s)
	* following the above three themes, we can:
		1. use **E-Greedy Exploration** of the arms
		2. maintain **Optimism Under Uncertainty**
			* intuition:
				* either you really are pulling the best arm and you'll get a large reward (on average)
				* or, you'll get information that allows you to revise your estimate
		3. maintain a distribution over worlds by being **Bayesian**, e.g., by using:
			* **Probability Matching / Thompson Sampling**
			* **Bayes-Optimal Bandits**: theoretically find optimal sequence of actions, but isn't computationally tractable
			* **(Bandit) Learning as Planning**
				* forms a **Partially Observable MDP (POMDP)** (*mentioned by John at the start of his Berkeley lecture above*)
				* is computationally tractable, provided we do not plan for continuous states exactly, and instead use approximate continuous-state MDLP or POMDP planning, e.g.: 
					* Sparse Sampling (Kearns, Mansour, Ng)
					* Monte Carlo Tree Search (Kocsis & Szepsesvari)
* **key ideas from above applied to RL**
	* same broad 1., 2., and 3., but RL-model- or Q-specific

#### 3. Evaluating an RL Algo

* ways to assess the performance of RL algos:
	* empirical
	* convergence
	* asymptotic optimality
	* probably approximately correct
	* regret
	* Bayes-optimality
	* best-in-class
	* efficiency is also important to some researchers, but is not the focus of Emma's workshop

<!---
---
### UC Berkeley CS294-112 Lecture II (1/23/17) on Supervised Learning of Behaviors: Deep Learning, Dynamical Systems, and Behavior Cloning

#### heading

* point
* point
--->



---
## Up Next

For our next session, we delve deeper into Reinforcement Learning. The recommended preparatory work is: 

1. The first two ([one](https://www.youtube.com/watch?v=Q4kF8sfggoI&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&index=1), [two](https://www.youtube.com/watch?v=C_LGsoe36I8&index=2&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3)) lectures from UC Berkeley CS294-112 **Fall 2017** Lectures, which are continuing to be released presently. Lecture 1 may be skimmed as it overlaps greatly with the material covered above. 
2. Andrej Karpathy's [blog post](http://karpathy.github.io/2016/05/31/rl/) *Deep Reinforcement Learning: Pong from Pixels* 
2. Additional resources to skim or dig your teeth into, depending on how much time you have: 
	* Kai Arulkumaran and colleagues' [arXiv review paper](https://arxiv.org/abs/1708.05866) *A Brief Survey of Deep Reinforcment Learning*
	* The *Reinforcement Learning* chapter (Chapter 16) from Aurélien Géron's [Hands-on Machine Learning book](http://shop.oreilly.com/product/0636920052289.do)
	* [OpenAI Lab](https://github.com/kengz/openai_lab)
	* [Coach](https://github.com/NervanaSystems/coach)
