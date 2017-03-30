# Session VIII: Unsupervised Learning, Regularisation, and Venture Capital

*Meeting date: February 7th, 2017*

With this session, we wrapped up our coverage of [CS231n](http://cs231n.github.io/) (Stanford) lectures, which were delivered by now-familiar faces Andrej Karpathy and Justin Johnson as well as guest lecturer, Google Senior Fellow Jeff Dean. 

In addition, we were delighted to hear from guest speakers of our own: 

1. **[Raphaela Sapire](https://angel.co/raphaela-sapire)** on her experience as a venture capitalist at Blue Seed Capital, particularly her insight into the machine- and deep-learning start-up market (slides [here](https://github.com/the-deep-learners/study-group/blob/master/slides/2017-02-07__raphaela_sapire__billion_dollar_AI.pdf))
2. **[Katya Vasilaky](https://kathrynthegreat.github.io/)** on her research into L2 Regularization, the popular method to avoid overfitting in a wide range of models, including the deep-learning variety (slides [here]())

A summary blog post, replete with photos of the session, can be found [here](https://medium.com/@jjpkrohn/deep-learning-study-group-viii-unsupervised-learning-regularisation-and-venture-capital-9aba67fc931c). 


## Recommended Preparatory Work

1. The final three lectures from CS231n ([13](https://www.youtube.com/watch?v=UFnO-ADC-k0&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA&index=13), [14](https://www.youtube.com/watch?v=I-i1KBuShCc&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA&index=14), and [15](https://www.youtube.com/watch?v=s63vOy1kvsU&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA&index=15))
2. as well as the final four sets of notes ([one](http://cs231n.github.io/neural-networks-case-study/), [two](http://cs231n.github.io/convolutional-networks/), [three](http://cs231n.github.io/understanding-cnn/), and [four](http://cs231n.github.io/transfer-learning/))


## Summary


Topic highlights of the session included: 


#### From Lecture 14

##### Karpathy on ConvNets applied to motion (videos)

* "fancy" spatio-temporal video ConvNets:
	* for detecting global motion: provide limited or no benefit over LSTM applied to individual video frames 
	* for detecting local motion, use a 3D ConvNet
	* try using Optical Flow in a second stream or GRU-RCN (the latter being Karpathy's favourite)

##### Johnson on unsupervised learning:

* autoencoder overview
	* traditional: 
		* try to reconstruct input
		* used to learn features, initialise supervised model
		* no longer predominant 
	* variational: 
		* Bayesian statistics crossed with Deep Learning (<3)
		* generate samples, e.g., images by sampling
	* Generative Adversarial Networks: Generate samples
* autoencoders in practice
	* input data (x) --> [encoder] --> features (z) --> [decoder] --> reconstructed input data (x)
	* the [encoder] and [decoder] often share weights
	* decoders evolved via this sequence: 
		1. linear + sigmoid neurons
		2. deep, fully-connected
		3. ReLU ConvNet ("upconv")
* Greedy Training of Autoencoders:
	* "Restricted Boltzmann Machines" (RBMs) were common in mid-2000s
		* train one layer at a time
		* start with first layer, freeze it, move to second layer, etc.
* Varational Autoencoders:
	* generate data by using Bayesian statistics within an autoencoder framework to sample from prior and posterior distributions
	* can, e.g., output smooth interpolations of input data
* Generative Adversarial Networks:
	* seminal paper is Goodfellow et al. (NIPS 2014)
	* random noise --> [generator] --> fake images (plus, separately, real images from a data set) --> [discriminator] --> trained to distinguish real images from fake
	* image generation with "less math"
	* train generator and discriminator jointly; after training, image generation is straightforward
	* Denton et al. (NIPS 2015): expanded work by enabling discriminators to work at every scale (applied to single classes of CIFAR-10 dataset)
	* Radford et al. (ICLR 2016): 
		* create realistic, latent space-interpolatable images of bedrooms
		* their generator: upsampling network with fractionally-strided convolutions
		* their discriminator: a ConvNet
		* "Architecture guidlines for stable Deep Conv GANs" (from Johnson):
			* replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator)
			* user batch normalisation in both the generator and discriminator
			* remove fully-connected hidden layers for deeper architectures
			* use ReLU activation in generator for all layers except for output (Tanh)
			* use Leaky ReLU activation in all layers of discriminator
		* vector math: 
			* [smiling woman] - [neutral woman] + [neutral man] = [smiling man]
			* [man with glasses] - [man without glasses] + [woman without glasses] = [woman with glasses]
	* Dosovitskiy & Brox (arXiv 2016):
		* creates convincing new ImageNet samples
		* trained on all ImageNet classes together
		* broadly a Variational Autoencoder fed into both (1.) a Discriminator network and (2.) a pretrained AlexNet (see slide 128 "Putting everything together" for diagram)


## Up Next

1. Richard Socher's [CS224d](https://cs224d.stanford.edu/) (also out of Stanford) on Deep Learning for Natural Language Processing, in early March
