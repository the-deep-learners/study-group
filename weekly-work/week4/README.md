# Week 4

meeting date: 10-20-2016

## Covered

* Recommended reading from Nielsen's electronic text: 
	* [Chapter Four](http://neuralnetworksanddeeplearning.com/chap4.html) 
	* [Chapter Five](http://neuralnetworksanddeeplearning.com/chap5.html) 
	
### Proof Neural Nets can Compute any Function

* Neural nets can compute any function (i.e., they are *universal*), assuming that:
	1. we accept they are an *approximation* (that can be improved by the inclusion of additional hidden neurons), as opposed to an *exact* solution
	2. the function they are explaining is *continuous* (e.g., no sharp jumps)
	
* For the first time in our study session, we moved from whiteboarding to a projector to cover this content
* In his fourth chapter, Michael Nielsen did a tremendous job of developing thematically-coherent, interactive Java applets that facilitate a clear visual understanding of this proof; try it for yourself!
* A fair bit of our discussion centered on the practicalities of expanding the proof beyond two inputs features into *n*-dimensional space

### Factors making Deep Neural Networks Difficult to Train

* We primarily discussed the causes of, implications of, and methods to mitigate *unstable* gradients, which in deep neural nets tend to *vanish* but under certain circumstances can instead *explode*
* We also touched on other factors that can make deep nets difficult to train, e.g., the propensity for sigmoid neurons to saturate in later layers, the perils of fully-random weight initialization 

### Visualizing the Function of Particular Hidden Layers

* [Thomas Balestri](https://www.linkedin.com/in/thomasbalestri) introduced us to Jason Yosinski's breathtaking [Deep Visualization Toolbox](https://www.youtube.com/watch?v=AgkfIQ4IGaM) for developing an understanding of how individual layers contribute to  a convolutional NN

## Applications

* We took a break from applications for this session to focus on finishing shortly Nielsen's text, but we'll return to practical work for the next session
