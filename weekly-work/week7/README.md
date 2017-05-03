# Session VII: Implementing Convolutional Nets

*Meeting date: January 12th, 2017*

By continuing to make our way through the material from Andrej Karpathy and Justin Johnson's CS231n (Stanford) lectures, we covered a broad range of practicalities and best practices for implementing convolutional neural nets. 

## Recommended Preparatory Work

1. The final thirteen minutes of the [sixth CS231n lecture](https://www.youtube.com/watch?v=KaR4lIdI1MQ&index=1&list=LLup-fnSNRaByeuXOWqfnykw) (i.e., starting from the 57:30 mark)
2. Lectures [seven](https://www.youtube.com/watch?v=AQirPKrAyDg) through [twelve](https://www.youtube.com/watch?v=XgFlBsl0Lq4) of CS231n 
3. CS231n lecture notes [five](http://cs231n.github.io/neural-networks-1/), [six](http://cs231n.github.io/neural-networks-2/), and [seven](http://cs231n.github.io/neural-networks-3/)

## Summary

Topic highlights of the session included: 

#### From Lecture 7

* common settings for the four hyperparameters of a convolutional layer, working through examples as the numbers must work out:
	* **K**: the number of filters (typically in powers of two -- some libraries optimise calculations to these levels)
	* **F**: spatial extent of the filters
	* **S**: stride length
	* **P**: the amount of zero padding
* famous convolutional net architectures, while focusing on their changes (associated with classification accuracy improvements in ILSVRC) over time: 
	* LeNet-5 (LeCun et al., 1998)
	* SuperVision / "AlexNet" (Krizhevsky et al., 2012)
	* ZFNet (Zeiler & Fergus, 2013)
	* VGGNet (Simonyan & Zisserman, 2014)
	* GoogLeNet (Szegedy et al., 2014)
	* ResNet (He et al., 2015)
* network depth versus ILSVRC classification accuracy over time
* ResNet network depth versus CIFAR-10 classification accuracy

#### From Lecture 8

* comparing computer vision tasks, e.g.: 
	* single object
		* classification
		* classification + localisation
	* multiple object
		* object detection
		* instance segmentation
* the ILSVRC localisation error of famous ConvNet architectures: 
	* AlexNet (2012)
	* Overfeat (2013)
	* VGG (2014)
	* ResNet (2015)
* object detection data sets: 
	* PASCAL VOC (2010): classic
	* ILSVRC *Detection* (2014): most classes and images per class
	* MS-COCO (2014): most objects per image
* as with image recognition, *R-CNN* greatly outperforms pre-ConvNet methods
* *Fast R-CNN* and the subsequent *Faster R-CNN* maintain classification accuracy but are 25 and 250 times faster than R-CNN, respectively
	* code for all three networks are available in the Caffe Zoo
	
#### From Lecture 9

* "deconvolutional" approaches for visualising and understanding individual neurons within convolutional neural networks: 
	1. feed an image into the net
	2. pick a layer, set the gradient there to be all zero except for one
	3. for some neuron of interest, backprop to image
* NeuralStyle (Gatys et al., 2015): set an image to any style
* intuitive explanations for fooling ConvNets (e.g., Nguyen, Yosinski & Clune, 2014; Szegedy et al., 2013):
	* visually: cases with parameters cleverly outside of the training set (Goodfellow, Shlens & Szegedy, 2014)
	* manually working through the arithmetic of fooling a binary linear classifier
	
#### From Lecture 10

* interpretable RNN neurons, as identified manually within text by [Karpathy, Johnson and Li (2015)](https://arxiv.org/abs/1506.02078):
	* quote detection 
	* line length 
	* if statements 
	* quotes or comments
	* code indent depth
* image captioning becomes possible by supplementing ConvNets with LSTMs (five key papers are provided on slide 51)
	* this requires image-sentence datasets, e.g.:
		* MS-COCO (2014), again (120k images, 5 sentences each)
	* ResNet is to vanilla ConvNet ~as LSTM is to RNN
	* GRUs (Cho et al., 2014) are the key alternative to LSTMs
	* Jozefowicz et al. (2015) provides a helpful, empirical comparison of RNN architectures
	
#### From Lecture 11

* NVIDIA chips are much more common than AMD for deep learning
* GPUs greatly outperform CPUs
* SSDs greatly outperform classic hard disks
* disk size can become a limiting factor
* floating point precision can go very low:
	* Courbariaux and Bengio (2016) train with single-bit activations and weights, so they are all simply either +1 or -1, though gradients require greater precision
	
#### From Lecture 12

* see [blog post](https://insights.untapt.com/fundamental-deep-learning-code-in-tflearn-keras-theano-and-tensorflow-66be10a03227) I (Jon Krohn) published that summarises the pros and cons of the four primary deep learning libraries (TensorFlow, Theano, Torch, and Caffe) as covered by Justin Johnson in this lecture
* in addition, here are Justin's broad recommendations: 
	* for feature extraction or fine-tuning existing models: use Caffe
	* for complex uses of pretrained models: use Lasagne or Torch
	* for writing your own layers: use Torch
	* for "crazy" RNNs: use Theano or TensorFlow
	* for a very large model that requires parallelism: use TensorFlow

## Up Next

1. the remaining lectures and notes of CS231n, in February
1. Richard Socher's CS224d (also out of Stanford) on Deep Learning for Natural Language Processing, in early March
