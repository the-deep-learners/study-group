# Session 6: Convolutional Neural Networks for Visual Recognition

Meeting date: November 30th, 2016

This was our first session since completing Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) text. 

## Recommended Preparatory Work

1. the [first six lectures](https://www.youtube.com/watch?v=g-PvXUjD6qg&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA) of Stanford's Winter 2016 CS231n course
1. the first four sets of [course notes](http://cs231n.github.io/), which cover:
  * [classification](http://cs231n.github.io/classification/)
  * [linear classification](http://cs231n.github.io/linear-classify/)
  * [optimization](http://cs231n.github.io/optimization-1/)
  * [more optimization](http://cs231n.github.io/optimization-2/)
1. optionally, [module 0](http://cs231n.github.io/) in the course notes, which provide an introduction to Python, NumPy, Jupyter Notebooks, the Unix command line, and Amazon Web Services

## Summary

The course notes linked to above provide excellent summaries of the material covered in the second lecture onward. For the first lecture, given by the illustrious Fei-Fei Li, here are my (i.e., Jon Krohn) own notes: 

#### Context

* Cisco: 85% of data on Internet is in the form of pixels ("dark matter")
* more video sensors on earth than people
* every minute, 150 hours of video are uploaded to YouTube

#### A History of Vision and Vision Research

* 543m years ago, explosion in speciation; Andrew Parker theorises this is due to the evolution of eyes (a simple pinhole light sensor in Trilobites)
* first well-documented effort to duplicate the visual world: da Vinci's Camera Obscura (15th century)
* Hubel & Wiesel (1959) Harvard postdocs 1981 Nobel Prize-winning work
	* vision starts with simple structures (edges) not fish

#### A History of Computer Vision

* Larry Roberts (1963) "Block World"
	* theorised that edge detection enables recognition of blocks from many angles
* first two AI labs:
	1. Marvin Minsky at MIT
	2. John McCarthy at Stanford: coined "Artificial Intelligence" term
* David Marr (1970s): 
	* "Stages of Visual Representation" (it is hierarchical)
	* first stage is "edge image" akin to H&W
	* second stage is 2-D sketch
	* final, third stage is 3D representation; enables guidance and manipulation in the real world
* David Lowe (1987): use edges to distinguish monochromatic razors
* Shi & Malik (1997): "Normalized Cut" was first stages of distinguishing objects in image (segmentation)
* Viola & Jones (2001): face detection within image
	* used in FujiFilm digital cameras in 2006; the first with face detection
	* first algorithm fast enough to be used for instantaneous machine vision
* David Lowe (1999): "SIFT" Object Recognition via (a handful of key) features, as opposed to full figure
	* this was the basis of machine vision for a decade -- until age of Deep Learning
	* features that Deep Learning networks learn are similar to features programmed by engineers
* prior to Deep Learning approaches, primary techniques were graphical models and SVMs
	* e.g., "Deformable Part Model", which used "something like" SVM
* PASCAL Video Object Challenge (2006-12) demonstrated improved classification performance on twenty object categories
	* IMAGENET (image-net.org) built in response by Fei-Fei Li and her colleagues (2009) with 22k categories and 14M images
	* IMAGENET Large Scale Visual Recognition Challenge: uses 1000 of IMAGENET object classes and 1.4M images
		* error rate decreases year-over-year, but in 2012 error rate was cut in half by ConvNet (*SuperVision* by Krizhevsky & Hinton; seven-layer)
		* CONVNET invented in '70s but confluence of techniques enabled it to be transformative in that year
		* in 2014, best architectures were GoogLeNet and VGG
		* winning architecture in 2015 is MRSA (Microsoft Asia Researchers), which has 100 layers
		
#### CS231n course overview

* focus on the visual recognition problem, specifically image classification, within IMAGENET
* also covers *object detection*, *image captioning*, and *action classification*
* CNNs were "not invented overnight"
	* major contributions were:
		* 1980s: LeCun and Hinton worked out backpropagation mathematics
		* LeCun et al. (1998): MNIST digit classification, eventually sold to U.S. Mail and banks (for cheques)
		* Krizhevsky et al. (2012): similar architecture to 1998, but able to leverage GPUs with three orders of magnitude more transistors, and able to train on IMAGENET, which has seven orders of magnitude more pixels than MNIST data; additional, but less important changes, include the use of ReLU in place of sigmoid neurons
* problems that still need to be solved in machine vision
	* classification of *all* objects in image
	* recognition within three dimensions, e.g., for use in robotics
	* anything related to motion
	* "understanding" the relationship between objects, as opposed to just labelling objects (e.g., Justin Jacobs' *Visual Genome* Project)
	* the "holy grail" is to be able to narrate a scene; people can write an essay after seeing a scene for just 500ms (Fei-Fei et al., 2007)
* machines vision facilitates better robots and will save lives

## Application

Our colourful study group member [Dmitri Nesterenko](https://www.linkedin.com/in/dmitri-nesterenko-7ba4484), who is Director of Software Engineering at the **XO Group** downtown, went into considerable, helpful detail describing his adventures writing a *k*-Nearest Neighbours algorithm [from scratch](https://www.linkedin.com/in/dmitri-nesterenko-7ba4484). 

## Up Next

1. the remaining lectures and notes of CS231n, split over two sessions, in January and February
1. RNN/LSTM courses and materials from Richard Socher and Chris Olah
1. a hands-on TensorFlow tutorial with engineers from the Google office in New York
1. the [Deep Learning Papers Reading Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)
