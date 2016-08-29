# Week 2
meeting date: *09-06-2016*

###Covered
- [Chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html) from Nielsen's Ebook
- Setup and get comfortable with [Keras](https://keras.io/)


###Nielsen Chapter 2
  - *backpropagation*
	- fast alogrithm for computing gradients
	- introduced in 1970s
	- [This 1986 paper](http://www.nature.com/nature/journal/v323/n6088/pdf/323533a0.pdf) recognized the usefulness of its application in neural nets

  - #####Warm up: a fast matrix-based approach to computing output of neural net
	- 3 neuron components
		- `w`: weight
		- `b`: bias
		- `a`: activation
	- `a_l_j = σ(∑_k w_l_jk * a_l-1_k + b_l_j)`
	- Weight matrix entry W_l_jk
		- `lth` layer
		- jth neruon in layer l
		- kth neuron in layer l - 1
	- `a_l = σ(w_l*a_l-1 + b_l)`
	- `z_l = w_l*a_l-1 + b_l`
		- Weighted input of the neurons in layer l
  - #####The two assumptions we need about the cost function
	- goal of backpropagation is to compute the partial derivatives of the cost function C with respect to any weight w aor bias b
	- 2 assumptions about cost function
		1. can be written as an average C = (1/n)∑_x C_x
			- allows us to get partial derivatives by averaging partial derivatives of individual training samples
		2. can be written as a function of the outputs from the neural network
  - #####The Hadamard product
	- `s⊙t` denots *elementwise* product of two vectors s and t
		- `(s⊙t)_j = s_j*t_j`
		- called *Hadamard* or *Schur* product
  - #####The four fundamental equations behind backpropagation
	- backpropagation is about understanding how changing the weights and biases in a network changes the cost function.
	- `δ_l_j` represents the *error* in the jth neuron in the lth layer.
		- with backpropagation we compute this error and then relate it to the partial derivatives
		- `δ_l_j ≡ ∂C/∂z_l_j`
			- error for jth neuron layer l
	- **4 fundamental equations of backpropagation**
		1. `δ_L_j = ∂C/∂a_L_j * σ′(z_L_j)`
			- `δ_L = ∇_aC⊙σ′(z_L)` in matrix form
		2. `δ_l=((w_l+1)^T*δ_l+1) ⊙ σ′(z_l)`
			- expresses the errors in layer *l* in terms of error in the next layer *l+1*
			- combining equation 1 and 2 allows us to compute the error for any layer in the net
		3. `∂C/∂b_l_j = δ_l_j`
			- rate of change of the cost with respect to any bias in the network
		4. `∂C/∂w_l_jk = a_l−1_k*δ_l_j`
			- rate of change of the cost with respect to any wight in the network
			- when activation is small, the gradient term with respect to w will tend to be small
			- weights output from low-activation neurons learn slowly

	- sigmoid function is very flat around 0 or 1, so `σ′(z_L_j) ≈ 0`
		- From equation 1 output neuron in final layer will learn slowly if the output neuron is either low or high activation (0 or 1).
		- From equation 2, error is likely to get small if neuron is near saturation
	- These 4 equations hold for any activation function, not just the sigmoid function
		- proofs don't use an special properties of σ
		- so we could pick an activation function whose derivative that is never close to 0 to prevent the slow-down of learning that occurs with saturated Sigmoid functions
