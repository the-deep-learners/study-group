import numpy as np
import math

inputs = {
    0: np.array([0.990, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]),
    1: np.array([0.009, 0.990, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]),
    2: np.array([0.009, 0.009, 0.990, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]),
    3: np.array([0.009, 0.009, 0.009, 0.990, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]),
    4: np.array([0.009, 0.009, 0.009, 0.009, 0.990, 0.009, 0.009, 0.009, 0.009, 0.009]),
    5: np.array([0.009, 0.009, 0.009, 0.009, 0.009, 0.990, 0.009, 0.009, 0.009, 0.009]),
    6: np.array([0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.990, 0.009, 0.009, 0.009]),
    7: np.array([0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.990, 0.009, 0.009]),
    8: np.array([0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.990, 0.009]),
    9: np.array([0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.990])
}

expected_outputs = {
    0: "0000",
    1: "0001",
    2: "0010",
    3: "0011",
    4: "0100",
    5: "0101",
    6: "0110",
    7: "0111",
    8: "1000",
    9: "1001"
}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def evaluate(W, B):
    """
    Evalute whether Neuron weights and biases generate
    expected output for each digit from 0 -> 9
    """
    for n in xrange(10):
        print "Determining if neurons produce proper bitwise representntation for: {}".format(n)

        neuron_zs = [np.dot(inputs[n], w) - b for w, b in zip(W, B)]
        neuron_outputs = [sigmoid(z) for z in neuron_zs]
        bitwise_string = ''.join([str(int(round(x))) for x in neuron_outputs])
        expected = expected_outputs[n]

        print "  neuron output: {}".format(bitwise_string)
        print "  expected output: {}".format(expected)
        assert(bitwise_string == expected)
        print "  correct!"


def main():
    # Neuron weights and biases that should be tuned to generate expected output
    w0, b0 = [-10, -10, -10, -10, -10, -10, -10, -10,  10,  10], 0
    w1, b1 = [-10, -10, -10, -10,  10,  10,  10,  10, -10, -10], 0
    w2, b2 = [-10, -10,  10,  10, -10, -10,  10,  10, -10, -10], 0
    w3, b3 = [-10,  10, -10,  10, -10,  10, -10,  10, -10,  10], 0

    W = np.array([w0, w1, w2, w3])
    B = np.array([b0, b1, b2, b3])

    evaluate(W, B)


if __name__ == "__main__":
    main()
