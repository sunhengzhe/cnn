import numpy as np

class SoftmaxLayer:

    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape

        input = input.flatten()

        totals = np.dot(input, self.weights) + self.biases

        exp = np.exp(totals)

        S = np.sum(exp, axis = 0)

        out = exp / S

        self.last_totals = totals
        self.last_input = input
        self.last_exp = exp
        self.last_S = S
        self.last_out = out

        return out

    def backprop(self, label, learn_rate):
        p = self.last_out[label]
        S = self.last_S
        exp = self.last_exp

        d_t_d_w = self.last_input
        d_t_d_b = 1
        d_t_d_inputs = self.weights

        # d_L_d_out = np.zeros(self.last_out.shape)
        # d_L_d_out[label] = - 1 / p

        d_L_d_out = - 1 / p

        d_out_d_t = - exp[label] * exp / (S ** 2)
        d_out_d_t[label] = exp[label] * (S - exp[label]) / (S ** 2)

        d_L_d_t = d_L_d_out * d_out_d_t

        d_L_d_w = d_t_d_w[np.newaxis].T.dot(d_L_d_t[np.newaxis])
        d_L_d_b = d_L_d_t * d_t_d_b
        d_L_d_inputs = d_t_d_inputs.dot(d_L_d_t)

        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.last_input_shape)

