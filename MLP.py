# coding: utf-8
import numpy as np
from function import Dendrite, Relu, Sigmoid, SoftmaxWithLoss, MSE
from collections import OrderedDict

class MultilayerPerceptron:
    def __init__(self, input_n, output_n, mode):
        w1_n = 32
        w2_n = 32
        std = 0.01
        self.mode = mode
        self.params = {}
        self.params['W1'] = np.random.normal(0, std, input_n*w1_n).reshape(input_n, w1_n)
        self.params['b1'] = np.random.normal(0, std, w1_n)
        self.params['W2'] = np.random.normal(0, std, w1_n*w2_n).reshape(w1_n, w2_n)
        self.params['b2'] = np.random.normal(0, std, w2_n)
        self.params['W3'] = np.random.normal(0, std, w2_n*output_n).reshape(w2_n, output_n)
        self.params['b3'] = np.random.normal(0, std, output_n)

        self.layers = OrderedDict()
        self.layers['Dendrite1'] = Dendrite(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Dendrite2'] = Dendrite(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Dendrite3'] = Dendrite(self.params['W3'], self.params['b3'])
        if mode == 1: ## classification
            self.lastLayer = SoftmaxWithLoss()
        elif mode == 0:
            self.lastLayer = MSE()
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        # print (dout.shape)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Dendrite1'].dW, self.layers['Dendrite1'].db
        grads['W2'], grads['b2'] = self.layers['Dendrite2'].dW, self.layers['Dendrite2'].db
        grads['W3'], grads['b3'] = self.layers['Dendrite3'].dW, self.layers['Dendrite3'].db

        return grads