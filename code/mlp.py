#This is code for a MLP with theano
#Heavily inspired by the code in the tutorial in:
#       http://deeplearning.net/tutorial/mlp.html#mlp
#Author : Vik Kamath

import cPickle
import gzip
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T


from logisticRegression import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self,rng, input, n_in,n_out, W=None, b= None, activation=T.tanh):
        """
        -Definition of a standard hidden layer. 
        -It's a fully connected network and have sigmoidal activation function. 
        -Weight matrix (W) is of shape (n_in,n_out)
        -Bias vector (b) is of shape (n_out,) 
        -Here, we use the tanh non-linearity
        -Hidden Unit activations are given by tanh(dot(input,W) + b)
        
        :param 'rng' : A random number generator used to initialize the weights. 
        :type 'rng' : numpy.random.RandomState

        :param 'input' : a symbolic tensor of shape (n_example, n_in)
        :type 'input' : theano.tensor.dmatrix

        :param 'n_in':
        :type 'n_in':




# vim : tabstop=8 expandtab shiftwidth=4 softabstop=4
