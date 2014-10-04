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

        :param 'n_in': dimensionality of input
        :type 'n_in': int

        :param 'n_out': number of hidden units
        :type 'n_out': int

        :param 'activation' : Non linearity to be applied on the hidden layer
        :type 'activation' : theano.Op or function
        """

        self.input = input

        # 'W' is initalized randomly (uniformly) from the range 
        #   sqrt(-5/[n_in + n_hidden]) to sqrt(+6/[n_in + n_hidden]) 
        #   for the tanh activation function
        # The output of rng.uniform is converted to an array and as floats
        #       to allow the code to be run on a GPU (using asarray and 
        #           dtype=theano.config.floatX respectively)
        # Optimal initizliation weights are dependent on the non-linearlity used
        #           refer Xavier, Bengio '10 for details. 
        
        if W is None: 
            W_values = np.asarray(rng.uniform(
                        low = -np.sqrt(6/ (n_in + n_out)),
                        high = np.sqrt(6/ (n_in + n_out)),
                        size = (n_in,n_out),
                        dtype = theano.config.floatX))
            if activation = theano.tensor.nnet.sigmoid:
                W_values = W_values * 4 #Refer [Xavier,Bengio '10]
            W = theano.shared(value = W_values,name = "W",borrow = True)
        
        if b is None: 
            b_values = np.zeros((n_out,),dtype=theano.config.floatX)
            b = theano.shared(value=b_values,name="b",borrow=True)

        self.W = W
        self.b = b

        #NOTE: I think the logic for this is that if the activation
        #       is 'none', then assume that it's a linear unit and just output
        #       lin_output (which is basically the non-transformed activation)
        lin_output = T.dot(input,self.W) + b
        self.output = (lin_output if activation is None
                      else activation(lin_output))
            




# vim : tabstop=8 expandtab shiftwidth=4 softabstop=4
