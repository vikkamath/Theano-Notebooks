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
        
        #Parameters of the model
        self.params = [self.W,self.b]


class MLP(object):
    """
    Multi Layer Perceptron Class
    Intermediate Layers have a non-linear activation fn - tanh or sigmoid
    The top layer is a softmax layer
    """

    def __init__(self,rng,input,n_in,n_hidden,n_out):
        """
        Initialize the parameters of the MLP
    
        :param rng : a random number generator used to initialize weights
        :type rng: numpy.random.RandomState
    
        :param input: symbolic variable that describes the input of the architecture
                        (one minibatch)
        :type input: theano.tensor.TensorType
    
        :param n_in: number of input units (the dimensionality of the space
        :           in which the datapoints live)
        :type n_in: int

        :param n_hidden: number of hidden units
        :type n_hidden: int
        
        :param n_out: number of hidden units in the layer. 
        :           i.e. the dimension of the space the labels lie in. 
        :type n_out: int
        """

        #Since we're dealing with an MLP with one hidden layer, 
        #   it is equivalent to being a network with one layer of 
        #   tanh activation units connected to a logistic regression layer
        self.hiddenLayer = HiddenLayer(rng=rng,input=input,n_in=n_in,n_out=n_hidden,
                activation=T.tanh)

        #The logisticRegression layer has as inputs, the hidden units of the hidden layer
        self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer.output,
                n_in = n_hidden,
                n_out = n_out)

        #Regularization:
        #1. L1 Norm
        #One method of regularization is to ensure that the L1 norm is small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()
        #2. Squared L2 Norm
        #Another regularization method is to ensure that the square of the L2 norm
        #   is small
        self.L2_sqr = (self.hiddenLayer.W**2).sum() \
                + (self.logRegressionLayer.W **2).sum()

        #Negative log likelihood of the MLP is given by the Negative log likelihood of 
        #   the output of the model, as computed by the LogisticRegression layer. 
        self.negative_log_likelihood = self.LogisticRegression.negative_log_likelihood
        #same holds for the funtion computing the number of errors
        self.errors = self.LogisticRegression.errors

        #the parameters of the model are the the parameters of the two layers
        #   that it is made up of 
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def test_mlp(learning_rate = 0.01, 
            L1_reg = 0.00,
            L2_reg = 0.0001,
            n_epochs = 1000,
            dataset = 'mnist.pkl.gz',
            batch_size = 20,
            n_hidden=500):
    """
    This method implements stochastic gradient descent for an MLP. 
    This is demonstrated on MNIST.

    :param learning_rate: Learning rate
    :type learning_rate: float

    :param L1_reg: l1 norm's weight when added to the cost 
    :type L1_reg: float

    :param L1_reg: l2 norm's weight when added to the cost
    :type L2_reg: float

    :param n_epoch: maximum number of epochs to run the optimizer
    :type n_epoch: int

    :param dataset: Path to the dataset, here MNIST. 
    :type dataset: string

    :param batch_size: size of the minibatches
    :type batch_size: int

    :param n_hidden: number of hidden units in the hidden layer
    :type n_hidden: int
    """

    datasets = load_data(dataset)

    train_set_x,train_set_y = datasets[0]
    valid_set_x,valid_set_y = datasets[1]
    test_set_x,test_set_y = datasets[2]

    #Compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[1]/batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[2]/batch_size


    ###############
    #-BUILD MODEL-#
    ###############

    print '.........BUILDING MODEL..............'

    #Allocate symbolic variables for the data





















        

            




# vim : tabstop=8 expandtab shiftwidth=4 softabstop=4
