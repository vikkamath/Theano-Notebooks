#Code that does logistic Regression. Duh
#From deeplearning.net

#Imports to process dataset - MNIST
import cPickle
import gzip
import os
import time
#Numpy
import numpy as np
#Theano imports
import theano
import theano.tensor as T

class LogisticRegression(object):

    def __init__(self,input,n_in,n_out):

        #Define model parameters W and b
        self.W = theano.shared(value=numpy.zeros((n_in,n_out), dtype=theano.config.floatX),name='W')
	self.b = theano.shared(value=numpy.zeros((n_out,),dtype=theano.config.floatX),name='b')

	#Compiled theano function that returns the vector of class probabilities
	self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)
	#Symbolic description of how to compute the prediction - i.e the class whose prediction is maximal
	self.ypred = T.argmax(self.p_y_given_x,axis=1)
        #Parameters of the model
        self.params = [self.W,self.b]


    def negative_log_likelihood(self.y):


        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
        #NOTE:
        #T.arange(y.shape[0]) is a vector of integers [0,1,...,len(y)]
        #Indexing a matrix M by vectors [1,2,3,...] and [a,b,c,...]
        #returns the elements M[1,a],M[2,b],M[3,c] etc as a vector
        #Here we use this syntax to retrieve the log probabilities of the correct labels 'y'
        #y.shape[0] is symbolically the number of examples (say 'n') in the minibatch
        #T.log(self.p_y_given_x) is a matrix of log probabilities - one row per example, one 
        #                           column per class (call it LP)
        #LP[T.arange(y.shape(a)),y] is a vector 'v' containing [LP[0,y(0)],LP[1,y(1)],...LP[n-1,y(n-1)]
        #T.mean(<term above>) calculates the mean of the elements of 'v'. i.e. the mean log likelihood 
        #               across the minibatch
        

    def errors(self,y):
        """
        Return a float representing the number of errors in the minibatch 
        over the total number of examples in the minibatch. 
        Zero-one loss over the size of the minibatch

        param y: corresponds to the label given for each point in the dataset
        """

        #Check if y has the same dimension as the predictions
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same dimension as self.y_pred')
        







# vim : tabstop=8 expandtab shiftwidth=4 softabsstop=4
        

        
	
