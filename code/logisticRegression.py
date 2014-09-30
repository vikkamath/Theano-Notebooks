#Code that does logistic Regression. Duh
#From deeplearning.net

import cPickle
import gzip
import os
import time

import numpy as np

import theano
import theano.tensor as T

class LogisticRegression(object):

    def __init__(self,input,n_in,n_out):

        self.W = theano.shared(value=numpy.zeros((n_in,n_out), dtype=theano.config.floatX),name='W')
	self.b = theano.shared(value=numpy.zeros((n_out,),dtype=theano.config.floatX),name='b')
	#Compiled theano function that returns the vector of class probabilities
	self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)
	#Symbolic description of how to compute the prediction - i.e the class whose prediction is maximal
	self.ypred = T.argmax(self.p_y_given_x,axis=1)
        #Parameters of the model
        self.params = [self.W,self.b]


    def negative_log_likelihood(self.y):






# vim : tabstop=8 expandtab shiftwidth=4 softabsstop=4
        

        
	
