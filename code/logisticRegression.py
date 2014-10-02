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

        #Check if y is of the correct datatype: 

        #Currently, this function (that calculates the error over the minibatch)
        #       is only implemented for integer targets
        if y.dtype.startswith('int'):
           #T.neq operator returns 1 when there two values (here at positions in vectors)
           #        don't equal each other and 0 if they do equal each other. 
           return T.mean(T.neq(self.y_pred,y))

        else:
           raise NotImplementedError()

def load_data(dataset):

    """
    Loads the dataset
    param: dataset - path to the dataset
    type: string
    """

    data_dir,data_file = os.path.split(dataset)
    f = gzip.open(data_file,'rb')
    train_set,valid_set,test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set are of the form tuple(input,target)
    #input : ndarray of dim 2 (matrix)
    #       each row of which corresponds to an example
    #target: np.ndarray of 1 dimensions that has as many rows as input 
    #       each elements corresponds to the label of a row of 'input'

    #Notice that the function below is a function inside a function
    def shared_dataset(data_xy,borrow=True):
        """
        Function that loads the dataset into shared variables
        
        The reason the dataset is stored into shared variables is
        to allow theano to copy it into the GPU memory (when the code
        is run on a GPU). Since copying data to GPU memory is slow, 
        initiating a copy process to GPU memory every time a 
        minibatch is operated on (the default behavior
        for non-shared variables), creates a huge overhead
        and hence a decrease in performance. 
        """
    
        data_x , data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shraed(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        #When storing data on the GPU, it has to be stored as floats. 
        #That's what shared_y does. But since the values in y are used also as indices, 
        #they have to be ints. The hack below works around this. 
    
        return shared_x,T.cast(shared_y,'int32')

    train_set_x , train_set_y = shared_dataset(train_set)
    valid_set_x , valid_set_y = shared_dataset(valid_set)
    test_set_x , test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x,test_set_y)]

    return rval


def sgd_optimization_mnist(learning_rate=0.13 ,
                            n_epochs=1000,
                            dataset='mnist.pkl.gz',
                            batch_size=600):

    """
    Demonstrates SGD optimization of a log-linear model.
    Here, this is demonstrated on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for SGD)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: The path to the pickled MNIST dataset file. 
                    Originally from: http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """

    datasets = load_data(dataset)
    train_set_x , train_set_y = datasets[0]
    valid_set_x , valid_set_y = datasets[1]
    test_set_x , test_set_y = datasets[2]

    #Compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size

    #######################
    #-BUILD ACTUAL MODEL-#
    #######################


    print "...BUILDING MODEL.."
    
    #Allocate symbolic variables for the data
    index = t.lscalar() #Index to a minibatch
    x = T.matrix('x') #Data is presented as rasterized images
    y = T.ivector('y') #Targets are provided as a 1D vector of integer values

    #Construct the logistic regression object from the class. 
    #Each MNIST image is a 28x28 image
    classifier = LogisticRegression(input=x, n_in=28*28, n_out = 10)

    #The cost we minimize during training is the negative log likelihood
    #           of the model in symbolic format - that's usually how Theano operations
    #           are defined. i.e. 'functions' are first defined symbolically and
    #           then the theano.function is used to create a 'compliled' function
    cost = classifier.negative_log_likelihood(y)

    #Compiling a theano function that computes the error made on the minibatch 
    #   by the model 
    #NOTE: I haven't understood what the function below is doing completely
    test_model = theano.function(inputs=[index],
            outputs = classifer.errors(y),
            givens = {
                x : test_set_x[index*batch_size : (index+1)*batchsize]
                y : test_set_y[index*batch_size : (index+1)*batchsize]})

    validate_model = theano.function(inputs=[index],
            outputs = classifier.errors(y),
            givens = {
                x : valid_set_x[index*batch_size : (index+1)*batchsize]
                y : valid_set_y[index*batch_size : (index+1)*batchsize]

    #Compute the gradient of the cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost,wrt=classifier.W)
    g_b = T.grad(cost=cost,wrt=classifier.b)

    #Specify how to update the parameters of the model 
    #   Updates are specified as a list of (variable, update_expression) pairs
    updates = [(classifier.W, classifer.W - learning_rate*g_W),
               (classifier.b, classifer.b - learning_rate*g_b)]


    #Compile a theano function train_model that returns the cost but 
    #   at the same time updates the parameters of the model using the update rule
    #   specified as above. 
    train_model = theano.function(inputs = [index],
            outputs = cost, 
            updates = updates, 
            givens = {
                x : train_set_x[index*batch_size : (index+1)*batch_size]
                y : train_set_y[index*batch_size : (index+1)*batch_size]})


    ###############
    #-TRAIN MODEL-#
    ###############

    print '....Training the model....'

    #Early stopping parameters
    patience = 5000 #Look at atleast 5000 examples
    patience_increase = 2 #Wait this much longer if a new best is found
    improvement_threshold = 0.095 #An improvement is considered significant
                                  #if it exceeds this threshold. 




    


























       
        
        







# vim : tabstop=8 expandtab shiftwidth=4 softabsstop=4
        

        
	
