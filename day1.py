from theano import *
import theano.tensor as T

x = T.dmatrix('x') #float 64 2D matrix
s = 1 / (1 + T.exp(-x)) #Define the logistic function
logistic = function([x],s)
#function returns an object that calculates outputs from ips
#Do: help(function) - function belongs to theano.compile

logistic([[0,1]])
 #Prints the value of logistic(0) and logistic(1). Note the double paranthesis

a,b = T.dmatrices('a','b')
#Notice how it's matrices - using this you can generate multiple matrices!
#dmatrices produces as many outputs as names that you provide. It is a shortcut for allocating symbolic variables that we will often use in the tutorials.

diff = a-b
abs_diff = abs(diff)
diff_squared = diff**2
#Self explanatory
f = function([a,b],[abs_diff,diff_squared])
#A function that performs multiple operations on multiple matrices

f([[1, 1], [1, 1]], [[0, 1], [2, 3]])


#The param class allows you to specify in greater detail the kinds of 


