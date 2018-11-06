import numpy as np  

####### generating the data ###############

N = 100 ### points in each class
K = 3 ###### number of classes
D = 2 #### dimension of the data

####### hyper-parameters ########

h = 100 ## number of hidden units
learning_rate = 1e-3

##### model parameters ########

Whx = np.random.randn(D,h) ## input to hidden
Wyh = np.random.randn(h,K) ## hidden to output
Bh = np.zeros((h,1)) ### hidden bias
By = np.zeros((K,1)) ## output bias

##### forward pass #########

def forward():
	