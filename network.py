import numpy as np  
 
#### setting up the input and dictionary ######

data = open('input.txt','r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_chars = {i:ch for i,ch in enumerate(chars)}

print vocab_size

H1 = 120 # no of hidden layer 1 neurons
H2 = 120 # no of hidden layer 2 neurons
D = vocab_size
Z1 = H1 + D
Z2 = H2 + H1

############ functions ##########
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def dtanh(x):
    return 1 - np.tanh(x)**2

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))



#### model parameters #####

Wf1=np.random.randn(Z1, H1) / np.sqrt(Z1 / 2.)
Wi1=np.random.randn(Z1, H1) / np.sqrt(Z1 / 2.)
Wc1=np.random.randn(Z1, H1) / np.sqrt(Z1 / 2.)
Wo1=np.random.randn(Z1, H1) / np.sqrt(Z1 / 2.)
bf1=np.zeros((1, H1))
bi1=np.zeros((1, H1))
bc1=np.zeros((1, H1))
bo1=np.zeros((1, H1))

Wf2=np.random.randn(Z2, H2) / np.sqrt(Z2 / 2.)
Wi2=np.random.randn(Z2, H2) / np.sqrt(Z2 / 2.)
Wc2=np.random.randn(Z2, H2) / np.sqrt(Z2 / 2.)
Wo2=np.random.randn(Z2, H2) / np.sqrt(Z2 / 2.)
bf2=np.zeros((1, H2))
bi2=np.zeros((1, H2))
bc2=np.zeros((1, H2))
bo2=np.zeros((1, H2))

Wy=np.random.randn(H2, D) / np.sqrt(D / 2.)
by=np.zeros((1, D))

def lstm_forward(X,state):
	X_one_hot = np.zeros(vocab_size)
	X_one_hot[X] = 1.
	X_one_hot = X_one_hot.reshape((1,-1))

	c1_old, h1_old, c2_old, h2_old = state

	X1 = np.column_stack((h1_old,X_one_hot))

	hf1 = sigmoid(np.dot(X1,Wf1) + bf1)
	hi1 = sigmoid(np.dot(X1,Wi1) + bi1)
	ho1 = sigmoid(np.dot(X1,Wo1) + bo1)
	hc1 = np.tanh(np.dot(X1,Wc1) + bc1)

	c1 = np.multiply(hf1,c1_old) + np.multiply(hi1,hc1)
	h1 = ho1*np.tanh(c1)
	X2 = np.column_stack((h2_old,h1))

	hf2 = sigmoid(np.dot(X2,Wf2) + bf2)
	hi2 = sigmoid(np.dot(X2,Wi2) + bi2)
	ho2 = sigmoid(np.dot(X2,Wo2) + bo2)
	hc2 = np.tanh(np.dot(X2,Wc2) + bc2)

	c2 = np.multiply(hf2,c2_old) + np.multiply(hi2,hc2)
	h2 = ho2*np.tanh(c2)

	y = np.dot(h2,Wy) + by
	prob = softmax(y)

	state = (c1,h1,c2,h2)
	cache = hf1
	return prob, state, cache


######## running the network #########
predictions = []
states = []
caches = []
c1 = np.zeros((1,H1))
h1 = np.zeros((1,H1))
c2 = np.zeros((1,H2))
h2 = np.zeros((1,H2))
state_prev = (c1,h1,c2,h2)

predictions.append(data[0])
for i in range(len(data)-1):
    prob, state_new, cache = lstm_forward(char_to_ix[predictions[i]],state_prev)
    y = data[i+1]
    x_pred = np.argmax(prob)
    x_pred = ix_to_chars[x_pred]    
    predictions.append(x_pred)
    state_prev = state_new

print ''.join(predictions)

