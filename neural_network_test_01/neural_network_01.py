import  numpy as np
import  time


#variables
h_hidden = 10
n_in =  10

#output
n_out = 10

#sample data
n_sampleData = 300

#hyperparameter
n_learn = 0.01
moment = 0.9

#seed
np.random.seed(0)


#function

def sigmoid(x):
    return  1.0/(1.0 * np.exp(-x))

def tnah_prime(x):
    return 1-np.tanh(x)**2

#train the model
#通过矩阵计算 和预测
'''
x -- its mean input data
t -- transpose data
V,W -- network layer
bv,bw -- biases
'''
def train(x , t , V , W , bv , bw):
    #forward  -- matrix multiply  + biases
    A = np.dot(x,V) + bv
    Z = np.tanh(A)

    B= np.dot(Z,W) + bw
    Y = sigmoid(B)


    # backward
    Ew = Y - t
    Ev = tnah_prime(A) * np.dot(W,Ew)

    #predict our loss
    dW = np.outer(Z,Ew)
    dV = np.outer(x,Ev)

    loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))

    return  loss, (dV, dW, Ev, Ew)


def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W  + bw)

    return (sigmoid(B) > 0.5).astype(int)

#create layes
V = np.random.normal(scale=0.1, size=(n_in,h_hidden))
W = np.random.normal(scale=0.1, size=(h_hidden, n_out))


bv = np.zeros(h_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]



#generate our last data
X = np.random.binomial(1, 0.5, (n_sampleData, n_in))
T = X ^ 1


#Training time
for epoch in range(10):
    err = []
    upd = [0] * len(params)

    t0 = time.clock()

    # for each data point
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i],  *params)
        #update loss
        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = n_learn * grad[j] + moment * upd[j]

        err.append(loss)


    print('Epoch: %d, Loss: %.f, Time: %.4fs'%(
        epoch,np.mean(err), time.clock() - t0))


#predict some data use in model
x = np.random.binomial(1, 0.5, n_in)
print('XOR prediction')
print(x)
print(predict(x, *params))