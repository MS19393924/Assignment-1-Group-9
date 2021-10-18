# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:40:33 2021

@author: f951
"""


import numpy as np
import os 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
import matplotlib as plt

def doubleToInt(yIn):
    tempY = np.array([])
    for sample in yIn:
        ySpecial = sample * 10
        tempY = np.append(tempY , ySpecial)
    return tempY


GWP = pd.read_csv("./garments_worker_productivity.csv")
GWP["wip"] = GWP["wip"].fillna(0)
GWP = GWP.drop(columns=["date"])
#may need to get rid of these functions...
lCoder = LabelEncoder()
lCoder.fit(GWP["quarter"])
GWP["quarter"] = lCoder.transform(GWP["quarter"])
lCoder.fit(GWP["department"])
GWP["department"] = lCoder.transform(GWP["department"])
lCoder.fit(GWP["day"])
GWP["day"] = lCoder.transform(GWP["day"])
np.random.seed(142)

X_orig = np.c_[GWP.drop(columns=["actual_productivity"])]
y_orig = GWP["actual_productivity"]
N = len(y_orig)
permute = np.random.choice(N, size=N, replace=False)
Ntrain = round(N * 0.8)
Ntest = N - Ntrain

Xtest = X_orig[permute[0:Ntest],:]
ytest = y_orig[permute[0:Ntest]]
print(ytest.shape)
X = X_orig[permute[Ntest:],:]
y = y_orig[permute[Ntest:]]
Xtrainmean = np.mean(X, axis=0)
Xtrainstd = np.std(X, axis=0)

X -= Xtrainmean
X /= Xtrainstd

Xtest -= Xtrainmean
Xtest /= Xtrainstd
d = X.shape[1]          
k = len(np.unique(y))   

W = 0.01 * np.random.randn(k, d)
b = np.zeros((k, 1))
#Setting up hyperparameters
step_size = 1e-0
reg = 1e-3

y = doubleToInt(y).astype(int)
ytest = doubleToInt(ytest).astype(int)

h = 150 # size of hidden layer
W0 = 0.01 * np.random.randn(h, d)
b0 = np.zeros((h, 1))
W1 = 0.01 * np.random.randn(k, h)
b1 = np.zeros((k, 1))


# gradient descent loop
num_examples = X.shape[0]
for i in range(500):
    a1 = np.maximum(0, X.dot(W0.T) + b0.T)
    scores = a1.dot(W1.T) + b1.T
        
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims = True)
    correct_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W0*W0) + 0.5 * reg * np.sum(W1*W1)
    loss = data_loss + reg_loss
    
    #gradient
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
    #backpropergation
    dW1 = dscores.T.dot(a1)
    db1 = np.sum(dscores, axis=0, keepdims=True).T
    da1 = dscores.dot(W1)
    #trigger function (ReLU)
    da1[a1 <= 0] = 0
    dW0 = da1.T.dot(X)
    db0 = np.sum(da1, axis=0, keepdims=True).T
    
    dW1 += reg * W1
    dW0 += reg * W0
    
    #Update parameter
    W0 += -step_size * dW0
    b0 += -step_size * db0
    W1 += -step_size * dW1
    b1 += -step_size * db1
    #Compare results to training true y
    a1 = np.maximum(0, X.dot(W0.T) + b0.T)
    scores = a1.dot(W1.T) + b1.T
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
    
    a1 = np.maximum(0, Xtest.dot(W0.T) + b0.T)
    scores = a1.dot(W1.T) + b1.T
    predicted_class = np.argmax(scores, axis=1)
    print('test accuracy: %.2f' % (np.mean(predicted_class == ytest)))
    print("Round " + str(i))
    
    h = 0.02
    i = j = 0
    for ix in range(4):
        iy = ix + 1 if ix < 3 else 0
        x_min, x_max = X[:, ix].min() - 1, X[:, ix].max() + 1
        y_min, y_max = X[:, iy].min() - 1, X[:, iy].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], 
                                        W0.T[(ix,iy),:]) + b0.T), W1.T) + b1.T
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        i = i + 1 if i == 0 else 0
        j = j + 1 if i == 0 else j
