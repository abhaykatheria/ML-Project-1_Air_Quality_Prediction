# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 01:14:43 2019

@author: Mithilesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df=pd.read_csv("Train.csv")

x=np.array(df)  #Converting dataframe into numpy array in order to access element
Y=df.target     #Saving the target values in Y
x=x[:,:5]       #Making x an array having all 5 features
ones=np.ones((x.shape[0],1))  
X=np.append(ones,x,axis=1)   #Since first element i.e x[0] should be 1, so we have added 1 as the first column
df_test=pd.read_csv("Test.csv")  #Reading Test data
X_test=np.array(df_test)
X_test=np.append(np.ones((X_test.shape[0],1)),X_test,axis=1)  #adding 1 as column 1 in test data
print("Shape of Training data " ,X.shape)

def hypothesis(x,theta):       #creating hypothesis function
    result=np.dot(theta,x.T)
    return(result)

def predictions(X,theta):     #creating prediction function which uses X_test and theta values to predict the Y value
    y_pred = []
    
    for i in range(X.shape[0]):
        pred = hypothesis(X[i],theta)
        y_pred.append(pred)
    y_pred = np.array(y_pred)
    return (y_pred)

def get_error(X,Y,theta):
    e = 0
    m = X.shape[0]
    for i in range(m):
        e += (hypothesis(X[i],theta)-Y[i])**2
        
    return(e)

def getGradients(X,Y,theta):    
    grads = np.zeros((X.shape[1]))
    n = len(grads) # no. of features inc. bias
    m = X.shape[0]
    for i in range(m):
        hx = hypothesis(X[i],theta)
        for grad_index in range(n):
            grads[grad_index] += (hx - Y[i])* X[i,grad_index]           
        
    return (grads)

def batchgrad(X,Y,theta,batch_size=100):
    m=Y.shape[0]
    indices=np.arange(m)
    np.random.shuffle(indices)
    indices=indices[:batch_size]
    grad=np.zeros((X.shape[1]))
    n=len(grad)
    for i in indices:
        hx=hypothesis(X[i],theta)
        for grad_index in range(n):
            grad[grad_index] += (hx - Y[i])* X[i,grad_index]           
        
    return (grad)

def miniBatchGradientDescent(X,Y,maxItr=100, learning_rate = 0.01):
    
    theta = np.zeros((X.shape[1]))
    error = []
    n=len(theta)
    for i in range(maxItr):
        grad=batchgrad(X,Y,theta)
        e=get_error(X,Y,theta)
        for theta_index in range(n):
            theta[theta_index]-=learning_rate*grad[theta_index]
        error.append(e)

    return (theta,error)

plt.figure(3)
theta_minibatch, error_minibatch miniBatchGradientDescent(X,Y,300,0.001)
sns.lineplot(data = np.array(error_minibatch))
plt.title("Mini Batch Gradient Descent")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()

ans=predictions(X_test,theta_minibatch)
print(ans)
