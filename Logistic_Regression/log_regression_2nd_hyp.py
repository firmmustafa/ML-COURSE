

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 18:41:08 2019

@author: Mustafa Saeed 34-5018
"""

# Importing libraries to work with
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

points = pd.read_csv(r"C:\Users\tiger\Downloads\heart_DD.csv")
# Indepedent matrix of features
X = np.array([ points['thalach'], points['sex'],points['ca'], points['age']**3,points['age']**2,points['age'], points['cp'], points['oldpeak']])   # Can take any number of numerical features 
X=X.T
# Dependent matrix of features
y = points.iloc[:, -1].values
#One value indicator
flag=1 
y= np.reshape(y,(len(y),1))
if sum(X.shape)==len(X):
    flag=0
    X= np.reshape(X,(len(y),1))
    X = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)

 


for i in range(0,X.shape[1]):
    X[:,i] = (X[:,i]-np.mean(X[:,i]))/(np.std(X[:,i]))



X = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
  
y_train= np.reshape(y_train,(len(y_train),1))
x_shape = X_train.shape
# vector of optimized parameters
theta = np.zeros([1,x_shape[1]])


def logistic(z):
    return 1 / (1 + np.exp(-z))

def fn_cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def predictor_f(X, theta):
    return logistic(np.dot(X, theta))

def predict(X, theta, threshold=0.5):
    return predictor_f(X, theta) >= threshold

    

def gradient_descent(X,y,h,learning_rate,theta):
    cost=[]
    o_error = fn_cost(h,y)
    i=-1
    while True:
        i+=1
        grad = np.dot(X.T, (h - y)) / len(y)
        grad=grad.T
        theta -= learning_rate*grad
        z=np.dot(X, theta.T)
        h = logistic(z)
        error = fn_cost(h,y)
        cost.append(error)
        if abs(error-o_error)<0.000001:
            break
        o_error = error
    return theta,cost,i+1,h


z = np.dot(X_train, theta.T)
h = logistic(z)
learning_rate = 0.01
opt_theta,cost,iters,h = gradient_descent(X_train,y_train,h,learning_rate,theta)
error = fn_cost(h, y_train)
y_train_pred = predict(X_train, opt_theta.T)
y_train_pred = y_train_pred*1
error_train_test = y_train^y_train_pred
error_train_sum = sum(error_train_test)/len(y_train)
print('error(train)->' , error_train_sum)
zz= np.dot(X_test, opt_theta.T)
hh=logistic(zz)
y_test_pred = fn_cost(hh,y_test)
y_test_pred = predict(X_test, opt_theta.T)
y_test_pred = y_test_pred*1
error_test = y_test^y_test_pred
error_test_sum = sum(error_test)/len(y_test)
print('error(test)->',error_test_sum)



print('my->',opt_theta)
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'b')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Error')  
ax.set_title('Error vs. Number of iterations') 

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print('built-in mse ->',metrics.mean_squared_error(y_test,y_pred))

logreg.fit(X_train, y_train)
print('built-in coef->',logreg.coef_)


