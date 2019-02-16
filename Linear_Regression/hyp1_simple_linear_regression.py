# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:40:25 2019

@author: Mustafa Saeed 34-5018
"""
# Importing libraries to work with
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# importing dataset
points = pd.read_csv(r"C:\Users\tiger\Downloads\house_prices_data_training_data.csv",parse_dates=['date'])
# Dropping NANs
points=points.dropna()
#One value indicator
flag=1
# Indepedent matrix of features
X = points.iloc[:,20].values # Can take any number of numerical features 
# Dependent matrix of features
y = points.iloc[:, 2].values   
y= np.reshape(y,(len(y),1))
if sum(X.shape)==len(X):
    flag=0
    X= np.reshape(X,(len(y),1))
    X = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)
y = np.append(arr = np.ones((len(y),1)).astype(int), values = y, axis = 1)


#Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
if flag==0:
    X = X[:, 1:]
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
y = y[:, 1:]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
x_shape = X.shape
# vector of optimized parameters
theta = np.zeros([1,x_shape[1]]) 

''' Function to compute value of squared error between hypothesis and target values'''
def fn_cost(X,y,theta):
    error= (np.dot(X,np.transpose(theta)) - y)**2
    error_r = 1/(len(X))*np.sum(error)
    return error_r
''' Function that optimizes model parameters for minimal error using gradient descent algorithm''' 
def gradientDescent(X,y,theta,learning_rate):
    cost = []
    i=-1
    o_error= (np.dot(X,np.transpose(theta)) - y)**2
    o_error = np.sum(o_error)
    while True:
        i+=1
        grad = 1/len(X) * np.sum(X * (np.dot(X,np.transpose(theta)) - y), axis=0)
        theta = theta - learning_rate*grad
        cost.append(fn_cost(X,y,theta))
        error= (np.dot(X,np.transpose(theta)) - y)**2
        error = np.sum(error)
        if abs(error-o_error)<0.000000001:
            break
        o_error=error
    return theta,cost,i+1
''' Function that uses normal equation for getting optimized model parameters'''
def normeq(X,y):

    theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta_best


#learning rate
learning_rate = 0.01
# Calling gradient descent fn which return -> optimized parameters, cost vector, and iterations
g,cost,iters = gradientDescent(X,y,theta,learning_rate)
print('Optimized thetas using gradient descent ->',g)
# Calling norm eq fn which return optimized parameters
theta_norm = normeq(X,y)
print('Optimized thetas using norm eq ->', theta_norm.T)
# computing final error for model parameters using norm eq
error_finale_norm = fn_cost(X,y,theta_norm.T)
print('final erorr: norm eq ->' , error_finale_norm)
# computing final error for model parameters using gradient descent
error_finale = fn_cost(X,y,g)
print('final erorr: gradient descent ->' ,error_finale)
## Plot
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'b')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Error')  
ax.set_title('Error vs. Number of iterations') 

#### From sklearn

X = X[:,1:]
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print('Optimized parameters usning sklearn',lin_reg.coef_)
print('Optimized intercept using sklearn', lin_reg.intercept_)
#f= lin_reg.intercept_
#theta_model = np.append(f,lin_reg.coef_)
#
#
#y_pred_linear = lin_reg.predict(X)
#
#from sklearn import metrics 
#print('linear->',metrics.mean_squared_error(y,y_pred_linear))


