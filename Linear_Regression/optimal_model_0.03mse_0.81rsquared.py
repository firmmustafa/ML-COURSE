# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 19:22:14 2019

@author: mustafa saeed 34-5018
"""

# Importing libraries to work with
import numpy as np
import pandas as pd
# importing dataset
points = pd.read_csv(r"C:\Users\tiger\Downloads\house_data_complete.csv",parse_dates=['date'])
# Dropping NANs
points=points.dropna()

def ziparea(zipcode):
    if zipcode <= 98033:
        return 'Area1'
    elif zipcode>98033 and zipcode <= 98065:
        return 'Area2'
    elif zipcode>98065 and zipcode<=98118:
        return 'Area3'
    elif zipcode>98118 and zipcode<=98199:
        return 'Area4'

points['Area'] = points['zipcode'].apply(lambda x:ziparea(x))
points['year_sold'] = pd.DatetimeIndex(points['date']).year
# Removing unwanted features
#del points['yr_renovated']
del points['id']
del points['zipcode']
del points['date']
#del points['condition']
#del points['yr_built']
#del points['long']
#One value indicator
flag=1
# Indepedent matrix of features
X = points.iloc[:, [2,3,4,6,7,8,9,12,13,14,15,16,18]
 ].values # Can take any number of numerical features 
# Dependent matrix of features
y = points.iloc[:, 0].values   
y= np.reshape(y,(len(y),1))
if sum(X.shape)==len(X):
    flag=0
    X= np.reshape(X,(len(y),1))
    X = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, X.shape[1]-1] = labelencoder_X.fit_transform(X[:, X.shape[1]-1])
onehotencoder = OneHotEncoder(categorical_features = [X.shape[1]-1])
X = onehotencoder.fit_transform(X).toarray()



for i in range(4,X.shape[1]):
    X[:,i] = (X[:,i]-np.mean(X[:,i]))/(np.std(X[:,i]))
y[:,0] = y[:,0]/np.mean(y[:,0])
X = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)

#for i in range(4,X.shape[1]):
#    X[:,i] = (X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i]))
#y[:,0] = (y[:,0]-min(y[:,0]))/(max(y[:,0])-min(y[:,0]))   

# vector of optimized parameters
theta = np.zeros([1,X.shape[1]]) 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
  
y_train= np.reshape(y_train,(len(y_train),1))

from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.linear_model import LinearRegression


def create_polynomial_regression_model(degree):
  "Creates a polynomial regression model for the given degree"
  poly_features = PolynomialFeatures(degree=degree)
  
  # transform the features to higher degree features.
  X_train_poly = poly_features.fit_transform(X_train)
  
  # fit the transformed features to Linear Regression
  poly_model = LinearRegression()
  poly_model.fit(X_train_poly, y_train)
  
  
  # predicting on training data-set
  y_train_predicted = poly_model.predict(X_train_poly)
  
  # predicting on test data-set
  y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
  r2_test = metrics.r2_score(y_test, y_test_predict)
  
  # evaluating the model on training dataset
   
  mses= metrics.mean_squared_error(y_train,y_train_predicted)
  print('poly->',mses/2)
  
  # evaluating the model on test dataset
  
  print('polyTEST->',metrics.mean_squared_error(y_test,y_test_predict)/2)
  print(r2_test)
  

create_polynomial_regression_model(2)
  

