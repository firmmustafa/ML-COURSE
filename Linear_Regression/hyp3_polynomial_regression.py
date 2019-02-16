
"""
Created on Fri Feb  8 17:40:25 2019

@author: Mustafa Saeed 34-5018
"""
# Importing libraries to work with
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
X = np.array([  points['view'], points['Area'], points['bedrooms'], points['sqft_above'], points['sqft_living'], points['grade'] , points['sqft_living']**2 , points['sqft_living']*points['grade'] , points['grade']**2 ]) # Can take any number of numerical features 
X = X.T
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
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Avoiding dummy variable trap
X=X[:,1:]

for i in range(3,X.shape[1]):
    X[:,i] = (X[:,i]-np.mean(X[:,i]))/(np.std(X[:,i]))
y[:,0] = y[:,0]/np.mean(y[:,0])
X = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)

# vector of optimized parameters
theta = np.zeros([1,X.shape[1]]) 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
  
y_train= np.reshape(y_train,(len(y_train),1))

''' Function to compute value of squared error between hypothesis and target values'''
def fn_cost(X,y,theta):
    error= (np.dot(X,np.transpose(theta)) - y)**2
    error_r = 1/(2*(len(X)))*np.sum(error)
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
        if abs(error-o_error)<0.0001:
            break
        o_error=error
    return theta,cost,i+1
''' Function that uses normal equation for getting optimized model parameters'''
def normeq(X,y):

    theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta_best

def predictor(opt_theta,X):
   return np.dot(X,np.transpose(opt_theta))

#learning rate
learning_rate = 0.01
# Calling gradient descent fn which return -> optimized parameters, cost vector, and iterations
g,cost,iters = gradientDescent(X_train,y_train,theta,learning_rate)
print('Optimized thetas using gradient descent ->',g)
# Calling norm eq fn which return optimized parameters
theta_norm = normeq(X_train,y_train)
print('Optimized thetas using norm eq ->', theta_norm.T)
model_acc_train = fn_cost(X_train,y_train,g)
print('Model mse(train)-> ', model_acc_train)
model_acc_test = fn_cost(X_test,y_test,g)
print('Model mse(test)-> ', model_acc_test)

# computing final error for model parameters using norm eq
error_finale_norm = fn_cost(X_train,y_train,theta_norm.T)
print('Model erorr: norm eq (train)->' , error_finale_norm)
model_acc_norm_test = fn_cost(X_test,y_test,theta_norm.T)
print('Model erorr : norm eq (test)->' , model_acc_norm_test )

y_pred_test = predictor(g,X_test)




### Plot
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'b')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Error')  
ax.set_title('Error vs. Number of iterations') 
r2_test = metrics.r2_score(y_test, y_pred_test)
print('r-squared ->', r2_test) # the higher the better
#print( metrics.mean_squared_error(y_test, y_pred_test)/2)








