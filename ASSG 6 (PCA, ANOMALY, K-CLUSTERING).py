
# coding: utf-8

# ## IMPORT LIBRARIES

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd


# ## GET DATA

# In[23]:


dataset= pd.read_csv(r"C:\Users\tiger\Downloads\house_prices_data_training_data.csv",parse_dates=['date'])
dataset.head()


# ## PREPARE

# In[24]:


dataset = dataset.dropna()


# In[25]:


msk = np.random.rand(len(dataset)) < 0.8


# In[26]:


train = dataset[msk]


# In[27]:


test = dataset[~msk]


# In[28]:


print(len(train), "train +", len(test), "test")


# In[29]:


# Matrix of features (test and training sets)
X_train = train.drop(['id','date','price'],axis=1)
X_test = test.drop(['id','date','price'],axis=1)
y_train = train[['price']].copy()
y_test = test[['price']].copy()


# In[30]:


# Feature Scaling X
mean = X_train.mean()
X_train_norm = X_train - mean
std = X_train_norm.std()
X_train_norm = X_train_norm / std
X_test_norm = (X_test - mean)/std
# Feature Scaling Y
mean_y = y_train.mean()
y_train_norm = (y_train - mean_y)
std_y = y_train_norm.std()
y_train_norm = y_train_norm / std_y
y_test_norm = (y_test-mean_y)/std_y


# In[31]:


# Linear Regression model before any adjustments:
# WIth normalization
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_norm, y_train_norm)
y_pred = regressor.predict(X_test_norm)
from sklearn import metrics
R_sq = metrics.r2_score(y_test_norm,y_pred)
mse = metrics.mean_squared_error(y_test_norm,y_pred)
print('MSE before PCA (18 features):', mse)
print('R-squared before PCA (18 features):', R_sq)

# # Without normalization
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# from sklearn import metrics
# R_sq = metrics.r2_score(y_test,y_pred)
# mse = metrics.mean_squared_error(y_test,y_pred)
# print('MSE before PCA (18 features):', mse)
# print('R-squared before PCA (18 features):', R_sq)


# ## TASKS:

# ### 1. PCA

# In[32]:


# TASK 1,2
# # Without Normalization
# corr_matrix = X_train.corr()
# corr_matrix
# WIth Normalization
corr_matrix = X_train_norm.corr()
corr_matrix


# In[33]:


# # TASK 3 
#Without Normalization
# cov_matrix = X_train.cov()
# cov_matrix
# With Normalization
cov_matrix = X_train_norm.cov()
cov_matrix


# In[34]:


# TASK 4
U, S, V = svd(cov_matrix)


# In[35]:


def alphacalc(S,K):
    alpha = 1 - (sum(S[0:K])/sum(S))
    return alpha


# In[36]:


# TASK 5
K=0
for i in range(1,X_train.shape[1]):
    alpha = alphacalc(S,i)
    if alpha <= 0.01:
        K = i
        break
K


# In[37]:


# TASK 6
# # # Without Normalization
# R = np.dot( U[:,0:K].T , X_train.T )
# R.shape
# With Normalization
R = np.dot( U[:,0:K].T , X_train_norm.T )
R.shape


# In[38]:


# TASK 7 
A = np.dot(R.T,U[:,0:K].T)
A.shape


# In[39]:


# TASK 8 
# # Without Normalization
# Error = sum(((X_train - A)**2).sum(axis=0))
# Error
# With Normalization
Error = sum(((X_train_norm - A)**2).sum(axis=0))
Error


# In[40]:


## TASK 9
# #Without Normalization
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(R.T, y_train)
# # With Normalization
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(R.T, y_train_norm)



# In[41]:


# PCA mapping for test dataset
# # Without normalization
# X_test_pca = np.dot( U[:,0:K].T , X_test.T )
# y_pred = regressor.predict(X_test_pca.T)
# With normalization
X_test_pca = np.dot( U[:,0:K].T , X_test_norm.T )
y_pred = regressor.predict(X_test_pca.T)


# In[42]:


# Evaluate Model 
from sklearn import metrics
# # Without normalization
# R_sq = metrics.r2_score(y_test,y_pred)
# mse = metrics.mean_squared_error(y_test,y_pred)
# # With normalization
R_sq = metrics.r2_score(y_test_norm,y_pred)
mse = metrics.mean_squared_error(y_test_norm,y_pred)
print('MSE after PCA (',(K),') features:', mse)
print('R-squared after PCA (',(K),') features:', R_sq)


# ### 2. K-Means Clustering

# In[64]:


def initalize_centroids(X,k):
    centroids = np.zeros([k,X.shape[1]]) # centroids matrix: K(no. of clusters) x no. of features
    rand_ind = np.random.permutation(X.shape[0])
    centroids = X[rand_ind[0:K],:]
    return centroids


# In[65]:


from numpy import linalg as LA
def closest_centroid(X,centroid):
    v = [None]*len(centroid)
    ind = [None]*len(X)
    for i in range(0,len(X)):
        for j in range(0,len(centroid)):
            v[j] = LA.norm(np.subtract(X[i,:],centroid[j,:])**2)
        ind[i] = np.argmin(v)
    return ind          


# In[66]:


def update_centroids(old_cent, X, k):
    new_cent = zeros([l,X.shape[1]])
    for i in range(0,k):
        new_cent[i,:] = np.mean(X[X==old_cent[i],:])
    return new_cent


# In[76]:


R.shape


# In[78]:


# Intialize Centroids
centroids = initalize_centroids(R.T,3)
centroids.shape

