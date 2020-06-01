#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[87]:


# Step 1: Import and Preprocessing
# Importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]


# In[88]:


X.head()


# In[89]:


X.nunique()


# In[90]:


X.columns


# In[91]:


one_hot_columns = [col for col in X.columns if col not in  ['CreditScore','Age','Balance', 'EstimatedSalary']]


# In[92]:


X = pd.get_dummies(X, columns = one_hot_columns)


# In[93]:


# Convert Ag in buckets
X['Age'] = pd.cut(X['Age'], 10)


# In[94]:


X = pd.get_dummies(X,columns = ['Age'])


# In[95]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns = X.columns)


# In[98]:


# Feature Scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[105]:


# There are two ways of initializing a neiral network:
# 1. by defining the sequence of layers
# 2. by defining the graph

# In this problem, we will intialize it by defining the seqquence of layers

# Step 2: Import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Step 3: Initialize
classifier = Sequential()

# Step 4: Add layers

# Input layer (designated by input_dim = 11) and first hidden layer
classifier.add(Dense(units = 19 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 37))
# second hidden layer
classifier.add(Dense(units = 19 , kernel_initializer = 'uniform', activation = 'relu'))
# output layer
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform', activation = 'sigmoid'))

# Step 5: compile ann - apply stochastic gradient descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Step 6: Fit the model:
classifier.fit(X_train,y_train, batch_size = 10, epochs = 100)


# In[108]:


# Step 7: Make Predictions
for layers in classifier.layers:
    weights = layers.get_weights()


# In[118]:


# shape of weights
for elements in classifier.get_weights():
    print(elements.shape)


# In[ ]:




