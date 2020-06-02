#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Step 1: Import and Preprocessing
# Importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]


# In[3]:


X.head()


# In[4]:


X.nunique()


# In[5]:


X.columns


# In[6]:


one_hot_columns = [col for col in X.columns if col not in  ['CreditScore','Age','Balance', 'EstimatedSalary']]


# In[7]:


X = pd.get_dummies(X, columns = one_hot_columns)


# In[8]:


# Convert Ag in buckets
X['Age'] = pd.cut(X['Age'], 10)


# In[9]:


X = pd.get_dummies(X,columns = ['Age'])


# In[10]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns = X.columns)


# In[11]:


# Feature Scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[26]:


# There are two ways of initializing a neiral network:
# 1. by defining the sequence of layers
# 2. by defining the graph

# In this problem, we will intialize it by defining the seqquence of layers

# Step 2: Import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Step 3: Initialize
classifier_initial = Sequential()

# Step 4: Add layers

# Input layer (designated by input_dim = 11) and first hidden layer
classifier_initial.add(Dense(units = 19 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 37))
# second hidden layer
classifier_initial.add(Dense(units = 19 , kernel_initializer = 'uniform', activation = 'relu'))
# output layer
classifier_initial.add(Dense(units = 1 , kernel_initializer = 'uniform', activation = 'sigmoid'))

# Step 5: compile ann - apply stochastic gradient descent
classifier_initial.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Step 6: Fit the model:
classifier_initial.fit(X_train,y_train, batch_size = 10, epochs = 10)


# In[27]:


# shape of weights
for elements in classifier_initial.get_weights():
    print(elements.shape)


# In[28]:


# Step 7: Make Predictions
y_pred = classifier_initial.predict(X_test)
# y_pred is probabolity --> convert it into class prediction to get y_pred_2 
y_pred_2 = pd.DataFrame(y_pred).apply(lambda row: 1 if row[0]>0.5 else 0, axis = 1)


# In[29]:


# Step 8: Accuracy
import sklearn.metrics as metrics
print(f'accuracy: {metrics.accuracy_score(y_test, y_pred_2)}')
print(f'f1_score: {metrics.f1_score(y_test, y_pred_2)}')
print(f'precision: {metrics.precision_score(y_test, y_pred_2)}')
print(f'recall: {metrics.recall_score(y_test, y_pred_2)}')


# # Step 9: Evaluating the ANN:
# Judging our models's performance on one accuracy and one test set is not the best way to evaluate the model. Changing the test set will change the accuracy of our model slightly. To curb this issue, we will use k-fold cross validation.
# 
# !['cv'](images/cv.png)
# 
# ### Cross validation steps:
# 1. Train on k-1 folds
# 2. Test on remaining one
# 3. Take mean of all k accuracies
# 4. Find standard deviation of all accuracies
# 5. Based on accuracy and standard deviations, we can see which of the following cases our model satisfies:
# 
# !['bias_variance'](images/bias_variance.JPG)
# 
# ### Implementation steps:
# 
# 1. cross_val_score is sklearn function
# 2. create a keras wrapper for sklearn so that the keras classifier can be used in sklearn cross_val_score
# 3. 

# In[16]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# In[17]:


# defining the function to be passed to KerasClassifier to convert it to sklearn classifier
# only define the nn architecture. training and testing will be done by cross_val_Score
def nn_architecture():
    #initialize
    classifier = Sequential()
    # add input and first hidden layer
    classifier.add(Dense(units = 19, activation = 'relu', kernel_initializer= 'uniform' , input_dim =  37))
    # add second hidden layer
    classifier.add(Dense(units = 19, activation = 'relu', kernel_initializer= 'uniform' ))
    # add output layer
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer= 'uniform'))
    # compile
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# In[18]:


classifier_2 = KerasClassifier(build_fn= nn_architecture, batch_size = 10, nb_epoch = 100)


# In[19]:


accuracies = cross_val_score(estimator = classifier_2,X = X_train, y =  y_train, cv = 10, n_jobs=-1)


# In[20]:


avg_accuracy = accuracies.mean()
std_accuracy = accuracies.std()

print(f'accuracy = {avg_accuracy*100:.2f}% +/- {std_accuracy*100:.2f}%')


# # Step 10: Improving Accuracy
# 1. Dropout Regularisation
# 2. Hyper parametric tuning

# In[39]:


from keras.layers import Dropout
#nn architecture with dropouts
def nn_architecture_2():
    classifier = Sequential()
    # add layers with dropout regularization
    classifier.add(Dense(units = 19, activation = 'relu', kernel_initializer = 'uniform', input_dim = 37))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 19, activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

    # compile
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier
# cross validation
classifier_wrapper = KerasClassifier(build_fn = nn_architecture_2, batch_size = 10, nb_epoch = 100)
accuracies_dropout = cross_val_score(estimator=classifier_wrapper, X = X_train, y = y_train, cv = 10)


# In[ ]:





# In[40]:


# print(f'training set initital accuracy: {metrics.accuracy_score(y_train, classifier_initial(X_train))}')
print(f'test set initital accuracy: {metrics.accuracy_score(y_test, y_pred_2)}')
print(f'cross validation accuracy: {avg_accuracy}')
print(f'training set accuracy after dropout regularisation: {accuracies_dropout.mean()}')


# In[41]:


accuracies.mean()


# ### Hyperparametric tuning

# In[55]:


def nn_architecture_3(optimizer):
    classifier = Sequential()
    # add layers with dropout regularization
    classifier.add(Dense(units = 19, activation = 'relu', kernel_initializer = 'uniform', input_dim = 37))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 19, activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

    # compile
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier


# In[59]:


from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn=nn_architecture_3) #dont add batch_size and nb_epochs as we will find them using grid_search
parameters = {'batch_size': [25,32],
             'nb_epoch':[100,500],
             'optimizer':['adam','rmsprop']}


# In[60]:


grid_search = GridSearchCV(estimator= classifier, param_grid= parameters, 
                          scoring = 'accuracy',
                          cv = 10)


# In[61]:


grid_search = grid_search.fit(X_train, y_train)


# In[63]:


best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[64]:


best_param


# In[65]:


best_accuracy


# In[ ]:




