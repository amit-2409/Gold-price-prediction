#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries
# 

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import os


# # Data collection and processing

# In[12]:


#loading the dataset 
gold_data=pd.read_csv(r'C:\Users\KIIT\Downloads\gld_price_data.csv')


# In[19]:



gold_data.head(-1)


# In[22]:


gold_data.shape


# In[25]:


gold_data.info()


# In[27]:


#number of null values
gold_data.isnull().sum()


# In[28]:


gold_data.describe()


# In[29]:


#checking for correlation
correlation=gold_data.corr()


# In[36]:


#heatmap to understand correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot_kws={'size':8},cmap='Blues')


# In[38]:


#correlation values of GLD
print(correlation['GLD'])


# In[39]:


sns.displot(gold_data['GLD'],color='green')


# # splitting the features and the target variable

# In[41]:


X=gold_data.drop(['Date','GLD'],axis=1)
X.head()


# In[43]:


Y=gold_data['GLD']
Y.head()


# splitting into train and test data

# In[44]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# # Model Training

# In[46]:


regressor=RandomForestRegressor(n_estimators=100)

training the model
# In[47]:


regressor.fit(X_train,Y_train)

evaluation of model-
prediction of test data
# In[48]:


test_data_prediction=regressor.predict(X_test)
print(test_data_prediction)


# In[49]:


test_data_prediction.shape


# calculation of error for performance of the model
# 

# In[51]:


error_score=metrics.r2_score(Y_test,test_data_prediction)
print(error_score)


# compare the actual values and predicted values in a plot

# In[54]:


Y_test=list(Y_test)
plt.plot(Y_test,color='blue',label='actual value')
plt.plot(test_data_prediction,color='red',label='predicted value')
plt.title('actual value vs predicted value')
plt.xlabel('number of values')
plt.ylabel('GLD price')
plt.legend()
plt.show()


# In[ ]:




