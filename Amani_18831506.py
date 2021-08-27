#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import csv
from scipy import stats
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# import dataset
wine=pd.read_csv("winequality-red.csv",delimiter=";")


# In[3]:


wine.head()


# In[4]:


wine.describe()


# In[5]:


wine.shape


# In[6]:


wine.dtypes


# In[7]:


wine.info()
#give info about dataset,dataset contains 1599 samples(rows) and 12 variables (columns)
# one variable is ordinal and the rest are numerical.


# In[8]:


correlation= wine.corr()['alcohol'].drop('alcohol')
print(correlation)


# In[9]:


x= wine.drop(['alcohol'],axis=1)
y= wine['alcohol']
x= sm.add_constant(x)
x_train, x_test, y_train, y_test = train_test_split(x,y)
model= sm.OLS(y_train,x_train).fit()
print(model.summary())
#put x and y
#drop alcohol from x
# treat y as alcohol variable
# split the data in training set and testing set
# 80 % trainig set
# 20% testing set


# In[10]:


model.params
# coefficients


# In[12]:



model.pvalues
# porint the p value


# In[13]:


y_pred=model.predict(x_test)
print(y_pred)


# In[14]:


sns.regplot(y_test,y_pred,line_kws={'color':'orange'},ci=None)
plt.xlabel('Actuall')
plt.ylabel('predictions')
plt.title('prediction vs Actuall')
plt.show()
# plot the data


# In[15]:


regressor= LinearRegression()
regressor.fit (x_train,y_train)
r2_score = regressor.score(x_train,y_train)
print("the accuracy of model is",r2_score*100,'%')


# In[16]:


train_pred = regressor.predict(x_train)
print(train_pred)


# In[17]:


test_pred= regressor.predict(x_test)
print(test_pred)


# In[18]:


train_rmse=mean_squared_error(train_pred,y_train)**0.5
print(train_rmse)
# calculate rmse


# In[19]:


pwd


# In[ ]:




