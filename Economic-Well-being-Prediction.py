#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import random as rm 
import yfinance as yf
import plotly.express as px 
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import seaborn as sns 


# ## Using information from Different African Countries to predict Economic Well being of the Country

# In[15]:


## Import the train and test variables form the .csv file


# In[16]:


train = pd.read_csv(r"C:\Users\ppiko\Documents\3. LBS - MBA 20\MBA 20 - Olumide Obanla\1. Data Analysis Club Exercise and Video\Zindi Africa\Done\Economic-Well-being-Prediction\Train.csv")


# In[17]:


test = pd.read_csv(r"C:\Users\ppiko\Documents\3. LBS - MBA 20\MBA 20 - Olumide Obanla\1. Data Analysis Club Exercise and Video\Zindi Africa\Done\Economic-Well-being-Prediction\Test.csv")


# In[18]:


train 


# In[19]:


test 


# In[20]:


train.info()


# In[21]:


test.info()


# In[22]:


train.isnull().sum()


# In[23]:


test.isnull().sum()


# In[24]:


train.describe()


# In[25]:


test.describe()


# In[49]:


## Exploratory Data Analysis of the different features with the target 
## GHSL stand for Global Human Settlement Layer 


# In[38]:


plt.figure(figsize=(15,15))
sns.heatmap(train.corr(), cmap='Blues', annot=True)
plt.title("Correlation Matrix");


# In[42]:


train.columns


# In[50]:


px.scatter(train, y="Target", x="year", hover_name="country", color="urban_or_rural", size="ghsl_built_pre_1975", log_x=True,
          height=1000,width=1000, title="GHSL in 1975 in Relation to Economical Wellbeing")


# In[52]:


px.scatter(train, y="Target", x="year", hover_name="country", color="urban_or_rural", size="ghsl_built_1975_to_1990", log_x=True,
          height=1000,width=1000, title="GHSL Btwn (1975 - 1990) in Relation to Economical Wellbeing Target")


# In[53]:


px.scatter(train, y="Target", x="year", hover_name="country", color="urban_or_rural", size="ghsl_built_2000_to_2014", log_x=True,
          height=1000,width=1000, title="GHSL Btwn (2000 - 2014) in Relation to Economical Wellbeing Target")


# In[77]:


get_ipython().run_cell_magic('time', '', 'px.histogram(train, x="year",color="urban_or_rural", hover_name="country", marginal="rug")')


# In[71]:


train.columns


# In[80]:


sns.scatterplot(data=train, x="ghsl_built_1975_to_1990", y="Target")


# In[81]:


sns.scatterplot(data=train, x="ghsl_built_1990_to_2000", y="Target")


# In[82]:


sns.scatterplot(data=train, x="ghsl_built_2000_to_2014", y="Target")


# In[84]:


train.groupby("country").sum()


# In[85]:


sns.scatterplot(data=train.groupby("country").sum(), x="year", y="Target")


# In[89]:


sns.boxplot(train.Target)


# In[96]:


train.head()


# In[97]:


test.head()


# In[106]:


sns.jointplot(x="urban_or_rural", y="Target", data=train, kind="hist");


# In[115]:


plt.figure(figsize=(15,15))
sns.countplot(x="urban_or_rural", data=train, hue="country");
plt.xlabel("Urban & Rural Areas")
plt.title("Urban and Rural Area by African Countries");


# In[120]:


rural = {"R":1, "U":0}


# In[121]:


rural_numeric = train.urban_or_rural.map(rural)


# In[122]:


rural_numeric


# In[123]:


train.Target.corr(rural_numeric)


# In[124]:


urban = {"U":1, "R":0}


# In[125]:


urban_numeric = train.urban_or_rural.map(urban)


# In[126]:


train.Target.corr(urban_numeric)


# In[127]:


train['urban_or_rural'] = train['urban_or_rural'].map({'U':1,'R':0})


# In[128]:


train 


# In[129]:


test['urban_or_rural'] = test['urban_or_rural'].map({'U':1,'R':0})


# In[130]:


test.head()


# In[133]:


id = test['ID']


# In[134]:


id


# In[135]:


train.columns


# In[139]:


X = train[['urban_or_rural', 'ghsl_water_surface','ghsl_built_pre_1975', 'ghsl_built_1975_to_1990','ghsl_built_1990_to_2000', 'ghsl_built_2000_to_2014',
       'ghsl_not_built_up', 'ghsl_pop_density', 'landcover_crops_fraction','landcover_urban_fraction', 'landcover_water_permanent_10km_fraction',
       'landcover_water_seasonal_10km_fraction', 'nighttime_lights','dist_to_capital', 'dist_to_shoreline']]


# In[140]:


X


# In[143]:


y = train[['Target']]


# In[144]:


y


# In[145]:


Test = test[['urban_or_rural', 'ghsl_water_surface','ghsl_built_pre_1975', 'ghsl_built_1975_to_1990','ghsl_built_1990_to_2000', 'ghsl_built_2000_to_2014',
       'ghsl_not_built_up', 'ghsl_pop_density', 'landcover_crops_fraction','landcover_urban_fraction', 'landcover_water_permanent_10km_fraction',
       'landcover_water_seasonal_10km_fraction', 'nighttime_lights','dist_to_capital', 'dist_to_shoreline']]


# In[146]:


Test


# In[150]:


Test.info()


# In[151]:


X.info()


# In[153]:


## Modelling of the Data Set
## We will be using Linear Regression to build the model


# In[154]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=101)


# In[155]:


X_train


# In[156]:


y_train 


# In[157]:


X_test


# In[158]:


y_test


# In[159]:


LR = LinearRegression()


# In[160]:


LR.fit(X_train, y_train)


# In[161]:


LR.intercept_


# In[162]:


LR.coef_


# In[164]:


Prediction = LR.predict(X_test)


# In[166]:


Prediction 


# In[168]:


plt.scatter(Prediction, y_test)


# In[169]:


## You can it is not in a Straight line. 
## So we deploy Lasso and Ridge 

from sklearn.linear_model import Lasso, Ridge


# In[171]:


LA = Lasso()
RD = Ridge()


# In[172]:


LA.fit(X_train, y_train)


# In[173]:


RD.fit(X_train, y_train)


# In[174]:


LA.intercept_


# In[175]:


LA.coef_


# In[176]:


RD.intercept_


# In[177]:


RD.coef_


# In[178]:


P_LA = LA.predict(X_test)


# In[179]:


P_LA


# In[180]:


P_RD = RD.predict(X_test)


# In[181]:


P_RD


# In[185]:


plt.scatter(y_test, P_LA)


# In[184]:


plt.scatter(y_test, P_RD)


# In[186]:


def rmse(target, prediction):
    return np.sqrt(np.mean(np.square(target,prediction)))


# In[187]:


rmse(y_test, Prediction)


# In[189]:


rmse(y_test,P_RD)


# In[ ]:





# In[ ]:




