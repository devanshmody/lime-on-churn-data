
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('churn.csv', na_values=' ')


# In[3]:


df=df.drop('customerID',axis=1)


# In[6]:


df[:2]


# In[7]:


df.isnull().sum()


# In[8]:


df=df.dropna()   # dropping NA rows


# In[9]:


# Converting string labels to integeral labels
df["gender"][df["gender"]=='Male']=0
df["gender"][df["gender"]=='Female']=1
df["Partner"][df["Partner"]=='Yes']= 1
df["Partner"][df["Partner"]=='No']= 0
df["Dependents"][df["Dependents"]=='Yes']=1
df["Dependents"][df["Dependents"]=='No']=0
df["PhoneService"][df["PhoneService"]=='Yes']=1
df["PhoneService"][df["PhoneService"]=='No']=0
df["PaperlessBilling"][df["PaperlessBilling"]=='Yes']=1
df["PaperlessBilling"][df["PaperlessBilling"]=='No']=0
df["Churn"][df["Churn"]=='Yes']=1
df["Churn"][df["Churn"]=='No']=0


# In[10]:


# Convert Object data types to integers
df=df.apply(pd.to_numeric, errors='ignore')


# In[11]:


df.dtypes


# In[13]:


#One-Hot-Encoding of Nominal Categorical Features 
df_encode=pd.get_dummies(df, columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod'])
df_encode.head(2)


# In[14]:


df_encode.dtypes


# In[15]:


df_temp=df_encode.drop('Churn',axis=1)
df_temp.head()


# In[16]:


from sklearn.cross_validation import train_test_split
X = df_temp.values #Feature Matrix
y=df['Churn'].values #Target Matrix
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0) #Splitting X,y into Train and Test sections


# In[66]:


df_temp.head()


# In[17]:


y


# In[18]:


X


# In[22]:


''''from skater import Interpretation
interpreter = Interpretation(X, feature_names=features)'''


# In[23]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier


# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting. 

# In[41]:


''''from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X, y)'''


# In[43]:


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation


# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting. 

# In[46]:


model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
accuracy_score(y_test, model_rf.predict(X_test))


# In[49]:


model_rf.fit(X, y)


# In[50]:


# LIME SECTION
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from __future__ import print_function


# In[51]:


predict_fn_rf = lambda x: model_rf.predict_proba(x).astype(float)


# In[65]:


#np.random.seed(1)
#i = 56
exp = explainer.explain_instance(X_test[700], predict_fn_rf, num_features=6)
exp.show_in_notebook(show_all=True)


# In[ ]:




