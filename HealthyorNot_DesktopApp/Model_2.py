#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import pickle


# In[14]:


dataset = pd.read_csv('heart.csv')


# In[16]:


dataset_X = dataset.iloc[:,[2,3,4,5,7,8,9,11,12]].values
dataset_Y = dataset.iloc[:,13].values


# In[4]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


# In[5]:


dataset_scaled = pd.DataFrame(dataset_scaled)


# In[6]:



X = dataset_scaled
Y = dataset_Y


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset['output'] )


# In[9]:



from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)


# In[10]:


svc.score(X_test, Y_test)


# In[11]:


Y_pred = svc.predict(X_test)


# In[12]:


pickle.dump(svc, open('model2.pkl','wb'))
model = pickle.load(open('model2.pkl','rb'))


# In[ ]:





# In[ ]:




