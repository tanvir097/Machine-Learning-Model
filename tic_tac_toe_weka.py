#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics


# In[2]:


df = pd.read_excel(r"tic_tac_toe_weka_dataset (1).xlsx")
df.head()


# In[5]:


X = df.drop('class', axis=1)
y = df['class']


# In[6]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X['top_left_square'] = encoder.fit_transform(X['top_left_square'])
X['top_middle_square'] = encoder.fit_transform(X['top_middle_square'])
X['top_right_square'] = encoder.fit_transform(X['top_right_square'])
X['middle_left_square'] = encoder.fit_transform(X['middle_left_square'])
X['middle_middle_square'] = encoder.fit_transform(X['middle_middle_square'])
X['middle_right_square'] = encoder.fit_transform(X['middle_right_square'])
X['bottom_left_square'] = encoder.fit_transform(X['bottom_left_square'])
X['bottom_middle_square'] = encoder.fit_transform(X['bottom_middle_square'])
X['bottom_right_square'] = encoder.fit_transform(X['bottom_right_square'])


# In[7]:


y = encoder.fit_transform(df['class'])


# In[10]:


print(X.shape)
print(y.shape)
print(X)
print(y)


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[8]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=2.0)
get_ipython().run_line_magic('time', 'nb.fit(X_train, y_train)')


# In[9]:


y_pred_class = nb.predict(X_test)
print("Training Accuracy:",nb.score(X_train,y_train))
print("Testing Accuracy:",nb.score(X_test,y_test))
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred_class))  
print(classification_report(y_test,y_pred_class)) 


# In[10]:


from sklearn import svm
clf = svm.SVC(C=3,kernel='poly')


# In[11]:


clf.fit(X_train, y_train)


# In[12]:


y_pred = clf.predict(X_test)


# In[13]:


print("Training Accuracy:",clf.score(X_train,y_train))
print("Testing Accuracy:",clf.score(X_test,y_test))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)


# In[15]:


y_pred = knn.predict(X_test)
print("Training Accuracy:",knn.score(X_train,y_train))
print("Testing Accuracy:",knn.score(X_test,y_test))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 


# In[16]:


from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[17]:


y_pred = dt.predict(X_test)
print("Training Accuracy:",dt.score(X_train,y_train))
print("Testing Accuracy:",dt.score(X_test,y_test))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 


# In[ ]:




