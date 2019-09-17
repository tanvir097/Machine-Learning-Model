#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


df = pd.read_csv('dataprep.csv')


# In[3]:


df.head()


# In[4]:


df.Label.value_counts()


# In[5]:


df['Label'] = df.Label.map({'H':0, 'N':1})


# In[6]:


df.head(25)


# In[7]:


X = df.Statement
y = df.Label
print(X.shape)
print(y.shape)


# In[8]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[9]:


vect = CountVectorizer()


# In[10]:


vect.fit(X_train)
X_train_dtm = vect.transform(X_train)


# In[11]:


X_train_dtm = vect.fit_transform(X_train)


# In[12]:


X_train_dtm


# In[13]:


print(X_train_dtm)


# In[14]:


X_test_dtm = vect.transform(X_test)
X_test_dtm


# In[15]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=2.0)


# In[16]:


get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')


# In[17]:


y_pred_class = nb.predict(X_test_dtm)


# In[18]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[19]:


metrics.confusion_matrix(y_test, y_pred_class)


# In[20]:


# print message text for the false positives (ham incorrectly classified as spam)
X_test[y_test < y_pred_class]


# In[21]:


# print message text for the false negatives (spam incorrectly classified as ham)
X_test[y_test > y_pred_class]


# In[22]:


# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# In[23]:


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# In[24]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=4,intercept_scaling=2)


# In[25]:


get_ipython().run_line_magic('time', 'logreg.fit(X_train_dtm, y_train)')


# In[26]:


y_pred_class = logreg.predict(X_test_dtm)


# In[27]:


y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# In[28]:


print("Accuracy:",metrics.recall_score(y_test, y_pred_class))


# In[29]:


metrics.accuracy_score(y_test, y_pred_class)


# In[30]:


metrics.roc_auc_score(y_test, y_pred_prob)


# In[31]:


from sklearn import svm
clf = svm.SVC(C=4,kernel='rbf',probability=True)


# In[32]:


clf.fit(X_train_dtm, y_train)


# In[33]:


y_pred = clf.predict(X_test_dtm)


# In[34]:


from sklearn import metrics


# In[35]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[36]:


print("Precision:",metrics.precision_score(y_test, y_pred))


# In[37]:


print("Recall:",metrics.recall_score(y_test, y_pred))


# In[38]:


print("F1 Score:",metrics.f1_score(y_test, y_pred))


# In[39]:


metrics.confusion_matrix(y_test, y_pred_class)


# In[ ]:


from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
estimators=[('lr', logreg), ('nb', nb), ('svc', clf)],
voting='soft'
)
voting_clf.fit(X_train_dtm, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
for clf in (logreg, nb, clf, voting_clf):
    clf.fit(X_train_dtm, y_train)
    y_pred = clf.predict(X_test_dtm)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(
LogisticRegression(), n_estimators=500,
max_samples=100, bootstrap=True, n_jobs=-1
)
bag_clf.fit(X_train_dtm, y_train)
y_pred = bag_clf.predict(X_test_dtm)


# In[ ]:


bag_clf = BaggingClassifier(
    LogisticRegression(), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf.fit(X_train_dtm, y_train)
bag_clf.oob_score_


# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


clf = MLPClassifier(hidden_layer_sizes=(100,100,100),activation='logistic', max_iter=500, alpha=0.001,
                     solver='adam', verbose=10,  random_state=21, tol=0.000000001, 
                    learning_rate_init=0.001)


# In[ ]:


clf.fit(X_train_dtm, y_train)
y_pred = clf.predict(X_test_dtm)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()


# In[ ]:




