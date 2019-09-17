#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D,Bidirectional
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("dataprep.csv")  


# In[27]:


df.Label.value_counts()


# In[3]:


df['Label'] = df.Label.map({'H':0, 'N':1})


# In[8]:


def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        #review = re.sub(r'^@', '', review, flags=re.MULTILINE)
        #review=re.sub(r'[^a-zA-Z]','',review)
        #review = re.sub(r'^https?:\/\/.*[\r\n]*', '', review, flags=re.MULTILINE)
        review = re.sub(r'RT', '', review, flags=re.MULTILINE)
        #' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)','',review).split())
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        #review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review = re.sub(r"(?:\@|https?\://)\S+", "", review)
        #review=''.join(review)
        review_corpus.append(review)
    return review_corpus


# In[9]:


df['clean_review']=clean_review(df.Statement.values)
df.head()


# In[10]:


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['clean_review'].values)
X = tokenizer.texts_to_sequences(df['clean_review'].values)
X = pad_sequences(X)


# In[18]:


embed_dim = 128
lstm_out = 196

from keras.layers import Bidirectional,GRU

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(64,return_sequences=True))
model.add(Bidirectional(GRU(lstm_out, return_sequences=True)))
#model.add(Bidirectional(LSTM(lstm_out, return_sequences=True)))
model.add(Bidirectional(GRU(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=False)))
#model.add(Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=False)))

model.add(Dense(2,activation='sigmoid',kernel_initializer='TruncatedNormal'))
model.compile(loss = 'categorical_crossentropy', optimizer='Adam',metrics = ['accuracy'])
print(model.summary())


# In[19]:


Y = pd.get_dummies(df['Label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 2)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[20]:


X_val = X_train[:4072]
Y_val = Y_train[:4072]


# In[21]:


partial_X_train = X_train[4072:]
partial_Y_train = Y_train[4072:]


# In[25]:


batch_size = 64
history = model.fit(X_train, 
                    Y_train, 
                    epochs = 50, 
                    batch_size=batch_size, 
                    validation_data=(X_val, Y_val))


# In[26]:


loss, accuracy = model.evaluate(X_train,Y_train , verbose=False)
print("Training Accuracy: {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test,Y_test, verbose=False)
print("Testing Accuracy:  {:.2f}".format(accuracy))


# In[ ]:




