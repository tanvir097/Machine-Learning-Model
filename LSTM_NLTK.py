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
df['Label'] = df.Label.map({'H':0, 'N':1})
df.head()


# In[3]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt
df['tidy_tweet'] = np.vectorize(remove_pattern)(df['Statement'], "@[\w]*")

# remove special characters, numbers, punctuations
df['tidy_tweet'] = df['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

df['tidy_tweet'] = df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

df.head(25)


# In[4]:


tokenized_tweet = df['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[5]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()


# In[6]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['tidy_tweet'] = tokenized_tweet


# In[7]:


df.head()


# In[8]:


all_words = ' '.join([text for text in df['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[9]:


max_fatures = 500
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['tidy_tweet'].values)
X = tokenizer.texts_to_sequences(df['tidy_tweet'].values)
X = pad_sequences(X)


# In[10]:


Y = pd.get_dummies(df['Label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[11]:


X_val = X_train[:4072]
Y_val = Y_train[:4072]


# In[12]:


partial_X_train = X_train[4072:]
partial_Y_train = Y_train[4072:]


# In[31]:


embed_dim = 64
lstm_out = 64

from keras.layers import Bidirectional,GRU, Activation

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(64,return_sequences=True))
#model.add(Bidirectional(GRU(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=True)))
#model.add(Bidirectional(GRU(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=False)))
model.add(Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=False)))

model.add(Dense(2, activation='softsign',kernel_initializer='he_uniform'))
model.compile(loss = 'mean_squared_error', optimizer='Adagrad',metrics = ['accuracy'])
print(model.summary())


# In[32]:


batch_size = 128
history = model.fit(partial_X_train, 
                    partial_Y_train, 
                    epochs = 20, 
                    batch_size=batch_size, 
                    validation_data=(X_val, Y_val))


# In[33]:


loss, accuracy = model.evaluate(X_train,Y_train , verbose=False)
print("Training Accuracy: {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test,Y_test, verbose=False)
print("Testing Accuracy:  {:.2f}".format(accuracy))


# In[ ]:





# In[ ]:





# In[ ]:




