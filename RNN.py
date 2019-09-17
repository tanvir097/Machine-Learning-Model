#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('dataprep.csv')
df.head()


# In[3]:


sns.countplot(df.Label)
plt.xlabel('Label')
plt.title('Number of H and N messages')


# In[4]:


X = df.Statement
Y = df.Label
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)


# In[5]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)


# In[6]:


max_words = 20000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)


# In[7]:


from keras.preprocessing.sequence import pad_sequences
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['Statement'].values)
X = tokenizer.texts_to_sequences(df['Statement'].values)
X = pad_sequences(X)


# In[8]:


from keras import layers
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = layers.GRU(100,return_sequences=True)(layer)
    layer = LSTM(512,return_sequences=True)(layer)
    layer = LSTM(256,return_sequences=True)(layer)
    layer = LSTM(128,return_sequences=True)(layer)
    layer = LSTM(64,return_sequences=True)(layer)
    layer = LSTM(32,return_sequences=False)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('softmax')(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


# In[9]:


model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])


# In[10]:


model.fit(sequences_matrix,Y_train,batch_size=64,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.001)])


# In[11]:


test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


# In[12]:


accr = model.evaluate(test_sequences_matrix,Y_test)


# In[13]:


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# In[ ]:




