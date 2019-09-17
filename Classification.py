#!/usr/bin/env python
# coding: utf-8

# In[57]:


import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.la
from keras.preprocessing.text import Tokenizeryers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv("dataprep.csv")  


# In[ ]:


df.head()


# In[ ]:


df['Label'] = df.Label.map({'H':0, 'N':1})


# In[ ]:


df.head()


# In[ ]:


print(len(df['Label']))


# In[ ]:


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['Statement'].values)
X = tokenizer.texts_to_sequences(df['Statement'].values)
X = pad_sequences(X)


# In[ ]:


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(Dropout(0.5))
model.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(2,activation='sigmoid',kernel_initializer='glorot_normal'))
model.compile(loss = 'categorical_crossentropy', optimizer='Nadam',metrics = ['accuracy'])
print(model.summary())


# In[ ]:


Y = pd.get_dummies(df['Label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


X_val = X_train[:4072]
Y_val = Y_train[:4072]


# In[ ]:


partial_X_train = X_train[4072:]
partial_Y_train = Y_train[4072:]


# In[32]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X_val)
rescaledX = scaler.fit_transform(partial_X_train)


# In[33]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X_val)
rescaledX = scaler.transform(partial_X_train)


# In[ ]:


batch_size = 64
history = model.fit(X_train, 
                    Y_train, 
                    epochs = 20, 
                    batch_size=batch_size, 
                    validation_data=(X_test, Y_test))


# In[35]:


loss, accuracy = model.evaluate(X_val,Y_val , verbose=False)
print("Training Accuracy: {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(partial_X_train, partial_Y_train, verbose=False)
print("Testing Accuracy:  {:.2f}".format(accuracy))


# In[38]:


history = model.fit(X_test, 
                    Y_test, 
                    epochs = 20, 
                    batch_size=batch_size, 
                    validation_data=(X_val, Y_val))


# In[41]:


history = model.fit(X_train,Y_train, epochs=10, validation_split=0.2, shuffle=True)


# In[43]:


model.test_on_batch(X_test, Y_test)
model.metrics_names


# In[44]:


import matplotlib.pyplot as plt
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

