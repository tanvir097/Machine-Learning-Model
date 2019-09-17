#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


data = pd.read_csv("DR_Data_Set.csv")
data.shape


# In[37]:


X = data.drop('label', axis=1)
y = data['label']


# In[38]:


X.shape


# In[39]:


y.shape


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[42]:


from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# In[43]:


scaler = MinMaxScaler()


# In[44]:


scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)


# # Abstraction
# 
# ### Estimator API

# In[11]:


from tensorflow import estimator 


# In[12]:


X_train.shape


# In[13]:


y_train.shape


# In[14]:


feat_cols = [tf.feature_column.numeric_column("x", shape=[66])]


# In[15]:


deep_model = estimator.DNNClassifier(hidden_units=[66,66,66],
                            feature_columns=feat_cols,
                            n_classes=6,
                            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01))


# In[16]:


input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train},y=y_train,shuffle=True,batch_size=10,num_epochs=5)


# In[17]:


deep_model.train(input_fn=input_fn,steps=100)


# In[18]:


input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test},shuffle=False)


# In[19]:


preds = list(deep_model.predict(input_fn=input_fn_eval))


# In[20]:


predictions = [p['class_ids'][0] for p in preds]


# In[21]:


from sklearn.metrics import confusion_matrix,classification_report


# In[22]:


print(classification_report(y_test,predictions))


# # TensorFlow Keras
# 
# ### Create the Model

# In[45]:


from tensorflow.contrib.keras import models
dnn_keras_model = models.Sequential()


# In[46]:


from tensorflow.contrib.keras import layers
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D


# In[47]:


dnn_keras_model.add(layers.Dense(units=66,input_dim=66,activation='relu'))


# In[48]:


dnn_keras_model.add(layers.Dense(units=66,activation='relu'))
dnn_keras_model.add(layers.Dense(units=66,activation='relu'))
dnn_keras_model.add(layers.Dense(units=66,activation='relu'))
dnn_keras_model.add(layers.Dense(units=66,activation='relu'))
dnn_keras_model.add(layers.Dense(units=6,activation='softmax'))


# In[49]:


from tensorflow.contrib.keras import losses,optimizers,metrics
losses.sparse_categorical_crossentropy


# In[50]:


dnn_keras_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[54]:


dnn_keras_model.fit(scaled_x_train,y_train,epochs=100)


# In[55]:


predictions = dnn_keras_model.predict_classes(scaled_x_test)


# In[56]:


from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(predictions,y_test))
print(confusion_matrix(predictions,y_test))


# In[ ]:




