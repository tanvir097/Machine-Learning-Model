#!/usr/bin/env python
# coding: utf-8

# In[62]:


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


# In[198]:


df = pd.read_csv("dataprep.csv")  


# In[199]:


df['Label'] = df.Label.map({'H':0, 'N':1})


# In[200]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt


# In[202]:


df['tidy_tweet'] = np.vectorize(remove_pattern)(df['Statement'], "@[\w]*")


# In[204]:


# remove special characters, numbers, punctuations
df['tidy_tweet'] = df['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")


# In[205]:


df['tidy_tweet'] = df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# In[207]:


df.head(25)


# In[208]:


tokenized_tweet = df['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[209]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()


# In[210]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['tidy_tweet'] = tokenized_tweet


# In[212]:


df.head()


# In[217]:


all_words = ' '.join([text for text in df['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[196]:


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
        review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        #review=''.join(review)
        review_corpus.append(review)
    return review_corpus


# In[197]:


df['clean_review']=clean_review(df.Statement.values)
df.head(25)


# In[213]:


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['tidy_tweet'].values)
X = tokenizer.texts_to_sequences(df['tidy_tweet'].values)
X = pad_sequences(X)


# In[216]:


embed_dim = 128
lstm_out = 196

from keras.layers import Bidirectional,GRU

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
#model.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
#model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(Bidirectional(GRU(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(GRU(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=False)))

model.add(Dense(2,activation='sigmoid',kernel_initializer='TruncatedNormal'))
model.compile(loss = 'categorical_crossentropy', optimizer='Adam',metrics = ['accuracy'])
print(model.summary())


# In[ ]:


Y = pd.get_dummies(df['Label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


X_val = X_train[:93]
Y_val = Y_train[:93]


# In[ ]:


partial_X_train = X_train[93:]
partial_Y_train = Y_train[93:]


# In[ ]:


batch_size = 64
history = model.fit(X_train, 
                    Y_train, 
                    epochs = 40, 
                    batch_size=batch_size, 
                    validation_data=(X_val, Y_val))


# In[ ]:


loss, accuracy = model.evaluate(X_train,Y_train , verbose=False)
print("Training Accuracy: {:.2f}".format(accuracy))*
loss, accuracy = model.evaluate(X_test,Y_test, verbose=False)
print("Testing Accuracy:  {:.2f}".format(accuracy))


# In[ ]:





# In[ ]:




