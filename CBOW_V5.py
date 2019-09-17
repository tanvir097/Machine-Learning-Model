#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().run_line_magic('autosave', '30')
# keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 

# bon
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence

# for python 2
import os
import subprocess

# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os
import math
from collections import Counter

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import re
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok


# In[2]:


new_text = open("input/short.txt", encoding="utf8").read()
new_text = new_text.split(".")
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(new_text)
word2id = tokenizer.word_index
#print(new_text)
# build vocabulary of unique words
word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
#wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in new_text]
wids = tokenizer.texts_to_sequences(new_text)
#print(wids)
vocab_size = len(word2id)
embed_size = 100
window_size = 2 # context window size

#print('Vocabulary Size:', vocab_size)
#print('Vocabulary Sample:', list(word2id.items()))


# In[3]:


import numpy as np
def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = int(window_size*2)
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word   = []
            #print(index)
            start = index - window_size
            #print(start)
            end = index + window_size + 1
            
            context_words.append([words[i] 
                                 for i in range(start, end) 
                                 if 0 <= i < sentence_length 
                                 and i != index])
            label_word.append(word)
            #print(label_word)

            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_word, vocab_size)
            #print(y)
            yield (x, y)
            
            
# Test this out for some samples
i = 0
#print("jhdhd")
for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
    #if 0 not in x[0]:
    print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])
    #print(x[0])
        
#print(x[0])
#print(id2word)


# In[4]:



# In[2]:


curr_dir = 'input/'
all_headlines = []
for filename in os.listdir(curr_dir):
    if 'short.txt' in filename:
        article_df = pd.read_csv(curr_dir + filename, encoding='utf-8')
#         article_df = pd.read_csv(filename, encoding='utf-8')
        article_df = article_df.replace('\n', '', regex=True)
        article_df = article_df.replace('\r', '', regex=True)
        #print(article_df)
        all_headlines.extend(list(article_df.headline.values))
        break

all_headlines = [h for h in all_headlines if h != "Unknown"]
len(all_headlines)
# print(all_headlines)


# In[3]:


def clean_text(txt):
    #txt = "".join(v for v in txt if v not in string.punctuation).lower()
    #print("1st ->" + txt)
    txt = txt.encode("utf8").decode("utf8",'ignore')
    #print("2nd ->" + txt)
    return txt 

corpus = [clean_text(x) for x in all_headlines]
corpus


# In[4]:


#corpus = "আমি গতকালকে ভাত খেয়েছিলাম।"
tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        print(token_list)
        for j in range(1, len(token_list)):
            for i in range(j, len(token_list)):
                n_gram_sequence = token_list[j-1:i+1]
                #print(n_gram_sequence)
                input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
# print(len(inp_sequences))
#inp_sequences[5150:]


# In[5]:


tokenizer.word_index.items()


# In[6]:


def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    #print(max_sequence_len)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    #print(input_sequences[0])
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
#print(predictors[0])
#print (max_sequence_len)
#print(label[0])


# In[7]:


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='nadam')
    
    return model

model = create_model(max_sequence_len, total_words)
model.summary()


# In[8]:


model.fit(predictors, label, epochs=100, verbose=5)


# In[9]:


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        #print(seed_text)
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        #print(token_list)
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        #print(token_list)
        predicted = model.predict_classes(token_list, verbose=0)
        #print(predicted)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


# In[ ]:





# In[ ]:





# In[13]:


get = input()
input_2 = get
get = get.split()
inp = get
i = 0
target_list = []
count = 0
if len(get)== 2:
    loop_var1 = 2
elif len(get) == 3:
    loop_var1 = 3
else:
    loop_var1 = 4
#loop_var1 = 2
loop_var2 = 0
while (loop_var1<=len(get)):
    temp1 = get.copy()
    temp2 = temp1[loop_var2:loop_var1]
    
    line_text = ""
    
    for z in range(len(temp2)):
        temp3 = temp2.copy()
        selected_element = temp3[z]
        print(selected_element)
        temp3.remove(selected_element)
        print(temp3)
        for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
            #if 0 not in x[0]:
            new_list=[]
            #set(temp3).intersection(set([id2word[w] for w in x[0]]))
            if set(temp3).intersection([id2word[w] for w in x[0]]):
                new_list.extend(set(temp3).intersection([id2word[w] for w in x[0]]))
            if len(temp3) > 2:
                if len(temp3)==len(new_list) or len(temp3)-1==len(new_list):
                    print('For context :', new_list,'Target (Y):', id2word[np.argwhere(y[0])[0][0]])
                    target_list.append(id2word[np.argwhere(y[0])[0][0]])
            else:
                if len(temp3)==len(new_list):
                    print('For context :', new_list,'Target (Y):', id2word[np.argwhere(y[0])[0][0]])
                    target_list.append(id2word[np.argwhere(y[0])[0][0]])

        # আমার ভাইবোন বা মাও সেটি মনে করতে পারেনি কাজেই সঠিক সময়টি বলা যাচ্ছে না
        print("target list: ", target_list)
#         print("Selected ele: ", selected_element)
        
        if selected_element in set(target_list):
            print("correct")
            line_text = line_text + selected_element + ' '
        else:
#             print("incorrect")
##########################################################################################################
            inp = selected_element
            f = open("input.txt","w",encoding='utf8')
            f.write(inp)
            f.close()

            command = "try.py"
            os.system(command)

            f = open("guru.txt", "r",encoding='utf8')
            out = f.read()
            similer_list = out.split()
            print(similer_list)
            
            flag = 0
            for i in target_list:
                word_1 = i
                if word_1 in similer_list:
                    line_text = line_text + word_1 + ' '
                    print("correct")
                    flag = 1
                    break
            if flag == 0:
                print("incorrect")
            line_text = line_text[:-1]
            print(line_text)
##########################################################################################################
        target_list.clear()
        
    loop_var1 = loop_var1+1
    loop_var2 = loop_var2+1
# আমি তখন ঢাকা বিশ্ববিদ্যালয়ের ছাত্র ।


# In[14]:


# inp = input()
# abc = inp.split()


abc = line_text.split()


length = len(abc)

keyword = ''
#print (generate_text(inp, 3 , model, max_sequence_len))
for i in range (0,length):
    keyword = keyword + abc[i] + ' '
    print ("Keyword -> " + keyword)
    string = generate_text(keyword, 5 , model, max_sequence_len)
    main_string = ''
    for i in string:
        main_string += i
        #print(i, end='')
        if i == '৷':
            break

    print(main_string)
    
# আলেক্সান্ডার বেলায়েভের লেখা উভচর মানব আমার পড়াপ্রথম পূর্ণাঙ্গ সায়েন্স ফিকশন উপন্যাস।
# সেই বইটিআমাকে এতই মুগ্ধ করল যে তার একটি গল্প “ম্যাক্সওয়েলেরসমীকরণ'কে নাট্যরূপ দিয়ে ঢাকা বিশ্ববিদ্যালয়ের টি.এস.সি.-তে নাটকহিসেবে মঞ্চস্থ করেছিলাম
# 'পাশাপাশিবসে এতোক্ষণ শামীমের সাথে কথা বলে এসেছে, শামীম পাস করাডাক্তার, ডাক্তারী না করে বিদেশে ল্যাবরেটরিতে গবেষণা করে দেশে চলেএসেছে, পরিবারের কেউ দেশে থাকে না এ-রকম একটা মানুষ হঠাৎ ট্রেনথেকে একটা ছোট গ্রাম্য স্টেশনে কোনো কারণ ছাড়াই নেমে পড়বে সেটাতার পক্ষে বিশ্বাস করা কঠিন।'


# In[16]:


# First, I removed the split... it is already an array
str1 = input_2
str2 = main_string

#then creating a new variable to store the result after  
#comparing the strings. You note that I added result2 because 
#if string 2 is longer than string 1 then you have extra characters 
#in result 2, if string 1 is  longer then the result you want to take 
#a look at is result 2

result1 = ''
result2 = ''

#handle the case where one string is longer than the other
maxlen=len(str2) if len(str1)<len(str2) else len(str1)

#loop through the characters
#use a slice rather than index in case one string longer than other
for i in range(maxlen):
    letter1=str1[i:i+1]
    letter2 =str2[i:i+1]
    #create string with differences
    if letter1 != letter2:
        result1+=letter1
        result2+=letter2

#print out result
print ("Given String:",result1)
print ("Correction:",result2)
print(input_2)
result2 = result2.split()

strs = ""
cnt = 0
for i in result2:
    if cnt != 0:
        strs = strs + i + " "
    cnt = cnt + 1
strs = strs[:-1]
# print(result2[1])
print ("Sentence: ", input_2 + " " + strs)
