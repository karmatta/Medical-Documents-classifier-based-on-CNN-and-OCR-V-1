
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import numpy as np


# In[2]:


df_master=pd.read_csv('/home/affine/Downloads/Deep_Learning/demo/demo/FSL/Document Classification/df_master.csv')


# In[3]:


df_master.head()


# In[21]:



import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, concatenate
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ModelCheckpoint


# In[9]:


EMBEDDING_FILE='/home/affine/Downloads/Deep_Learning/demo/demo/Toxic_Comment/glove.6B/glove.6B.300d.txt'

embed_size = 300 # how big is each word vector
max_features = 814 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use
list_classes = ["HCFA", "HCFASB","MEDICARE","UB","USB"]

#Read in our data and replace missing values
train_X,test_X,train_Y,test_Y=train_test_split(df_master[['File_name','text']],df_master[list_classes], test_size=0.33, random_state=42)
# print(train_X)
list_sentences_train = train_X['text'].fillna("_na_").values

list_sentences_test = test_X['text'].fillna("_na_").values



# In[10]:


y=train_Y.values
# y


# In[11]:


#Preprocessing
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[12]:


#Read the glove word vectors (space delimited strings) into a dictionary from word->vector.
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


# In[13]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


# In[14]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[15]:


print(embedding_matrix.shape)


# In[22]:


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool]) 

# x = GlobalMaxPool1D()(x)
x = Dense(300, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(5, activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
adam=Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam , metrics=['accuracy'])


# In[23]:


model.fit(X_t, y, batch_size=1, epochs=10, validation_split=0.25);


# In[40]:


# from keras.models import load_model

# model.save('LSTM_text_v4.h5')
# # model=load_model('LSTM_text_v1.h5')
# model.save_weights('weights_LSTM_text_v4.hdf5')
# # weights=model.load_weights('weights_LSTM_text_v1.hdf5')


# In[24]:


y_test = model.predict([X_te], batch_size=1, verbose=1)


# In[25]:


print("predicted:")
print(y_test)


# In[26]:


print("Actual")
print(test_Y)


# In[27]:


y_pred=np.array(y_test)
y_pred1=y_pred.argmax(1)
print('test_pred  ',y_pred1)
y_actual=test_Y.values
y_actual1=y_actual.argmax(1)
print('test_actual',y_actual1)


# In[28]:


from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_actual1,y_pred1)
print(conf_mat)


# In[29]:


from sklearn.metrics import classification_report
target_names=["HCFA", "HCFASB","MEDICARE","UB","USB"]
print(classification_report(y_actual1,y_pred1,target_names=target_names))


# In[30]:


df1=pd.DataFrame(y_test)
df1.columns=list_classes
# print(df1)
df2=pd.DataFrame((test_Y))
df2.columns=list_classes
# print(df2)
df3=test_X
# print(df3)
df4=pd.concat([df3,df2], axis=1)
df4=pd.DataFrame(np.array(df4))
# print(df4)
df=pd.concat([df4,df1], axis=1)


# In[31]:


df.columns=['File_name','text',"HCFA", "HCFASB","MEDICARE","UB","USB","HCFA_pred", "HCFASB_pred","MEDICARE_pred","UB_pred","USB_pred"]
df


# In[94]:


df.to_csv('test_FSL.csv',index=False)


# In[32]:


#Train Prediction
y_train1 = model.predict([X_t], batch_size=1, verbose=1)


# In[33]:


y_pred_train=np.array(y_train1)
y_pred_train1=y_pred_train.argmax(1)
print('train_pred  ',y_pred_train1)
y_actual_train=train_Y.values
y_actual_train1=y_actual_train.argmax(1)
print('train_actual',y_actual_train1)


# In[34]:


from sklearn.metrics import confusion_matrix
conf_mat_train=confusion_matrix(y_actual_train1,y_pred_train1)
print(conf_mat_train)


# In[35]:


from sklearn.metrics import classification_report
target_names=["HCFA", "HCFASB","MEDICARE","UB","USB"]
print(classification_report(y_actual_train1,y_pred_train1,target_names=target_names))


# In[36]:


df11=pd.DataFrame(y_train1)
df11.columns=list_classes
# print(df1)
df21=pd.DataFrame((train_Y))
df21.columns=list_classes
# print(df2)
df31=train_X
# print(df3)
df41=pd.concat([df31,df21], axis=1)
df41=pd.DataFrame(np.array(df41))
# print(df4)
df_1=pd.concat([df41,df11], axis=1)


# In[37]:


df_1.columns=['File_name','text',"HCFA", "HCFASB","MEDICARE","UB","USB","HCFA_pred", "HCFASB_pred","MEDICARE_pred","UB_pred","USB_pred"]
df_1


# In[93]:


df_1.to_csv('train_FSL.csv',index=False)

