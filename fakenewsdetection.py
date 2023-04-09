import pandas as pd
df=pd.read_csv("C:/Users/purus/OneDrive/desktop/WELFake_Dataset.csv")
# print(df)
print(df.head())
print(df.shape)
print(df.isnull().sum())
df=df.dropna()
print(df.head())
x=df.drop('label',axis=1)
y=df['label']
print(x.shape)
print(y.shape)


import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot

voc_size=5000
messages=x.copy()
# messages['title'][1]
messages.reset_index(inplace=True)
#print(messages)

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word for word in review if not word in stopwords('english'))]
    review=' '.join(review)
    corpus.append(review)

# print(corpus)

onehot_repr=[one_hot(words,voc_size) for words in corpus]
print(onehot_repr)


sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=sent_length)
print(embedded_docs)


embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

print(len(embedded_docs),y.shape)

import numpy as np
x_final=np.array(embedded_docs)
y_final=np.array(y)

print(x_final.shape,y_final.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size=0.33,random_state=42)
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)


from tensorflow.keras.layers import Dropout
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
y_pred=model.predict(x_test)
y_pred=np.where(y_pred>0.6,1.0)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


