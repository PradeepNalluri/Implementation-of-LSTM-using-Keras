from __future__ import print_function
from sklearn.preprocessing import OneHotEncoder
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import pandas as pd
import numpy as np
import os
maxlen = 30
labels = 2
input = pd.read_csv("gender_data.csv",header=None)
input.columns = ['name','m_or_f']
input['namelen']= [len(str(i)) for i in input['name']]
input1 = input[(input['namelen'] >= 2) ]
input1.groupby('m_or_f')['name'].count()
names = input['name']
gender = input['m_or_f']
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)
print(vocab)
print("vocab length is ",len_vocab)
print ("length of input is ",len(input1))
char_index = dict((c, i) for i, c in enumerate(vocab))
print(char_index)
msk = np.random.rand(len(input1)) < 0.8
train = input1[msk]
test = input1[~msk]     
def encoding(i):
    tmp = np.zeros(39);
    tmp[i] = 1
    return(tmp)
encoding(3)
train_X = []
train_Y = []
trunc_train_name = [str(i)[0:maxlen] for i in train.name]
for i in trunc_train_name:
    tmp = [encoding(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(encoding(char_index["END"]))
    train_X.append(tmp)
for i in train.m_or_f:
    if i == 'm':
        train_Y.append([1,0])
    else:
        train_Y.append([0,1])
np.asarray(train_X).shape
np.asarray(train_Y).shape
print('Build Network...')
Network = Sequential()
Network.add(LSTM(512, return_sequences=True, input_shape=(maxlen,len_vocab)))
Network.add(Dropout(0.2))
Network.add(LSTM(512, return_sequences=False))
Network.add(Dropout(0.2))
Network.add(Dense(2))
Network.add(Activation('softmax'))
Network.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
test_X = []
test_Y = []
trunc_test_name = [str(i)[0:maxlen] for i in test.name]
for i in trunc_test_name:
    tmp = [encoding(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(encoding(char_index["END"]))
    test_X.append(tmp)
for i in test.m_or_f:
    if i == 'm':
        test_Y.append([1,0])
    else:
        test_Y.append([0,1])
print(np.asarray(test_X).shape)
print(np.asarray(test_Y).shape)
batch_size=1000
#Network.fit(train_X, train_Y,batch_size=batch_size,nb_epoch=10,validation_data=(test_X, test_Y))
#score, acc = Network.evaluate(test_X, test_Y)
#print('Test score:', score)
#print('Test accuracy:', acc)
name=["pradeep","ravi","srinu","udaya","lakshmi","siva","kumari","pamulaiah","vivek","ramarao","krish","mrs. rajini","pretham","sravan","ragini rajaram"]
X=[]
trunc_name = [i[0:maxlen] for i in name]
for i in trunc_name:
    tmp = [encoding(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(encoding(char_index["END"]))
    X.append(tmp)
prediction=Network.predict(np.asarray(X))
Network.save_weights('gender_Network',overwrite=True)