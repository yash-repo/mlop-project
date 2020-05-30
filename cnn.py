#!/usr/bin/env python
# coding: utf-8

# In[2]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from tweak import *

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test  = X_test.reshape(X_test.shape[0],1,28,28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = makeModel()

model.summary()

# In[3]:

model.fit(X_train, Y_train,batch_size=32, nb_epoch=epochs, verbose=1)

#model.save('my_model.h5')

# In[5]:


x = model.evaluate(X_test, Y_test)
acc = x[1]
#storing accuracy
text_file = open("acc.txt", "w")
n = round(acc,2)
text_file.write(str(n))
text_file.close()

#create log
f = open("log.txt","a+")
f.write("Total Accuracy for {0} is {1}\n".format(combNum, acc))
f.close()

#increment combination number
combNum = combNum + 1
text_file = open("combNum.txt", "w")
text_file.write(str(combNum))
text_file.close()