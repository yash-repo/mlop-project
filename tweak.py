from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

#read combination number
input_file = open("combNum.txt", "r")
Num = input_file.read()
Num = int(Num)
input_file.close()

#All Combinations
combArray = [[1,1],
             [2,2],
             [3,3]]

#assign values to factors
layers = combArray[Num][0]
epochs = combArray[Num][1]
combNum = Num
def makeModel():
    model = Sequential()
    model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(1,28,28),dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    for i in range(0, layers):
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
