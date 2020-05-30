from keras.datasets import mnist
dataset = mnist.load_data('mymnist.db')
train , test = dataset
X_train , y_train = train
X_train.shape
X_test , y_test = test
X_test.shape
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_train_cat

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.summary()
print("##########################################################")
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
print("##########################################################")
print("ADDED 3 LAYERS of DENSE relu")
print("##########################################################")
model.summary()
model.add(Dense(units=10, activation='softmax'))
print("############# ADDED softmax function layer ###########")
model.summary()
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy',metrics=['accuracy'])
h = model.fit(X_train, y_train_cat, epochs=2)

