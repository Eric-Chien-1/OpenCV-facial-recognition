import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

#load the save numpy arrrays in the previous model
data = np.load('data.npy')
target = np.load('target.npy')

model = Sequential()
model.add(Conv2D(200,(3,3), input_shape= data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Creating the first CNN layer

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Creating the 2nd layer of the CNN

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second
model.add(Dense(50, activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The final layer with two outputs for two categories

model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#compiler thhat make sure the output is accurate.
train_data, test_data, train_target, test_target=train_test_split(data, target, test_size=0.1)
#model checkpoint to show the predictions
checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor= 'val_loss', verbose= 0,save_best_only= True,mode='auto')
history = model.fit(train_data, train_target, epochs=100, callbacks=[checkpoint], validation_split=0.2)
