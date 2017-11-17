from keras.utils import np_utils
from keras.models import Model,Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout,Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
import os
import numpy as np
from math import log, floor
import pandas as pd
import sys



def load_data(training_data_path):
	df = pd.read_csv(training_data_path)
	X_train = []
	for index in range(len(df)):
		X_train.append(np.asarray(list(map(int, df.feature[index].split()))))
	y_train = df.label

	return X_train,y_train

def preprocess(X_train,y_train):

	X_train = np.asarray(X_train)
	X_train = X_train.reshape(-1,48,48,1)/255.
	# One-Hot encoding
	y_train = np_utils.to_categorical(y_train, num_classes=7)

	# Data Augmentation
	X_train = np.concatenate((X_train, X_train), axis=0)
	y_train = np.concatenate((y_train, y_train), axis=0)

	test_size = 0

	train_pixels,valid_pixels,train_labels,valid_labels = train_test_split(X_train,y_train,test_size=test_size,random_state=42)
	# save_pickle(train_pixels,train_labels,valid_pixels,valid_labels)
	return train_pixels,valid_pixels,train_labels,valid_labels
def build_model():

	model = Sequential()
	# Conv layer 1 output shape (32, 48, 48)
	model.add(Conv2D(batch_input_shape=(None, 48,48,1),filters=32,kernel_size=3,padding='same',data_format='channels_last',activation='relu'))
	# Conv layer 2 output shape (32, 48, 48)
	model.add(Conv2D(32,3,strides=1,padding='same',data_format='channels_last',activation='relu'))
	model.add(BatchNormalization())
	# Pooling layer 1 (max pooling) output shape (32, 24, 24)
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.1))

	# Conv layer 3 output shape (64, 24, 24)
	model.add(Conv2D(64, 3, strides=1, padding='same', data_format='channels_last',activation='relu'))
	# Conv layer 4 output shape (64, 24, 24)
	model.add(Conv2D(64, 3, strides=1, padding='same', data_format='channels_last',activation='relu'))
	model.add(BatchNormalization())
	# Pooling layer 2 (max pooling) output shape (64, 12, 12)
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.2))

	# Conv layer 5 output shape (128, 12, 12)
	model.add(Conv2D(128, 3, strides=1, padding='same', data_format='channels_last',activation='relu'))
	# Conv layer 6 output shape (128, 12, 12)
	model.add(Conv2D(128, 3, strides=1, padding='same', data_format='channels_last',activation='relu'))
	model.add(BatchNormalization())
	# Pooling layer 2 (max pooling) output shape (128, 6, 6)
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.3))

	# Conv layer 7 output shape (256, 6, 6)
	model.add(Conv2D(256, 3, strides=1, padding='same', data_format='channels_last',activation='relu'))
	# Conv layer 8 output shape (256, 6, 6)
	model.add(Conv2D(256, 3, strides=1, padding='same', data_format='channels_last',activation='relu'))
	model.add(BatchNormalization())
	# Pooling layer 3 (max pooling) output shape (256, 6, 6)
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.4))

	# Fully connected layer 1 input shape (256 * 3 * 3) = ()
	model.add(Flatten())

	model.add(Dense(1024))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(7))
	model.add(Activation('softmax'))


	return model

def main(training_data_path):

	# parameter
	save_every = 20
	batch_size = 128
	num_epoch = 60 

	X_train,y_train = load_data(training_data_path)
	print('Load Data Successful!')

	train_pixels,valid_pixels,train_labels,valid_labels = preprocess(X_train,y_train)
	print('Data PreProcess Successful!')

	# Data Augmentation
	datagen = ImageDataGenerator(
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True)

	datagen.fit(train_pixels)

	model = build_model()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	# callback 
	earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5,verbose=1, mode='auto')
	filepath = "model/model-{epoch:02d}-{acc:.2f}.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1,save_best_only=False,mode='auto', period=save_every)

	# 
	model_info = model.fit_generator(datagen.flow(train_pixels, train_labels, batch_size=batch_size), 
		steps_per_epoch=len(train_pixels)//batch_size,
		epochs=num_epoch,
		# validation_data=(valid_pixels, valid_labels),
		# callbacks=[checkpoint]
		)
	print('Model Train Successful!')
	# Evaluate
	score = model.evaluate(train_pixels,train_labels,batch_size=128)
	print ('\nTotal loss on Testing Set :', score[0])
	print ('Train Acc:', score[1])

	model.save('best_model.h5')
	print('Model Save!')

if __name__ == '__main__':
	training_data_path = sys.argv[1]
	main(training_data_path)