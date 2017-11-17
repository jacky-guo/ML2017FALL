# -- coding: utf-8 --
from keras.utils import np_utils
from keras.models import Model,Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout,Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os
import numpy as np
import pandas as pd
import sys


def outputfile(prediction_file_path,y_classes):
    with open(prediction_file_path, 'w') as f:
        print('id,label', file=f)
        print('\n'.join(['{},{}'.format(i, p) for (i, p) in enumerate(y_classes)]), file=f)


def main(testing_data_path,prediction_file_path):
	test = pd.read_csv(testing_data_path)
	X_test = []
	for index in range(len(test)):
		X_test.append(np.asarray(list(map(int, test.feature[index].split()))))

	X_test = np.asarray(X_test)
	X_test = X_test.reshape(-1,48,48,1)/255.

	model = load_model("model/best_model.h5")

	output = model.predict(X_test,batch_size=64)
	y_classes = output.argmax(axis=-1)
	outputfile(prediction_file_path,y_classes)


if __name__ == '__main__':
	testing_data_path = sys.argv[1]
	prediction_file_path = sys.argv[2]
	main(testing_data_path,prediction_file_path)
