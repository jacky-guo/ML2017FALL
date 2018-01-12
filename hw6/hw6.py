import numpy as np
import pandas as pd
np.random.seed(1337)
from sklearn.cluster import KMeans
from keras.layers import Dense,Input
from keras import Model
from keras.callbacks import EarlyStopping
import sys

if __name__ == "__main__":

	data2 = np.load(sys.argv[1]) 
	test = pd.read_csv(sys.argv[2])
	output_file = sys.argv[3]
	ans = []
	# Autoencoder

	x_train = data2.astype('float32') / 255. -0.5
	x_train.reshape((x_train.shape[0],-1))

	encoding_dim = 32

	input_img = Input(shape=(784,))

	encoded = Dense(256,activation='relu')(input_img)
	encoded = Dense(128,activation='relu')(encoded)
	encoded = Dense(64,activation='relu')(encoded)
	encoder_output = Dense(encoding_dim)(encoded)

	decoded = Dense(64,activation='relu')(encoder_output)
	decoded = Dense(128,activation='relu')(decoded)
	decoded = Dense(256,activation='tanh')(decoded)
	decoder_output = Dense(784)(decoded)

	autoencoder = Model(input_img,decoder_output)
	encoder = Model(input_img,encoder_output)

	autoencoder.compile(optimizer='adam',loss='mse')

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(x_train, x_train, test_size=0.1, random_state=42)

	earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

	autoencoder.fit(X_train,X_train,epochs=100,batch_size=1024,validation_data=(y_train,y_train),callbacks=[earlystop])

	data = encoder.predict(x_train)

	# KMeans
	kmeans = KMeans(n_clusters = 2,random_state=5566).fit(data)
	label = kmeans.labels_
	for i in range(len(test)):
		if label[test.iloc[i][1]] == label[test.iloc[i][2]]:
			ans.append(1)
		else:
			ans.append(0)

	with open(output_file, 'w') as f:
		print('ID,Ans',file=f)
		print('\n'.join(['{},{}'.format(i, p) for (i, p) in enumerate(ans)]), file=f)

