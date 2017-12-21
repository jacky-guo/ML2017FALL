import numpy as np
import pandas as pd
import keras 
import keras.backend as K
from keras.layers import Input, Flatten, Embedding, Dropout, Concatenate, Dot, Add, Dense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1, l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import sys
from keras.models import load_model
import keras.backend as K

def get_model(n_users,n_items,latent_dim=15):
	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
	user_vec = Dropout(0.1)(user_vec)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
	item_vec = Dropout(0.1)(item_vec)
	item_vec = Flatten()(item_vec)
	user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
	item_bias = Flatten()(item_bias)
	r_hat = Dot(axes=1)([user_vec,item_vec])
	r_hat = Add()([r_hat,user_bias,item_bias])
	model = keras.models.Model([user_input,item_input],r_hat)
	model.compile(loss='mse',optimizer='adamax',metrics=[rmse])
	model.summary()
	return model

def rmse(y_true,y_pred):
    return K.sqrt(K.mean((y_pred - y_true)**2))

def main(test_path,output_path):

	test = pd.read_csv(test_path)

	user = test['UserID'].values
	movie = test['MovieID'].values

	# model = load_model('mf_best.h5', custom_objects={'rmse':rmse})
	model = get_model(6041,3953)
	model.load_weights('mf_best.h5')
	result = model.predict([user, movie], batch_size=128, verbose=1)
	result = result.reshape(len(result))
	result = np.clip(result,1.0,5.0)

	of = open(output_path,'w')
	out_txt = 'TestDataID,Rating\n'
	for i in range(len(result)):
	    out_txt += str(i+1) +','+ str(result[i]) + '\n'

	of.write(out_txt)
	of.close()

if __name__ == '__main__':
	testing_data_path = sys.argv[1]
	prediction_file_path = sys.argv[2]
	main(testing_data_path,prediction_file_path)