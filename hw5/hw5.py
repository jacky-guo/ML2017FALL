import numpy as np
import pandas as pd
import keras 
import keras.backend as K
from keras.layers import Input, Flatten, Embedding, Dropout, Concatenate, Dot, Add, Dense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1, l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

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
# def read_data()

def rmse(y_true,y_pred):
    return K.sqrt(K.mean((y_pred - y_true)**2))

def split_data(movie,user,Y,split_ratio=0.1):
    indices = np.arange(movie.shape[0])  
    np.random.shuffle(indices) 
    
    movie_data = movie[indices]
    user_data = user[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * movie_data.shape[0] )
    
    movie_train = movie_data[num_validation_sample:]
    user_train = user_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    movie_val = movie_data[:num_validation_sample]
    user_val = user_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (user_train,movie_train,Y_train),(user_val,movie_val,Y_val)
# get embedding
# user_emb = np.array(model.layers[2].get_weights()).squeeze()
# print('user embedding shape:',user_emb.shape)
ratings = pd.read_csv('data/train.csv')
user = ratings['UserID'].values
movie = ratings['MovieID'].values

Y = np.float32(ratings['Rating'].values)

n_movie = np.max(ratings['MovieID']) + 1
n_users = np.max(ratings['UserID'])  + 1
print(n_movie,n_users)
model = get_model(n_users, n_movie)

(user_train, movie_train,Y_train),(user_val,movie_val,Y_val) = split_data(movie,user,Y)

earlystopping = EarlyStopping(monitor='val_rmse', patience = 5, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='mf_test.h5',
                             verbose=1,
                             save_best_only=True,
                             monitor='val_rmse',
                             save_weights_only=True,
                             mode='min')

model.fit([user_train, movie_train],Y_train,epochs=400, batch_size=1024,
          validation_data=([user_val,movie_val],Y_val),
          callbacks=[earlystopping,checkpoint])


