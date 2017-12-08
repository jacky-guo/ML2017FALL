import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import _pickle as pk
import math
import gensim


class DataManager:
    def __init__(self):
        self.data = {}
    # Read data from data_path
    #  name       : string, name of data
    #  with_label : bool, read data with label or without label
    def add_data(self,name, data_path, with_label=True):
        print ('read data from %s...'%data_path)
        X, Y = [], []
        with open(data_path,'r',encoding="utf-8") as f:
            index = -1
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                else:
                    if name == 'test_data':
                        if index >= 0:
                            if index == 0:
                                X.append(line[int(math.log10(index+1))+2:])
                            else:
                                X.append(line[int(math.log10(index))+2:])
                        index += 1
                    else:
                        X.append(line)
        if with_label:
            self.data[name] = [X,Y]
        else:
            self.data[name] = [X]
    # Build dictionary
    #  vocab_size : maximum number of word in yout dictionary
    def tokenize(self, vocab_size):
        print ('create new tokenizer')
        self.tokenizer = Tokenizer(num_words=vocab_size,filters='\t\n')
        for key in self.data:
            print ('tokenizing %s'%key)
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)
        
    # Save tokenizer to specified path
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump(self.tokenizer, open(path, 'wb'))
            
    # Load tokenizer from specified path
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pk.load(open(path, 'rb'))
    
    # Convert words in data to index and pad to equal size
    #  maxlen : max length after padding
    def to_sequence(self, maxlen):
        self.maxlen = maxlen
        for key in self.data:
            print ('Converting %s to sequences'%key)
            tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen))
    
    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to tfidf'%key)
            self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')
    
    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))

    #Functions for semi-supervised learning
    def get_semi_data(self,name,label,threshold,loss_function) : 
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)

    # get data by name
    def get_data(self,name):
        return self.data[name]

    # split data to two part by a specified ratio
    #  name  : string, same as add_data
    #  ratio : float, ratio to split
    def split_data(self, name, ratio):
        data = self.data[name]
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])
    def get_sentencesList(self):
        sentencesList = []
        for key in self.data:
            for sentence in self.data[key][0]:
                sentencesList.append(text_to_word_sequence(sentence,filters='\t\n'))
        return sentencesList

    def get_embedding_matrix(self,sentencesList,embedding_dim,nb_words):
        word_index = self.tokenizer.word_index
        model = gensim.models.Word2Vec(sentencesList,size=embedding_dim,min_count=5,workers=4)
        embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
        for word, i in word_index.items():
            if i > nb_words:
                continue
            if word in model:
                embedding_vector = model[word]
            # if embedding_vector is not None:
            #     # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix 

    