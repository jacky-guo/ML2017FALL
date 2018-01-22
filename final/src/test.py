import pandas as pd
import numpy as np
import jieba
import h5py
from gensim.models import word2vec
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, Input, Flatten, Dot, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys



def text2encode_voc(x, voc):
    max_length = 0
    new_x = []
    keys = voc.keys()
    for text in x:
        max_length = max_length if max_length > len(text) else len(text)
        indexed = []
        for word in text:
            if(word not in keys):
                indexed.append(0)
            else:
                indexed.append(voc[word] + 1)
        new_x.append(indexed)
    return new_x

def main(testing_data_path,prediction_file_path):
    max_sentence_length = 27
    jieba.set_dictionary('data/jieba_dict/dict.txt.big.txt')
    test_X1 = []
    test_X2 = []
    with open(testing_data_path, encoding='utf-8') as f:
        f.readline()
        maxcount = 0
        for line in f:
            count = 0
            line_arr = ((line.replace('A:', '')).replace('B:', '')).split(',')
            tab_seperate = line_arr[1].split('\t')
            space_seperate = []
            for sen in tab_seperate:
                space_seperate = space_seperate+sen.split(' ')
            if(len(space_seperate)>1):
                q_words = jieba.cut(space_seperate[-2]+space_seperate[-1], cut_all=False)
            else:
                q_words = jieba.cut(space_seperate[-1], cut_all=False)
            question = []
            for word in q_words:
                if(word != '\n' and word != ' ' and word != '\t'):
                    question.append(word)
                    count += 1
            if(count > maxcount):
                print(q_words)
                maxcount = count
            test_X1.append(question)
            test_X1.append(question)
            test_X1.append(question)
            test_X1.append(question)
            test_X1.append(question)
            test_X1.append(question)

            ans_arr = line_arr[2].split('\t')
            for ans in ans_arr:
                ans_tmp = []
                a_words = jieba.cut(ans, cut_all=False)
                for word in a_words:
                    if(word != '\n' and word != ' ' and word != '\t'):
                        ans_tmp.append(word)
                test_X2.append(ans_tmp)


    model = word2vec.Word2Vec.load('model/word_model.w2v')
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    test_X1 = text2encode_voc(test_X1, vocab)
    test_X2 = text2encode_voc(test_X2, vocab)
    test_X1 = pad_sequences(sequences=test_X1, maxlen=max_sentence_length, padding='post')
    test_X2 = pad_sequences(sequences=test_X2, maxlen=max_sentence_length, padding='post')


    DNN_model = load_model('model.h5')

    test_Y = DNN_model.predict([test_X1, test_X2], batch_size=128, verbose=1)

    with open(prediction_file_path,'w',encoding='utf-8') as f:
        f.write('id,ans\n')
        for i in range(5060):
            ans = np.argmax(test_Y[i*6:(i*6)+5])
            print(ans)
            f.write(str(i+1)+','+str(ans)+'\n')

if __name__ == '__main__':
    testing_data_path = sys.argv[1]
    prediction_file_path = sys.argv[2]
    main(testing_data_path,prediction_file_path)