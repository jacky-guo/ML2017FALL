
import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
from math import log, floor

# If you wish to get the same shuffle result
# np.random.seed(2401)

delList = [' Never-married',' Own-child',' Not-in-family',' Other-service',' Unmarried',' HS-grad',' Divorced',' Adm-clerical',' Black',' Handlers-cleaners',' 11th']
getList = [' Married-civ-spouse',' Husband','age','hours_per_week','capital_gain','sex',' Exec-managerial',' Prof-specialty',' Bachelors',' Masters',' Prof-school','capital_loss']
def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    # delList
    for column in delList:
    	del X_train[column]
    	del X_test[column]
    
    ## getList 	
    # X_train = X_train[[_ for _ in getList]]
    # X_test = X_test[[_ for _ in getList]]
    
    X_train = np.array(X_train.values)
    Y_train = np.array(Y_train.values)
    X_test = np.array(X_test.values)
    
    return (X_train, Y_train, X_test)

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))
    
    X_all, Y_all = _shuffle(X_all, Y_all)
    
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    
    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return

def train(X_all, Y_all):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    
    # Gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((95,))
    mu2 = np.zeros((95,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((95,95))
    sigma2 = np.zeros((95,95))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    N1 = cnt1
    N2 = cnt2

    # print('=====Saving Param=====')
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # param_dict = {'mu1':mu1, 'mu2':mu2, 'shared_sigma':shared_sigma, 'N1':[N1], 'N2':[N2]}
    # for key in sorted(param_dict):
    #     print('Saving %s' % key)
    #     np.savetxt(os.path.join(save_dir, ('%s' % key)), param_dict[key])
    
    print('=====Validating=====')
    valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)


    return mu1,mu2,shared_sigma,N1,N2

def infer(X_test,save_path,mu1,mu2,shared_sigma,N1,N2):
    # Load parameters
    # print('=====Loading Param from %s=====' % save_dir)
    # mu1 = np.loadtxt(os.path.join(save_dir, 'mu1'))
    # mu2 = np.loadtxt(os.path.join(save_dir, 'mu2'))
    # shared_sigma = np.loadtxt(os.path.join(save_dir, 'shared_sigma'))
    # N1 = np.loadtxt(os.path.join(save_dir, 'N1'))
    # N2 = np.loadtxt(os.path.join(save_dir, 'N2'))

    # Predict
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    # print('=====Write output to %s =====' % output_dir)
    # # Write output
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # output_path = os.path.join(output_dir, 'prediction.csv')
    with open(save_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return

def main(train_data_path,train_label_path,test_data_path,save_path):
    # Load feature and label
    X_all, Y_all, X_test = load_data(train_data_path, train_label_path, test_data_path)
    # Normalization
    X_all, X_test = normalize(X_all, X_test)
    
    # To train or to infer
    mu1,mu2,shared_sigma,N1,N2 = train(X_all, Y_all)
    infer(X_test,save_path,mu1,mu2,shared_sigma,N1,N2)

    return

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #             description='Probabilistic Generative Model for Binary Classification'
    #          )
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument('--train', action='store_true', default=False,
    #                     dest='train', help='Input --train to Train')
    # group.add_argument('--infer', action='store_true',default=False,
    #                     dest='infer', help='Input --infer to Infer')
    # parser.add_argument('--train_data_path', type=str,
    #                     default='feature/X_train', dest='train_data_path',
    #                     help='Path to training data')
    # parser.add_argument('--train_label_path', type=str,
    #                     default='feature/Y_train', dest='train_label_path',
    #                     help='Path to training data\'s label')
    # parser.add_argument('--test_data_path', type=str,
    #                     default='feature/X_test', dest='test_data_path',
    #                     help='Path to testing data')
    # parser.add_argument('--save_dir', type=str, 
    #                     default='generative_params/', dest='save_dir',
    #                     help='Path to save the model parameters')
    # parser.add_argument('--output_dir', type=str, 
    #                     default='generative_output/', dest='output_dir',
    #                     help='Path to save output')
    # opts = parser.parse_args()
    train_data_path = sys.argv[3]
    train_label_path = sys.argv[4]
    test_data_path = sys.argv[5]
    save_path = sys.argv[6]
    main(train_data_path,train_label_path,test_data_path,save_path)