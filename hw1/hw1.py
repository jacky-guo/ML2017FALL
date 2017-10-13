import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

def training():
	data = []
	for i in range(18):
	    data.append([])
	n_row = 0
	text = open('./data/train.csv', 'r', encoding='big5') 
	row = csv.reader(text , delimiter=",")
	for r in row:
	    if n_row != 0:
	        for i in range(3,27):
	            if r[i] != "NR":
	                data[(n_row-1)%18].append(float(r[i]))
	            else:
	                data[(n_row-1)%18].append(float(0))
	    n_row = n_row+1
	text.close()
	x = []
	y = []
	for i in range(12):
	    for j in range(471):
	        x.append([])
	        for t in range(18):
	            if t == 9:
	                for s in range(9):
	                    x[471*i+j].append(data[t][480*i+j+s] )
	                y.append(data[9][480*i+j+9])
	        
	x = np.array(x)
	y = np.array(y)

	# add square term
	x = np.concatenate((x,x**2), axis=1)

	# add bias
	x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)


	w = np.zeros(len(x[0]))

	l_rate = 10
	repeat = 10000
	x_t = x.transpose()
	s_gra = np.zeros(len(x[0]))


	for i in range(repeat):
	    hypo = np.dot(x,w)
	    loss = hypo - y
	    cost = np.sum(loss**2) / len(x)
	    cost_a  = math.sqrt(cost)
	    gra = 2*np.dot(x_t,loss)
	    s_gra += gra**2
	    ada = np.sqrt(s_gra)
	    w = w - l_rate * gra/ada
	    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

	np.save('./model.npy',w)
	return w



if __name__ == "__main__":

	w = np.load('model.npy')
	test_x = []
	n_row = 0
	text = open(sys.argv[1] ,"r")
	row = csv.reader(text , delimiter= ",")

	for r in row:
	    if n_row %18 == 9:
	        test_x.append([])
	        for i in range(2,11):
	            test_x[n_row//18].append(float(r[i]))
	    n_row = n_row+1
	    
	text.close()
	test_x = np.array(test_x)

	# add square term
	test_x = np.concatenate((test_x,test_x**2), axis=1)

	# add bias
	test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

	ans = []
	for i in range(len(test_x)):
	    ans.append(["id_"+str(i)])
	    a = np.dot(w,test_x[i])
	    ans[i].append(a)

	filename = sys.argv[2]
	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","value"])
	for i in range(len(ans)):
	    s.writerow(ans[i]) 
	text.close()