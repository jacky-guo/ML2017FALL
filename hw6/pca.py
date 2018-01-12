from skimage import io
import numpy as np
import os
import sys

if __name__ == "__main__":
	img_list = []
	for dirPath, dirNames, fileNames in os.walk(sys.argv[1]):
		for i, f in enumerate(fileNames):
			load_file = os.path.join(dirPath, f)
			img = np.array(io.imread(load_file))
			img_fla = img.flatten().reshape(1,1080000)
			if i == 0:
				img_list = img_fla
			else:
				img_list = np.concatenate((img_list, img_fla), axis=0)
	img_list = img_list.T
	new_mean = np.mean(img_list, axis=1)
	img_mean_np = np.repeat(np.array([new_mean]).T,415,axis=1)
	X = img_list - img_mean_np
	U, s, V = np.linalg.svd(X, full_matrices=False)
	
	test_img = np.array(io.imread(sys.argv[2]))
	test_img_fla = test_img.flatten()
	
	weight = []
	for column in range(4):
		weight.append(np.dot(test_img_fla, U.T[column]))

	eigenfaces = np.zeros(1080000,)
	eigenface_li = []

	for column, w in enumerate(np.array(weight)):
		eigenfaces += w * U.T[column]
		eigenface_li.append(w * U.T[column])

	eigenfaces += new_mean
	eigenfaces -= np.min(eigenfaces)
	eigenfaces /= np.max(eigenfaces)
	eigenfaces = (eigenfaces*255).astype(np.uint8)
	print(eigenfaces.shape)
	eigenfaces = eigenfaces.reshape(600, 600, 3)
	
	io.imsave('reconstruction.jpg', eigenfaces)
