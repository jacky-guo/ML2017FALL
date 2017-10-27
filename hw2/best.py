import pandas as pd
import xgboost as xgb
import os,sys

if __name__ == '__main__':

	train_data_path = sys.argv[3]
	train_label_path = sys.argv[4]
	test_data_path = sys.argv[5]
	save_path = sys.argv[6]

	df_X = pd.read_csv(train_data_path).as_matrix()
	df_Y = pd.read_csv(train_label_path).as_matrix().reshape(len(df_X))
	df_Xtest = pd.read_csv(test_data_path).as_matrix()


	xgb_model = xgb.XGBClassifier(learning_rate=0.28,max_depth=4,n_estimators=100,min_child_weight=6)
	xgb_model.fit(df_X,df_Y)
	output = xgb_model.predict(df_Xtest)
	
	with open(save_path, 'w') as f:
		f.write('id,label\n')
		for i, v in  enumerate(output):
			f.write('%d,%d\n' %(i+1, v))
