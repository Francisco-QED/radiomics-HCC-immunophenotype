import os, sys, glob
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from joblib import dump, load

# import tensorflow as tf
# # import keras

# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential, model_from_json
# from tensorflow.keras.utils import to_categorical


sys.path.append('/home/fauvev01/BMEII-Functions/')

#Custom modules
import bmeii_io as bio
import bmeii_ml_functions as bml

def main():

	# Loop over the three feature selection methods and run the three classifiers for each

	# Feature selection methods 'ap', 'pca', 'lasso'
	feat_sel_methods = ['lasso']

	############################
	# Input Values
	############################

	# Pre or post
	# pre, post

	# Sequence
	# adc, dwi_high_bv, hbp, hbp_peri, t1_ap2, t1_pre, t1_pvp, t1_tp

	pre_post = 'pre'
	sequence = 'hbp'

	############################

	dict_res_auc = {}
	dict_res_sens = {}
	dict_res_spec = {}

	for feat_sel_method in feat_sel_methods:

		# Run the classifier
		aucs, sens, spec, = run_classifier(feat_sel_method, pre_post, sequence)

		# pdb.set_trace()

		# Append the results to the lists
		dict_res_auc[feat_sel_method] = [aucs]
		dict_res_sens[feat_sel_method] = [sens]
		dict_res_spec[feat_sel_method] = [spec]

		# Set up index



		# Save the results as a heatmap
	pdb.set_trace()
	dict_auc = {}
	dict_sens = {}
	dict_spec = {}

	dict_auc['AP'] = dict_res_auc['ap'][0]
	dict_auc['PCA'] = dict_res_auc['pca'][0]
	dict_auc['Lasso'] = dict_res_auc['lasso'][0]

	dict_sens['AP'] = dict_res_sens['ap'][0]
	dict_sens['PCA'] = dict_res_sens['pca'][0]
	dict_sens['Lasso'] = dict_res_sens['lasso'][0]

	dict_spec['AP'] = dict_res_spec['ap'][0]
	dict_spec['PCA'] = dict_res_spec['pca'][0]
	dict_spec['Lasso'] = dict_res_spec['lasso'][0]

	fig1, ax1 = plt.subplots(figsize=(8, 6))
	ax1.set_ylim([0, 1])
	df_auc = pd.DataFrame(data=dict_auc)
	df_auc.index = ['MLP', 'SVM', 'RF']

	sns.heatmap(df_auc, annot=True, cmap='viridis', ax=ax1, vmin=0, vmax=1)
	plt.savefig('/home/fauvev01/Bachir/SBRT/output/results/auc_'+pre_post+'_'+sequence+'_results.png')

	fig2, ax2 = plt.subplots(figsize=(8, 6))
	ax2.set_ylim([0, 1])
	df_sens = pd.DataFrame(data=dict_sens)
	df_sens.index = ['MLP', 'SVM', 'RF']

	# Set the scale legend to be from 0 to 1
	sns.heatmap(df_sens, annot=True, cmap='viridis', ax=ax2, vmin=0, vmax=1)
	plt.savefig('/home/fauvev01/Bachir/SBRT/output/results/sens_'+pre_post+'_'+sequence+'_results.png')

	fig3, ax3 = plt.subplots(figsize=(8, 6))
	ax3.set_ylim([0, 1])
	df_spec = pd.DataFrame(data=dict_spec)
	df_spec.index = ['MLP', 'SVM', 'RF']

	sns.heatmap(df_spec, annot=True, cmap='viridis', ax=ax3, vmin=0, vmax=1)
	plt.savefig('/home/fauvev01/Bachir/SBRT/output/results/spec_'+pre_post+'_'+sequence+'_results.png')

	# pdb.set_trace()
	print('>>> Done <<<')


def run_classifier(feat_sel_method, pre_post, seq):

	test_name = pre_post+'_'+seq+'_'+feat_sel_method

	data_lst_dir = '/home/fauvev01/Bachir/SBRT/data/lists/data_split/'+pre_post+'_tr_lst.csv'
	tst_data_lst_dir = '/home/fauvev01/Bachir/SBRT/data/lists/data_split/'+pre_post+'_tst_lst.csv'
	
	best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/'+feat_sel_method+'/'+pre_post+'_'+seq+'_best_feats_'+feat_sel_method+'.csv'
	means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq+'_means_and_stds.csv'


	#Reads the list
	data_lst = bio.read_csv(data_lst_dir, type='list')
	tst_data_lst = bio.read_csv(tst_data_lst_dir, type='list')

	best_feats = bio.read_csv(best_feats_dir, type='list')
	means_stds = np.array(bio.read_csv(means_stds_dir, type='list')).astype(float)

	means = means_stds[:,0]
	stds = means_stds[:,1]

	auc_out = []
	sens_out = []
	spec_out = []

	######################

	#Creates the radiomics matrix from list
	#Sequences 
	#Pre: DWI High B value, ADC, HBP, HBP peri-lesion, PVP, PVP peri-lesion, T2, T2 peri-lesion
	#Post: DWI, DWI peri-lesion, HBP, HBP peri-lesion, PVP, PVP peri-lesion, T2, T2 peri-lesion

	if seq == 'hbp':
		X, y, feat_names = extract_radiomics_table(data_lst, ['HBP'])
		X_tst, y_tst, feat_names_tst = extract_radiomics_table(tst_data_lst, ['HBP'])
	elif seq == 'adc':
		X, y, feat_names = extract_radiomics_table(data_lst, ['ADC'])
		X_tst, y_tst, feat_names_tst = extract_radiomics_table(tst_data_lst, ['ADC'])
	elif seq == 'hbp_peri':
		X, y, feat_names = extract_radiomics_table(data_lst, ['HBP Peri-lesion'])
		X_tst, y_tst, feat_names_tst = extract_radiomics_table(tst_data_lst, ['HBP Peri-lesion'])
	elif seq == 't1_pvp':
		X, y, feat_names = extract_radiomics_table(data_lst, ['T1 PVP'])
		X_tst, y_tst, feat_names_tst = extract_radiomics_table(tst_data_lst, ['T1 PVP'])
	elif seq == 't1_ap2':
		X, y, feat_names = extract_radiomics_table(data_lst, ['T1 AP2'])
		X_tst, y_tst, feat_names_tst = extract_radiomics_table(tst_data_lst, ['T1 AP2'])
	elif seq == 't1_pre':
		X, y, feat_names = extract_radiomics_table(data_lst, ['T1 PRE'])
		X_tst, y_tst, feat_names_tst = extract_radiomics_table(tst_data_lst, ['T1 PRE'])
	elif seq == 't1_pvp':
		X, y, feat_names = extract_radiomics_table(data_lst, ['T1 PVP'])
		X_tst, y_tst, feat_names_tst = extract_radiomics_table(tst_data_lst, ['T1 PVP'])
	elif seq == 't1_tp':
		X, y, feat_names = extract_radiomics_table(data_lst, ['T1 TP'])
		X_tst, y_tst, feat_names_tst = extract_radiomics_table(tst_data_lst, ['T1 TP'])
	elif seq == 'dwi_high_bv':
		X, y, feat_names = extract_radiomics_table(data_lst, ['DWI High B value'])
		X_tst, y_tst, feat_names_tst = extract_radiomics_table(tst_data_lst, ['DWI High B value'])
	elif seq == 't1_pre-t1_pvp':
		X, y, feat_names, subj_lst = extract_radiomics_table_multiseq(data_lst, ['T1 PRE', 'T1 PVP'])
		X_tst, y_tst, feat_names_tst, subj_lst_tst = extract_radiomics_table_multiseq(tst_data_lst, ['T1 PRE', 'T1 PVP'])
	elif seq == 't1_tp-t1_pvp':
		X, y, feat_names, subj_lst = extract_radiomics_table_multiseq(data_lst, ['T1 TP', 'T1 PVP'])
		X_tst, y_tst, feat_names_tst, subj_lst_tst = extract_radiomics_table_multiseq(tst_data_lst, ['T1 TP', 'T1 PVP'])
	elif seq == 'hbp-t1_pvp':
		X, y, feat_names, subj_lst = extract_radiomics_table_multiseq(data_lst, ['HBP', 'T1 PVP'])
		X_tst, y_tst, feat_names_tst, subj_lst_tst = extract_radiomics_table_multiseq(tst_data_lst, ['HBP', 'T1 PVP'])

	###################################

	# pdb.set_trace()

	#Avoid errors of nan or inf
	X = np.nan_to_num(X)
	X_tst = np.nan_to_num(X_tst)

	####### Standardization ############

	X = (X - means) / stds
	X_tst = (X_tst - means) / stds

	####### Standardization ############


	######## Get only best feats #######

	#Gets only the top feats based on the previous pca analysis
	#array of best feats nbs
	arr_best_feats = [int(i[1]) for i in best_feats]

	##### Taking best feats only
	#>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	X = X[:,arr_best_feats]
	X_tst = X_tst[:,arr_best_feats]



	#####################
	### MLP classifier
	#####################

	params_grid = {
		'hidden_layer_sizes': [(100, 10), (100, 50, 10), (100, 50, 25, 10)],
		'alpha': [0.001, 0.01],
		'batch_size': [8, 16],
		'max_iter': [500],
		'random_state': [3],
		'solver': ['adam'],
		'activation': ['tanh', 'relu'],
	}

	mlp_cls = MLPClassifier(random_state=3)

	#Runs the random search
	# random_search = RandomizedSearchCV(mlp_cls, param_distributions=params_dist, n_iter=20, cv=5, random_state=42)
	grid_search = GridSearchCV(mlp_cls, params_grid, cv=5)

	#Fits the model
	grid_search.fit(X, y)

	# print('Best parameters:', grid_search.best_params_)
	# print('Best score:', grid_search.best_score_)

	grid_preds = grid_search.predict(X_tst)
	grid_auc = roc_auc_score(y_tst, grid_preds)

	# pdb.set_trace()

	mlp_cls = MLPClassifier(random_state=3, max_iter=500, hidden_layer_sizes=(100, 50, 10), alpha=0.001, batch_size=16)
	auc_dir_out = '/home/fauvev01/Bachir/SBRT/output/mlp/'+test_name+'_mlp.png'
	best_model = bml.roc_ic(X, y, mlp_cls, odir='', N=20, IC=0.95, model_out='best')

	# pdb.set_trace()
	preds = best_model.predict(X_tst)
	AUC_mlp = roc_auc_score(y_tst, preds)

	if grid_auc > AUC_mlp:
		best_model = grid_search
		AUC_mlp = grid_auc

		best_params_mlp = grid_search.best_params_
	else:
		best_params_mlp = best_model.get_params()

	#Test set
	auc_ic = bml.roc_cis(y_tst, preds)
	TP_mlp, TN_mlp, FP_mlp, FN_mlp = bml.confusion_matrix(preds, y_tst)
	
	print('MLP AUC: '+str(AUC_mlp))
	print('MLP AUC IC: [' + str(auc_ic[0]) + ' - ' + str(auc_ic[1]) + ']')
	print('MLP Sens: ' + str(TP_mlp/(TP_mlp+FN_mlp)))
	print('MLP Spec: ' + str(TN_mlp/(TN_mlp+FP_mlp)))
	print('MLP PPV: ' + str(TP_mlp/(TP_mlp+FP_mlp)))
	print('MLP NPV: ' + str(TN_mlp/(TN_mlp+FN_mlp)))
	
	# Saving the parameters of the best model	
	best_params_dir_out = '/home/fauvev01/Bachir/SBRT/output/mlp/'+test_name+'_mlp.txt'

	best_params_mlp['auc'] = AUC_mlp
	best_params_mlp['auc_ic'] = [auc_ic[0], auc_ic[1]]
	best_params_mlp['sens'] = TP_mlp/(TP_mlp+FN_mlp)
	best_params_mlp['sens_ic'] = [auc_ic[2], auc_ic[3]]
	best_params_mlp['spec'] = TN_mlp/(TN_mlp+FP_mlp) 
	best_params_mlp['spec_ic'] = [auc_ic[4], auc_ic[5]]
	
	# with open(best_params_dir_out, 'w') as f:
	# 	best_params_str = str(best_params_mlp)
	# 	f.write(best_params_str)


	pdb.set_trace()

	# model_dirout = '/home/fauvev01/Bachir/SBRT/output/best_models/' + test_name + '_mlp.joblib'
	# dump(best_model, model_dirout)

	# pdb.set_trace()

	#####################
	### SVM classifier
	#####################

	# SVM grid search
	params_grid = {
		'kernel': ['linear', 'rbf'],
		'C': [0.1, 1, 10],
		'gamma': ['auto', 'scale'],
		'random_state': [42],
	}

	svm_cls = svm.SVC(probability=True)
	
	#Runs the grid search
	grid_search = GridSearchCV(svm_cls, params_grid, cv=5)
	
	#Fits the model
	grid_search.fit(X, y)

	preds_grid = grid_search.best_estimator_.predict(X_tst)
	grid_auc = roc_auc_score(y_tst, preds_grid)

	svm_cls = svm.SVC(kernel='linear', C=0.1, gamma='auto', random_state=42, probability=True)
	model_bml = bml.roc_ic(X, y, svm_cls, odir='', N=20, IC=0.95, model_out='best')

	preds_bml = model_bml.predict(X_tst)
	bml_auc = roc_auc_score(y_tst, preds_bml)

	if grid_auc > bml_auc:
		best_model = grid_search.best_estimator_
		best_params_svm = grid_search.best_params_
	else:
		best_model = model_bml
		best_params_svm = best_model.get_params()	

	#Test set
	preds = best_model.predict(X_tst)
	auc_ic = bml.roc_cis(y_tst, preds)
	TP_svm, TN_svm, FP_svm, FN_svm = bml.confusion_matrix(preds, y_tst)
	AUC_svm = roc_auc_score(y_tst, preds)

	print('SVM AUC: '+str(AUC_svm))
	print('SVM AUC IC: [' + str(auc_ic[0]) + ' - ' + str(auc_ic[1]) + ']')
	print('SVM Sens: ' + str(TP_svm/(TP_svm+FN_svm)))
	print('SVM Spec: ' + str(TN_svm/(TN_svm+FP_svm)))
	print('SVM PPV: ' + str(TP_svm/(TP_svm+FP_svm)))
	print('SVM NPV: ' + str(TN_svm/(TN_svm+FN_svm)))

	# Saving the parameters of the best model	
	best_params_dir_out = '/home/fauvev01/Bachir/SBRT/output/svm/'+test_name+'_svm.txt'

	best_params_svm['auc'] = AUC_svm
	best_params_svm['auc_ic'] = [auc_ic[0], auc_ic[1]]
	best_params_svm['sens'] = TP_svm/(TP_svm+FN_svm)
	best_params_svm['sens_ic'] = [auc_ic[2], auc_ic[3]]
	best_params_svm['spec'] = TN_svm/(TN_svm+FP_svm) 
	best_params_svm['spec_ic'] = [auc_ic[4], auc_ic[5]]
	
	# with open(best_params_dir_out, 'w') as f:
	# 	best_params_str = str(best_params_svm)
	# 	f.write(best_params_str)

	# auc_dir_out_tst = '/home/fauvev01/Bachir/SBRT/output/svm/'+test_name+'_svm.png'
	# bml.gen_roc_curve(auc_dir_out_tst, preds, y_tst)

	# pdb.set_trace()

	# model_dirout = '/home/fauvev01/Bachir/SBRT/output/best_models/' + test_name + '_svm.joblib'
	# dump(best_model, model_dirout)

	# pdb.set_trace()

	#####################
	### Random Forest classifier
	#####################

	param_grid = {
		'n_estimators': [100, 200, 300],
		'max_depth': [None, 5, 10],
		'max_features': ['sqrt', 'log2'],
		'bootstrap': [True, False]
	}

	rf_cls = RandomForestClassifier(random_state=42)

	grid_search = GridSearchCV(estimator=rf_cls, param_grid=param_grid, cv=5)

	# Fit the GridSearchCV object on training data
	grid_search.fit(X, y)

	# Print the best parameters and score
	print("Best parameters: ", grid_search.best_params_)
	print("Best score: ", grid_search.best_score_)

	# Predict on the test data using the best model

	best_model = grid_search.best_estimator_
	best_params_rf = grid_search.best_params_

	preds_grid = best_model.predict(X_tst)
	auc_grid = roc_auc_score(y_tst, preds_grid)

	# auc_dir_out = '/home/fauvev01/Bachir/SBRT/output/rf/'+test_name+'_rf.png'
	best_model_bml = bml.roc_ic(X, y, rf_cls, odir='', N=20, IC=0.95, model_out='best')

	preds_bml = best_model_bml.predict(X_tst)
	auc_bml = roc_auc_score(y_tst, preds_bml)

	if auc_grid > auc_bml:
		best_model = best_model
		best_params_rf = best_params_rf
	else:
		best_model = best_model_bml
		best_params_rf = best_model_bml.get_params()

	# pdb.set_trace()

	#Test set
	preds = best_model.predict(X_tst)
	auc_ic = bml.roc_cis(y_tst, preds)
	TP_rf, TN_rf, FP_rf, FN_rf = bml.confusion_matrix(preds, y_tst)
	AUC_rf = roc_auc_score(y_tst, preds)

	print('RF AUC: '+str(AUC_rf))
	print('RF AUC IC: [' + str(auc_ic[0]) + ' - ' + str(auc_ic[1]) + ']')
	print('RF Sens: ' + str(TP_rf/(TP_rf+FN_rf)))
	print('RF Spec: ' + str(TN_rf/(TN_rf+FP_rf)))
	print('RF PPV: ' + str(TP_rf/(TP_rf+FP_rf)))
	print('RF NPV: ' + str(TN_rf/(TN_rf+FN_rf)))

	# Saving the parameters of the best model
	best_params_dir_out = '/home/fauvev01/Bachir/SBRT/output/rf/'+test_name+'_rf.txt'

	best_params_rf['auc'] = AUC_rf
	best_params_rf['auc_ic'] = [auc_ic[0], auc_ic[1]]
	best_params_rf['sens'] = TP_rf/(TP_rf+FN_rf)
	best_params_rf['sens_ic'] = [auc_ic[2], auc_ic[3]]
	best_params_rf['spec'] = TN_rf/(TN_rf+FP_rf)
	best_params_rf['spec_ic'] = [auc_ic[4], auc_ic[5]]

	# with open(best_params_dir_out, 'w') as f:
	# 	best_params_str = str(best_params_rf)
	# 	f.write(best_params_str)

	# auc_dir_out_tst = main_tst_dirout+test_name+'_rf.png'
	# bml.gen_roc_curve(auc_dir_out_tst, preds, y_tst)

	pdb.set_trace()

	# model_dirout = '/home/fauvev01/Bachir/SBRT/output/best_models/' + test_name + '_rf.joblib'
	# dump(best_model, model_dirout)

	# Creates a 

	# print(">>> DONE <<<")

	auc_out = [AUC_mlp, AUC_svm, AUC_rf]
	sens_out = [TP_mlp/(TP_mlp+FN_mlp), TP_svm/(TP_svm+FN_svm), TP_rf/(TP_rf+FN_rf)]
	spec_out = [TN_mlp/(TN_mlp+FP_mlp), TN_svm/(TN_svm+FP_svm), TN_rf/(TN_rf+FP_rf)]
	return auc_out, sens_out, spec_out



#From a list of Table.tsv files it creates a matrix of radiomics data
def extract_radiomics_table_multiseq(lst, sequences):

	out = []
	labels = []
	feat_names = []
	subjs_lst = []
	start = 1
	done = 0
	seqs_done = 0
	feat_names_done = 0
	subj_temp = ''

	# pdb.set_trace()

	for i in lst:

		if not sequences:
			try:
				temp = pd.read_csv(i[2], header=None, skiprows=18, sep=';')

				if done == 0:
					feat_names = [n[0]+'_'+n[1]+'_'+n[2] for n in temp]
					done = 1

				temp = temp[:,3].astype(float).tolist()

				out.append(temp)
				labels.append(int(i[2]))

			except:
				continue
		else:
			try:

				# pdb.set_trace()

				dir_temp = i[2].split('/')
				seq = i[1]
				pre_post = dir_temp[-3]
				subj = i[0]

				if not seq in sequences:
					continue

				if subj in subjs_lst:
					continue

				# Read the csv file from line 37
				# Header is: 'Radiomics Name' 'Value'
				# Separated by semicolon
				temp = pd.read_csv(i[2], header=None, skiprows=18, sep=';')

				# pdb.set_trace()

				rows, cols = temp.shape
				range_val = sequences.index(seq)

				if pre_post == 'Pre-Tx':
					pre_post = 'pre'
				elif pre_post == 'Post Tx':
					pre_post = 'post'
				else:
					pdb.set_trace()

				if start:
					#Empty sample
					sample = np.zeros(len(temp)*len(sequences))
					subj_temp = subj
					start = 0

					if feat_names_done == 0:
						feat_names = [0]*(len(temp)*len(sequences))

				# sample = np.zeros(len(temp)*len(sequences))
				# feat_names = [0]*(len(temp)*len(sequences))
				# subj_temp = subj
				# start = 0

				# feat_names_temp2 = temp[0].tolist()
				# feat_names[(range_val)*rows:(range_val+1)*rows] = feat_names_temp2

				# sample[(range_val)*rows:(range_val+1)*rows] = temp[1].astype(float).tolist()

				# if len(sample) != 108:
				# 	continue

				# out.append(sample)
				# labels.append(int(i[3]))

				if seqs_done < len(sequences):
					# pdb.set_trace()
					# Get all the features names corresponding to the first column
					# feat_names_temp = [pre_post + '_' + seq + '_' + n[0] for n in temp]

					# pdb.set_trace()

					if feat_names_done == 0:
						feat_names_temp2 = temp[0].tolist()
						# feat_names_temp2 = [pre_post + '_' + seq + '_' + n[0] for n in temp]
						feat_names_temp2 = [seq + '_' + n for n in feat_names_temp2]
						feat_names[(range_val)*rows:(range_val+1)*rows] = feat_names_temp2

					try:
						sample[(range_val)*rows:(range_val+1)*rows] = temp[1].astype(float).tolist()
					except:
						continue

					seqs_done+=1

				if seqs_done == len(sequences):
					out.append(sample)
					labels.append(int(i[3]))
					subjs_lst.append(subj)
					feat_names_done = 1
					subj_temp = ''
					start = 1
					seqs_done = 0

			except:
				continue


	return out, labels, feat_names, subjs_lst


#From a list of Table.tsv files it creates a matrix of radiomics data
def extract_radiomics_table(lst, sequences):

	out = []
	labels = []
	feat_names = []
	start = 1
	done = 0
	seqs_done = 0
	subj_temp = ''

	# pdb.set_trace()

	for i in lst:

		if not sequences:
			try:
				temp = pd.read_csv(i[2], header=None, skiprows=18, sep=';')

				if done == 0:
					feat_names = [n[0]+'_'+n[1]+'_'+n[2] for n in temp]
					done = 1

				temp = temp[:,3].astype(float).tolist()

				out.append(temp)
				labels.append(int(i[2]))

			except:
				continue
		else:
			try:
				dir_temp = i[2].split('/')
				seq = i[1]
				pre_post = dir_temp[-3]
				subj = i[0]

				if not seq in sequences:
					continue

				# Read the csv file from line 37
				# Header is: 'Radiomics Name' 'Value'
				# Separated by semicolon
				temp = pd.read_csv(i[2], header=None, skiprows=18, sep=';')

				# pdb.set_trace()

				rows, cols = temp.shape
				range_val = sequences.index(seq)

				if pre_post == 'Pre-Tx':
					pre_post = 'pre'
				elif pre_post == 'Post Tx':
					pre_post = 'post'
				else:
					pdb.set_trace()

				#Checks if it is the same subject
				# if subj != subj_temp:
				# 	start = 1

				# if start:
				# 	#Empty sample
				# 	sample = np.zeros(len(temp)*len(sequences))
				# 	feat_names = [0]*(len(temp)*len(sequences))
				# 	subj_temp = subj
				# 	start = 0

				sample = np.zeros(len(temp)*len(sequences))
				feat_names = [0]*(len(temp)*len(sequences))
				subj_temp = subj
				start = 0

				feat_names_temp2 = temp[0].tolist()
				feat_names[(range_val)*rows:(range_val+1)*rows] = feat_names_temp2

				sample[(range_val)*rows:(range_val+1)*rows] = temp[1].astype(float).tolist()

				if len(sample) != 108:
					continue

				out.append(sample)
				labels.append(int(i[3]))

				# if seqs_done < len(sequences):
				# 	# pdb.set_trace()
				# 	# Get all the features names corresponding to the first column
				# 	# feat_names_temp = [pre_post + '_' + seq + '_' + n[0] for n in temp]

				# 	feat_names_temp2 = temp[0].tolist()

				# 	# feat_names_temp2 = [pre_post + '_' + seq + '_' + n[0] for n in temp]
				# 	feat_names[(range_val)*rows:(range_val+1)*rows] = feat_names_temp2
				# 	seqs_done+=1

				# 	try:
				# 		sample[(range_val)*rows:(range_val+1)*rows] = temp[1].astype(float).tolist()
				# 	except:
				# 		continue

				# if seqs_done == len(sequences):
				# 	out.append(sample)
				# 	labels.append(int(i[2]))
				# 	feat_names_done = 0

			except:
				continue


	return out, labels, feat_names


def best_pred_seq(pred, gt, seq):

	dict_out = {}

	for i in range(len(gt)):
		gt_temp = gt[i]
		pred_temp = pred[i]
		seq_temp = seq[i]

		if gt_temp == 0:
			if gt_temp != pred_temp:
				# print('Pred=GT=0, ' + str(seq_temp))
				try:
					dict_out['0, ' + str(seq_temp)] += 1
				except:
					dict_out['0, ' + str(seq_temp)] = 1
		else:
			if gt_temp != pred_temp:
				# print('Pred=GT=1, ' + str(seq_temp))

				try:
					dict_out['1, ' + str(seq_temp)] += 1
				except:
					dict_out['1, ' + str(seq_temp)] = 1

	return dict_out


if __name__ == "__main__":
	main()