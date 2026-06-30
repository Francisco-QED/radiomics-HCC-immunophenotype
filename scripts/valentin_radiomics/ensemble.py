import os, sys, glob
import numpy as np
import pandas as pd
import pdb
from joblib import dump, load

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

sys.path.append('/home/fauvev01/BMEII-Functions/')

#Custom modules
import bmeii_io as bio
import bmeii_ml_functions as bml

def main():

	# test_name = 'prepost_ensemble_predwi_postdwiperi_posthbp_posthbpperi_postpvp_postpvpperi_pret2_postt2peri'
	test_name = 'post_ensemble_dwiperi_hbp_hbpperi'
	tgt_feats = []

	main_dirout = '/home/fauvev01/Bachir/Radiomics/output/radiomics/roc/ensemble/new/'
	main_tst_dirout = '/home/fauvev01/Bachir/Radiomics/output/radiomics/roc/ensemble/new/tst2_'

	data_lst_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/main_lsts/post_all_data_lst.csv'
	tst_data_lst_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/main_lsts/tst_post_data_lst.csv'


	dwi_best_feats_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/best_feats/indep/ind_post_dwi_best_feats_pca.csv'
	dwi_means_stds_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/means_stds/indep/ind_post_dwi_means_and_stds.csv'
	dwiperi_best_feats_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/best_feats/indep/ind_post_dwiperi_best_feats_pca.csv'
	dwiperi_means_stds_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/means_stds/indep/ind_post_dwiperi_means_and_stds.csv'
	hbp_best_feats_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/best_feats/indep/ind_post_hbp_best_feats_pca.csv'
	hbp_means_stds_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/means_stds/indep/ind_post_hbp_means_and_stds.csv'
	hbpperi_best_feats_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/best_feats/indep/ind_post_hbpperi_best_feats_pca.csv'
	hbpperi_means_stds_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/means_stds/indep/ind_post_hbpperi_means_and_stds.csv'
	pvp_best_feats_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/best_feats/indep/ind_post_pvp_best_feats_pca.csv'
	pvp_means_stds_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/means_stds/indep/ind_post_pvp_means_and_stds.csv'
	pvpperi_best_feats_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/best_feats/indep/ind_post_pvpperi_best_feats_pca.csv'
	pvpperi_means_stds_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/means_stds/indep/ind_post_pvpperi_means_and_stds.csv'
	t2_best_feats_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/best_feats/indep/ind_post_t2_best_feats_pca.csv'
	t2_means_stds_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/means_stds/indep/ind_post_t2_means_and_stds.csv'
	t2peri_best_feats_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/best_feats/indep/ind_post_t2peri_best_feats_pca.csv'
	t2peri_means_stds_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/means_stds/indep/ind_post_t2peri_means_and_stds.csv'

	#All feats names
	all_feats_names_dir = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/lists/best_feats/indep/all_feats.csv'


	#Best models
	#Pre
	# best_dwi = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/pre_dwi_allfeats_ind_mlp.joblib'
	# best_dwiperi = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/pre_dwiperi_bestfeats_ind_mlp.joblib'
	# best_hbp = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/pre_hbp_bestfeats_ind_mlp.joblib'
	# best_hbpperi = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/pre_hbpperi_allfeats_ind_rf.joblib'
	# best_pvp = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/pre_pvp_bestfeats_ind_rf.joblib'
	# best_pvpperi = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/pre_pvpperi_allfeats_ind_rf.joblib'
	# best_t2 = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/pre_t2_allfeats_ind_mlp.joblib'
	# best_t2peri = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/pre_t2peri_bestfeats_ind_rf.joblib'

	#Post
	best_dwi = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/post_dwi_bestfeats_ind_svm.joblib'
	best_dwiperi = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/post_dwiperi_bestfeats_ind_rf.joblib'
	best_hbp = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/post_hbp_allfeats_ind_mlp.joblib'
	best_hbpperi = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/post_hbpperi_bestfeats_ind_mlp.joblib'
	best_pvp = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/post_pvp_bestfeats_ind_rf.joblib'
	best_pvpperi = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/post_pvpperi_bestfeats_ind_svm.joblib'
	best_t2 = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/post_t2_bestfeats_ind_rf.joblib'
	best_t2peri = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/post_t2peri_allfeats_ind_mlp.joblib'

	#Load models
	model_dwi = load(best_dwi)
	model_dwiperi = load(best_dwiperi)
	model_hbp = load(best_hbp)
	model_hbpperi = load(best_hbpperi)
	model_pvp = load(best_pvp)
	model_pvpperi = load(best_pvpperi)
	model_t2 = load(best_t2)
	model_t2peri = load(best_t2peri)


	#Reads the list
	data_lst = bio.read_csv(data_lst_dir, type='list')
	tst_data_lst = bio.read_csv(tst_data_lst_dir, type='list')

	dwi_best_feats = bio.read_csv(dwi_best_feats_dir, type='list')
	dwiperi_best_feats = bio.read_csv(dwiperi_best_feats_dir, type='list')
	hbp_best_feats = bio.read_csv(hbp_best_feats_dir, type='list')
	hbpperi_best_feats = bio.read_csv(hbpperi_best_feats_dir, type='list')
	pvp_best_feats = bio.read_csv(pvp_best_feats_dir, type='list')
	pvpperi_best_feats = bio.read_csv(pvpperi_best_feats_dir, type='list')
	t2_best_feats = bio.read_csv(t2_best_feats_dir, type='list')
	t2peri_best_feats = bio.read_csv(t2peri_best_feats_dir, type='list')


	#All feats names
	feats_names = bio.read_csv(all_feats_names_dir, type='list')
	feats_names = np.squeeze(feats_names)

	#Sequences 
	#Pre: DWI, DWI peri-lesion, HBP, HBP peri-lesion, PVP, PVP peri-lesion, T2, T2 peri-lesion
	#Post: DWI, DWI peri-lesion, HBP, HBP peri-lesion, PVP, PVP peri-lesion, T2, T2 peri-lesion

	# pdb.set_trace()

	X_dict, y, feat_names_dict = extract_radiomics_table_ensemble(data_lst, ['DWI', 'DWI peri-lesion', 'HBP', 'HBP peri-lesion', 'PVP', 'PVP peri-lesion', 'T2', 'T2 peri-lesion'])
	X_dict_tst, y_tst, feat_names_dict_tst = extract_radiomics_table_ensemble(tst_data_lst, ['DWI', 'DWI peri-lesion', 'HBP', 'HBP peri-lesion', 'PVP', 'PVP peri-lesion', 'T2', 'T2 peri-lesion'])


	#Extracting the data
	X_dwi = np.nan_to_num(X_dict['DWI'])
	X_dwi_tst = np.nan_to_num(X_dict_tst['DWI'])

	X_dwiperi = np.nan_to_num(X_dict['DWI peri-lesion'])
	X_dwiperi_tst = np.nan_to_num(X_dict_tst['DWI peri-lesion'])

	X_hbp = np.nan_to_num(X_dict['HBP'])
	X_hbp_tst = np.nan_to_num(X_dict_tst['HBP'])

	X_hbpperi = np.nan_to_num(X_dict['HBP peri-lesion'])
	X_hbpperi_tst = np.nan_to_num(X_dict_tst['HBP peri-lesion'])

	X_pvp = np.nan_to_num(X_dict['PVP'])
	X_pvp_tst = np.nan_to_num(X_dict_tst['PVP'])

	X_pvpperi = np.nan_to_num(X_dict['PVP peri-lesion'])
	X_pvpperi_tst = np.nan_to_num(X_dict_tst['PVP peri-lesion'])

	X_t2 = np.nan_to_num(X_dict['T2'])
	X_t2_tst = np.nan_to_num(X_dict_tst['T2'])

	X_t2peri = np.nan_to_num(X_dict['T2 peri-lesion'])
	X_t2peri_tst = np.nan_to_num(X_dict_tst['T2 peri-lesion'])


	####### Standardization ############

	scaler_dwi = StandardScaler()
	scaler_dwi.fit(X_dwi)
	means_dwi = scaler_dwi.mean_
	stds_dwi = scaler_dwi.scale_

	scaler_dwiperi = StandardScaler()
	scaler_dwiperi.fit(X_dwiperi)
	means_dwiperi = scaler_dwiperi.mean_
	stds_dwiperi = scaler_dwiperi.scale_

	scaler_hbp = StandardScaler()
	scaler_hbp.fit(X_hbp)
	means_hbp = scaler_hbp.mean_
	stds_hbp = scaler_hbp.scale_

	scaler_hbpperi = StandardScaler()
	scaler_hbpperi.fit(X_hbpperi)
	means_hbpperi = scaler_hbpperi.mean_
	stds_hbpperi = scaler_hbpperi.scale_

	scaler_pvp = StandardScaler()
	scaler_pvp.fit(X_pvp)
	means_pvp = scaler_pvp.mean_
	stds_pvp = scaler_pvp.scale_

	scaler_pvpperi = StandardScaler()
	scaler_pvpperi.fit(X_pvpperi)
	means_pvpperi = scaler_pvpperi.mean_
	stds_pvpperi = scaler_pvpperi.scale_

	scaler_t2 = StandardScaler()
	scaler_t2.fit(X_t2)
	means_t2 = scaler_t2.mean_
	stds_t2 = scaler_t2.scale_

	scaler_t2peri = StandardScaler()
	scaler_t2peri.fit(X_t2peri)
	means_t2peri = scaler_t2peri.mean_
	stds_t2peri = scaler_t2peri.scale_

	######## Get only best feats #######

	#Gets only the top feats based on the previous pca analysis
	#array of best feats nbs
	dwi_best_feats = [int(i[1]) for i in dwi_best_feats]
	dwiperi_best_feats = [int(i[1]) for i in dwiperi_best_feats]
	hbp_best_feats = [int(i[1]) for i in hbp_best_feats]
	hbpperi_best_feats = [int(i[1]) for i in hbpperi_best_feats]
	pvp_best_feats = [int(i[1]) for i in pvp_best_feats]
	pvpperi_best_feats = [int(i[1]) for i in pvpperi_best_feats]
	t2_best_feats = [int(i[1]) for i in t2_best_feats]
	t2peri_best_feats = [int(i[1]) for i in t2peri_best_feats]

	###########################
	##### Data Preparation ####
	###########################

	#DWI
	X_dwi = (X_dwi - means_dwi) / stds_dwi
	X_dwi_tst = (X_dwi_tst - means_dwi) / stds_dwi
	X_dwi = X_dwi[:,dwi_best_feats]
	X_dwi_tst = X_dwi_tst[:,dwi_best_feats]

	#DWI peri
	X_dwiperi = (X_dwiperi - means_dwiperi) / stds_dwiperi
	X_dwiperi_tst = (X_dwiperi_tst - means_dwiperi) / stds_dwiperi
	X_dwiperi = X_dwiperi[:,dwiperi_best_feats]
	X_dwiperi_tst = X_dwiperi_tst[:,dwiperi_best_feats]
	dwiperi_feats_names = feats_names[dwiperi_best_feats]

	#HBP
	X_hbp = (X_hbp - means_hbp) / stds_hbp
	X_hbp_tst = (X_hbp_tst - means_hbp) / stds_hbp
	# X_hbp = X_hbp[:,hbp_best_feats]
	# X_hbp_tst = X_hbp_tst[:,hbp_best_feats]
	hbp_feats_names = feats_names

	#HBP peri
	X_hbpperi = (X_hbpperi - means_hbpperi) / stds_hbpperi
	X_hbpperi_tst = (X_hbpperi_tst - means_hbpperi) / stds_hbpperi
	X_hbpperi = X_hbpperi[:,hbpperi_best_feats]
	X_hbpperi_tst = X_hbpperi_tst[:,hbpperi_best_feats]
	hbpperi_feats_names = feats_names[hbpperi_best_feats]

	#PVP 
	X_pvp = (X_pvp - means_pvp) / stds_pvp
	X_pvp_tst = (X_pvp_tst - means_pvp) / stds_pvp
	X_pvp = X_pvp[:,pvp_best_feats]
	X_pvp_tst = X_pvp_tst[:,pvp_best_feats]

	#PVP peri
	X_pvpperi = (X_pvpperi - means_pvpperi) / stds_pvpperi
	X_pvpperi_tst = (X_pvpperi_tst - means_pvpperi) / stds_pvpperi
	X_pvpperi = X_pvpperi[:,pvpperi_best_feats]
	X_pvpperi_tst = X_pvpperi_tst[:,pvpperi_best_feats]

	#T2
	X_t2 = (X_t2 - means_t2) / stds_t2
	X_t2_tst = (X_t2_tst - means_t2) / stds_t2
	X_t2 = X_t2[:,t2_best_feats]
	X_t2_tst = X_t2_tst[:,t2_best_feats]

	#T2 peri
	X_t2peri = (X_t2peri - means_t2peri) / stds_t2peri
	X_t2peri_tst = (X_t2peri_tst - means_t2peri) / stds_t2peri
	# X_t2peri = X_t2peri[:,t2peri_best_feats]
	# X_t2peri_tst = X_t2peri_tst[:,t2peri_best_feats]


	###############################################################
	############### RUN MODELS ####################################
	###############################################################

	pred_dwi = model_dwi.predict(X_dwi)
	pred_dwi_tst = model_dwi.predict(X_dwi_tst)

	pred_dwiperi = model_dwiperi.predict(X_dwiperi)
	pred_dwiperi_tst = model_dwiperi.predict(X_dwiperi_tst)

	pred_hbp = model_hbp.predict(X_hbp)
	pred_hbp_tst = model_hbp.predict(X_hbp_tst)

	pred_hbpperi = model_hbpperi.predict(X_hbpperi)
	pred_hbpperi_tst = model_hbpperi.predict(X_hbpperi_tst)

	pred_pvp = model_pvp.predict(X_pvp)
	pred_pvp_tst = model_pvp.predict(X_pvp_tst)

	pred_pvpperi = model_pvpperi.predict(X_pvpperi)
	pred_pvpperi_tst = model_pvpperi.predict(X_pvpperi_tst)

	pred_t2 = model_t2.predict(X_t2)
	pred_t2_tst = model_t2.predict(X_t2_tst)

	pred_t2peri = model_t2peri.predict(X_t2peri)
	pred_t2peri_tst = model_t2peri.predict(X_t2peri_tst)


	#########################
	### Feature Importance
	#########################

	# pdb.set_trace()

	#DWI-peri
	f_dwiperi = model_dwiperi.feature_importances_

	#HBP
	f_hbp = []
	for j in range(X_hbp_tst.shape[1]):
		f_j = get_feature_importance(model_hbp, X_hbp_tst, y_tst, pred_hbp_tst, j, 100)
		f_hbp.append(f_j)

	#HBP-peri
	f_hbpperi = []
	for j in range(X_hbpperi_tst.shape[1]):
		f_j = get_feature_importance(model_hbpperi, X_hbpperi_tst, y_tst, pred_hbpperi_tst, j, 100)
		f_hbpperi.append(f_j)

	# pdb.set_trace()

	df_featsimp_dwiperi = pd.DataFrame({'feats':dwiperi_feats_names, 'feat_importance':f_dwiperi})
	df_featsimp_hbp = pd.DataFrame({'feats':hbp_feats_names, 'feat_importance':f_hbp})
	df_featsimp_hbpperi = pd.DataFrame({'feats':hbpperi_feats_names, 'feat_importance':f_hbpperi})


	#########################
	### Prepare Ensemble data
	#########################

	# X_ens = np.vstack((pred_hbp, pred_t2peri))
	X_ens = np.vstack((pred_dwiperi, pred_hbp, pred_hbpperi))
	X_ens = X_ens.T
	X_ens_tst = np.vstack((pred_dwiperi_tst, pred_hbp_tst, pred_hbpperi_tst))
	X_ens_tst = X_ens_tst.T


	#####################
	### Voting strategy
	#####################
	
	vote_cls = Ensemble_Vote()
	auc_dir_out = main_dirout+test_name+'_vote.png'
	best_model = bml.roc_ic(X_ens, y, vote_cls, X_ens_tst, y_tst,odir=auc_dir_out, N=10, IC=0.95)

	# pdb.set_trace()

	preds = (np.sum(X_ens_tst, axis=1) > np.shape(X_ens)[1]/2).astype(int)
	TP, TN, FP, FN = bml.confusion_matrix(preds, y_tst)
	AUC = roc_auc_score(y_tst, preds)
	res = bml.roc_cis(y_tst, preds)
	print('Vote AUC: '+str(AUC))
	print('CI: ' + str(res[0]) + ' - ' + str(res[1]))
	print('Vote Sens: ' + str(TP/(TP+FN)))
	print('Vote Spec: ' + str(TN/(TN+FP)))
	print('Vote PPV: ' + str(TP/(TP+FP)))
	print('Vote NPV: ' + str(TN/(TN+FN)))

	auc_dir_out_tst = main_tst_dirout+test_name+'_vote.png'
	# bml.gen_roc_curve(auc_dir_out_tst, preds, y_tst)

	pdb.set_trace()

	#####################
	### MLP ensemble classifier
	#####################

	mlp_cls = MLPClassifier(random_state=3, max_iter=1000, hidden_layer_sizes=(16, 16, 16), alpha=0.001, batch_size=8)
	auc_dir_out = main_dirout+test_name+'_mlp.png'
	best_model = bml.roc_ic(X_ens, y, X_ens_tst, y_tst, mlp_cls, odir=auc_dir_out, N=10, IC=0.95)

	#Test set
	preds = best_model.predict(X_ens_tst)
	TP, TN, FP, FN = bml.confusion_matrix(preds, y_tst)
	AUC = roc_auc_score(y_tst, preds)
	print('MLP AUC: '+str(AUC))

	auc_dir_out_tst = main_tst_dirout+test_name+'_mlp.png'
	# bml.gen_roc_curve(auc_dir_out_tst, preds, y_tst)

	# pdb.set_trace()

	# model_dirout = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/' + test_name + '_mlp.joblib'
	# dump(best_model, model_dirout)

	#####################
	### SVM ensemble classifier
	#####################

	svm_cls = svm.SVC(kernel='sigmoid')
	auc_dir_out = main_dirout+test_name+'_svm.png'
	best_model = bml.roc_ic(X_ens, y, X_ens_tst, y_tst, svm_cls, odir=auc_dir_out, N=10, IC=0.95)

	#Test set
	preds = best_model.predict(X_ens_tst)
	TP, TN, FP, FN = bml.confusion_matrix(preds, y_tst)
	AUC = roc_auc_score(y_tst, preds)
	print('SVM AUC: '+str(AUC))

	auc_dir_out_tst = main_tst_dirout+test_name+'_svm.png'
	# bml.gen_roc_curve(auc_dir_out_tst, preds, y_tst)

	# pdb.set_trace()

	# model_dirout = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/' + test_name + '_svm.joblib'
	# dump(best_model, model_dirout)

	#####################
	### Random Forest ensemble classifier
	#####################

	rf_cls = RandomForestClassifier(max_depth=np.floor(np.shape(X_ens)[1]/2), random_state=0)

	auc_dir_out = main_dirout+test_name+'_rf.png'
	best_model = bml.roc_ic(X_ens, y, X_ens_tst, y_tst, rf_cls, odir=auc_dir_out, N=20, IC=0.95)

	#Test set
	preds = best_model.predict(X_ens_tst)
	TP, TN, FP, FN = bml.confusion_matrix(preds, y_tst)
	AUC = roc_auc_score(y_tst, preds)
	print('RF AUC: '+str(AUC))

	auc_dir_out_tst = main_tst_dirout+test_name+'_rf.png'
	# bml.gen_roc_curve(auc_dir_out_tst, preds, y_tst)

	# model_dirout = '/home/fauvev01/Bachir/Radiomics/output/radiomics/models/' + test_name + '_rf.joblib'
	# dump(best_model, model_dirout)

	print(">>> DONE <<<")
	

def get_feature_importance(clf, X_test, y_test, y_pred, j, n):
	s = accuracy_score(y_test, y_pred) # baseline score
	total = 0.0
	for i in range(n):
		perm = np.random.permutation(range(X_test.shape[0]))
		X_test_ = X_test.copy()
		X_test_[:, j] = X_test[perm, j]
		y_pred_ = clf.predict(X_test_)
		s_ij = accuracy_score(y_test, y_pred_)
		total += s_ij
	return s - total / n


#From a list of Table.tsv files it creates a matrix of radiomics data
def extract_radiomics_table_ensemble(lst, sequences):

	dict_out = {}
	dict_out_temp = {}
	labels = []
	dict_feat_names = {}
	start = 0
	end = 0
	feat_names_done = 0
	subj_temp = ''

	for seq in sequences:
		dict_out[seq] = []
		dict_out_temp[seq] = []
		dict_feat_names[seq] = []

	for i in lst:

		try:
			dir_temp = i[1].split('/')
			seq = dir_temp[-2]
			pre_post = dir_temp[-3]
			subj = i[0]

			if not seq in sequences:
				continue

			temp = pd.read_csv(i[1], sep='\t')
			temp = temp.iloc[37:].values

			rows, cols = temp.shape
			range_val = sequences.index(seq)

			if pre_post == 'pre-treatment':
				pre_post = 'pre'
			elif pre_post == 'post treatment':
				pre_post = 'post'
			else:
				pdb.set_trace()

			#Checks if it is the same subject
			if subj != subj_temp:
				subj_temp = subj
				checklst = [0]*len(sequences)
				start = 1

			if start:
				#Empty sample
				checklst[sequences.index(seq)] = 1
				sample = temp[:,3].astype(float).tolist()
				
				dict_out_temp[seq] = sample

				if feat_names_done == 0:
					feat_names = [pre_post + '_' + seq + '_' + n[0]+'_'+n[1]+'_'+n[2] for n in temp]
					dict_feat_names[seq] = feat_names

				if sum(checklst) == len(sequences):
					end = 1

			if end:
				feat_names_done = 1

				labels.append(int(i[2]))
				
				for seq in sequences:
					temp = dict_out[seq]
					temp.append(dict_out_temp[seq])
					dict_out[seq] = temp

				end = 0


		except:
			continue

	return dict_out, labels, dict_feat_names

class Ensemble_Vote():

	def __init__(self):
		return

	def fit(self, X, y):
		return

	def predict(self, X):
		preds = (np.sum(X, axis=1) > np.shape(X)[1]/2).astype(int)
		return preds


if __name__ == "__main__":
	main()