import sys, os
import numpy as np
import pdb
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.cluster import AffinityPropagation
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/fauvev01/BMEII-Functions/')

#Custom modules
import bmeii_io as bio


##################################################

#Main function
def main():
	data_lst_dir = '/home/fauvev01/Bachir/SBRT/data/lists/data_split/pre_tr_lst.csv'

	test_name = 'pre_t1_tp-t1_ap2'

	#Pvalue threshold. For pca it would take all the components with a pvalue equal or lower to P_thres.
	#For univariate analysis it would take all feautures with a pvalue equal or lower to P_thres.
	P_thres = 0.05

	#Extract the top N features
	N = 20

	#Reads the list
	data_lst = bio.read_csv(data_lst_dir, type='list')
	# tst_data_lst = bio.read_csv(tst_data_lst_dir, type='list')

	#Creates the radiomics matrix from list
	#Sequences to extract
	X, y, feat_names, subjs_lst = extract_radiomics_table_multiseq(data_lst, ['T1 TP', 'T1 AP2'])

	pdb.set_trace()

	# # Checks if all samples in X have the same number of features
	# n_feats = [len(x) for x in X]

	#Scaling the data 
	#Z-score
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	XT = np.ascontiguousarray(X.T)

	means = scaler.mean_
	stds = scaler.scale_

	#Saves means and stds for all features
	means_stds = np.stack((means, stds)).T
	bio.write_csv('/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+test_name+'_means_and_stds.csv', means_stds, ['Means','Stds'])

	
	##################################################
	#Affinity Propagation
	##################################################

	clustering = AffinityPropagation(random_state=3, damping = 0.9, max_iter = 1000).fit(XT)

	gps = clustering.predict(XT)
	clust_inds = clustering.cluster_centers_indices_

	feat_names_out = np.array(feat_names)[clust_inds]

	ap_lst_feats = pd.DataFrame({'BestFeats':feat_names_out, 'FeatNb':clust_inds})

	#pdb.set_trace()

	#Saves the best feats names
	bio.write_csv('/home/fauvev01/Bachir/SBRT/data/lists/best_feats/ap/'+test_name+'_best_feats_ap.csv', ap_lst_feats, ['BestFeats', 'FeatNb'])

	#pdb.set_trace()

	#Univariate Analysis
	#TODO

	##################################################
	#PCA
	##################################################
	
	#getting the number of features and samples
	n_samps, n_feats = np.shape(X) 

	#Max number of components must be the min between number of samples and number of features
	max_n_comps = min(n_samps, n_feats)

	pca = PCA(n_components=max_n_comps)
	princ_comps = pca.fit(X)

	all_comps = princ_comps.components_

	var_ratio = pca.explained_variance_ratio_

	n_components_retained = np.sum(var_ratio > 0.5)

	#Performs an ANOVA analysis for each one of the components to check the top features that better separate the two groups
	v_norm = np.sqrt(np.sum(all_comps**2, axis=1))
	X_proj = (np.dot(X, all_comps.T)/v_norm)
	Fs, ps = f_classif(X_proj, y)

	comps_n = range(len(ps))

	df = pd.DataFrame()
	df['PCnb'] = comps_n
	df['Pvalues'] = ps

	#Sorting all pvalues to determine the principal components that better separate the two groups
	df = df.sort_values(by='Pvalues')

	#Extracting the top N principal components based on the pvalue threshold 
	df_best = df.loc[(df['Pvalues'] <= P_thres)]

	# pdb.set_trace()

	#Based on the best components, we quantify the importance of each feature for all components and we extract the top N features
	#to train the predictive models

	importance_sum = np.zeros(n_feats)

	for c in df_best.values:

		nb_temp = int(c[0])
		comp_temp = all_comps[nb_temp] 

		importance_sum += abs(comp_temp)

	df_best_feats = pd.DataFrame()
	df_best_feats['BestFeats'] = feat_names
	df_best_feats['FeatNb'] = range(len(feat_names))
	df_best_feats['Importance'] = importance_sum

	df_best_feats = df_best_feats.sort_values(by='Importance', ascending=False)
	df_best_feats = df_best_feats[:N]

	#Plots the pvalues vs pc_nb
	# df_best_feats.plot(x='FeatName',y='Importance',kind='bar',figsize=(15,8)) 
	# plt.savefig('/home/fauvev01/Bachir/Radiomics/output/radiomics/pca/'+test_name+'_Best_Feats.png')

	out_lst_feats = [[f[0], f[1]] for f in df_best_feats.values]

	#Saves the best feats names
	bio.write_csv('/home/fauvev01/Bachir/SBRT/data/lists/best_feats/pca/'+test_name+'_best_feats_pca.csv', out_lst_feats, ['BestFeats', 'FeatNb'])

	##################################################
	#LASSO
	##################################################

	# pdb.set_trace()

	sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
	sel_.fit(X, y)

	#Get the values and indexes of the coefficients that are not null
	coef = sel_.estimator_.coef_

	#Get the indexes of the features that are not null
	coef_inds = np.nonzero(coef)[1]
	lasso_feats = np.array(feat_names)[coef_inds]

	lasso_lst_feats = pd.DataFrame({'BestFeats':lasso_feats, 'FeatNb':coef_inds})

	#Saves the best feats names
	bio.write_csv('/home/fauvev01/Bachir/SBRT/data/lists/best_feats/lasso/'+test_name+'_best_feats_lasso.csv', lasso_lst_feats, ['BestFeats', 'FeatNb'])


	# Create a heatmap table with all feature names in columns and rows correspond to the best features selected by the LASSO, PCA and Affinity Propagation
	dict_heatmap = {}

	dict_heatmap['AP'] = [0]*len(feat_names)
	dict_heatmap['PCA'] = [0]*len(feat_names)
	dict_heatmap['LASSO'] = [0]*len(feat_names)

	#pdb.set_trace()

	out_lst_feats_temp = np.array(out_lst_feats)

	for i in range(len(feat_names)):
		
		feat = feat_names[i]

		if feat in lasso_feats:
			dict_heatmap['LASSO'][i] = 1

		if feat in feat_names_out:
			dict_heatmap['AP'][i] = 1

		if feat in list(out_lst_feats_temp[:,0]):
			dict_heatmap['PCA'][i] = 1


	# Features heatmap

	# data = pd.DataFrame(dict_heatmap, index=feat_names)
	# fig, ax = plt.subplots(figsize=(18, 14))
	# sns.heatmap(data, ax=ax, yticklabels=data.index, cmap="YlGnBu")

	# plt.tight_layout()

	# ax.figure.savefig('/home/fauvev01/Bachir/SBRT/output/feat_sel_heatmaps/'+test_name+'_heatmap.jpg')

	#Avoid that the names of the features are cut
	# plt.tight_layout()

	# pdb.set_trace()

	print(">>> DONE <<<")






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

				if len(sample) != 216:
					continue

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


if __name__ == '__main__':
	main()