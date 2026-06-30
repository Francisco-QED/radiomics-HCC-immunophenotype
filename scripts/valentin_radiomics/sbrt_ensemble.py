import os, sys, glob
import numpy as np
import pandas as pd
import pdb
import ast
from joblib import dump, load
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

sys.path.append('/home/fauvev01/BMEII-Functions/')

#Custom modules
import bmeii_io as bio
import bmeii_ml_functions as bml

def main():

    #Sequences: T1 TP, T1 AP2, T1 PVP, T1 PRE, ADC, HBP, HBP Peri-lesion, DWI High B value

    lst_seqs = ['T1 TP', 'T1 AP2', 'T1 PVP', 'T1 PRE', 'HBP', 'HBP Peri-lesion']

    # create 20 random combinations of sequences without repeats

    pre_post_tg = 'both'

    combs = []

    dict_res_auc = {}
    dict_res_sens = {}
    dict_res_spec = {}

    for i in range(2):

        # pdb.set_trace()

        comb = np.random.choice(lst_seqs, 3, replace=False)

        if i == 0:
            comb = ['T1 PRE', 'T1 PVP', 'T1 AP2']
        
        if pre_post_tg == 'both':
            pre_post_lst = np.random.choice(['pre', 'post'], len(comb))

            if i == 0:
                pre_post_lst = ['pre', 'post', 'pre']
            
            # check there is at least one pre and one post
            # while 'pre' not in pre_post_lst or 'post' not in pre_post_lst:
            #     pre_post_lst = np.random.choice(['pre', 'post'], len(comb))
        else:
            pre_post_lst = [pre_post_tg]

        print(str(comb))
        print(str(pre_post_lst))

        while str(comb) in combs:
            comb = np.random.choice(lst_seqs, 3, replace=False)
        combs.append(str(comb))
        
        # sequences = ['T1 PVP', 'T1 PRE']
        test_name = 'ensemble'
        
        aucs, sens, spec = create_ensemble_cls(comb, pre_post_lst, test_name)

        pdb.set_trace()

        dict_res_auc[str(comb)] = aucs
        dict_res_sens[str(comb)] = sens
        dict_res_spec[str(comb)] = spec

    df_auc = pd.DataFrame(data=dict_res_auc)
    df_sens = pd.DataFrame(data=dict_res_sens)
    df_spec = pd.DataFrame(data=dict_res_spec)

    # Invert rows and columns
    df_auc = df_auc.T
    df_sens = df_sens.T
    df_spec = df_spec.T

    # pdb.set_trace()

    if pre_post_tg == 'both':
        
        # Add the pre or post info to the index
        for i in range(len(df_auc.index)):
            temp = df_auc.index[i]
            temp = temp.replace('[', '')
            temp = temp.replace(']', '')
            temp = temp.split("' '")
            temp = [temp[j].replace("'", '') for j in range(len(temp))]
            temp = [pre_post_lst[j] + ' ' + str(temp[j])  for j in range(len(temp))]

            df_auc.index.values[i] = temp
            df_sens.index.values[i] = temp
            df_spec.index.values[i] = temp


    # pdb.set_trace()

    # Put column names
    df_auc.columns = ['MLP', 'SVM', 'RF']
    df_sens.columns = ['MLP', 'SVM', 'RF']
    df_spec.columns = ['MLP', 'SVM', 'RF']

    # Change the font size of the axes
    sns.set(font_scale=0.5)

    # Save the results as heatmap
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1 = sns.heatmap(df_auc, annot=True, cmap='Blues', fmt='.3f', vmin=0.5, vmax=0.9)
    ax1.set_title('AUC')

    plt.savefig('/home/fauvev01/Bachir/SBRT/output/ensemble/auc_'+pre_post_tg+'_results.png')

    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2 = sns.heatmap(df_sens, annot=True, cmap='Blues', fmt='.3f', vmin=0.5, vmax=0.9)
    ax2.set_title('Sensitivity')

    plt.savefig('/home/fauvev01/Bachir/SBRT/output/ensemble/sens_'+pre_post_tg+'_results.png')

    fig3, ax3 = plt.subplots(figsize=(15, 6))
    ax3 = sns.heatmap(df_spec, annot=True, cmap='Blues', fmt='.3f', vmin=0.5, vmax=0.9)
    ax3.set_title('Specificity')

    plt.savefig('/home/fauvev01/Bachir/SBRT/output/ensemble/spec_'+pre_post_tg+'_results.png')

    pdb.set_trace()

    print('>>> Done <<<')


def create_ensemble_cls(sequences, pre_post_tg, test_name):

    # Output variables
    aucs = []
    sens = []
    spec = []

    # Extract the datalists

    if len(pre_post_tg) > 1:
        
        prepost_data_lst_dir = '/home/fauvev01/Bachir/SBRT/data/lists/data_split/pre&post_tr_lst.csv'
        prepost_tst_data_lst_dir = '/home/fauvev01/Bachir/SBRT/data/lists/data_split/pre&post_tst_lst.csv'

        data_lst = bio.read_csv(prepost_data_lst_dir, type='list')
        tst_data_lst = bio.read_csv(prepost_tst_data_lst_dir, type='list')
        
        # Create a random pre post list with the length of sequence
        pre_post_lst = pre_post_tg

    else:

        data_lst_dir = '/home/fauvev01/Bachir/SBRT/data/lists/data_split/'+pre_post_tg+'_tr_lst.csv'
        tst_data_lst_dir = '/home/fauvev01/Bachir/SBRT/data/lists/data_split/'+pre_post_tg+'_tst_lst.csv'

        data_lst = bio.read_csv(data_lst_dir, type='list')
        tst_data_lst = bio.read_csv(tst_data_lst_dir, type='list')

        # pre_post_lst = [pre_post_tg]*len(sequences)
        pre_post_lst = [pre_post_tg]

    # Extract the radiomics table for the combination of sequences
    #Creates the radiomics matrix from list

    X, y, subjs_lst = extract_radiomics_table_multiseq(data_lst, sequences, pre_post_lst)
    X_tst, y_tst, subjs_lst_tst = extract_radiomics_table_multiseq(tst_data_lst, sequences, pre_post_lst)

    # Reshape the data, switch the axes 0 and 1
    X = np.swapaxes(X, 0, 1)
    X_tst = np.swapaxes(X_tst, 0, 1)

    new_X = []
    new_X_tst = []

    # pdb.set_trace()
    
    for i in range(len(sequences)):

        # pdb.set_trace()

        # Load the best feat sel method according to the sequence

        seq = sequences[i]

        if len(pre_post_tg) > 1:
            pre_post = pre_post_lst[i]
        else:
            pre_post = pre_post_tg[0]

        if seq == 'ADC':
            seq_tmp = 'adc'

            if pre_post == 'pre':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/ap/'+pre_post+'_'+seq_tmp+'_best_feats_ap.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'
            elif pre_post == 'post':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/lasso/'+pre_post+'_'+seq_tmp+'_best_feats_lasso.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'
                
        elif seq == 'HBP':
            seq_tmp = 'hbp'

            if pre_post == 'pre':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/lasso/'+pre_post+'_'+seq_tmp+'_best_feats_lasso.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'
            elif pre_post == 'post':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/ap/'+pre_post+'_'+seq_tmp+'_best_feats_ap.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'

        elif seq == 'DWI High B value':
            seq_tmp = 'dwi_high_bv'

            if pre_post == 'pre':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/lasso/'+pre_post+'_'+seq_tmp+'_best_feats_lasso.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'
            elif pre_post == 'post':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/pca/'+pre_post+'_'+seq_tmp+'_best_feats_pca.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'

        elif seq == 'HBP Peri-lesion':
            seq_tmp = 'hbp_peri'

            if pre_post == 'pre':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/lasso/'+pre_post+'_'+seq_tmp+'_best_feats_lasso.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'
            elif pre_post == 'post':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/lasso/'+pre_post+'_'+seq_tmp+'_best_feats_lasso.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'

        elif seq == 'T1 AP2':
            seq_tmp = 't1_ap2'

            if pre_post == 'pre':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/ap/'+pre_post+'_'+seq_tmp+'_best_feats_ap.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'
            elif pre_post == 'post':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/lasso/'+pre_post+'_'+seq_tmp+'_best_feats_lasso.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'

        elif seq == 'T1 PVP':
            seq_tmp = 't1_pvp'

            if pre_post == 'pre':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/ap/'+pre_post+'_'+seq_tmp+'_best_feats_ap.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'
            elif pre_post == 'post':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/lasso/'+pre_post+'_'+seq_tmp+'_best_feats_lasso.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'

        elif seq == 'T1 PRE':
            seq_tmp = 't1_pre'

            if pre_post == 'pre':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/lasso/'+pre_post+'_'+seq_tmp+'_best_feats_lasso.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'
            elif pre_post == 'post':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/pca/'+pre_post+'_'+seq_tmp+'_best_feats_pca.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'

        elif seq == 'T1 TP':
            seq_tmp = 't1_tp'

            if pre_post == 'pre':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/pca/'+pre_post+'_'+seq_tmp+'_best_feats_pca.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'
            elif pre_post == 'post':
                best_feats_dir = '/home/fauvev01/Bachir/SBRT/data/lists/best_feats/ap/'+pre_post+'_'+seq_tmp+'_best_feats_ap.csv'
                means_stds_dir = '/home/fauvev01/Bachir/SBRT/data/lists/scalers/'+pre_post+'_'+seq_tmp+'_means_and_stds.csv'

        # Reads the best features list
        best_feats = bio.read_csv(best_feats_dir, type='list')

        # Reads the means and stds
        means_stds = pd.read_csv(means_stds_dir, header=None)
        means_stds = np.array(bio.read_csv(means_stds_dir, type='list')).astype(float)

        means = means_stds[:,0]
        stds = means_stds[:,1]

        # pdb.set_trace()
        
        # Extract the features corresponding to the series
        X_tmp = X[i]
        X_tst_tmp = X_tst[i]

        #Avoid errors of nan or inf
        X_tmp = np.nan_to_num(X_tmp)
        X_tst_tmp = np.nan_to_num(X_tst_tmp)

	    ####### Standardization ############

        X_tmp = (X_tmp - means) / stds
        X_tst_tmp = (X_tst_tmp - means) / stds
        
        ######## Get only best feats #######

        #Gets only the top feats based on the previous feat selection analysis
        #array of best feats nbs
        arr_best_feats = [int(i[1]) for i in best_feats]

        ##### Taking best feats only
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        X_tmp = X_tmp[:,arr_best_feats]
        X_tst_tmp = X_tst_tmp[:,arr_best_feats]

        ################################
        # Running the best models
        ################################

        if seq == 'ADC':
            if pre_post == 'pre':
                # AP + MLP
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/pre_adc_ap_mlp.joblib'
            elif pre_post == 'post':
                # Lasso + RF
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/post_adc_lasso_rf.joblib'
        elif seq == 'HBP':
            if pre_post == 'pre':
                # Lasso + MLP
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/pre_hbp_lasso_mlp.joblib'
            elif pre_post == 'post':
                # AP + RF
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/post_hbp_ap_rf.joblib'
        elif seq == 'T1 PVP':
            if pre_post == 'pre':
                # AP + MLP
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/pre_t1_pvp_ap_mlp.joblib'
            elif pre_post == 'post':
                # Lasso + MLP
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/post_t1_pvp_lasso_mlp.joblib'
        elif seq == 'T1 PRE':
            if pre_post == 'pre':
                # Lasso + MLP
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/pre_t1_pre_lasso_mlp.joblib'
            elif pre_post == 'post':
                # PCA + MLP
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/post_t1_pre_pca_mlp.joblib'
        elif seq == 'T1 TP':
            if pre_post == 'pre':
                # PCA + SVM
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/pre_t1_tp_pca_svm.joblib'
            elif pre_post == 'post':
                # AP + MLP
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/post_t1_tp_ap_mlp.joblib'
        elif seq == 'T1 AP2':
            if pre_post == 'pre':
                # AP + RF
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/pre_t1_ap2_ap_rf.joblib'
            elif pre_post == 'post':
                # Lasso + MLP
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/post_t1_ap2_lasso_mlp.joblib'
        elif seq == 'HBP Peri-lesion':
            if pre_post == 'pre':
                # Lasso + RF
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/pre_hbp_peri_lasso_rf.joblib'
            elif pre_post == 'post':
                # Lasso + RF
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/post_hbp_peri_lasso_rf.joblib'
        elif seq == 'DWI High B value':
            if pre_post == 'pre':
                # Lasso + SVM
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/pre_dwi_high_bv_lasso_svm.joblib'
            elif pre_post == 'post':
                # PCA + RF
                best_mod_dir = '/home/fauvev01/Bachir/SBRT/output/best_models/post_dwi_high_bv_pca_rf.joblib'

        #Load the best model
        best_mod = joblib.load(best_mod_dir)

        # pdb.set_trace()

        #Predict the labels
        y_pred = best_mod.predict_proba(X_tmp)
        y_pred_tst = best_mod.predict_proba(X_tst_tmp)

        new_X.append(y_pred)
        new_X_tst.append(y_pred_tst)

    # pdb.set_trace()

    #Concatenate the predictions
    new_X = np.concatenate(new_X, axis=1)
    new_X_tst = np.concatenate(new_X_tst, axis=1)


    #####################
    ### MLP classifier
    #####################

    params_grid = {
        'hidden_layer_sizes': [(100, 10), (100, 50, 10), (100, 50, 25, 10)],
        'alpha': [0.001, 0.01],
        'batch_size': [4, 8, 16],
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
    grid_search.fit(new_X, y)

    # print('Best parameters:', grid_search.best_params_)
    # print('Best score:', grid_search.best_score_)

    grid_preds = grid_search.predict(new_X_tst)
    grid_auc = roc_auc_score(y_tst, grid_preds)

    # pdb.set_trace()

    best_model = grid_search
    AUC_mlp = grid_auc

    best_params_mlp = grid_search.best_params_

    preds = best_model.predict(new_X_tst)

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
    best_params_dir_out = '/home/fauvev01/Bachir/SBRT/output/ensemble/'+test_name+'_mlp.txt'

    best_params_mlp['auc'] = AUC_mlp
    best_params_mlp['auc_ic'] = [auc_ic[0], auc_ic[1]]
    best_params_mlp['sens'] = TP_mlp/(TP_mlp+FN_mlp)
    best_params_mlp['sens_ic'] = [auc_ic[2], auc_ic[3]]
    best_params_mlp['spec'] = TN_mlp/(TN_mlp+FP_mlp) 
    best_params_mlp['spec_ic'] = [auc_ic[4], auc_ic[5]]

    # with open(best_params_dir_out, 'w') as f:
    #     best_params_str = str(best_params_mlp)
    #     f.write(best_params_str)

    # pdb.set_trace()

    a = 0

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
    grid_search.fit(new_X, y)

    preds_grid = grid_search.best_estimator_.predict(new_X_tst)
    grid_auc = roc_auc_score(y_tst, preds_grid)

    best_model = grid_search.best_estimator_
    best_params_svm = grid_search.best_params_

    #Test set
    preds = best_model.predict(new_X_tst)
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
    #     best_params_str = str(best_params_svm)
    #     f.write(best_params_str)

    # auc_dir_out_tst = '/home/fauvev01/Bachir/SBRT/output/svm/'+test_name+'_svm.png'
    # bml.gen_roc_curve(auc_dir_out_tst, preds, y_tst)

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
    grid_search.fit(new_X, y)

    # Print the best parameters and score
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    # Predict on the test data using the best model

    best_model = grid_search.best_estimator_
    best_params_rf = grid_search.best_params_

    preds_grid = best_model.predict(new_X_tst)
    auc_grid = roc_auc_score(y_tst, preds_grid)

    # auc_dir_out = '/home/fauvev01/Bachir/SBRT/output/rf/'+test_name+'_rf.png'
    # best_model_bml = bml.roc_ic(X, y, rf_cls, odir='', N=20, IC=0.95, model_out='best')

    preds_bml = best_model.predict(new_X_tst)
    auc_bml = roc_auc_score(y_tst, preds_bml)


    # pdb.set_trace()

    #Test set
    preds = best_model.predict(new_X_tst)
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

    pdb.set_trace()

    # Output
    auc_out = [AUC_mlp, AUC_svm, AUC_rf]
    sens_out = [TP_mlp/(TP_mlp+FN_mlp), TP_svm/(TP_svm+FN_svm), TP_rf/(TP_rf+FN_rf)]
    spec_out = [TN_mlp/(TN_mlp+FP_mlp), TN_svm/(TN_svm+FP_svm), TN_rf/(TN_rf+FP_rf)]

    return auc_out, sens_out, spec_out

    
#From a list of Table.tsv files it creates a matrix of radiomics data
def extract_radiomics_table_multiseq(lst, sequences, pre_post_lst):

    out = []
    labels = []
    subjs_lst = []

    # pdb.set_trace()

    # Find unique patients in list
    unique_subjs = np.unique([i[0] for i in lst])

    for i in unique_subjs:

        if len(pre_post_lst) > 1:
            # Select all the sequences for a given patient
            seqs_subj = [j for j in lst if j[0] == i]

            # Check if the target sequences are present for the patient
            seqs_subj = [j for j in seqs_subj if j[1] in sequences]

            seqs_sub_out = []

            # Use the pre_post list to select the sequences
            for pp in range(len(pre_post_lst)):

                pre_post_temp = pre_post_lst[pp]
                seq_temp = sequences[pp]

                if pre_post_temp == 'pre':
                    seqs_subj_new = [j for j in seqs_subj if j[2].split('/')[-3] == 'Pre-Tx']
                    seqs_subj_new = [j for j in seqs_subj_new if j[1] == seq_temp]
                elif pre_post_temp == 'post':
                    seqs_subj_new = [j for j in seqs_subj if j[2].split('/')[-3] == 'Post Tx']
                    seqs_subj_new = [j for j in seqs_subj_new if j[1] == seq_temp]

                if len(seqs_subj_new) == 0:
                    break
                elif len(seqs_subj_new) > 1:
                    seqs_subj_new = seqs_subj_new[0]

                seqs_sub_out.append(list(np.squeeze(seqs_subj_new)))

            # pdb.set_trace()

            if len(seqs_sub_out) != len(sequences):
                continue
            else:
                seqs_subj = seqs_sub_out
        
        else:
            # Select all the sequences for a given patient
            seqs_subj = [j for j in lst if j[0] == i]

            # Check if the target sequences are present for the patient
            seqs_subj = [j for j in seqs_subj if j[1] in sequences]

            # If not, skip the patient
            if len(seqs_subj) != len(sequences):
                continue

        sample = []

        counter = 0

        # Get the label for that patient
        try:
            label_tmp = seqs_subj[0][3]
        except:
            pdb.set_trace()

        # pdb.set_trace()

        for m in range(len(seqs_subj)):

            seq_to_get = sequences[m]

            # Find the sequence in the list
            try:
                seq = [k for k in seqs_subj if k[1] == seq_to_get]
            except:
                pdb.set_trace()

            try:
                seq = seq[0]
            except:
                continue

            subj = seq[0]
            dir_temp = seq[2].split('/')
            seqq = seq[1]
            pre_post = dir_temp[-3]

            # pdb.set_trace()

            # Read the csv file from line 37
            # Header is: 'Radiomics Name' 'Value'
            # Separated by semicolon
            try:
                temp = pd.read_csv(seq[2], header=None, skiprows=18, sep=';')
            except:
                break
            

            if pre_post == 'Pre-Tx':
                pre_post = 'pre'
            elif pre_post == 'Post Tx':
                pre_post = 'post'
            else:
                pdb.set_trace()

            rows, cols = temp.shape

            if rows != 108:
                break

            # Fill the sample with the features
            sample.append(temp[1].astype(float).tolist())

            counter += 1

        if counter != len(sequences):
            continue
        else:
            out.append(sample)
            subjs_lst.append(subj)
            labels.append(int(label_tmp))

    return out, labels, subjs_lst


if __name__ == "__main__":
    main()