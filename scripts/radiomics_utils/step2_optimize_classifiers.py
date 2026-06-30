import pickle
import argparse
import pandas as pd
import os
import json
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

### Model optimization ####
# In this project, I consider various classification models like XGBoost, SVM, etc. These have parameters that need to be tuned to find optimal performance,
# with a balance between accuracy and generalizability.

# This step will follow after extracting the most stable, predictive radiomics features, which is done in step1_cleanup_extraction.py
# Now this step 1 is usually run for each binary variable, and the extracted features (as well as the trained imputer and scaler) are specific to that specific endpoint variable.
# So this script (step 2) must be re-run every time step 1 is run with a different binary variable.
# The outputs of step1 are stored in a folder of the form "binary_endpoint/train_test_split/binwidth/mr_sequences/selected_features"
# All we need are the features selected and the binary outcome in the development and test cohots:
    # X_dev, y_dev.  (test set will not be touched)
# These will be generated and stored as csv's in step 1.


# Load results from step 1.
def load_preprocessed_data(in_dir):
    print(f"\nLoading preprocessed data from: {in_dir}")
    
    X_dev = pd.read_csv(f"{in_dir}/X_dev_processed.csv", index_col=0)
    y_dev = pd.read_csv(f"{in_dir}/y_dev.csv", index_col=0).squeeze()
    
    print(f"✓ Loaded X_dev: {X_dev.shape}")
    print(f"✓ Loaded y_dev: {len(y_dev)} samples")
   
    return X_dev,y_dev

#Run GridSearchCV for a set of models and save best params/estimators.

# X, y: training data (numpy / pandas)
# model_names: list of model keys among ['svm_rbf','xgboost','logistic']
# out_dir: directory where results and pickled estimators will be saved
# cv: number of CV folds (int) or a cv splitter
# scoring: scoring metric for GridSearchCV
# n_jobs: parallel jobs for GridSearchCV
# save: whether to write results to disk

def perform_grid_search(X, y, model_names=None, out_dir=None, cv=5, scoring=None, n_jobs=1, random_state=42, save=True):
 
    if model_names is None:
        model_names = ['logistic', 'svm_rbf', 'rf', 'xgboost']
    if scoring is None:
        scoring = 'roc_auc'

    os.makedirs(out_dir, exist_ok=True)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []
    for m in model_names:
        print(f"\nRunning grid search for: {m}")
        if m == 'svm_rbf':
            estimator = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=random_state)
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        elif m == 'xgboost':
            estimator = XGBClassifier(use_label_encoder=False, scale_pos_weight=4, eval_metric='logloss', random_state=random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [1, 2, 3],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0]
            }
        elif m == 'logistic':
            estimator = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        else:
            print(f"Skipping unknown model: {m}")
            continue

        gs = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring,
                          cv=cv_splitter, n_jobs=n_jobs, refit=True)

        gs.fit(X, y)
        best_score = gs.best_score_
        best_params = gs.best_params_
        print(f"Best {scoring} for {m}: {best_score:.4f} with params: {best_params}")

        row = {
            'model': m,
            'best_score': best_score,
            'best_params': json.dumps(best_params)
        }
        results.append(row)

        if save:
            # Save best estimator
            with open(os.path.join(out_dir, f"best_estimator_{m}.pkl"), 'wb') as f:
                pickle.dump(gs.best_estimator_, f)

    results_df = pd.DataFrame(results)

    if save:
        csvp = os.path.join(out_dir, 'grid_search_results.csv')
        results_df.to_csv(csvp, index=False)
        print(f"Saved grid search summary to: {csvp}")

    return results_df


def grid_search_from_dir(in_dir, out_dir, model_names, cv, scoring, n_jobs, random_state, save=True):
    if scoring is None:
        scoring = 'roc_auc'
    X_dev,  y_dev = load_preprocessed_data(in_dir)
    return perform_grid_search(X_dev, y_dev, model_names, out_dir, cv, scoring, n_jobs, random_state, save)


#-------------------------------------------------------------------------------------------------------------------------------------#

# --- EXECUTION ---------------------------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Optimize radiomics model using parameter search',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog='''''')
    parser.add_argument('--input-dir', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/test_3_vs_0-2/dev110_test30/binwidth5/pvp_tp_hbp_dwi_b1_adc/clean_partition_select')
    parser.add_argument('--output-dir', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/test_3_vs_0-2/dev110_test30/binwidth5/pvp_tp_hbp_dwi_b1_adc/optimized_params')
    parser.add_argument('--models',nargs='+',default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    print("\n" + "="*100)
    print("RADIOMICS MODEL OPTIMIZER")
    print("="*100)
    print(f"Output directory: {args.output_dir}")
    if args.models:
        print(f"Models: {', '.join(args.models)}")
    print("="*100)
    
    results_df = grid_search_from_dir(
        in_dir=args.input_dir,
        out_dir=args.output_dir,
        model_names = args.models,
        cv=5,
        scoring='roc_auc',n_jobs=1, random_state=42, save=True
    )
    print(results_df)
    
    print(f"{'='*100}")
    print("Optimization Complete")
    print(f"{'='*100}\n")