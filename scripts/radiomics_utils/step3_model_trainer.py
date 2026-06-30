# Loads preprocessed data and trains multiple models without rerunning feature selection.
# This enables fast iteration over different model architectures.

#Usage:
 #   python model_trainer.py [--output-dir PATH] [--models model1 model2 ...]
    
#Examples:
 #   python model_trainer.py  # Uses default output directory, trains all models
 #   python model_trainer.py --models logistic svm rf  # Train specific models only
 #   python model_trainer.py --output-dir /custom/path --models logistic xgboost

import os
import shap
import pickle
import argparse
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

def load_preprocessed_data(in_dir):
    """Load all preprocessed artifacts saved by main_orchestrator2.py"""
    print(f"\nLoading preprocessed data from: {in_dir}")
    
    X_dev = pd.read_csv(f"{in_dir}/X_dev_processed.csv", index_col=0)
    X_lock = pd.read_csv(f"{in_dir}/X_lock_processed.csv", index_col=0)
    y_dev = pd.read_csv(f"{in_dir}/y_dev.csv", index_col=0).squeeze()
    y_lock = pd.read_csv(f"{in_dir}/y_lock.csv", index_col=0).squeeze()

    print(f"✓ Loaded X_dev: {X_dev.shape}")
    print(f"✓ Loaded X_lock: {X_lock.shape}")
    print(f"✓ Loaded y_dev: {len(y_dev)} samples")
    print(f"✓ Loaded y_lock: {len(y_lock)} samples")
    
    return X_dev, X_lock, y_dev, y_lock

def load_saved_estimator(model_name, input_dir):
    """Load a pickled best estimator for the given model from the input directory."""
    if input_dir is None:
        return None

    estimator_path = os.path.join(input_dir, f"best_estimator_{model_name}.pkl")
    if os.path.exists(estimator_path):
        print(f"Loading saved estimator for {model_name} from: {estimator_path}")
        with open(estimator_path, 'rb') as f:
            return pickle.load(f)
    return None

# Compute bootstrap confidence intervals for a binary classification metric.
# y_true : array-like            True binary labels
# y_pred_proba : array-like      Predicted probabilities
# metric_fn : callable           Function to compute metric from (y_true, y_pred_proba)
# n_bootstrap : int              Number of bootstrap iterations
# ci : float                     Confidence interval (e.g., 95)
# seed : int                     Random seed
    
def bootstrap_ci(y_true, y_pred_proba, metric_fn, n_bootstrap=1000, ci=95, seed=42):
    import warnings
    np.random.seed(seed)
    bootstrap_scores = []
    n_samples = len(y_true)
    n_valid = 0
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true.iloc[indices].values if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_boot = y_pred_proba[indices] if isinstance(y_pred_proba, np.ndarray) else y_pred_proba.iloc[indices].values
        
        # Compute metric, suppressing warnings for undefined metrics (e.g., AUC with only 1 class)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = metric_fn(y_true_boot, y_pred_boot)
                # Only keep valid (non-NaN) scores
                if not np.isnan(score):
                    bootstrap_scores.append(score)
                    n_valid += 1
        except:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # If we have valid bootstrap scores, use them; otherwise return NaN
    if len(bootstrap_scores) > 0:
        point_estimate = metric_fn(y_true.values if hasattr(y_true, 'values') else y_true, y_pred_proba.values if hasattr(y_pred_proba, 'values') else y_pred_proba)
        
        # Warn if very few valid bootstrap samples (indicates small test set)
        if n_valid < n_bootstrap * 0.5:
            print(f"    ⚠ Warning: Only {n_valid}/{n_bootstrap} bootstrap samples were valid.")
            print(f"      (Small test set may have bootstrap samples with only one class)")
        
        alpha = (100 - ci) / 2
        ci_lower = np.percentile(bootstrap_scores, alpha)
        ci_upper = np.percentile(bootstrap_scores, 100 - alpha)
    else:
        point_estimate = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
    
    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_scores': bootstrap_scores,
        'n_valid': n_valid
    }

# Compute sensitivity and specificity for binary classification.
# y_true :         True binary labels
# y_pred_proba :   Predicted probabilities
# threshold :      Threshold for converting probabilities to binary predictions
def compute_sensitivity(y_true, y_pred_proba, threshold=0.3):
    """Compute sensitivity (recall for positive class)."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0
def compute_specificity(y_true, y_pred_proba, threshold=0.3):
    """Compute specificity (recall for negative class)."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# Create a SHAP explainer based on the model type and data.
def create_shap_explainer(model, X, model_name=None):
    model_type = (model_name or type(model).__name__).lower()
    # Group Random Forest with XGBoost under TreeExplainer
    if any(keyword in model_type for keyword in ['xgb', 'xgboost', 'booster', 'rf', 'randomforest']):
        return shap.TreeExplainer(model)
    if 'logistic' in model_type or ('linear' in model_type and hasattr(model, 'coef_')):
        # LinearExplainer works best with raw numpy arrays for background data
        return shap.LinearExplainer(model, X.values if hasattr(X, 'values') else X)
    if 'svm' in model_type or 'svc' in model_type:
        if hasattr(model, 'coef_') and getattr(model, 'kernel', None) == 'linear':
            return shap.LinearExplainer(model, X.values if hasattr(X, 'values') else X)    
        # Convert background sample to a simple matrix format to avoid predict_proba format crashes
        bg_sample = shap.sample(X, min(50, len(X)))
        bg_data = bg_sample.values if hasattr(bg_sample, 'values') else bg_sample
        return shap.KernelExplainer(model.predict_proba, bg_data)

    # Fallback to generic explainer
    return shap.Explainer(model, X)

def plot_shap_summary(model, X, out_dir, model_name, plot_type='bar'):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    explainer = create_shap_explainer(model, X, model_name=model_name)
    shap_values = explainer(X)

    if isinstance(X, np.ndarray):
        X_plot = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    else:
        X_plot = X

    # Initialize the figure size
    fig = plt.figure(figsize=(14, 10))
    
    # Draw SHAP plot (mean absolute value for bar plot)
    shap.summary_plot(shap_values, X_plot, plot_type=plot_type, show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP value", fontsize=12, labelpad=12)

    # Handle the multi-line wrapping for your radiomics features on the Y-axis
    labels = [item.get_text() for item in ax.get_yticklabels()]
    wrapped_labels = []
    for label in labels:
        if "_3D_" in label:
            wrapped_labels.append(label.replace("_3D_", "_\n3D_"))
        elif "wavelet-" in label:
            wrapped_labels.append(label.replace("wavelet-", "wavelet-\n"))
        elif "original_" in label:
            wrapped_labels.append(label.replace("original_", "original_\n"))
        else:
            wrapped_labels.append('\n'.join(textwrap.wrap(label, width=28)))
            
    ax.set_yticklabels(wrapped_labels, fontsize=10, ha='right', va='center')

    # Set manual layout bounds so nothing is truncated on the margins
    plt.subplots_adjust(left=0.45, right=0.95, top=0.95, bottom=0.15)
    
    # Save to your output directory
    filename = f'{model_name}_shap_summary_{plot_type}.png'
    plot_path = os.path.join(out_dir, filename)
    
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path

def create_roc_comparison_plot(results_list, out_dir, filename='model_roc_comparison.png'):
    plt.figure(figsize=(10, 8))
    sns.set_style('whitegrid')

    plotted = 0
    for result in results_list:
        if result.get('status') != 'success':
            continue
        fpr = result.get('fpr')
        tpr = result.get('tpr')
        auc = result.get('lock_auc')
        model_name = result.get('model_name', 'model').upper()

        if fpr is None or tpr is None or np.isnan(auc):
            continue

        plt.plot(fpr, tpr,
                 linewidth=2.5,
                 label=f"{model_name} (AUC={auc:.3f})")
        plotted += 1

    if plotted == 0:
        print("⚠ No valid ROC curves to plot")
        return

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Comparison of Models on Lock Set', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    plot_path = f"{out_dir}/{filename}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ ROC comparison plot saved to: {plot_path}")
    return plot_path


#  Train a single model and return comprehensive metrics with confidence intervals
#  X_dev :           Development set features (imputed and scaled)
#  y_dev :           Development set labels
#  X_lock :          Lock set features (imputed and scaled)
#  y_lock :          Lock set labels
#  model_name :      Name of model to train (logistic, svm, rf, xgboost, tabpfn)
#  out_dir :         Directory where result plots are saved
#  random_state :    Random state for reproducibility
#  n_bootstrap :     Number of bootstrap iterations for CI estimation
#  generate_shap :   Whether to compute and save SHAP summary plots for the trained model
     
def train_and_evaluate_model(X_dev, y_dev, X_lock, y_lock, model_name, out_dir, param_dir, random_state=42, n_bootstrap=1000, generate_shap=False):
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load a saved estimator first, otherwise build and fit a new one
        model = load_saved_estimator(model_name, param_dir)
        if model is not None:
            print(f"Using pickled model for {model_name} from param_dir.")
        else:
            model = get_model(model_name, random_state=random_state)
            model.fit(X_dev, y_dev)
        
        # Get predictions on lock set
        y_pred_proba = model.predict_proba(X_lock)[:, 1]  ######### Warning: smotimes I cahgne to X_dev to check on development set ###########

        TARGET_THRESHOLD = 0.25
        y_pred = (y_pred_proba >= TARGET_THRESHOLD).astype(int)
        #y_pred = model.predict(X_lock)
        
        # Calculate point estimates
        auc = roc_auc_score(y_lock, y_pred_proba)        ########### see warning above
        print(f"DEBUG DEV AUC: {auc}")
        fpr, tpr, thresholds = roc_curve(y_lock, y_pred_proba)
        
        # Confusion matrix
        sensitivity = compute_sensitivity(y_lock, y_pred_proba, threshold=TARGET_THRESHOLD)
        specificity = compute_specificity(y_lock, y_pred_proba, threshold=TARGET_THRESHOLD)
        
        # Bootstrap confidence intervals
        print(f"  Computing bootstrap CIs ({n_bootstrap} iterations)...")
        auc_ci = bootstrap_ci(y_lock, y_pred_proba, roc_auc_score, n_bootstrap=n_bootstrap)
        #sens_ci = bootstrap_ci(y_lock, y_pred_proba, compute_sensitivity, n_bootstrap=n_bootstrap)
        #spec_ci = bootstrap_ci(y_lock, y_pred_proba, compute_specificity, n_bootstrap=n_bootstrap)

        # Use lambda wrappers to force the bootstrap iterations to calculate using the exact same threshold
        sens_ci = bootstrap_ci(
            y_lock, 
            y_pred_proba, 
            lambda y_true, y_prob: compute_sensitivity(y_true, y_prob, threshold=TARGET_THRESHOLD), 
            n_bootstrap=n_bootstrap
        )
        spec_ci = bootstrap_ci(
            y_lock, 
            y_pred_proba, 
            lambda y_true, y_prob: compute_specificity(y_true, y_prob, threshold=TARGET_THRESHOLD), 
            n_bootstrap=n_bootstrap
        )


        # Format CI strings, handling NaN cases
        def format_metric(val, ci_lower, ci_upper):
            if np.isnan(val) or np.isnan(ci_lower) or np.isnan(ci_upper):
                return f"{val:.2f}", "(95% CI: N/A)"
            else:
                return f"{val:.2f}", f"(95% CI: {ci_lower:.2f} - {ci_upper:.2f})"
        
        auc_str, auc_ci_str = format_metric(auc, auc_ci['ci_lower'], auc_ci['ci_upper'])
        sens_str, sens_ci_str = format_metric(sensitivity, sens_ci['ci_lower'], sens_ci['ci_upper'])
        spec_str, spec_ci_str = format_metric(specificity, spec_ci['ci_lower'], spec_ci['ci_upper'])
        
        results = {
            'model_name': model_name,
            'model': model,
            'lock_auc': auc,
            'auc_ci_lower': auc_ci['ci_lower'],
            'auc_ci_upper': auc_ci['ci_upper'],
            'sensitivity': sensitivity,
            'sens_ci_lower': sens_ci['ci_lower'],
            'sens_ci_upper': sens_ci['ci_upper'],
            'specificity': specificity,
            'spec_ci_lower': spec_ci['ci_lower'],
            'spec_ci_upper': spec_ci['ci_upper'],
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'fpr': fpr,
            'tpr': tpr,
            'n_dev_samples': len(X_dev),
            'n_lock_samples': len(X_lock),
            'status': 'success'
        }
        
        print(f"✓ Lock Set AUC:       {auc_str} {auc_ci_str}")
        print(f"✓ Sensitivity:        {sens_str} {sens_ci_str}")
        print(f"✓ Specificity:        {spec_str} {spec_ci_str}")
        print(f"  Dev Samples:        {len(X_dev)}")
        print(f"  Lock Samples:       {len(X_lock)}")

        if generate_shap:
            try:
                shap_plot_path = plot_shap_summary(model, X_dev, out_dir, model_name, plot_type='bar')
                results['shap_plot'] = shap_plot_path
                print(f"✓ SHAP summary plot saved to: {shap_plot_path}")
            except Exception as exc:
                print(f"⚠ SHAP plot skipped: {exc}")
        return results
        
    except Exception as e:
        print(f"✗ Error training {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }
    
# Train multiple models on preprocessed data and return comparison with CIs.
# in_dir : Directory containing saved preprocessing artifacts
# model_list : List of model names to train. If None, trains all available models.
# n_bootstrap : Number of bootstrap iterations for CI estimation

def train_multiple_models(in_dir, param_dir, out_dir, model_list=None, n_bootstrap=1000, generate_shap=True):
    if model_list is None:
        model_list = ['logistic', 'svm_linear','svm_rbf', 'rf', 'xgboost']
    
    # Load preprocessed data
    X_dev, X_lock, y_dev, y_lock = load_preprocessed_data(in_dir)
    
    # Train models
    results_list = []
    formatted_results = []
    model_objects = {}
    
    for model_name in model_list:
        result = train_and_evaluate_model(
            X_dev,
            y_dev,
            X_lock,
            y_lock,
            model_name,
            out_dir,
            param_dir=param_dir,
            n_bootstrap=n_bootstrap,
            generate_shap=generate_shap
        )
        results_list.append(result)  # Store full result dict for plotting
        
        if result['status'] == 'success':
            formatted_results.append({
                'Model': model_name,
                'AUC': f"{result['lock_auc']:.2f}",
                'AUC 95% CI': f"({result['auc_ci_lower']:.2f}, {result['auc_ci_upper']:.2f})",
                'Sensitivity': f"{result['sensitivity']:.2f}",
                'Sens 95% CI': f"({result['sens_ci_lower']:.2f}, {result['sens_ci_upper']:.2f})",
                'Specificity': f"{result['specificity']:.2f}",
                'Spec 95% CI': f"({result['spec_ci_lower']:.2f}, {result['spec_ci_upper']:.2f})",
                'N Dev': result['n_dev_samples'],
                'N Lock': result['n_lock_samples']
            })
            model_objects[model_name] = result['model']
        else:
            formatted_results.append({
                'Model': model_name,
                'AUC': '—',
                'AUC 95% CI': '—',
                'Sensitivity': '—',
                'Sens 95% CI': '—',
                'Specificity': '—',
                'Spec 95% CI': '—',
                'Error': result.get('error', 'Unknown'),
                'N Dev': '—',
                'N Lock': '—'
            })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(formatted_results)
    
    # Create ROC comparison plot
    create_roc_comparison_plot(results_list, out_dir)
    
    # Print summary
    print(f"\n{'='*100}")
    print("MODEL COMPARISON SUMMARY (with 95% Bootstrap Confidence Intervals)")
    print(f"{'='*100}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*100}\n")
    
    # Save results
    comparison_df.to_csv(f"{out_dir}/model_comparison.csv", index=False)
    print(f"✓ Comparison saved to: {out_dir}/model_comparison.csv")
    
    # Save best model (by AUC)
    successful_results = [r for r in formatted_results if 'N Dev' in r and isinstance(r['N Dev'], int)]
    if len(successful_results) > 0:
        # Parse AUC values from string
        aucs = {}
        for i, r in enumerate(formatted_results):
            if 'N Dev' in r and isinstance(r['N Dev'], int):
                auc_str = r['AUC']
                aucs[i] = float(auc_str)
        
        best_idx = max(aucs, key=aucs.get)
        best_model_name = formatted_results[best_idx]['Model']
        best_model = model_objects[best_model_name]
        
        with open(f"{out_dir}/best_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        
        best_auc = aucs[best_idx]
        print(f"✓ Best model ({best_model_name}) saved: {best_auc:.2f} AUC\n")
    
    return comparison_df, model_objects

#==============================================================================================#
#######--------------------------------EXECUTION-------------------------------#######
#==============================================================================================#

def get_model(model_name, random_state=42):
    if model_name == 'logistic':
        return LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', max_iter=1000)
    elif model_name == 'svm_linear':
        # Probability=True is required for ROC-AUC
        return SVC(kernel='linear', probability=True, random_state=random_state)
    elif model_name == 'svm_rbf':
        # Requires tuning C (regularization) and gamma (boundary flexibility)
        return SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)
    elif model_name == 'rf':
        return RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    elif model_name == 'xgboost':
        # Low depth to prevent overfitting on N=89
        return XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.01, subsample = 0.6,
                             use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    elif model_name == 'tabpfn':
        # TabPFN is typically used as a classifier without much tuning
        return TabPFNClassifier(device='cpu')
    else:
        raise ValueError(f"Model {model_name} not recognized.")

# Define all the arguments for the script, including output directory, models to train, bootstrap iterations, and SHAP computation.
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train multiple models on preprocessed radiomics data with bootstrap CI.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog='''''')
    parser.add_argument('--input-data-dir', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/test_3_vs_0-2/dev110_test30/binwidth5/pvp_tp_hbp_dwi_b1_adc/clean_partition_select')
    parser.add_argument('--input-params-dir', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/test_3_vs_0-2/dev110_test30/binwidth5/pvp_tp_hbp_dwi_b1_adc/optimized_params')
    parser.add_argument('--output-dir', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/test_3_vs_0-2/dev110_test30/binwidth5/pvp_tp_hbp_dwi_b1_adc/trained_model_outputs')
    parser.add_argument('--models',nargs='+',default=None)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--shap',action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print("\n" + "="*100)
    print("RADIOMICS MODEL TRAINER (with Bootstrap Confidence Intervals)")
    print("="*100)
    print(f"Bootstrap iterations: {args.bootstrap}")
    print(f"Output directory: {args.output_dir}")
    if args.models:
        print(f"Models: {', '.join(args.models)}")
    print("="*100)
    
    comparison_df, model_objects = train_multiple_models(
        in_dir=args.input_data_dir,
        param_dir=args.input_params_dir,
        out_dir=args.output_dir,
        model_list=args.models,
        n_bootstrap=args.bootstrap,
        generate_shap=True
    )
    
    print(f"{'='*100}")
    print("Training Complete")
    print(f"{'='*100}\n")
