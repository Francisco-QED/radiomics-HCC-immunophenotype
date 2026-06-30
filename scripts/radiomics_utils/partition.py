from sklearn.model_selection import train_test_split
import pandas as pd

# Endpoint-aware partitioning function into test set and development set. We want to ensure 
# that the distribution of responders vs non-responders is balanced across the two sets, so we 
# will use stratified sampling based on the response label.

def partition_data(df_reliable, outcomes_path, label_col, cutoff,test_size=25):
    df_outcomes = pd.read_excel(outcomes_path)
    df_merged = df_reliable.reset_index().merge(
        df_outcomes[['PatientID', label_col]], on='PatientID', how='inner'
    ).set_index('PatientID')

    df_merged[label_col] = (df_merged[label_col] >= cutoff).astype(int)
    print(df_merged)
    X = df_merged.drop(columns=[label_col])
    y = df_merged[label_col]
    
    # Stratified split to ensure even distribution of responders
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"Partitioning Complete: Dev={len(X_dev)}, Test={len(X_test)}")
    print(y_dev)
    return X_dev, y_dev, X_test, y_test

if __name__ == '__main__':
    # Example usage
    df_reliable = pd.read_csv("/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/reliable_feature_pool.csv", index_col=0)
    outcomes_path = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/immunophenotype_outcomes.xlsx"
    X_dev, y_dev, X_test, y_test = partition_data(df_reliable, outcomes_path, label_col='TILs_resection', cutoff=3, test_size=5)
