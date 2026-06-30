import os
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from pingouin import intraclass_corr
from tqdm import tqdm


## This is a somewhat overkill function to reconstuct the name of the the list off icc's for each radiomics feature calculated between two different settings

def icc_filename(filenames):
    # 1. Split path and keep the filename
    name1 = os.path.basename(filenames[0]).replace('.csv', '')
    ext = '.csv'
    
    parts1 = name1.split('_')
    parts2 = os.path.basename(filenames[1]).replace('.csv', '').split('_')
    
    new_filename_parts = []
    
    for p1, p2 in zip(parts1, parts2):
        if p1 == p2:
            # If the part is identical, keep it
            new_filename_parts.append(p1)
        else:
            # We found the change! 
            # Separate the prefix (e.g., 'W' or 'I') from the value
            prefix1, val1 = p1.split('-', 1)
            _, val2 = p2.split('-', 1)
            
            # --- Special Handling for Interpolator (sitkBSpline vs sitkBLinear) ---
            if prefix1 == 'I' and 'sitk' in val1 and 'sitk' in val2:
                # This logic cleans 'sitkBLinear' to 'Linear'
                clean_val2 = val2.replace('sitkB', '') 
                merged_val = f"{val1}-{clean_val2}"
            else:
                # Standard merging for other parameters (e.g., W-10 vs W-25 -> W-10-25)
                merged_val = f"{val1}-{val2}"
                
            new_filename_parts.append(f"{prefix1}-{merged_val}")
            
    return "_".join(new_filename_parts) + ext

def calculate_stability_scorecard(csv_paths, settings):
    """
    Conductor for sequential ICC calculation.
    """
    # 1. Silence the warnings globally for this session
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='pingouin')
    warnings.filterwarnings('ignore', message='divide by zero encountered in scalar divide')
    warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')

    all_data = []
    
    # 2. Load and Combine Data
    for path, v_name in tqdm(zip(csv_paths, settings), desc="Loading Files", total=len(csv_paths)):
        df = pd.read_csv(path)
        melted = df.melt(id_vars=['PatientID'], var_name='Feature', value_name='Value')
        melted['VariantID'] = v_name 
        all_data.append(melted)
    
    full_df = pd.concat(all_data)
    unique_features = full_df['Feature'].unique()
    
    # 3. Sequential Loop (No Multiprocessing)
    results = {}
    print(f"🔬 Analyzing {len(unique_features)} features sequentially...")
    
    # tqdm handles the progress bar for the single loop
    for feature_name in tqdm(unique_features, desc="Calculating ICC"):
        try:
            # Filter data for just this feature
            feat_data = full_df[full_df['Feature'] == feature_name]
            
            # Calculate ICC(2,1) - Fixed raters, agreement
            icc = intraclass_corr(data=feat_data, targets='PatientID', raters='VariantID', ratings='Value')
            
            # Extract the specific ICC value
            icc_val = icc.set_index('Type').loc['ICC2', 'ICC']
            results[feature_name] = icc_val
            
        except Exception:
            # Captures math errors (like IMC2 NaNs) and assigns NaN
            results[feature_name] = np.nan

    results = pd.DataFrame(list(results.items()), columns=['Feature', 'ICC'])  
    results = results.sort_values(by='ICC')
    return results

### Now plotting function for the stability spider plot ###

def visualize_sequence_stability_spider(scorecard_path, threshold):
    """
    Calculates percentage of stable features (>threshold) per sequence 
    and generates a spider plot.
    """
    # 1. LOAD THE SCORECARD
    print(f"📖 Loading stability scorecard from: {os.path.basename(scorecard_path)}")
    df = pd.read_csv(scorecard_path, names=['FeatureName', 'ICC_Score'], header=0)
    
    # Define your 9 sequences (Must match the prefixes used during extraction)
    sequences = ['pre', 'ap', 'pvp', 'tp', 'hbp', 't2', 'dwi_b1', 'dwi_b2','adc'] 
    
    stability_matrix = []
    
    print(f"📊 Analyzing {len(sequences)} sequences...")
    for seq in tqdm(sequences, desc="Calculating Percentages"):
        # a. Filter the dataframe for features starting with the sequence prefix (e.g., 'hbp_')
        # We ensure it's seq + '_' to avoid 't1' matching 't1_in' by accident.
        seq_data = df[df['FeatureName'].str.startswith(f"{seq}_")]
        
        total_features = len(seq_data)
        
        if total_features == 0:
            print(f"  ⚠️ Warning: No features found for sequence '{seq}'. Skipping.")
            stability_matrix.append({'Sequence': seq.upper(), 'Stable_Percent': 0, 'Count': 0})
            continue
            
        # b. Count features with ICC > Threshold
        stable_count = (seq_data['ICC_Score'].abs() > threshold).sum()
        
        # c. Calculate percentage
        percent_stable = (stable_count / total_features) * 100
        
        stability_matrix.append({
            'Sequence': seq.upper(), 
            'Stable_Percent': round(percent_stable, 1),
            'Count': stable_count
        })

    # 2. CREATE THE DATAFRAME FOR PLOTTING
    plot_df = pd.DataFrame(stability_matrix)
    
    # To close the loop in Plotly, we append the first row to the end
    plot_df_closed = pd.concat([plot_df, plot_df.iloc[[0]]])

    # 3. GENERATE THE SPIDER PLOT (RADAR)
    print("🎨 Generating interactive Spider Plot...")
    
    fig = go.Figure()

    # Add the single trace for stability percentages
    fig.add_trace(go.Scatterpolar(
        r=plot_df_closed['Stable_Percent'],
        theta=plot_df_closed['Sequence'],
        fill='toself', # Fill the area inside
        fillcolor='rgba(135, 206, 250, 0.4)', # Light sky blue, semi-transparent
        line=dict(color='rgb(70, 130, 180)', width=2), # Steel blue line
        name='Stable Features (%)',
        marker=dict(size=8),
        # Interactive hover text: showing sequence, percentage, and exact count
        hoverinfo='text',
        text=[f"Sequence: {s}<br>Stable: {p}%<br>Count: {c}" 
              for s, p, c in zip(plot_df_closed['Sequence'], 
                                plot_df_closed['Stable_Percent'], 
                                plot_df_closed['Count'])]
    ))

    # Configure the Layout
    fig.update_layout(
        title={
            'text': f"Percentage of Stable Features (ICC > {threshold}) by HCC MRI Sequence",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100], # Radius range (0% to 100%)
                tickprefix="",
                ticksuffix="%", # Add percentage sign to ticks
                tickfont=dict(size=10, color='gray'),
                gridcolor='rgba(0, 0, 0, 0.1)'
            ),
            angularaxis=dict(
                gridcolor='rgba(0, 0, 0, 0.1)',
                rotation=90, # Starts at 'North'
                direction='clockwise' # Moves clockwise
            ),
            bgcolor='white'
        ),
        showlegend=False,
        width=700,
        height=700
    )

    # 4. SHOW AND SAVE THE PLOT
    # Shows an interactive plot on your iMac's default browser
    fig.show() 
    
    # Also save a static PNG for a presentation or proposal document
    #fig.write_image("hcc_sequence_stability_spider.png")
    print("✅ Done. Interactive plot opened. Static PNG saved as 'hcc_sequence_stability_spider.png'.")

# --- EXECUTION ---
if __name__ == '__main__':
    csv_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets"
    results_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/stability_analysis/icc_sheets"
    settings = ["R-Reader1_I-sitkBSpline_S-1mm_W-5.csv", "R-Reader1_I-sitkBSpline_S-1.5mm_W-5.csv"]

    paths = [os.path.join(csv_dir, sub) for sub in settings]
    
    if paths:
        stability_results = calculate_stability_scorecard(paths, settings)
        
        # Save the final scorecard
        stability_results.to_csv(results_dir+"/"+icc_filename(settings), index=False)
        
        # Quick summary print
        stable_count = (stability_results['ICC'] > 0.90).sum()
        print(f"✅ Analysis Complete. {stable_count} features are highly stable (ICC > 0.90).")
    else:
        print("❌ No CSV files found in the results directory.")
