import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

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

#Plot Spider plot of ICC ####
if __name__ == '__main__':
    # Adjust this path to point to your saved scorecard CSV
    base_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/stability_analysis/icc_sheets"
    scorecard_path = os.path.join(base_dir, "R-Reader1_I-sitkBSpline_S-1.5mm_W-5-10.csv")
    if os.path.exists(scorecard_path):
        visualize_sequence_stability_spider(scorecard_path, threshold=0.85)
    else:
        print(f"❌ Could not find scorecard at: {scorecard_path}")