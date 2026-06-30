import os
import yaml
import logging
import itertools
from tqdm import tqdm
import pandas as pd
from radiomics import featureextractor

# Silence featureextractor logs to avoid cluttering the console
logging.getLogger('radiomics').setLevel(logging.ERROR)

def extract_features_single(image_path, mask_path, settings):
    """Worker: Extracts features for a single patient."""
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        return None
    extractor = featureextractor.RadiomicsFeatureExtractor(settings)
    try:
        result = extractor.execute(image_path, mask_path)       
        return {k: v for k, v in result.items() if not k.startswith('diagnostics_')}
    except Exception as e:
        print(f"  ⚠️ Extraction error: {e}")
        return None

def run_batch_extraction(base_dir, results_dir, config):
    """The Boss: Orchestrates the parameter combinations."""
    exec_cfg = config['execution']

    # 1. Prepare the static parts of the config that never change per loop
    # We rename 'extraction_settings' to 'setting' because PyRadiomics expects 'setting'
    base_settings = config.get('extraction_settings', {}).copy()
    image_types = config.get('imageType', {})

    # 1. Flatten the loops using itertools.product
    # This creates a flat list of every unique combination of params
    param_combinations = itertools.product(
        exec_cfg['readers'],
        exec_cfg['interpolators'],
        exec_cfg['spacings'],
        exec_cfg['bin_widths']
    )

    for reader, interp, space, bw in param_combinations:
        variant_name = f"R-{reader}_I-{interp}_S-{space[0]}mm_W-{bw}"
        csv_path = os.path.join(results_dir, f"{variant_name}.csv")
        
        if os.path.exists(csv_path): continue

        # 2. Setup Data Collection
        patient_list = sorted(os.listdir(os.path.join(base_dir, reader)))
        variant_results = []

        # 3. Process Patients
        for pid in tqdm(patient_list, desc=f"🧬 {variant_name}"):
            patient_row = {'PatientID': pid}
            
            # Apply dynamic settings
            current_settings = base_settings.copy()
            current_settings.update({
                'interpolator': interp,
                'resampledPixelSpacing': space,
                'binWidth': bw
            })
            good_settings = {
            'imageType': image_types,
            'setting': current_settings
            }

            for seq in exec_cfg['sequences']:
                img_path = os.path.join(base_dir, reader, pid, f"{pid}_{seq}_N4.nrrd")
                mask_path = os.path.join(base_dir, reader, pid, f"{pid}_{seq}.seg.nrrd")
                
                features = extract_features_single(img_path, mask_path, good_settings)
                if features:
                    for k, v in features.items():
                        patient_row[f"{seq}_{k}"] = v
            
            if len(patient_row) > 1:
                variant_results.append(patient_row)
        
        # 4. Save results
        if variant_results:
            pd.DataFrame(variant_results).to_csv(csv_path, index=False)


if __name__ == '__main__':
    # Load config once
    with open("/Users/Francisco/Desktop/RestrepoProjects/Active/Regeneron retrospective radiomics/scripts/radiomics_params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    run_batch_extraction(
        base_dir="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations",
        results_dir="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/SA2",
        config=config
    )