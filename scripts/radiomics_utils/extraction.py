import os
import yaml
import logging
import itertools
from tqdm import tqdm
import pandas as pd
from radiomics import featureextractor

logging.getLogger('radiomics').setLevel(logging.ERROR)


def extract_features_single(image_path, mask_path, settings):
    """Worker: extracts features for a single patient."""
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        return None
    extractor = featureextractor.RadiomicsFeatureExtractor(settings)
    try:
        result = extractor.execute(image_path, mask_path)
        return {k: v for k, v in result.items() if not k.startswith('diagnostics_')}
    except Exception as e:
        print(f"  ⚠️ Extraction error: {e}")
        return None


def run_batch_extraction(config):
    """
    Runs extraction for every combination of (target × reader × interp × spacing × bin_width).
    Targets, readers, and paths are read from config['extraction_targets'].
    Shared sweep params (interpolators, spacings, bin_widths, sequences) come from config['execution'].
    Output: results_base_dir/reader/R-{reader}_I-{interp}_S-{spacing}mm_W-{bw}.csv
    """
    exec_cfg = config['execution']
    base_settings = config.get('extraction_settings', {}).copy()
    image_types = config.get('imageType', {})
    targets = config.get('extraction_targets', [])

    shared_combos = list(itertools.product(
        exec_cfg['interpolators'],
        exec_cfg['spacings'],
        exec_cfg['bin_widths'],
    ))

    for target in targets:
        label = target.get('label', '?')
        base_dir = target['base_dir']
        results_base_dir = target['results_base_dir']
        readers = target['readers']

        no_n4 = set(exec_cfg.get('sequences_without_n4', []))
        images_dir = target.get('images_dir')  # None for SA1/SA2; shared folder for inter-reader

        for reader, (interp, space, bw) in itertools.product(readers, shared_combos):
            perseq_widths = exec_cfg.get('per_sequence_bin_widths', {}) if bw == 'perseq' else {}

            variant_name = f"R-{reader}_I-{interp}_S-{space[0]}mm_W-{bw}"
            reader_results_dir = os.path.join(results_base_dir, reader)
            os.makedirs(reader_results_dir, exist_ok=True)
            csv_path = os.path.join(reader_results_dir, f"{variant_name}.csv")

            if os.path.exists(csv_path):
                print(f"[{label}] {reader} | {variant_name} — already exists, skipping.")
                continue

            patient_list = sorted(os.listdir(os.path.join(base_dir, reader)))
            variant_results = []

            for pid in tqdm(patient_list, desc=f"[{label}] {reader} | {variant_name}"):
                patient_row = {'PatientID': pid}

                current_settings = base_settings.copy()
                current_settings.update({
                    'interpolator': interp,
                    'resampledPixelSpacing': space,
                    'binWidth': bw,
                })
                good_settings = {'imageType': image_types, 'setting': current_settings}

                for seq in exec_cfg['sequences']:
                    img_suffix = '' if seq in no_n4 else '_N4'
                    img_root = images_dir if images_dir else os.path.join(base_dir, reader)
                    img_path  = os.path.join(img_root, pid, f"{pid}_{seq}{img_suffix}.nrrd")
                    mask_path = os.path.join(base_dir, reader, pid, f"{pid}_{seq}.seg.nrrd")

                    if perseq_widths:
                        seq_settings = current_settings.copy()
                        seq_settings['binWidth'] = perseq_widths.get(seq, bw)
                        good_settings = {'imageType': image_types, 'setting': seq_settings}

                    features = extract_features_single(img_path, mask_path, good_settings)
                    if features:
                        for k, v in features.items():
                            patient_row[f"{seq}_{k}"] = v

                if len(patient_row) > 1:
                    variant_results.append(patient_row)

            if variant_results:
                pd.DataFrame(variant_results).to_csv(csv_path, index=False)
                print(f"  Saved {len(variant_results)} patients → {csv_path}")
            else:
                print(f"  ⚠️ No results for [{label}] {reader} | {variant_name}")


if __name__ == '__main__':
    _yaml_path = os.path.join(os.path.dirname(__file__), '..', 'radiomics_params.yaml')
    with open(_yaml_path) as f:
        config = yaml.safe_load(f)
    run_batch_extraction(config)
