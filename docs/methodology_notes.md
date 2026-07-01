# Methodology notes: feature extraction QC

Running log of decisions made while auditing the radiomics extraction settings
(`scripts/radiomics_params.yaml`) before trusting `main_pipeline.py`'s modeling stages.
Update this as decisions change — it's meant to travel with the repo across machines/sessions.

## Context

`main_pipeline.py` (partition -> gridsearch -> train) assumes radiomics features were
already extracted correctly. That extraction currently uses a single fixed `binWidth: 5`
across all 9 sequences (`pre, ap, pvp, tp, hbp, t2, dwi_b1, dwi_b2, adc`), plus LoG
(sigma 2-5) and full wavelet decompositions, producing a very large feature space per
sequence. Two things are under active review: whether bin width should be sequence-
specific, and whether N4 bias correction should be applied uniformly.

## Bin width: should be sequence-specific, and how

- Problem: intensity scale and noise level differ a lot across the 9 sequences (e.g.
  post-contrast phases vs. ADC maps vs. DWI), so one fixed `binWidth=5` over- or
  under-resolves different sequences.
- Existing infra already in the repo for this (previously unused/half-wired):
  - `scripts/radiomics_utils/snr_binwidth.py` -- noise-based per-sequence bin width
    recommendation (IBSI/Boellaard-style: `binWidth = 2 x noise sigma`).
  - `scripts/radiomics_utils/binwidth_texture_gallery.py` -- visual sanity check of
    candidate bin widths on one patient.
  - `scripts/radiomics_utils/stability.py`, `feature_stability.py`,
    `icc_stability_per_sequence_category.py` -- ICC-based stability comparison across
    extraction-setting variants (bin width, reader, interpolator, spacing).
  - `scripts/radiomics_utils/extraction.py` / `main_orchestrator.py` -- already has the
    `itertools.product` sweep machinery over readers/interpolators/spacings/bin_widths;
    currently collapsed to single values in the YAML.
- Noise is estimated from small homogeneous **liver patches** (not tumor ROI, and not a
  whole-liver mask) -- patches were deliberately delineated away from vessels/other
  structures specifically for this calibration. Do NOT use tumor segmentations for the
  noise estimate: tumor heterogeneity is the actual signal of interest (related to the
  immunophenotype endpoint), so treating it as "noise" would bias bin width upward and
  wash out the texture differences the study is trying to detect.
- `snr_binwidth.py` was rewritten to:
  1. Compute noise sigma in PyRadiomics-**normalized** units (rescaled by
     `normalizeScale / whole_image_std`, matching `extraction_settings.normalize: true`,
     `normalizeScale: 100` in `radiomics_params.yaml`), since PyRadiomics z-scores the
     whole image before applying `binWidth` -- the calibration needs to be measured in
     that same space to be comparable. Caveat: this uses the pre-resampling image std;
     PyRadiomics normalizes post-resampling-to-[1,1,1], so treat results as a good
     starting estimate, not an exact reproduction.
  2. Use **pooled within-patch variance** (via `scipy.ndimage.label` on the patch mask)
     instead of a naive global std over all pooled voxels -- avoids conflating true
     voxel-level noise with real between-patch/slice differences. Feasible now because
     patch voxel counts are consistently well above the `>10` sanity threshold (multiple
     areas delineated across many slices per patient).
  3. Aggregate sigma across the calibration cohort via **median**, not `max()` -- a
     single atypical calibration patient shouldn't set the bin width for a whole
     sequence. (Open option: delineate ~10 more calibration patients if the current
     10-15 patient cohort turns out too small to trust.)
- Normalization does NOT make bin width sequence-agnostic: `normalize=true` equalizes
  total variance (signal + noise) across sequences/patients, but not the noise-to-signal
  ratio -- two sequences can both land at std=100 post-normalization with very different
  fractions of that variance coming from real noise. Per-sequence calibration is still
  necessary.
- Still open: the drive holding the calibration set (`/Volumes/T7 Shield/Restrepo/
  Calibration_Set`) hasn't been mounted yet during this work, so `snr_binwidth.py`
  hasn't actually been run against real data. Once it has, validate the recommended
  per-sequence bin widths against the current uniform `binWidth=5` using the
  ICC-stability sweep (`stability.py`/`feature_stability.py`) before changing
  `radiomics_params.yaml`.

## N4 bias correction: should not be applied uniformly

- `scripts/radiomics_utils/n4_correction.py` currently applies N4 to every
  non-segmentation `.nrrd` file with no sequence-type branching -- it runs on ADC and
  DWI exactly like the anatomical sequences.
- **ADC should not get N4.** ADC is a derived quantitative map
  (`ADC = -ln(S(b2)/S(b1))/(b2-b1)`), not raw signal -- coil-sensitivity shading mostly
  cancels out already in that ratio/log computation. Running N4 on top fits a smooth
  bias field to a map that largely doesn't have one, risking distortion of the actual
  quantitative ADC values (which matter because ADC's value as a biomarker depends on
  being an absolute, comparable physical measurement across patients).
- **Raw DWI b-value images are a judgment call, not resolved yet.** They are real
  magnitude signal (so coil-sensitivity bias is real and N4 is defensible in principle),
  but low SNR (especially high-b images), Rician noise statistics, and EPI geometric/
  susceptibility distortion (which N4 does not model) make bias-field estimation less
  reliable there than on anatomical sequences.
- Decision on how to route around this in code: rather than adding sequence-type
  branching logic to `n4_correction.py`, the plan is to keep that script's logic
  unchanged and instead rename the raw ADC/DWI files to include the `_N4` suffix
  (whether or not N4 was actually run on them), so downstream filename-matching logic
  in `extraction.py`/`snr_binwidth.py` (which looks for `"n4"` in the filename) doesn't
  need to change. This is a deliberate minimal-footprint choice -- revisit if it causes
  confusion later (the filename would then lie about whether N4 was actually applied).

## `feature_stability.py` bug fixes (applied)

- **Sequence-name parsing bug (fixed):** `feat.split('_')[0]` truncated `dwi_b1`/
  `dwi_b2` feature names to a bogus `'dwi'` bucket, since those two sequence names
  contain an underscore. Fixed by matching against an explicit `SEQUENCES` list via
  `startswith(seq + "_")`, same approach already used correctly in
  `icc_stability_per_sequence_category.py`.
- **Missing patient alignment across ICC variants (fixed):** the script didn't ensure
  all input CSVs (extraction variants being compared) shared the same patients before
  running `pingouin.intraclass_corr`, which expects a balanced target x rater design. A
  patient missing from even one variant (e.g. a failed extraction run) could silently
  unbalance the ICC calculation (caught by a bare `except`, silently producing NaN, or
  worse computing on a coincidentally-balanced subset). Fixed by intersecting
  `PatientID`s across all input CSVs before melting, with a printed warning listing any
  dropped patients per file.
- Minor cleanups: removed a leftover `print(melted)` debug dump, removed an unused
  `variant_id` variable, tightened a bare `except:` to `except Exception:`.
- **Still open / not changed:** the `'wavelet'/'log-sigma'` check overwrites whatever
  category (`First-Order`/`Texture`) was already assigned, collapsing all transform-
  domain features into a single `'Transform'` bucket. This is inconsistent with
  `icc_stability_per_sequence_category.py`, where the equivalent line is commented out
  (so that script keeps the First-Order/Texture/Shape breakdown even for
  wavelet/LoG-derived features). Needs a decision on which convention to standardize on
  across both scripts.

## Environment

- No `requirements.txt`/`environment.yml` exists in the repo yet. Project imports
  `pingouin`, `SimpleITK`, `pyradiomics`, `xgboost`, `shap`, `tabpfn`, `torch`, and
  standard scientific-Python packages, none of which are installed in the system
  Python on this machine (no venv/conda env found either) -- that's why VSCode/Pylance
  shows unresolved-import warnings; it's an environment issue, not a code bug. A
  `requirements.txt` generated from actual imports across `scripts/` would fix this but
  hasn't been created yet.
