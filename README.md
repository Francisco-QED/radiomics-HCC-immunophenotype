# radiomics-HCC-immunophenotype

# radiomics-HCC-immunophenotype

Predicting the immune microenvironment (tumor-infiltrating lymphocytes / tertiary
lymphoid structures) of hepatocellular carcinoma (HCC) from pre-treatment,
multi-sequence MRI using radiomics and machine learning.

## Overview

- Extracts PyRadiomics features from 9 MRI sequences per patient (pre-contrast,
  arterial, portal venous, transitional/delayed, hepatobiliary phase, T2, two DWI
  b-values, and ADC), across original, LoG, and wavelet image transforms.
- Builds a binary classification pipeline — partition, feature selection,
  hyperparameter tuning, locked-set evaluation — around a TILs-based immunophenotype
  endpoint, with a parallel pipeline for a recurrence/survival endpoint.
- Prioritizes methodological rigor over raw model complexity: sequence-specific
  bin-width calibration from noise estimates, inter-reader ICC filtering, leakage-safe
  fold-specific feature selection, nested cross-validation for honest performance
  estimates, and bootstrap confidence intervals on a held-out lock set.
- Compares eight model families (logistic regression, linear/RBF SVM, random forest,
  extra trees, XGBoost, Gaussian process, TabPFN) under identical preprocessing.

## Repository layout

- `scripts/dicom_processing/` — raw DICOM to NRRD conversion, sequence extraction,
  ADC calculation, and other imaging preprocessing.
- `scripts/radiomics_utils/` — the core pipeline: feature extraction, extraction-setting
  QC (bin-width calibration, ICC stability analysis), the main modeling pipeline, and
  sweep scripts for exploring cutoff/sequence-subset combinations.
- `scripts/radiomics_params.yaml` — PyRadiomics extraction configuration (image
  filters, discretization, normalization).
- `docs/` — running notes on methodology decisions.

## Data

Patient imaging and clinical data are not included in this repository and are handled
separately.

## Built with

Python, PyRadiomics, SimpleITK, scikit-learn, XGBoost, TabPFN, pandas.
