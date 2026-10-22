# Vendored function snapshot (frozen)

Two modules copied once (October 2026, when asked to mimic the "gpt" radiomics procedure) into this project so the pipeline does not depend on any other
agent's live files. They are plain function / class definitions, no data and no results.

| file | used for |
|---|---|
| run_compact_ridge_sequence_search.py | reader-ICC filter, QC filter, median imputer, within-sequence clustering + outcome-ranked representatives, ridge wrapper, feasible-k grid, metrics, development-set loader |
| run_sparse_dynamic_extensions.py | lasso / elastic-net pipeline with the 10-20 non-zero-coefficient constraint (fit_one, refit_sparse_candidate) |

Local edits (by this project, not upstream): path constants only (radiomics-sheet folder names, split-file location, expected 111 dev patients / 963 features, results root).
Nothing here is updated from upstream: if the other agent changes its own files, this copy is unaffected. Pinned by checksum (below); every ladder run also records these checksums in its manifest.json.
Use with: --gpt-dir <this folder>.

sha256:
    b9a5f2a68186cbe7cccc468c863236f8ae7481012be9116ebce7e2cc163d36f8  run_compact_ridge_sequence_search.py
    e80f90e2fd4b63d3349ee724914d464caa000114bbe0bf259e78ef5600051c52  run_sparse_dynamic_extensions.py
