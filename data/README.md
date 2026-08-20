# `data/`

This directory is a local drop-zone for input data and is **not** tracked by
git (see the repo's `.gitignore`). Expected inputs:

| File | Format | Description |
|---|---|---|
| `*.h5ad` | AnnData (`.h5ad`) | Spatial transcriptomics counts matrix. Must contain `adata.obsm["spatial"]` (2D coordinates) and a `adata.obs["cell_type"]` column for cell-type-prior-aware training. |
| `*.csv`  | Ligand-receptor pair table | Two columns naming the ligand and receptor gene for each pair (matched against `adata.var_names`). |

Run `python -m svrn --run_example` to generate a small synthetic dataset here
and confirm the pipeline runs end-to-end before pointing it at real data.
