# Project setup and how to access parts A / B / C

Paper: https://arxiv.org/pdf/2002.03230.pdf

## What to set up first

- Create and activate a Python environment (conda recommended).
- Install RDKit (via conda-forge) and a suitable PyTorch build that matches your CUDA (or CPU-only) setup.
- Install Python packages from `requirements.txt` and install the package in editable mode: `pip install -e .`.
- Ensure `data/` and `ckpt/` folders are present and populated with any checkpoints or datasets you plan to use (examples are included in the repository).

## Quick installation (recommended)

```bash
conda create -n hgraph python=3.8 -y
conda activate hgraph
conda install -c conda-forge rdkit -y
# Install PyTorch according to your CUDA version; see https://pytorch.org/
pip install -r requirements.txt
pip install -e .
```

If you plan to use property-guided finetuning, install `chemprop` separately following its documentation.

## Important repository layout

- `data/` — datasets (chembl, qed, drd2, polymers, scraped_data)
- `ckpt/` — checkpoints and saved models (e.g. `ckpt/chembl-pretrained/model.ckpt`, `ckpt/antibiotc/`)
- `hgraph/` — core model code (encoder/decoder, dataset, vocab)
- `part_a/`, `part_b/`, `part_c/` — analysis and evaluation scripts

## Running the main workflows (short commands)

- Extract vocabulary:

```bash
python get_vocab.py --ncpu 16 < data/chembl/all.txt > data/chembl/vocab.txt
```

- Preprocess (example):

```bash
python preprocess.py --train data/chembl/all.txt --vocab data/chembl/vocab.txt --ncpu 16 --mode single
mkdir train_processed
mv tensor* train_processed/
```

- Train generator:

```bash
mkdir -p ckpt/chembl-pretrained
python train_generator.py --train train_processed/ --vocab data/chembl/vocab.txt --save_dir ckpt/chembl-pretrained
```

- Generate samples:

```bash
python generate.py --vocab data/chembl/vocab.txt --model ckpt/chembl-pretrained/model.ckpt --nsamples 1000
```

## Part-specific scripts (how to run)

- Part A — Evaluation grid and summary

  - Script: `part_a/eval_partA.py`
  - Purpose: Reads `data/scraped_data/scrapedData_clean.txt`, simulates model outputs (replace the `model_predict` stub with your model invocation), computes validity/ similarity metrics, and writes `results.csv`, `similarity_distribution.png`, and `qualitative_grid.png`.
  - Run:

  ```bash
  python part_a/eval_partA.py
  ```

  - Notes: Ensure `data/scraped_data/scrapedData_clean.txt` exists. Replace `model_predict` in the script with a call to your model (e.g., reconstruction/generation function) to evaluate real model outputs.

- Part B — Checkpoint / information-loss plots

  - Script: `part_b/information_loss_partB.py`
  - Purpose: Generates example plots (`exact_match_trend.png`, `tanimoto_trend.png`) showing training progress or checkpoint trends.
  - Run:

  ```bash
  python part_b/information_loss_partB.py
  ```

- Part C — Hierarchical probing and probing plots
  - Script: `part_c/part_c.py`
  - Purpose: Loads a trained `HierVAE` model, extracts hidden states, trains simple Ridge probes to predict chemical properties (e.g., molecular weight, ring count), saves `probing_metrics.csv` and `probing_results.png`.
  - Run:
  ```bash
  python part_c/part_c.py
  ```
  - Notes:
    - The script loads model and vocab from constants near the top: `MODEL_PATH`, `VOCAB_PATH`, `DATA_PATH`. Update those paths or the file to point to your checkpoint and vocab if needed.
    - The script uses `cuda` if available; ensure PyTorch and GPU drivers are configured for GPU runs.

## Accessing checkpoints and outputs

- Example pretrained checkpoint: `ckpt/chembl-pretrained/model.ckpt`
- Antibiotic checkpoints: `ckpt/antibiotc/` contains `model.ckpt.*` files shown in the repo
- Finetune outputs: `ckpt/finetune/` and `results.csv` (from translation) are used to collect generated molecules and metrics

## Troubleshooting & tips

- RDKit: install via `conda install -c conda-forge rdkit` for best compatibility on Windows and Linux.
- If `part_c/part_c.py` fails to find your model, update `MODEL_PATH` or pass a checkpoint to the script (it currently uses a constant). You can also load the checkpoint manually in an interactive session to validate.
- Use `requirements.txt` to view pinned package versions.

## Next steps (optional)

- I can add a `environment.yml` for conda reproducibility, make a small wrapper CLI for running the parts with options, or add a CONTRIBUTING guide. Tell me which you'd prefer.
