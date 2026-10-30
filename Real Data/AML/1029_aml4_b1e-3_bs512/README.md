# aml4, beta 1e-3, batch size 512

Native Decipher (PyPI `scdecipher`, batch-blind) and three `decipher_models3` batch arms
(model5, model4r, model5r; batch = `patient`) on AML1-4 (64,803 cells x 6,341 genes, all genes),
lr 5e-3, beta 1e-3, batch size 512, decipherseeds 0-2, early stopping on validation NLL.

## Layout

| path | content |
|---|---|
| `1027_build_aml4.py` | builds `aml4/` from `../AML1-4_preprocessed.h5ad` |
| `1027_aml4_common.py` | constants, the leukemic trajectory and the evaluation shared by both drivers |
| `scib_metrics.py` | scIB metrics, DPT, decipher time on a trajectory, reconstruction R^2 and plot |
| `d3_helpers.py` | parser, run names, file split and batch norms of the `decipher_models3` driver |
| `1027_aml4_d3_decipher.py` | one `decipher_models3` run (preset x seed) |
| `1027_aml4_native_decipher.py` | one `scdecipher` run (seed) |
| `1029_aml4_b1e-3_bs512_jobs_d3.txt`, `1029_aml4_b1e-3_bs512_jobs_native.txt` | driver arguments, one run per line |
| `aml4/` | `aml4_shared.h5ad` (written by the build), its `.sha256`, `patient_sizes.csv`, `build_meta.json` |
| `model5/`, `model4r/`, `model5r/`, `native/` | per run: `seed<k>_meta.json`, `csv/seed<k>_{loss,metrics,traj_clusters,vspace}.csv`, `reconstruction/seed<k>_recon.png`, `vspace/seed<k>_vspace.png`, `logs/seed<k>.log`, `_models/` |

## Environments

Two Python environments:

1. `decipher_models3`: `pip install -e .` at the repository root. Runs the build and
   `1027_aml4_d3_decipher.py`.
2. `scdecipher`: `pip install scdecipher`. Runs `1027_aml4_native_decipher.py`.

## Commands

Run from this folder. Build the shared h5ad (environment 1):

```bash
python 1027_build_aml4.py
```

Then each job file, one run after another, each run's output in `<out-dir>/logs/seed<k>.log`.
`decipher_models3` arms (environment 1):

```bash
grep -v '^#' 1029_aml4_b1e-3_bs512_jobs_d3.txt | while read -r a; do
  out=$(echo "$a" | sed -E 's/.*--out-dir ([^ ]+).*/\1/')
  seed=$(echo "$a" | sed -E 's/.*--seed ([0-9]+).*/\1/')
  mkdir -p "$out/logs"
  python 1027_aml4_d3_decipher.py $a > "$out/logs/seed$seed.log" 2>&1 < /dev/null
done
```

Native (environment 2):

```bash
grep -v '^#' 1029_aml4_b1e-3_bs512_jobs_native.txt | while read -r a; do
  out=$(echo "$a" | sed -E 's/.*--out-dir ([^ ]+).*/\1/')
  seed=$(echo "$a" | sed -E 's/.*--seed ([0-9]+).*/\1/')
  mkdir -p "$out/logs"
  python 1027_aml4_native_decipher.py $a > "$out/logs/seed$seed.log" 2>&1 < /dev/null
done
```

At these settings native seeds 0 and 1 stop with NaN in epoch 0; only native seed 2 completes.
