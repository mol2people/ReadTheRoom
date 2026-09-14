# ecg4cluster

_Last updated: 2026-09-14._

A small ZIH GPU workflow around
[Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer) v1.9.3.

It digitizes full-width ECG traces, exports every displayed lead, selects the
most complete preferred lead, and combines the selected waveforms into one CSV.
Pages remain separate: time restarts at zero and `is_scan_start` marks the first
sample of every page.

## Setup

On Capella, unpack the data, create the venv, and request a GPU:

```bash
cd /data/horse/ws/buza314h-ecg
unzip Archive.zip -d data

module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
python3 -m venv --system-site-packages venv
srun --partition=capella --nodes=1 --gres=gpu:1 --time=03:00:00 --pty bash
```

Allocation actually used on Sep 14 (job 4213240, node c29: one H100, 10 CPUs,
100 GB, 6 h): allocate first, then open a shell on the assigned node with
`srun` from inside the allocation:

```bash
salloc --account=p_epoch_data --partition=capella --job-name=ecg --nodes=1 \
  --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --mem=100G --time=06:00:00
srun --pty bash
```

Do all of this inside `tmux` (`tmux new -s ecg`; detach with `Ctrl-b d`,
reattach with `tmux attach -t ecg`): if SSH drops, the allocation shell — and
the run with it — dies otherwise.

On the GPU node:

```bash
source /data/horse/ws/buza314h-ecg/venv/bin/activate
cd /data/horse/ws/buza314h-ecg/experiment_sep14/ReadTheRoom/ecg4cluster
bash fetch_open_ecg.sh
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.cuda.get_device_name(0))'
```

On Alpha instead: `salloc --account=p_epoch_data --partition=alpha
--cpus-per-task=6 --gres=gpu:1 --mem=64G --time=03:00:00`, then set
`OMP/MKL/OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK`. Alpha allows max 6 CPUs
per GPU and uses A100 40 GiB cards.

To push, cache a classic token (`repo` scope) in memory only:
`git config --global credential.helper 'cache --timeout=28800'`, then `git push`
uses the GitHub username plus the token as password.

## Run

Smoke-test one page per layout (CPU and GPU), then the full directories.
Batch size is fixed at 2:

```bash
python digitize.py --input <ws>/data/12lead/DIII --output <ws>/experiment_sep14/output/smoke \
  --file DIII_adelya_0007.jpg --device cuda:0 --batch-size 2

python digitize.py --input <ws>/data/12lead/DII --output <ws>/experiment_sep14/output/DII_b2_full_gpu \
  --device cuda:0 --batch-size 2

python digitize.py --input <ws>/data/12lead/DIII --output <ws>/experiment_sep14/output/DIII_b2_full_gpu \
  --device cuda:0 --batch-size 2
```

Use absolute `--input`/`--output` paths. A single `--file` run always uses an
actual batch of 1, so CPU smoke with `--batch-size 2` is safe.

Monitor runs with `python ../watch_experiments.py` for a one-shot status of all
`experiment_sep14/output` runs, or add `--watch 5` to refresh every 5 seconds.

## Segmentation batches

`--batch-size` overrides `batch_size` in [config.yml](config.yml) (default 2).
Pages batch only when resized dimensions match (no padding); CPU extraction
stays per-page, so batching only speeds up segmentation (~0.5 s/page vs
~3–16 s/page postprocessing). Max working batch for 7016×4964 pages on an
H100 95 GB card is **3**; `b=4` fails with
`input tensor must fit into 32-bit index math`, not OOM.

`run.log` records seg/post/total time, images/s, and CUDA peak
allocated/reserved MiB. `run_config.yml` records the effective batch size.

## Results (Sep 14, H100, b=2)

- Full DIII (56 files): 379 s, 0.148 img/s, 23647/34944 MiB.
- Full DII (87 files): 800 s, 0.109 img/s, same peaks.
- Smoke IDX=7: GPU ~63 s total, CPU ~104 s total for 4 files.
- 3/4 smokes agree CPU vs GPU (`max|ΔmV|` ≤ 0.002); `DII_tamir_eI_0007`
  flips (CPU `III`/13 rows vs GPU `V6`/12 rows) — 12-row detection instability.

## Layouts

[config.yml](config.yml) maps filenames to profiles: `tamir_adelya` (12-row
Tamir/Adelya) and `azamat_aygul` (3-row Azamat/Aygul). Pass
`--profile tamir_adelya` or `--profile azamat_aygul` when a filename does not
identify its profile. Only single-column full-width layouts are supported.

## Outputs

Each output directory contains `intermediate/<scan>_times_s.csv` and
`intermediate/<scan>_<lead>_mV.csv` per lead, `ecg_selected_waveforms.csv`
(frozen columns), `qc/lead_quality.csv` plus `qc/<scan>_<lead>.png`,
`run_config.yml`, and `run.log`. Lead selection measures coverage, not
R-peak quality; R-peaks and RR intervals are a later stage.

## Known bugs and limitations

Open-ECG-Digitizer scales voltage by mean H/V pixel density instead of
vertical only (upstream, unchanged; 25 mm/s and 10 mm/mV assumed). Missing
rows shift lead labels positionally — check `detected_rows` warnings and
`qc/*.png`; the wrapper does not relabel.
