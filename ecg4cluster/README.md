# ecg4cluster

ZIH GPU wrapper around
[Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer) v1.9.3.
Full-width traces → all leads → most-complete preferred lead in
`ecg_selected_waveforms.csv`. Time restarts per page (`is_scan_start`).

## Setup

### First time

```bash
ws_allocate --filesystem horse ecg 100
ws_list
scp ~/Documents/ReadTheRoom_PhysiologicalSynchrony/Archive.zip \
  scpTUD:/data/horse/ws/buza314h-ecg/

cd /data/horse/ws/buza314h-ecg
unzip Archive.zip -d data

module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
python3 -m venv --system-site-packages venv
source venv/bin/activate
cd experiment_sep14/ReadTheRoom/ecg4cluster
bash fetch_open_ecg.sh
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.cuda.get_device_name(0))'
```

### Every session

Use `tmux` (`tmux new -s ecg`; detach `Ctrl-b d`; reattach
`tmux attach -t ecg`). SSH drop kills the allocation.

```bash
salloc --account=p_epoch_data --partition=capella --job-name=ecg --nodes=1 \
  --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --mem=100G --time=06:00:00
srun --pty bash
```

100G host RAM is overkill: `b=2` reserves ~35 GB GPU; 32–64G host is enough.

```bash
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
source /data/horse/ws/buza314h-ecg/venv/bin/activate
cd /data/horse/ws/buza314h-ecg/experiment_sep14/ReadTheRoom/ecg4cluster
```

Alpha: `salloc --account=p_epoch_data --partition=alpha --cpus-per-task=6
--gres=gpu:1 --mem=64G --time=03:00:00`, then
`OMP/MKL/OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK` (max 6 CPUs/GPU, A100 40 GiB).

Push: `git config --global credential.helper 'cache --timeout=28800'`, then
username + classic PAT (`repo` scope) as password.

## Run

`--batch-size` overrides [config.yml](config.yml) (`batch_size: 2`). No config
edit. Use absolute `--input`/`--output`. `--file` always runs as batch 1.

Pages of equal resized shape stack; no padding. CPU extraction stays per-page,
so batching only speeds segmentation (~0.5 s/page vs ~3–16 s postprocess).
`run.log`: seg/post/total, images/s, CUDA peak alloc/reserved MiB.
`run_config.yml` records the effective batch size.

### Smoke test

One page per layout, CPU and GPU:

```bash
python digitize.py --input <ws>/data/12lead/DIII --output <ws>/experiment_sep14/output/smoke \
  --file DIII_adelya_0007.jpg --device cuda:0 --batch-size 2
```

### Full run

```bash
python digitize.py --input <ws>/data/12lead/DII --output <ws>/experiment_sep14/output/DII_b2_full_gpu \
  --device cuda:0 --batch-size 2

python digitize.py --input <ws>/data/12lead/DIII --output <ws>/experiment_sep14/output/DIII_b2_full_gpu \
  --device cuda:0 --batch-size 2
```

Monitor: `python ../watch_experiments.py` or `--watch 5`.

## Results (Sep 14, H100, b=2)

- DIII 56 files: 379 s, 0.148 img/s, 23.6/34.9 GB
- DII 87 files: 800 s, 0.109 img/s
- Smoke (4 files): GPU 63 s vs CPU 104 s. 3/4 same lead (`max|ΔmV|` ≤ 0.002);
  `DII_tamir_eI_0007` CPU III/13 rows vs GPU V6/12 rows.

## Layouts

[config.yml](config.yml): `tamir_adelya` (12-row Tamir/Adelya), `azamat_aygul`
(3-row Azamat/Aygul). Pass `--profile` if the filename does not match. Only
single-column full-width layouts.

## Outputs

- `intermediate/<scan>_times_s.csv`, `intermediate/<scan>_<lead>_mV.csv`
- `ecg_selected_waveforms.csv` (frozen columns)
- `qc/lead_quality.csv`, `qc/<scan>_<lead>.png`
- `run_config.yml`, `run.log`

Selection is coverage, not R-peak quality. R-peaks/RR are a later stage.

## Known bugs and limitations

Open-ECG-Digitizer scales voltage by mean H/V pixel density, not vertical only
(upstream, unchanged; 25 mm/s, 10 mm/mV assumed). Missing rows shift lead
labels positionally — check `detected_rows` and `qc/*.png`; the wrapper does
not relabel.

### 32-bit tensor ceiling

`b=4` on 7016×4964 pages fails in the segmentation conv:
`RuntimeError: input tensor must fit into 32-bit index math`. Not GPU OOM —
stacked UNet feature maps exceed 2³¹ elements. More VRAM does not help. Max
working batch is **3** (~34 GB reserved at `b=2`, ~49 GB at `b=3` on H100 95 GB).
Keep default `b=2`: `b=2→3` was only ~2.5% faster because postprocess dominates.

If raising the cap later: split shape-groups in `segment_batch` when
`N×C×H×W` would exceed 2³¹ (conservative C ≈ 128–320); optionally reject
`--batch-size > 3` in CLI. Do not tile pages or lower `resample_size` — both
risk waveform quality for unused throughput. PyTorch CUDA conv stays 32-bit.
