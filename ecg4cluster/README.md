# ecg4cluster

A small Capella workflow around
[Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer) v1.9.3.

It digitizes full-width ECG traces, exports every displayed lead, selects the
most complete preferred lead, and combines the selected waveforms into one CSV.
Pages remain separate: time restarts at zero and `is_scan_start` marks the first
sample of every page.

## Capella setup

On the login node, allocate a workspace and confirm its actual path:

```bash
ws_allocate --filesystem horse ecg 100
ws_list
```

Copy the archive from the local computer:

```bash
scp ~/Documents/ReadTheRoom_PhysiologicalSynchrony/Archive.zip \
  scpTUD:/data/horse/ws/buza314h-ecg/
```

On Capella, unpack the data, load modules, create the venv, and request a GPU:

```bash
cd /data/horse/ws/buza314h-ecg
unzip Archive.zip -d data

module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
python3 -m venv --system-site-packages venv
srun --partition=capella --nodes=1 --gres=gpu:1 --time=03:00:00 --pty bash
```

On the GPU node, activate the venv and enter this project:

```bash
source /data/horse/ws/buza314h-ecg/venv/bin/activate
cd /data/horse/ws/buza314h-ecg/ecg4cluster

git --version
git lfs version
bash fetch_open_ecg.sh
python -m pip install -r requirements.txt
```

Confirm that PyTorch sees the H100:

```bash
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

These commands assume this repository is at `.../ecg4cluster` and the extracted
images are under `.../data`. Adjust those two paths to match the archive.

## Run

Test one page from each supplied layout before processing the full directory:

```bash
python digitize.py --input ../data/12lead/DII --output output/test_tamir \
  --file DII_tamir_eI_0011.jpg --device cuda:0

python digitize.py --input ../data/12lead/DII --output output/test_azamat \
  --file DII_azamat_eI_0011.jpg --device cuda:0
```

Then run the complete directory:

```bash
python digitize.py --input ../data/12lead/DII \
  --output output/output_DII --device cuda:0
```

Archive the complete run, including its log and configuration snapshot:

```bash
tar -czf output_DII.tar.gz -C output output_DII
```

## Layouts

The Python is generic; [config.yml](config.yml) is specific to these filenames
and page formats. Each profile defines a filename regex, a layout file, and an
ordered list of preferred leads. Pass `--profile tamir` or `--profile azamat`
when a filename does not identify its profile.

Layout files live under `layouts/`. A profile's layout file may contain several
candidate layouts, from which OpenECG selects one for each page.

Only single-column layouts containing full-width signal rows are supported and
tested. This dataset has no multi-column ECG pages. Standard 4x3 and other
short-segment layouts are outside the current scope.

## Outputs

Each output directory contains:

- `intermediate/<scan>_times_s.csv`: one page-local time array;
- `intermediate/<scan>_<lead>_mV.csv`: one array for each displayed lead;
- `ecg_selected_waveforms.csv`: selected waveform from every page;
- `qc/lead_quality.csv`: coverage, selected lead, layout, and row counts;
- `qc/<scan>_<lead>.png`: selected-lead plot;
- `run_config.yml`: configuration, paths, Torch/CUDA versions, and GPU name;
- `run.log`: processing log.

Selection measures data coverage, not R-peak visibility or physiological
quality. Configured lead preference is used only between leads with nearly
equal coverage. R-peak detection and R-R intervals remain a later stage.

## Known bugs and limitations

### Amplitude calibration

Open-ECG-Digitizer converts signal height using the average of horizontal and
vertical pixel densities. Voltage would theoretically use only vertical pixel
density. This upstream behavior is intentionally left unchanged. The pipeline
assumes 25 mm/s and the confirmed 10 mm/mV calibration, but mV values may still
be biased when detected horizontal and vertical grid scaling differ.

### Missing rows can scramble lead labels

A full 12-lead page does not always produce 12 detected signal rows. OpenECG
assigns the rows it did detect positionally to the configured layout. If one row
is missing, later signals can shift into the wrong canonical labels—for example,
the true lead III may be exported as II or I. This has occurred in Tamir's scans.

The warning that the number of detected peaks differs from the number of merged
lines is therefore important. Row counts and mismatches are also saved in
`qc/lead_quality.csv` and `run.log`. Visually compare affected 12-lead outputs
with the source page; this wrapper does not automatically correct label shifts.
