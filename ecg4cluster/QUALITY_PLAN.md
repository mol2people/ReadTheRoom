# ECG quality plan

Goal: validate digitized beat timing and page alignment before synchrony analysis.

## 1. Establish a reference

- [ ] Annotate representative clean, noisy, short, and missing-row pages: lead identity, beats, and grid scale.
- [ ] Set acceptance tolerances for missed/extra beats, timing error, and calibration.
- [ ] Check the current wrapper against these references; keep older `test/` results separate.

## 2. Expose uncertainty

- [ ] Export `row_count_matches`, assignment status (`unverified` / `verified` / `failed`), and QC reasons.
- [ ] Do not infer label confidence from counts: upstream ignores lead I; matching counts cannot prove identity.
- [ ] Handle empty input and all-invalid signals explicitly. Preserve raw exports; flag segments excluded from analysis.

## 3. Establish alignment

- [ ] Transcribe or OCR printed timestamps, paper speed, and gain into a reviewed page manifest.
- [ ] Verify timestamp meaning, waveform crop offset, and `.atc` clock offset/drift; record alignment uncertainty.
- [ ] Preserve gaps and overlaps; compute no RR interval across an unverified page boundary or missing segment.

## 4. Validate beat QC

- [ ] Evaluate beat detection against the annotated reference before using it for lead selection.
- [ ] Record beat count, detector agreement, coverage, and rejection reasons per lead/segment.
- [ ] Do not reward low RR variability or use a universal R-amplitude cutoff as proof of quality.

## 5. Make calibration explicit

- [ ] Export per-page `fs_hz = (N - 1) / duration_s`, grid spacing, retained pixel width/crop bounds, and gain/speed source.
- [ ] Validate timing and vertical voltage scaling against the printed grid/calibration pulse before changing calibration.
- [ ] Add fixed-rate output only if downstream needs it; preserve duration and missing segments.

## 6. Make results reproducible

- [ ] Record actual wrapper/upstream revisions and dirty states, input/layout/weight hashes, and dependency versions.
- [ ] Pin the validated environment; add focused regression and failure-case checks.
- [ ] Verify references and QC decisions before processing the full dataset.

Later: wire in DI/DIII layouts and measure cluster performance.
