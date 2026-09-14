#!/usr/bin/env python3
import argparse
import logging
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from open_ecg_runner import OpenECGRunner


PROJECT_DIR = Path(__file__).resolve().parent
OPEN_ECG_DIR = PROJECT_DIR / "vendor" / "Open-ECG-Digitizer"
OPEN_ECG_REVISION = "v1.9.3 (97a15087d4abcda843da8c58ee74b1d8f47e6f9a)"
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digitize ECG scans with Open-ECG-Digitizer.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, help="Maximum segmentation batch size; defaults to config.yml.")
    parser.add_argument("--file", help="Process only this exact filename.")
    parser.add_argument("--profile", help="Use one named profile instead of matching filenames.")
    return parser.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ecg4cluster")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_file, mode="w"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def metadata(path: Path, config: dict) -> dict[str, str]:
    pattern = config.get("metadata_regex")
    match = re.fullmatch(pattern, path.name) if pattern else None
    return match.groupdict() if match else {}


def sort_key(path: Path, config: dict) -> tuple:
    fields = metadata(path, config)
    experiments = config.get("experiment_order", [])
    experiment = fields.get("experiment", "")
    experiment_index = experiments.index(experiment) if experiment in experiments else len(experiments)
    scan_index = fields.get("scan_index", "")
    scan_order = (0, int(scan_index)) if scan_index.isdigit() else (1, scan_index)
    return fields.get("participant", ""), experiment_index, scan_order, path.name


def profile_for(path: Path, profiles: dict, override: str | None) -> tuple[str, dict]:
    if override:
        return override, profiles[override]
    for name, profile in profiles.items():
        if re.search(profile["filename_regex"], path.name):
            return name, profile
    raise ValueError(f"No profile matches {path.name}; add one to config.yml or pass --profile")


def layout_leads(layout: dict) -> list[str]:
    leads = []
    for row in layout["leads"]:
        leads.extend(row if isinstance(row, list) else [row])
    leads.extend(layout.get("rhythm_leads", []))
    leads = [lead.removeprefix("-") for lead in leads]
    return list(dict.fromkeys(lead for lead in leads if lead in CANONICAL_LEADS))


def lead_stats(values: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values)
    completeness = float(finite.mean())
    padded = np.r_[False, finite, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    longest = int(np.max(edges[1::2] - edges[::2])) if finite.any() else 0
    return completeness, longest / len(values)


def choose_lead(stats: dict[str, tuple[float, float]], preferred: list[str]) -> str:
    best_coverage = max(longest for _, longest in stats.values())
    near_best = [lead for lead in stats if stats[lead][1] >= best_coverage - 0.01]
    for lead in preferred:
        if lead in near_best:
            return lead
    return max(near_best, key=lambda lead: (stats[lead][1], stats[lead][0]))


def save_plot(path: Path, time_s: np.ndarray, values: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time_s, values, color="black", linewidth=0.7)
    ax.set(xlabel="Time (s)", ylabel="mV", title=title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def segmented_pages(runner: OpenECGRunner, paths: list[Path], batch_size: int, logger: logging.Logger):
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        started = time.perf_counter()
        probabilities = runner.segment_batch(batch_paths)
        if runner.device.startswith("cuda"):
            torch.cuda.synchronize(runner.device)
        shapes = {tuple(probability.shape[2:]) for probability in probabilities}
        logger.info(
            "segmentation pages=%d shape_groups=%d elapsed_s=%.3f",
            len(batch_paths), len(shapes), time.perf_counter() - started,
        )
        yield from zip(batch_paths, probabilities)
        # Release the complete previous batch before allocating the next one.
        del probabilities


def main() -> None:
    args = arguments()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    config_dir = config_path.parent
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    intermediate_dir = output_dir / "intermediate"
    qc_dir = output_dir / "qc"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "run.log")

    profiles = config["profiles"]
    layouts = {
        name: OpenECGRunner.load_layout(config_dir / profile["layout_file"])
        for name, profile in profiles.items()
    }
    extensions = {suffix.lower() for suffix in config["image_extensions"]}
    paths = [input_dir / args.file] if args.file else [
        path for path in input_dir.iterdir() if path.suffix.lower() in extensions
    ]
    paths.sort(key=lambda path: sort_key(path, config))
    if not paths:
        raise ValueError(f"No input images found in {input_dir}")

    first_profile, _ = profile_for(paths[0], profiles, args.profile)
    run_config = dict(config)
    run_config["runtime"] = {
        "device": args.device,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "open_ecg_revision": OPEN_ECG_REVISION,
        "file": args.file,
        "profile_override": args.profile,
        "batch_size": batch_size,
    }
    (output_dir / "run_config.yml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    logger.info("files=%d device=%s batch_size=%d config=%s", len(paths), args.device, batch_size, config_path)

    runner = OpenECGRunner(
        repository=OPEN_ECG_DIR,
        device=args.device,
        resample_size=config["resample_size"],
        target_samples=config["target_samples"],
        initial_layout_file=config_dir / profiles[first_profile]["layout_file"],
    )

    combined_parts = []
    quality_rows = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    processing_started = time.perf_counter()
    for image_path, probabilities in segmented_pages(runner, paths, batch_size, logger):
        started = time.perf_counter()
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])

        profile_name, profile = profile_for(image_path, profiles, args.profile)
        result = runner.digitize_probabilities(probabilities, layouts[profile_name])
        del probabilities
        layout_name = result["layout_name"]
        layout = layouts[profile_name][layout_name]
        leads = layout_leads(layout)
        expected_rows = layout["total_rows"]
        if result["detected_rows"] != expected_rows:
            logger.warning(
                "file=%s expected_rows=%d detected_rows=%d lead labels may be shifted",
                image_path.name,
                expected_rows,
                result["detected_rows"],
            )
        canonical_uv = result["signal"]["canonical_lines"].squeeze().cpu().numpy()
        canonical_mv = canonical_uv / 1000.0 * (10.0 / config["gain_mm_mV"])
        time_s = runner.time_axis(result, config["paper_speed_mm_s"])
        base = image_path.stem

        pd.DataFrame({"time_s": time_s}).to_csv(
            intermediate_dir / f"{base}_times_s.csv", index=False
        )
        values_by_lead = {}
        stats = {}
        for lead in leads:
            values = canonical_mv[CANONICAL_LEADS.index(lead)]
            values_by_lead[lead] = values
            stats[lead] = lead_stats(values)
            pd.DataFrame({"mV": values}).to_csv(
                intermediate_dir / f"{base}_{lead}_mV.csv", index=False
            )

        selected = choose_lead(stats, profile.get("preferred_leads", []))
        selected_values = values_by_lead[selected]
        scan_metadata = metadata(image_path, config)
        part = pd.DataFrame(
            {
                "scan_file": image_path.name,
                "profile": profile_name,
                **scan_metadata,
                "selected_lead": selected,
                "sample_index": np.arange(len(time_s)),
                "time_s": time_s,
                "mV": selected_values,
                "is_scan_start": np.arange(len(time_s)) == 0,
            }
        )
        combined_parts.append(part)

        for lead in leads:
            completeness, longest = stats[lead]
            quality_rows.append(
                {
                    "scan_file": image_path.name,
                    "profile": profile_name,
                    "lead": lead,
                    "completeness": completeness,
                    "longest_complete_fraction": longest,
                    "selected": lead == selected,
                    "duration_s": time_s[-1],
                    "layout": layout_name,
                    "layout_cost": float(result["signal"]["layout_matching_cost"]),
                    "expected_rows": expected_rows,
                    "detected_rows": result["detected_rows"],
                    "detected_lead_labels": result["detected_lead_labels"],
                }
            )

        save_plot(
            qc_dir / f"{base}_{selected}.png",
            time_s,
            selected_values,
            f"{base} — {selected}",
        )
        logger.info(
            "file=%s profile=%s layout=%s cost=%.4f duration_s=%.3f selected=%s completeness=%.4f postprocess_elapsed_s=%.1f",
            image_path.name,
            profile_name,
            layout_name,
            float(result["signal"]["layout_matching_cost"]),
            time_s[-1],
            selected,
            stats[selected][0],
            time.perf_counter() - started,
        )

    combined = pd.concat(combined_parts, ignore_index=True)
    combined.to_csv(output_dir / "ecg_selected_waveforms.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(qc_dir / "lead_quality.csv", index=False)
    elapsed = time.perf_counter() - processing_started
    logger.info(
        "complete rows=%d output=%s elapsed_s=%.3f images_per_s=%.4f",
        len(combined), output_dir, elapsed, len(paths) / elapsed,
    )
    if args.device.startswith("cuda"):
        logger.info(
            "peak_gpu_allocated_mib=%.1f peak_gpu_reserved_mib=%.1f",
            torch.cuda.max_memory_allocated(args.device) / 1024**2,
            torch.cuda.max_memory_reserved(args.device) / 1024**2,
        )


if __name__ == "__main__":
    main()
