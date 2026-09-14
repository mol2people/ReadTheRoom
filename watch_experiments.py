#!/usr/bin/env python3
"""Live status monitor for experiment_sep14 digitization runs."""

import argparse
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


_HERE = Path(__file__).resolve().parent
if (_HERE / "experiment_sep14").is_dir():
    # Script lives at the workspace root: <ws>/watch_experiments.py.
    PROJECT_DIR = _HERE
    EXPERIMENT_DIR = PROJECT_DIR / "experiment_sep14"
else:
    # Script lives inside the checked-out repo: <ws>/experiment_sep14/ReadTheRoom.
    PROJECT_DIR = _HERE.parent
    EXPERIMENT_DIR = PROJECT_DIR
LOG_NAME = "run.log"


def find_output_dirs(base: Path) -> list[Path]:
    """Find directories that look like digitization output folders."""
    dirs: list[Path] = []
    if not base.exists():
        return dirs
    for path in base.rglob("*"):
        if not path.is_dir():
            continue
        # Output dirs either contain run.log or are named like benchmark/retest runs.
        if (path / LOG_NAME).exists() or re.search(r"_(b\d+|e[IV]+_\d+|retest)", path.name):
            dirs.append(path)
    # Prefer directories closer to the experiment root; stable sort.
    dirs.sort(key=lambda p: (len(p.parts), str(p)))
    return dirs


def parse_run_log(log_path: Path) -> dict:
    """Extract status information from a run.log file."""
    status: dict = {
        "has_log": True,
        "total_files": None,
        "processed_files": 0,
        "last_file": None,
        "last_time": None,
        "complete": False,
        "warnings": 0,
        "errors": 0,
        "latest_lines": [],
    }

    try:
        text = log_path.read_text()
    except OSError as exc:
        status["latest_lines"] = [f"cannot read log: {exc}"]
        return status

    lines = text.splitlines()
    if not lines:
        return status

    # First line typically has files=N device=... batch_size=N
    first = lines[0]
    m = re.search(r"files=(\d+)", first)
    if m:
        status["total_files"] = int(m.group(1))

    # Count processed files and warnings/errors.
    file_re = re.compile(r"\bfile=([^\s]+)")
    for line in lines:
        if "file=" in line and "INFO" in line and "profile=" in line:
            fm = file_re.search(line)
            if fm:
                status["processed_files"] += 1
                status["last_file"] = fm.group(1)
                status["last_time"] = line.split()[0]
        if "WARNING" in line:
            status["warnings"] += 1
        if "ERROR" in line:
            status["errors"] += 1

    # A run is complete if any log message starts with "complete".
    # Log format: "<timestamp> <LEVEL> <message...>", so the message starts at index 2.
    status["complete"] = any(
        len(line.split()) >= 4 and line.split()[3] == "complete" for line in lines
    )

    # Keep the last few relevant lines for context.
    status["latest_lines"] = lines[-3:]
    return status


def describe_output_dir(out_dir: Path) -> dict:
    """Return a status record for an output directory."""
    log_path = out_dir / LOG_NAME
    if log_path.exists():
        record = parse_run_log(log_path)
        record["mtime"] = log_path.stat().st_mtime
    else:
        record = {
            "has_log": False,
            "total_files": None,
            "processed_files": 0,
            "last_file": None,
            "last_time": None,
            "complete": False,
            "warnings": 0,
            "errors": 0,
            "latest_lines": [],
            "mtime": None,
        }
    record["path"] = out_dir
    return record


def find_active_processes() -> list[dict]:
    """Find running digitization/benchmark processes for this experiment."""
    try:
        proc = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    matches: list[dict] = []
    for line in proc.stdout.splitlines():
        if "watch_experiments.py" in line:
            continue
        if "grep" in line:
            continue
        if "digitize.py" in line or "ecg4cluster" in line or str(EXPERIMENT_DIR) in line:
            parts = line.split(None, 10)
            if len(parts) >= 11:
                matches.append(
                    {
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "time": parts[9],
                        "cmd": parts[10],
                    }
                )
    return matches


def fmt_time(s: str | None) -> str:
    if not s:
        return "—"
    return s


def progress_bar(done: int, total: int | None, width: int = 20) -> str:
    if total is None or total <= 0:
        return "?".ljust(width)
    filled = int(round(width * done / total))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def print_report(records: list[dict], processes: list[dict], args: argparse.Namespace) -> None:
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f" experiment_sep14 monitor — {now_str} ".center(80, "="))
    print()

    # Active processes
    if processes:
        print("ACTIVE PROCESSES")
        print("-" * 80)
        for p in processes:
            print(f"  pid {p['pid']:>7}  cpu {p['cpu']:>5}  mem {p['mem']:>5}  time {p['time']:>8}  {p['cmd'][:55]}")
        print()
    else:
        print("ACTIVE PROCESSES: none")
        print()

    # Per-run status
    print("RUN STATUS")
    print("-" * 80)
    for rec in records:
        rel = rec["path"].relative_to(PROJECT_DIR)
        total = rec["total_files"]
        done = rec["processed_files"]
        if rec["has_log"]:
            if rec["complete"]:
                state = "DONE"
            elif rec["mtime"] is not None and (now.timestamp() - rec["mtime"]) <= args.stale:
                state = "RUNNING"
            else:
                state = "PARTIAL"
            bar = progress_bar(done, total)
            pct = f"{done}/{total}" if total else f"{done}/?"
            warn_err = []
            if rec["warnings"]:
                warn_err.append(f"{rec['warnings']} warn")
            if rec["errors"]:
                warn_err.append(f"{rec['errors']} err")
            extra = f"  ({', '.join(warn_err)})" if warn_err else ""
            print(f"  {state:8} {bar} {pct:>8}  {rel}{extra}")
            if args.verbose:
                print(f"           last: {fmt_time(rec['last_file'])} @ {fmt_time(rec['last_time'])}")
        else:
            print(f"  PENDING  {'░' * 20}     ?/?  {rel}")
    print()

    # Summary
    completed = sum(1 for r in records if r["complete"])
    running = sum(1 for r in records if not r["complete"] and r["has_log"])
    pending = sum(1 for r in records if not r["has_log"])
    total_warns = sum(r["warnings"] for r in records)
    total_errs = sum(r["errors"] for r in records)
    print("SUMMARY")
    print("-" * 80)
    print(f"  output dirs: {len(records)}   done: {completed}   running/partial: {running}   pending: {pending}")
    print(f"  warnings: {total_warns}   errors: {total_errs}")
    print()

    if args.verbose:
        print("LATEST LOG LINES (last 3 per run)")
        print("-" * 80)
        for rec in records:
            if rec["has_log"] and rec["latest_lines"]:
                rel = rec["path"].relative_to(PROJECT_DIR)
                print(f"  {rel}:")
                for line in rec["latest_lines"]:
                    print(f"    {line}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor experiment_sep14 digitization runs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show latest log lines per run")
    parser.add_argument(
        "--watch",
        type=int,
        metavar="SEC",
        help="refresh every SEC seconds (like watch); omit for one-shot report",
    )
    parser.add_argument(
        "--stale",
        type=int,
        default=60,
        metavar="SEC",
        help="treat a log as RUNNING if modified within SEC seconds (default: 60)",
    )
    args = parser.parse_args()

    if not EXPERIMENT_DIR.exists():
        print(f"experiment dir not found: {EXPERIMENT_DIR}", file=sys.stderr)
        return 1

    def tick():
        out_dirs = find_output_dirs(EXPERIMENT_DIR)
        records = [describe_output_dir(d) for d in out_dirs]
        processes = find_active_processes()
        print_report(records, processes, args)

    if args.watch:
        import time

        try:
            while True:
                # Clear screen between refreshes for a dashboard feel.
                print("\033[2J\033[H", end="")
                tick()
                sys.stdout.flush()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nmonitor stopped.")
            return 0
    else:
        tick()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
