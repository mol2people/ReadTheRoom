#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$project_dir/vendor/Open-ECG-Digitizer"

git clone --depth 1 --branch v1.9.3 https://github.com/Ahus-AIM/Open-ECG-Digitizer.git "$repository"
git -C "$repository" lfs pull
