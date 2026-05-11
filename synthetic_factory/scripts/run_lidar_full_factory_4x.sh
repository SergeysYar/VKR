#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/configs/presets/scene_lidar_full_factory_4x.json"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Full factory 4x config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

SEED_VALUE="${SEED:-42}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

uv sync --project "${PROJECT_ROOT}" --all-extras
uv run --project "${PROJECT_ROOT}" -m synthetic_factory \
  --config "${CONFIG_PATH}" \
  --seed "${SEED_VALUE}" \
  --log-level INFO \
  "$@"
