# Pointcloud Scanner

Standalone project for LiDAR-like scanning of industrial 3D scenes.

For a full Windows/Linux runbook for the whole stack (factory generation + scanning), see:
- `../docs/FULL_GUIDE_RU.md`

Input:
- OBJ scene file

Output:
- per-station circle clouds (`.ply`)
- stitched cloud (`combined/interior/exterior`)
- metadata JSON
- run summary JSON

## 1) Installation

```powershell
cd pointcloud_scanner
uv sync
```

## 2) Run

```powershell
uv run pointcloud-scanner --config configs/scanner.default.yaml --input-obj ../out/factory.obj
```

Quick test:

```powershell
uv run pointcloud-scanner --config configs/scanner.fast.yaml --input-obj ../out/factory.obj
```

## 3) Output structure

By default:

`out/pointcloud_scans/<YYYY-MM-DD>/<run_timestamp>/`

Contains:
- `circles/*.ply` - separate station scans
- `combined/factory_stitched.ply` - merged cloud
- `interior/factory_stitched_interior.ply`
- `exterior/factory_stitched_exterior.ply`
- `scan_dataset.json` - stations/circles metadata
- `run_summary.json` - compact run report

## 4) Config

Main config sections:
- `paths` - input/output paths and filenames
- `output` - export toggles and folder strategy
- `lidar` - scanner geometry, density, station planning, semantic labels

Default config:
- `configs/scanner.default.yaml`

Fast config:
- `configs/scanner.fast.yaml`

More details:
- `docs/CONFIG.md`

## 5) CLI options

```powershell
uv run pointcloud-scanner --help
```

Useful flags:
- `--input-obj` override source OBJ
- `--output-dir` override output root
- `--write-default-config path.yaml` write template config and exit
