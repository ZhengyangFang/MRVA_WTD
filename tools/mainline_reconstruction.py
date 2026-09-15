from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Runs the canonical groundwater reconstruction.
MAINLINE_RECON_NAME = "RECON_MAIN_2011_2023"
MAINLINE_RECON_ROOT = ROOT / "outputs" / MAINLINE_RECON_NAME

MAINLINE_RECON_DESCRIPTION = (
    "13 January anchors from 2011-01 to 2023-01, 2023-12 final endpoint anchor, "
    "and well-derived regional storage-mass correction. GRACE is reserved for validation."
)
