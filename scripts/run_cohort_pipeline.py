"""Thin wrapper — prefer ``halfmap-qc cohort`` (installed via cryoem-halfmap-qc)."""

from __future__ import annotations

from cryoem_mrc.cohort_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
