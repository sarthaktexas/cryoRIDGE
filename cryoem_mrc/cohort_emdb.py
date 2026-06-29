"""EMDB metadata helpers for the thesis cohort."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

EMDB_ENTRY_API = "https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"

# Free-text patterns for depositor-reported map sharpening B (Å²), when present.
_SHARPENING_B_PATTERNS = (
    r"map\s+sharpening\s+b[\s\-]*factor[^0-9\-+]*([+-]?\d+(?:\.\d+)?)",
    r"sharpening\s+b[\s\-]*factor[^0-9\-+]*([+-]?\d+(?:\.\d+)?)",
    r"b[\s\-]*factor\s+sharpening[^0-9\-+]*([+-]?\d+(?:\.\d+)?)",
    r"sharpened\s+(?:with|using|at)\s+b\s*[=:]?\s*([+-]?\d+(?:\.\d+)?)",
)


def _walk_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            out.extend(_walk_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(_walk_strings(value))
    elif isinstance(obj, str):
        out.append(obj)
    return out


def parse_emdb_reported_sharpening_b(entry_json: dict[str, Any]) -> float | None:
    """
    Extract depositor-reported map sharpening B (Å²) from an EMDB entry JSON blob.

    EMDB rarely stores this in a structured field; we scan all string values for
    common publication phrases (e.g. "map sharpening B-factor: -67").
    """
    import re

    for text in _walk_strings(entry_json):
        for pattern in _SHARPENING_B_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    return None


def fetch_emdb_entry_json(
    emdb_id: str | int,
    *,
    timeout_s: float = 30.0,
    retries: int = 2,
    retry_delay_s: float = 0.5,
) -> dict[str, Any]:
    """Fetch the full EMDB REST entry JSON for one map."""
    eid = str(emdb_id).strip().removeprefix("EMD-").removeprefix("emd-")
    url = EMDB_ENTRY_API.format(emdb_id=eid)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(retry_delay_s)
    if last_err is not None:
        raise RuntimeError(f"EMDB API failed for EMD-{eid}: {last_err}") from last_err
    return {}


def fetch_emdb_reported_sharpening_b(
    emdb_id: str | int,
    *,
    timeout_s: float = 30.0,
    retries: int = 2,
    retry_delay_s: float = 0.5,
) -> float | None:
    """Query EMDB for a depositor-reported map sharpening B (Å²), if present."""
    data = fetch_emdb_entry_json(
        emdb_id,
        timeout_s=timeout_s,
        retries=retries,
        retry_delay_s=retry_delay_s,
    )
    return parse_emdb_reported_sharpening_b(data)


def parse_emdb_global_resolution_a(entry_json: dict[str, Any]) -> float | None:
    """
    Extract author-reported global resolution (Å) from an EMDB entry JSON blob.

    Uses ``final_reconstruction.resolution`` from the first structure determination.
    """
    try:
        sd = entry_json["structure_determination_list"]["structure_determination"]
        if not sd:
            return None
        ip = sd[0]["image_processing"]
        if not ip:
            return None
        res = ip[0]["final_reconstruction"]["resolution"]
        val = res.get("valueOf_")
        return float(val) if val is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def fetch_emdb_global_resolution_a(
    emdb_id: str | int,
    *,
    timeout_s: float = 30.0,
    retries: int = 2,
    retry_delay_s: float = 0.5,
) -> float | None:
    """Query the EMDB REST API for one entry's global resolution (Å)."""
    data = fetch_emdb_entry_json(
        emdb_id,
        timeout_s=timeout_s,
        retries=retries,
        retry_delay_s=retry_delay_s,
    )
    return parse_emdb_global_resolution_a(data)


def _contour_level_from_list(contours: list[dict[str, Any]] | None) -> float | None:
    """Pick the author primary contour level from an EMDB ``contour_list``."""
    if not contours:
        return None
    for item in contours:
        if item.get("primary") and item.get("level") is not None:
            try:
                return float(item["level"])
            except (TypeError, ValueError):
                continue
    for item in contours:
        if item.get("level") is not None:
            try:
                return float(item["level"])
            except (TypeError, ValueError):
                continue
    return None


def parse_emdb_recommended_contour(entry_json: dict[str, Any]) -> float | None:
    """
    Extract the depositor-recommended primary map contour from an EMDB entry JSON blob.

    Uses ``map.contour_list`` (sharpened primary map). Falls back to the first half-map
    primary contour when the primary map level is absent.
    """
    try:
        level = _contour_level_from_list(entry_json["map"]["contour_list"].get("contour"))
        if level is not None:
            return level
    except (KeyError, TypeError, AttributeError):
        pass
    try:
        for half_map in entry_json["interpretation"]["half_map_list"]["half_map"]:
            level = _contour_level_from_list(half_map.get("contour_list", {}).get("contour"))
            if level is not None:
                return level
    except (KeyError, TypeError, AttributeError):
        pass
    return None


def fetch_emdb_recommended_contour(
    emdb_id: str | int,
    *,
    timeout_s: float = 30.0,
    retries: int = 2,
    retry_delay_s: float = 0.5,
) -> float | None:
    """Query EMDB for the depositor-recommended primary map contour level."""
    data = fetch_emdb_entry_json(
        emdb_id,
        timeout_s=timeout_s,
        retries=retries,
        retry_delay_s=retry_delay_s,
    )
    return parse_emdb_recommended_contour(data)
