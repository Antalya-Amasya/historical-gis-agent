"""Read-only provenance schema and inventory helpers for local HGT directories."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Any


UNKNOWN = "UNKNOWN"
_TILE_NAME = re.compile(r"([NS])(\d{2})([EW])(\d{3})\.hgt$", re.IGNORECASE)


@dataclass(frozen=True)
class DEMProvenance:
    source_name: str = UNKNOWN
    source_url: str = UNKNOWN
    acquisition_date: str = UNKNOWN
    acquisition_method: str = UNKNOWN
    license: str = UNKNOWN


@dataclass(frozen=True)
class DEMInventory:
    directory: str
    generated_at: str
    tile_count: int
    total_bytes: int
    filename_valid_count: int
    raster_shape: str
    byte_order: str
    sample_type: str
    latitude_southwest_bounds: tuple[int, int] | None
    longitude_southwest_bounds: tuple[int, int] | None
    provenance: DEMProvenance

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_hgt_inventory(directory: str | Path) -> DEMInventory:
    """Inventory file metadata only; it does not infer acquisition provenance."""
    root = Path(directory)
    tiles = sorted(root.glob("*.hgt")) if root.is_dir() else []
    coordinates: list[tuple[int, int]] = []
    valid = 0
    for tile in tiles:
        match = _TILE_NAME.fullmatch(tile.name)
        if match is None:
            continue
        valid += 1
        latitude = int(match.group(2)) * (1 if match.group(1).upper() == "N" else -1)
        longitude = int(match.group(4)) * (1 if match.group(3).upper() == "E" else -1)
        coordinates.append((latitude, longitude))
    return DEMInventory(
        directory=str(root),
        generated_at=datetime.now(timezone.utc).isoformat(),
        tile_count=len(tiles),
        total_bytes=sum(tile.stat().st_size for tile in tiles),
        filename_valid_count=valid,
        raster_shape="1201x1201 expected for SRTM3-style HGT",
        byte_order="big-endian",
        sample_type="signed int16",
        latitude_southwest_bounds=(min(lat for lat, _ in coordinates), max(lat for lat, _ in coordinates)) if coordinates else None,
        longitude_southwest_bounds=(min(lon for _, lon in coordinates), max(lon for _, lon in coordinates)) if coordinates else None,
        provenance=DEMProvenance(),
    )


def load_dem_manifest(path: str | Path | None) -> dict[str, Any]:
    """Read an optional manifest without manufacturing acquisition metadata."""
    if path is None or not Path(path).is_file():
        return {"provenance": asdict(DEMProvenance())}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DEM manifest must be a JSON object")
    return payload
