"""Reproducibly acquire the official Itiner-e nightly NDJSON export.

This module deliberately does not interpret the network as a routing graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ITINER_E_EXPORT_URL = "https://itiner-e.org/route-segments/download"
ITINER_E_LICENSE = "CC BY 4.0"
ITINER_E_CITATION = "Brughmans, T., de Soto, P., Pažout, A. and Bjerregaard Vahlstrup (2024), Itiner-e: the digital atlas of ancient roads, https://itiner-e.org"


@dataclass(frozen=True)
class DownloadResult:
    output: Path
    byte_size: int
    sha256: str | None
    skipped: bool


def _response_status(response: object) -> int:
    return int(getattr(response, "status", getattr(response, "code", 200)))


def download_export(
    output: Path,
    *,
    url: str = ITINER_E_EXPORT_URL,
    timeout: float = 60.0,
    force: bool = False,
    calculate_sha256: bool = False,
    opener: Callable[..., BinaryIO] = urlopen,
) -> DownloadResult:
    """Stream an export to a sibling temporary file, then atomically replace."""
    output = Path(output)
    if output.exists() and not force:
        return DownloadResult(output, output.stat().st_size, None, True)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256() if calculate_sha256 else None
    temp_path: Path | None = None
    try:
        with opener(url, timeout=timeout) as response:
            status = _response_status(response)
            if not 200 <= status < 300:
                raise RuntimeError(f"Itiner-e download failed with HTTP {status}")
            with NamedTemporaryFile("wb", delete=False, dir=output.parent, prefix=f".{output.name}.", suffix=".part") as temp:
                temp_path = Path(temp.name)
                byte_size = 0
                while chunk := response.read(1024 * 1024):
                    temp.write(chunk)
                    byte_size += len(chunk)
                    if digest:
                        digest.update(chunk)
        temp_path.replace(output)
        return DownloadResult(output, byte_size, digest.hexdigest() if digest else None, False)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Itiner-e download failed: {exc}") from exc
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def iter_ndjson(path: Path) -> Iterator[dict]:
    """Yield NDJSON objects and identify malformed input by its one-based line."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed NDJSON at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"NDJSON object required at line {line_number}")
            yield value


def write_metadata(result: DownloadResult, *, url: str, metadata_path: Path, data_format: str = "ndjson", accessed_at: str | None = None) -> None:
    payload = {
        "source_name": "Itiner-e — The Digital Atlas of Ancient Roads",
        "source_url": url,
        "accessed_at": accessed_at or datetime.now(UTC).isoformat(),
        "license": ITINER_E_LICENSE,
        "citation": ITINER_E_CITATION,
        "local_filename": result.output.name,
        "byte_size": result.byte_size,
        "sha256": result.sha256,
        "format": data_format,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/itiner_e/itiner_e_latest.ndjson"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--format", choices=("ndjson", "geojson"), default="ndjson")
    parser.add_argument("--url", default=ITINER_E_EXPORT_URL, help="Official Itiner-e export URL; useful for controlled tests only.")
    args = parser.parse_args()
    result = download_export(args.output, url=args.url, force=args.force, calculate_sha256=args.sha256)
    if not result.skipped:
        write_metadata(result, url=args.url, metadata_path=args.output.parent / "metadata.json", data_format=args.format)
    print(json.dumps({"output": str(result.output), "byte_size": result.byte_size, "sha256": result.sha256, "skipped": result.skipped}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
