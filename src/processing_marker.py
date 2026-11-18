# Vendored from mouse_bluesky, src/mouse_bluesky/processing_marker.py at 282a365.
# Update from that source rather than editing this copy here.
"""Portable YMD manifest writer and reference check for a synced data share."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MARKER_NAME = "READY_FOR_PROCESSING.json"
MARKER_FORMAT = "mouse-ymd-complete/1"
UNRESOLVED_YMD_SENTINEL = "20261232"


def ymd_dir(root: Path, ymd: str) -> Path:
    """Return the instrument's date directory without creating it."""
    return root / ymd[:4] / ymd


def validate_ymd(ymd: str) -> None:
    """Require an eight-digit calendar date, excluding the unresolved sentinel."""
    if ymd == UNRESOLVED_YMD_SENTINEL:
        raise ValueError(f"Unresolved YMD sentinel {UNRESOLVED_YMD_SENTINEL} cannot be marked complete")
    if not isinstance(ymd, str) or re.fullmatch(r"[0-9]{8}", ymd) is None:
        raise ValueError(f"Invalid YMD {ymd!r}: expected an eight-digit date (%Y%m%d)")
    try:
        datetime.strptime(ymd, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"Invalid YMD {ymd!r}: expected a real date (%Y%m%d)") from exc


def scan_ymd_dir(path: Path) -> dict[str, tuple[int, int]]:
    """Snapshot visible regular files, without following symbolic links."""
    files: dict[str, tuple[int, int]] = {}

    def scan(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name.startswith(".") or entry.name == MARKER_NAME:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    scan(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    info = entry.stat(follow_symlinks=False)
                    files[Path(entry.path).relative_to(path).as_posix()] = (info.st_size, info.st_mtime_ns)

    scan(path)
    return files


def build_manifest(path: Path, ymd: str, marker_id: str, files: dict[str, tuple[int, int]]) -> dict:
    """Describe the settled files and completion flags of direct sequence directories."""
    sequences = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.name.startswith(".") or not entry.is_dir(follow_symlinks=False):
                continue
            complete = Path(entry.path) / "COMPLETE"
            try:
                is_complete = stat.S_ISREG(complete.lstat().st_mode)
            except FileNotFoundError:
                is_complete = False
            sequences.append({"name": entry.name, "complete": is_complete})
    return {
        "format": MARKER_FORMAT,
        "marker_id": marker_id,
        "ymd": ymd,
        "created": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": name, "size": files[name][0]} for name in sorted(files)],
        "sequences": sorted(sequences, key=lambda sequence: sequence["name"]),
    }


def write_marker_atomically(path: Path, manifest: dict) -> Path:
    """Flush a hidden temporary file and atomically replace the published marker."""
    temporary = path / f".READY_FOR_PROCESSING.{manifest['marker_id']}.tmp"
    marker = path / MARKER_NAME
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)
    return marker


@dataclass(frozen=True)
class MarkerCheck:
    ready: bool
    marker_id: str | None
    missing: list[str]
    size_mismatch: list[str]
    problem: str | None


def _manifest_problem(manifest: object, path: Path) -> str | None:
    if not isinstance(manifest, dict):
        return "Marker must be a JSON object"
    if manifest.get("format") != MARKER_FORMAT:
        return f"Unknown marker format: {manifest.get('format')!r}"
    if manifest.get("ymd") != path.name:
        return f"Marker YMD {manifest.get('ymd')!r} does not match directory {path.name!r}"
    if not isinstance(manifest.get("marker_id"), str) or not manifest["marker_id"]:
        return "Marker has no marker_id"
    if not isinstance(manifest.get("files"), list):
        return "Marker files must be a list"
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            return "Invalid file entry"
        name, size = entry.get("path"), entry.get("size")
        if not isinstance(name, str) or not name or "\\" in name:
            return "Invalid file path"
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative == PurePosixPath("."):
            return f"Invalid relative file path: {name!r}"
        if type(size) is not int or size < 0:
            return f"Invalid file size: {name!r}"
    return None


def check_marker(path: Path) -> MarkerCheck:
    """Check share readiness using listed file sizes; share mtimes are irrelevant."""
    try:
        manifest = json.loads((path / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return MarkerCheck(False, None, [], [], str(exc))
    marker_id = manifest.get("marker_id") if isinstance(manifest, dict) else None
    if not isinstance(marker_id, str):
        marker_id = None
    problem = _manifest_problem(manifest, path)
    if problem:
        return MarkerCheck(False, marker_id, [], [], problem)
    missing, size_mismatch = [], []
    try:
        for entry in manifest["files"]:
            file = path / entry["path"]
            try:
                info = file.lstat()
            except FileNotFoundError:
                missing.append(entry["path"])
                continue
            if not stat.S_ISREG(info.st_mode):
                missing.append(entry["path"])
            elif info.st_size != entry["size"]:
                size_mismatch.append(entry["path"])
    except OSError as exc:
        return MarkerCheck(False, marker_id, missing, size_mismatch, str(exc))
    return MarkerCheck(not missing and not size_mismatch, marker_id, missing, size_mismatch, None)
