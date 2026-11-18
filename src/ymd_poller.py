"""Poll synced YMD manifests and run the existing processing presets."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from defaults_carrier import DefaultsCarrier, load_config_from_yaml
from processing_marker import MARKER_NAME, MarkerCheck, check_marker, validate_ymd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = Path("~/.local/state/mouse-ymd-poller")
TERMINAL_STATUSES = {"processed", "failed", "skipped"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_state(state_dir: Path, ymd: str) -> dict | None:
    path = state_dir / f"{ymd}.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read state {path}: {exc}") from exc
    if not isinstance(state, dict) or state.get("status") not in TERMINAL_STATUSES | {"waiting", "running"}:
        raise ValueError(f"Invalid state in {path}")
    if state.get("marker_id") is not None and not isinstance(state["marker_id"], str):
        raise ValueError(f"Invalid marker_id in {path}")
    return state


def _write_state(state_dir: Path, ymd: str, state: dict) -> None:
    """Publish only complete JSON, leaving the old state intact on write failure."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=state_dir, prefix=f".{ymd}.", delete=False
        ) as f:
            temporary = Path(f.name)
            json.dump(state, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, state_dir / f"{ymd}.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _lock(state_dir: Path):
    with (state_dir / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another poll is running; no changes made.")
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _markers(data_dir: Path) -> dict[str, Path]:
    markers = {}
    # iterdir surfaces an unavailable share instead of silently treating it as empty.
    for year in sorted(data_dir.iterdir()):
        if not year.is_dir():
            continue
        for directory in sorted(year.iterdir()):
            if directory.is_dir() and (directory / MARKER_NAME).exists():
                validate_ymd(directory.name)
                if year.name != directory.name[:4]:
                    raise ValueError(f"Marker is outside its YMD year directory: {directory}")
                if directory.name in markers:
                    raise ValueError(f"Duplicate YMD marker: {directory.name}")
                markers[directory.name] = directory
    return dict(sorted(markers.items()))


def _manifest(directory: Path) -> dict:
    try:
        value = json.loads((directory / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _batches(manifest: dict, ymd: str) -> tuple[list[int], list[str]]:
    pattern = re.compile(rf"{re.escape(ymd)}_(\d+)_(\d+)")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError(f"{ymd}: marker sequences must be a list")
    batches = set()
    incomplete = []
    for sequence in sequences:
        if not isinstance(sequence, dict) or not isinstance(sequence.get("name"), str):
            raise ValueError(f"{ymd}: invalid marker sequence")
        match = pattern.fullmatch(sequence["name"])
        if match:
            if type(sequence.get("complete")) is not bool:
                raise ValueError(f"{ymd}: sequence {sequence['name']} has no boolean completion flag")
            batches.add(int(match[1]))
            if not sequence["complete"]:
                incomplete.append(sequence["name"])
    return sorted(batches), sorted(incomplete)


def _processor() -> Path:
    override = os.environ.get("MOUSE_DIRECTORY_PROCESSOR_BIN")
    if override is None:
        path = Path(sys.executable).parent / "mouse-directory-processor"
    else:
        # Bare command names may be looked up; relative paths are resolved before changing cwd.
        path = Path(shutil.which(override) or override).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Processor executable not found or not executable: {path}")
    return path


def _run_batches(
    ymd: str, marker_id: str, batches: list[int], config: Path, state_dir: Path, processor: Path
) -> tuple[list[dict], Path]:
    logs = state_dir / "logs"
    logs.mkdir(exist_ok=True)
    # A marker_id normally is UUID hex, but the reference checker accepts any nonempty string.
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", marker_id[:8])
    stamp = _now().strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = logs / f"{ymd}_{prefix}_{stamp}.log"
    environment = os.environ.copy()
    environment["PATH"] = str(processor.parent) + os.pathsep + environment.get("PATH", os.defpath)
    results = []
    print(f"{ymd}: log {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        for batch in batches:
            result = {"batch": batch}
            for preset in ("preprocess", "stackonly"):
                if preset == "stackonly" and result["preprocess"]["status"] == "failed":
                    result[preset] = {"status": "skipped", "exit_code": None}
                    print(f"{ymd} batch {batch}: stackonly SKIPPED (preprocess failed)", flush=True)
                    continue
                command = [
                    str(processor),
                    "--config",
                    str(config),
                    "--ymd",
                    ymd,
                    "--batch",
                    str(batch),
                    "--parallel",
                    "--step-preset",
                    preset,
                ]
                log.write(f"\n{ymd} batch {batch}: {preset}\n")
                log.flush()
                try:
                    completed = subprocess.run(
                        command, cwd=REPOSITORY_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                    exit_code = completed.returncode
                    result[preset] = {"status": "processed" if exit_code == 0 else "failed", "exit_code": exit_code}
                except OSError as exc:
                    log.write(f"Could not start processor: {exc}\n")
                    result[preset] = {"status": "failed", "exit_code": None, "error": str(exc)}
                outcome = result[preset]
                print(
                    f"{ymd} batch {batch}: {preset} {'OK' if outcome['status'] == 'processed' else 'FAIL'} "
                    f"(exit {outcome['exit_code']})",
                    flush=True,
                )
            results.append(result)
    return results, log_path


def _waiting(state_dir: Path, ymd: str, check: MarkerCheck, old: dict, incomplete: list[str], hours: float) -> bool:
    now = _now()
    first_seen = old.get("first_seen", now.isoformat())
    first_time = datetime.fromisoformat(first_seen)
    if first_time.tzinfo is None:
        raise ValueError(f"{ymd}: first_seen must have a timezone")
    newly_overdue = now - first_time > timedelta(hours=hours) and not old.get("overdue", False)
    state = {
        "ymd": ymd,
        "marker_id": check.marker_id,
        "status": "waiting",
        "first_seen": first_seen,
        "overdue": bool(old.get("overdue", False) or newly_overdue),
        "problem": check.problem,
        "missing_count": len(check.missing),
        "missing": check.missing[:10],
        "size_mismatch_count": len(check.size_mismatch),
        "size_mismatch": check.size_mismatch[:10],
        "incomplete_sequences": incomplete,
    }
    _write_state(state_dir, ymd, state)
    print(
        f"{ymd}: waiting; missing={len(check.missing)} {check.missing[:10]}; "
        f"size_mismatch={len(check.size_mismatch)} {check.size_mismatch[:10]}"
        + (f"; {check.problem}" if check.problem else "")
        + ("; OVERDUE" if newly_overdue else ""),
        flush=True,
    )
    return newly_overdue


def _poll(markers: dict[str, Path], states: dict[str, dict], args, processor: Path) -> int:
    attention = False
    for ymd, directory in markers.items():
        manifest = _manifest(directory)
        old = states.get(ymd, {})
        marker_id = manifest.get("marker_id")
        if marker_id and old.get("marker_id") == marker_id and old.get("status") in TERMINAL_STATUSES:
            print(f"{ymd}: {old['status']} (already recorded)")
            continue
        check = check_marker(directory)
        if check.marker_id is None and ymd in states:
            print(
                f"{ymd}: marker unreadable ({check.problem}); keeping recorded {old['status']} state",
                flush=True,
            )
            continue
        if manifest.get("marker_id") != check.marker_id:
            # Never use sequences from a different marker than the one whose files were checked.
            check = MarkerCheck(
                False, check.marker_id, check.missing, check.size_mismatch, "Marker changed during check"
            )
            manifest = {}
        if check.ready:
            batches, incomplete = _batches(manifest, ymd)
        else:
            try:
                _, incomplete = _batches(manifest, ymd)
            except ValueError:
                incomplete = []
        if incomplete:
            print(f"{ymd}: incomplete sequences: {', '.join(incomplete)}", flush=True)
        if old.get("marker_id") != check.marker_id:
            old = {}
        if not check.ready:
            attention = _waiting(args.state_dir, ymd, check, old, incomplete, args.overdue_hours) or attention
            continue
        if old.get("status") == "running" and not old.get("finished"):
            print(f"{ymd}: recovering an unfinished running attempt", flush=True)
        state = {
            "ymd": ymd,
            "marker_id": check.marker_id,
            "status": "running",
            "started": _now().isoformat(),
            "incomplete_sequences": incomplete,
        }
        _write_state(args.state_dir, ymd, state)
        results, log_path = _run_batches(ymd, check.marker_id, batches, args.config, args.state_dir, processor)
        failed = any(
            preset["status"] == "failed" for batch in results for preset in (batch["preprocess"], batch["stackonly"])
        )
        state.update(
            status="failed" if failed else "processed", finished=_now().isoformat(), batches=results, log=str(log_path)
        )
        _write_state(args.state_dir, ymd, state)
        print(f"{ymd}: {state['status']}; log {log_path}", flush=True)
        attention = attention or failed
    return int(attention)


def _all_states(state_dir: Path) -> dict[str, dict]:
    states = {}
    for path in sorted(state_dir.glob("*.json")):
        validate_ymd(path.stem)
        states[path.stem] = _read_state(state_dir, path.stem)
    return states


def _status(markers: dict[str, Path], state_dir: Path) -> int:
    ymds = sorted(set(markers) | {path.stem for path in state_dir.glob("*.json")})
    for ymd in ymds:
        try:
            validate_ymd(ymd)
            state = _read_state(state_dir, ymd) or {}
            manifest = _manifest(markers[ymd]) if ymd in markers else {}
            current_id = manifest.get("marker_id")
            if current_id and current_id != state.get("marker_id"):
                state = {"marker_id": current_id, "status": "new"}
            marker_id = state.get("marker_id") or current_id or "unknown"
            timestamp = (
                state.get("first_seen") or state.get("started") or state.get("finished") or manifest.get("created")
            )
            age = "unknown"
            if timestamp:
                age = f"{(_now() - datetime.fromisoformat(timestamp)).total_seconds() / 3600:.1f}h"
            incomplete = state.get("incomplete_sequences")
            if incomplete is None:
                try:
                    _, incomplete = _batches(manifest, ymd)
                except ValueError:
                    incomplete = []
            print(f"{ymd} {marker_id[:8]} {state.get('status', 'new')} age={age} incomplete={len(incomplete)}")
        except (OSError, ValueError, TypeError) as exc:
            print(f"{ymd} ERROR {exc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("poll", "status", "mark-done", "retry"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True, help="Pipeline YAML configuration")
        command.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR, help="Local state directory")
        if name == "poll":
            command.add_argument(
                "--overdue-hours", type=float, default=24, help="Alert once after this wait (default: 24)"
            )
        elif name == "mark-done":
            command.add_argument("ymds", metavar="YMD", nargs="*")
            command.add_argument("--all-present", action="store_true")
        elif name == "retry":
            command.add_argument("ymd", metavar="YMD")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.config = args.config.expanduser().resolve()
        args.state_dir = args.state_dir.expanduser().resolve()
        config = load_config_from_yaml(str(args.config))
        if not isinstance(config, dict):
            raise ValueError("Pipeline configuration must be a YAML mapping")
        # Match the processor's cwd for relative pipeline path settings.
        for key in (
            "vsi_root",
            "post_translation_dir",
            "translator_template_dir",
            "saxs_dir",
            "data_dir",
            "masks_dir",
            "projects_dir",
            "logbook_file",
            "stacker_config_file",
            "log_file",
        ):
            if config.get(key) and not Path(config[key]).is_absolute():
                config[key] = str(REPOSITORY_ROOT / config[key])
        defaults = DefaultsCarrier(**config)
        if args.state_dir.is_relative_to(defaults.data_dir.resolve()):
            raise ValueError("State directory must be outside the synced data tree")
        if args.command == "status":
            return _status(_markers(defaults.data_dir), args.state_dir)
        if args.command == "poll" and (not math.isfinite(args.overdue_hours) or args.overdue_hours < 0):
            raise ValueError("--overdue-hours must be a finite nonnegative number")
        if args.command == "mark-done" and bool(args.ymds) == args.all_present:
            raise ValueError("mark-done requires either YMD arguments or --all-present")
        args.state_dir.mkdir(parents=True, exist_ok=True)
        with _lock(args.state_dir) as acquired:
            if not acquired:
                return 0
            states = _all_states(args.state_dir)
            if args.command == "retry":
                validate_ymd(args.ymd)
                (args.state_dir / f"{args.ymd}.json").unlink(missing_ok=True)
                print(f"{args.ymd}: state cleared; eligible for the next poll")
                return 0
            markers = _markers(defaults.data_dir)
            if args.command == "poll":
                return _poll(markers, states, args, _processor())
            selected = list(markers) if args.all_present else list(dict.fromkeys(args.ymds))
            checks = {}
            for ymd in selected:
                validate_ymd(ymd)
                if ymd not in markers:
                    raise ValueError(f"{ymd}: no marker found")
                check = check_marker(markers[ymd])
                if not check.marker_id or check.problem:
                    raise ValueError(f"{ymd}: cannot mark done: {check.problem or 'no marker_id'}")
                manifest = _manifest(markers[ymd])
                if manifest.get("marker_id") != check.marker_id:
                    raise ValueError(f"{ymd}: marker changed during mark-done; try again")
                _, incomplete = _batches(manifest, ymd)
                checks[ymd] = (check, incomplete)
            for ymd, (check, incomplete) in checks.items():
                _write_state(
                    args.state_dir,
                    ymd,
                    {
                        "ymd": ymd,
                        "marker_id": check.marker_id,
                        "status": "skipped",
                        "finished": _now().isoformat(),
                        "incomplete_sequences": incomplete,
                    },
                )
                print(f"{ymd}: skipped marker {check.marker_id}")
            return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"mouse-ymd-poller: {exc}", file=sys.stderr)
        return 0 if args.command == "status" else 1


if __name__ == "__main__":
    sys.exit(main())
