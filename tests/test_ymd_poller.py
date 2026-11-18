from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import ymd_poller
from processing_marker import MARKER_FORMAT, MARKER_NAME, build_manifest, scan_ymd_dir, write_marker_atomically

YMD = "20261118"


@pytest.fixture
def poller(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "vsi_root": str(tmp_path),
                "data_dir": str(data),
                "saxs_dir": str(tmp_path),
                "masks_dir": str(data),
                "projects_dir": str(tmp_path),
                "logbook_file": str(config),
                "stacker_config_file": str(config),
                "post_translation_dir": str(tmp_path),
                "translator_template_dir": str(tmp_path),
                "logging_level": "ERROR",
            }
        )
    )
    state_dir = tmp_path / "state"
    calls_file = tmp_path / "calls.jsonl"
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    processor = binary_dir / "mouse-directory-processor"
    processor.write_text(
        f"#!{sys.executable}\n"
        + """
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
ymd = args[args.index("--ymd") + 1]
batch = args[args.index("--batch") + 1]
preset = args[args.index("--step-preset") + 1]
with open(os.environ["POLLER_CALLS"], "a") as stream:
    stream.write(json.dumps({"args": args, "cwd": os.getcwd(), "path": os.environ["PATH"]}) + "\\n")
print("processor stdout", flush=True)
print("processor stderr", file=sys.stderr, flush=True)
replacement = os.environ.get("POLLER_REPLACEMENT")
if replacement:
    info = json.loads(replacement)
    marker = Path(info["path"])
    guard = marker.parent / ".fake_processor_replaced"
    if not guard.exists():
        guard.touch()
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(info["manifest"]))
        os.replace(temporary, marker)
failures = json.loads(os.environ.get("POLLER_FAILURES", "{}"))
sys.exit(failures.get(f"{ymd}:{batch}:{preset}", 0))
"""
    )
    processor.chmod(0o755)
    monkeypatch.setenv("MOUSE_DIRECTORY_PROCESSOR_BIN", str(processor))
    monkeypatch.setenv("POLLER_CALLS", str(calls_file))
    monkeypatch.delenv("POLLER_FAILURES", raising=False)
    monkeypatch.delenv("POLLER_REPLACEMENT", raising=False)

    def marker(ymd=YMD, batches=(2,), incomplete=(), marker_id="a" * 32, extra=()):
        directory = data / ymd[:4] / ymd
        directory.mkdir(parents=True, exist_ok=True)
        for batch in batches:
            sequence = directory / f"{ymd}_{batch}_0"
            sequence.mkdir(exist_ok=True)
            (sequence / "image.h5").write_bytes(b"sample-data")
            if batch not in incomplete:
                (sequence / "COMPLETE").write_text("ok")
        for name in extra:
            (directory / name).mkdir(exist_ok=True)
        manifest = build_manifest(directory, ymd, marker_id, scan_ymd_dir(directory))
        write_marker_atomically(directory, manifest)
        return directory, manifest

    def invoke(command="poll", *arguments):
        return ymd_poller.main([command, "--config", str(config), "--state-dir", str(state_dir), *arguments])

    def state(ymd=YMD):
        return json.loads((state_dir / f"{ymd}.json").read_text())

    def calls():
        return [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []

    return SimpleNamespace(
        data=data,
        config=config,
        state_dir=state_dir,
        processor=processor,
        calls_file=calls_file,
        marker=marker,
        invoke=invoke,
        state=state,
        calls=calls,
    )


def _runs(poller):
    return [(call["args"][call["args"].index("--batch") + 1], call["args"][-1]) for call in poller.calls()]


def test_ready_batches_order_arguments_environment_and_idempotency(poller):
    poller.marker(batches=(10, 2, 1))
    assert poller.invoke() == 0
    assert _runs(poller) == [(str(batch), preset) for batch in (1, 2, 10) for preset in ("preprocess", "stackonly")]
    for call in poller.calls():
        assert call["args"] == [
            "--config",
            str(poller.config.resolve()),
            "--ymd",
            YMD,
            "--batch",
            call["args"][5],
            "--parallel",
            "--step-preset",
            call["args"][-1],
        ]
        assert call["cwd"] == str(ymd_poller.REPOSITORY_ROOT)
        assert call["path"].split(os.pathsep)[0] == str(poller.processor.parent)
    state = poller.state()
    assert state["status"] == "processed"
    assert state["marker_id"] == "a" * 32
    assert state["started"] <= state["finished"]
    assert [batch["batch"] for batch in state["batches"]] == [1, 2, 10]
    output = Path(state["log"]).read_text()
    assert "processor stdout" in output
    assert "processor stderr" in output
    assert Path(state["log"]).parent == poller.state_dir / "logs"
    assert Path(state["log"]).name.startswith(f"{YMD}_aaaaaaaa_")
    assert poller.invoke() == 0
    assert len(poller.calls()) == 6


@pytest.mark.parametrize("change", ["missing", "smaller", "larger"])
def test_waiting_for_missing_or_wrong_sized_file(poller, change):
    directory, _ = poller.marker()
    image = directory / f"{YMD}_2_0" / "image.h5"
    if change == "missing":
        image.unlink()
    else:
        image.write_bytes(b"x" if change == "smaller" else b"x" * 30)
    assert poller.invoke() == 0
    state = poller.state()
    assert state["status"] == "waiting"
    key = "missing" if change == "missing" else "size_mismatch"
    assert state[key] == [f"{YMD}_2_0/image.h5"]
    assert state[key + "_count"] == 1
    assert poller.calls() == []
    first_seen = state["first_seen"]
    assert poller.invoke() == 0
    assert poller.state()["first_seen"] == first_seen


def test_waiting_details_are_bounded(poller):
    directory, manifest = poller.marker()
    manifest["files"] = [{"path": f"missing_{index}", "size": 1} for index in range(14)]
    write_marker_atomically(directory, manifest)
    assert poller.invoke() == 0
    assert poller.state()["missing_count"] == 14
    assert len(poller.state()["missing"]) == 10


def test_waiting_resumes_when_sync_finishes(poller):
    directory, _ = poller.marker()
    image = directory / f"{YMD}_2_0/image.h5"
    image.unlink()
    assert poller.invoke() == 0
    image.write_bytes(b"sample-data")
    assert poller.invoke() == 0
    assert poller.state()["status"] == "processed"


def test_overdue_alerts_once_and_new_marker_resets_timer(poller):
    directory, manifest = poller.marker()
    (directory / f"{YMD}_2_0/image.h5").unlink()
    assert poller.invoke() == 0
    state = poller.state()
    state["first_seen"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    (poller.state_dir / f"{YMD}.json").write_text(json.dumps(state))
    assert poller.invoke() == 1
    assert poller.state()["overdue"] is True
    assert poller.invoke() == 0
    manifest["marker_id"] = "b" * 32
    write_marker_atomically(directory, manifest)
    assert poller.invoke() == 0
    assert poller.state()["overdue"] is False
    assert poller.state()["first_seen"] != state["first_seen"]
    assert poller.calls() == []


@pytest.mark.parametrize("preset", ["preprocess", "stackonly"])
def test_failure_continues_other_batches_and_is_not_retried(poller, monkeypatch, preset):
    poller.marker(batches=(2, 3))
    monkeypatch.setenv("POLLER_FAILURES", json.dumps({f"{YMD}:2:{preset}": 7}))
    assert poller.invoke() == 1
    expected = [("2", "preprocess")]
    if preset == "stackonly":
        expected.append(("2", "stackonly"))
    assert _runs(poller) == expected + [("3", "preprocess"), ("3", "stackonly")]
    state = poller.state()
    assert state["status"] == "failed"
    assert state["batches"][0][preset] == {"status": "failed", "exit_code": 7}
    if preset == "preprocess":
        assert state["batches"][0]["stackonly"]["status"] == "skipped"
    assert poller.invoke() == 0
    assert len(poller.calls()) == len(expected) + 2


@pytest.mark.parametrize("recovery", ["retry", "new_marker"])
def test_failed_marker_recovery(poller, monkeypatch, recovery):
    directory, manifest = poller.marker()
    monkeypatch.setenv("POLLER_FAILURES", json.dumps({f"{YMD}:2:preprocess": 1}))
    assert poller.invoke() == 1
    monkeypatch.delenv("POLLER_FAILURES")
    if recovery == "retry":
        assert poller.invoke("retry", YMD) == 0
        assert not (poller.state_dir / f"{YMD}.json").exists()
    else:
        manifest["marker_id"] = "b" * 32
        write_marker_atomically(directory, manifest)
    assert poller.invoke() == 0
    assert poller.state()["status"] == "processed"
    assert len(poller.calls()) == 3


def test_non_sequences_ignored_incomplete_sequences_run_and_are_reported(poller, capsys):
    poller.marker(batches=(2, 4), incomplete=(4,), extra=("autoproc", "20261117_99_0", f"{YMD}_1_0_extra"))
    assert poller.invoke() == 0
    assert _runs(poller) == [(str(batch), preset) for batch in (2, 4) for preset in ("preprocess", "stackonly")]
    assert poller.state()["incomplete_sequences"] == [f"{YMD}_4_0"]
    assert f"incomplete sequences: {YMD}_4_0" in capsys.readouterr().out


def test_marker_replacement_during_processing_keeps_authorising_id(poller, monkeypatch):
    directory, manifest = poller.marker()
    replacement = {**manifest, "marker_id": "b" * 32, "sequences": [{"name": f"{YMD}_8_0", "complete": True}]}
    monkeypatch.setenv(
        "POLLER_REPLACEMENT", json.dumps({"path": str(directory / MARKER_NAME), "manifest": replacement})
    )
    assert poller.invoke() == 0
    assert poller.state()["marker_id"] == manifest["marker_id"]
    assert _runs(poller) == [("2", "preprocess"), ("2", "stackonly")]
    assert poller.invoke() == 0
    assert poller.state()["marker_id"] == replacement["marker_id"]
    assert _runs(poller)[2:] == [("8", "preprocess"), ("8", "stackonly")]


def test_marker_replacement_during_readiness_check_does_not_mix_sequences(poller, monkeypatch):
    directory, manifest = poller.marker()
    check = ymd_poller.check_marker

    def replace_before_check(path):
        write_marker_atomically(directory, {**manifest, "marker_id": "b" * 32})
        return check(path)

    monkeypatch.setattr(ymd_poller, "check_marker", replace_before_check)
    assert poller.invoke() == 0
    assert poller.calls() == []
    assert poller.state()["status"] == "waiting"
    assert poller.state()["marker_id"] == "b" * 32
    assert "changed" in poller.state()["problem"]
    monkeypatch.setattr(ymd_poller, "check_marker", check)
    assert poller.invoke() == 0


@pytest.mark.parametrize("all_present", [True, False])
def test_mark_done(poller, all_present):
    poller.marker()
    poller.marker(ymd="20261117")
    assert poller.invoke("mark-done", *(["--all-present"] if all_present else [YMD])) == 0
    assert poller.state()["status"] == "skipped"
    assert (poller.state_dir / "20261117.json").exists() is all_present
    assert poller.calls() == []
    assert poller.invoke() == 0
    assert len(poller.calls()) == (0 if all_present else 2)


def test_mark_done_accepts_marker_still_waiting_for_sync(poller):
    directory, _ = poller.marker()
    (directory / f"{YMD}_2_0/image.h5").unlink()
    assert poller.invoke("mark-done", YMD) == 0
    assert poller.state()["status"] == "skipped"
    assert poller.invoke() == 0
    assert poller.calls() == []


def test_held_lock_skips_poll_without_processing(poller, capsys):
    poller.marker()
    poller.state_dir.mkdir()
    with (poller.state_dir / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert poller.invoke() == 0
    assert "Another poll is running" in capsys.readouterr().out
    assert poller.calls() == []
    assert not (poller.state_dir / f"{YMD}.json").exists()


@pytest.mark.parametrize("contents", ["{broken", "[]", '{"status": "unknown"}'])
def test_invalid_state_is_error_and_not_overwritten(poller, contents):
    poller.marker()
    poller.state_dir.mkdir()
    state_file = poller.state_dir / f"{YMD}.json"
    state_file.write_text(contents)
    assert poller.invoke() == 1
    assert state_file.read_text() == contents
    assert poller.calls() == []


@pytest.mark.parametrize("executable", ["missing", "not_executable"])
def test_processor_must_exist_and_be_executable(poller, monkeypatch, executable):
    poller.marker()
    if executable == "missing":
        monkeypatch.setenv("MOUSE_DIRECTORY_PROCESSOR_BIN", str(poller.processor.parent / "absent"))
    else:
        poller.processor.chmod(0o644)
    assert poller.invoke() == 1
    assert poller.calls() == []


def test_default_processor_is_next_to_active_interpreter(poller, monkeypatch):
    poller.marker()
    monkeypatch.delenv("MOUSE_DIRECTORY_PROCESSOR_BIN")
    monkeypatch.setattr(ymd_poller.sys, "executable", str(poller.processor.parent / "python"))
    assert poller.invoke() == 0
    assert len(poller.calls()) == 2


def test_stale_running_state_is_recovered(poller, capsys):
    poller.marker()
    poller.state_dir.mkdir()
    (poller.state_dir / f"{YMD}.json").write_text(
        json.dumps(
            {
                "marker_id": "a" * 32,
                "status": "running",
                "started": "2026-11-17T00:00:00+00:00",
            }
        )
    )
    assert poller.invoke() == 0
    assert "recovering an unfinished running attempt" in capsys.readouterr().out
    assert poller.state()["status"] == "processed"


def test_status_lists_union_and_reports_new_markers_and_incomplete_counts(poller, capsys):
    directory, manifest = poller.marker(incomplete=(2,))
    poller.marker(ymd="20261117")
    assert poller.invoke("mark-done", "20261117") == 0
    # State-only dates remain visible even when the marker is no longer on the share.
    (poller.data / "2026/20261117" / MARKER_NAME).unlink()
    capsys.readouterr()
    assert poller.invoke("status") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("20261117 aaaaaaaa skipped age=")
    assert lines[1].startswith(f"{YMD} aaaaaaaa new age=")
    assert lines[1].endswith("incomplete=1")
    assert poller.invoke() == 0
    write_marker_atomically(directory, {**manifest, "marker_id": "b" * 32})
    capsys.readouterr()
    assert poller.invoke("status") == 0
    assert f"{YMD} bbbbbbbb new" in capsys.readouterr().out


def test_status_exits_zero_with_corrupt_state(poller, capsys):
    poller.marker()
    poller.state_dir.mkdir()
    (poller.state_dir / f"{YMD}.json").write_text("bad json")
    assert poller.invoke("status") == 0
    assert "ERROR" in capsys.readouterr().out


def test_ymds_run_in_order_and_new_failures_do_not_stop_later_ymds(poller, monkeypatch):
    poller.marker()
    poller.marker(ymd="20261117")
    monkeypatch.setenv("POLLER_FAILURES", json.dumps({"20261117:2:preprocess": 9}))
    assert poller.invoke() == 1
    assert [call["args"][3] for call in poller.calls()] == ["20261117", YMD, YMD]
    assert poller.state("20261117")["status"] == "failed"
    assert poller.state()["status"] == "processed"
    assert poller.invoke() == 0


def test_unreadable_marker_waits_and_nothing_to_do_is_success(poller):
    assert poller.invoke() == 0
    directory, _ = poller.marker()
    (directory / MARKER_NAME).write_text("{partial")
    assert poller.invoke() == 0
    assert poller.state()["status"] == "waiting"
    assert poller.state()["problem"]
    assert poller.calls() == []


def test_state_must_not_be_in_data_tree(poller):
    poller.marker()
    assert poller.invoke("poll", "--state-dir", str(poller.data / "state")) == 1
    assert not (poller.data / "state").exists()


def test_cli_process_exit_status_and_help(poller):
    poller.marker()
    command = [
        sys.executable,
        str(Path(ymd_poller.__file__)),
        "poll",
        "--config",
        str(poller.config),
        "--state-dir",
        str(poller.state_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert poller.state()["status"] == "processed"
    help_result = subprocess.run(command[:2] + ["--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "mark-done" in help_result.stdout


def test_vendored_format_constant():
    assert MARKER_FORMAT == "mouse-ymd-complete/1"


def test_distinct_repetitions_do_not_repeat_a_batch(poller):
    directory, manifest = poller.marker()
    other_repetition = directory / f"{YMD}_2_1"
    other_repetition.mkdir()
    manifest = build_manifest(directory, YMD, manifest["marker_id"], scan_ymd_dir(directory))
    write_marker_atomically(directory, manifest)
    assert poller.invoke() == 0
    assert _runs(poller) == [("2", "preprocess"), ("2", "stackonly")]
    assert poller.state()["incomplete_sequences"] == [f"{YMD}_2_1"]


def test_relative_config_path_becomes_absolute_for_processor(poller, monkeypatch):
    poller.marker()
    monkeypatch.chdir(poller.config.parent)
    assert (
        ymd_poller.main(
            [
                "poll",
                "--config",
                poller.config.name,
                "--state-dir",
                str(poller.state_dir),
            ]
        )
        == 0
    )
    assert poller.calls()[0]["args"][1] == str(poller.config)


@pytest.mark.parametrize("contents", [None, "[invalid: yaml", "- a list"])
def test_broken_config_is_poller_error(poller, capsys, contents):
    if contents is None:
        poller.config.unlink()
    else:
        poller.config.write_text(contents)
    assert poller.invoke() == 1
    assert "mouse-ymd-poller:" in capsys.readouterr().err
    assert poller.calls() == []


def test_failed_atomic_state_publish_keeps_previous_state(poller, monkeypatch):
    poller.marker()
    assert poller.invoke("mark-done", YMD) == 0
    state_path = poller.state_dir / f"{YMD}.json"
    previous = state_path.read_bytes()

    def fail_replace(source, destination):
        # The previous file remains usable until the replacement is published.
        assert json.loads(Path(source).read_text())["status"] == "processed"
        assert Path(destination).read_bytes() == previous
        raise OSError("simulated replace failure")

    monkeypatch.setattr(ymd_poller.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        ymd_poller._write_state(poller.state_dir, YMD, {"status": "processed"})
    assert state_path.read_bytes() == previous
    assert not list(poller.state_dir.glob(f".{YMD}.*"))
