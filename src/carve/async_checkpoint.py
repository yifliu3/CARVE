"""Immutable CPU snapshots, local checkpoints, and a CPU-only background writer.

Inspired by master_leo_1.5's subprocess writer. CARVE adapters and optimizer
states are replicated, so only rank zero writes the large tensors. Each
snapshot includes every rank's RNG/cursor and fixes EMA at the same step.
"""

import copy
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import time
import traceback
from pathlib import Path

import torch


def cpu_snapshot(value):
    """Copy all tensor storage, including already-shared CPU tensors."""
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device="cpu", copy=True).contiguous()
        return result.share_memory_()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _sync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_local(root, files, metadata):
    root = Path(root)
    if (root / "LOCAL_COMPLETE.json").exists() or (root / "COMPLETE.json").exists():
        raise FileExistsError(f"Refusing to overwrite complete checkpoint: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, value in files.items():
        temporary = root / (name + ".tmp")
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(root / name)
        manifest[name] = {"bytes": (root / name).stat().st_size, "sha256": digest(root / name)}
    metadata = {**metadata, "files": manifest}
    atomic_json(root / "LOCAL_COMPLETE.json", metadata)
    return metadata


def upload_local(local, remote, metadata):
    """Retain local files; publish the remote completion marker last."""
    local, remote = Path(local), Path(remote)
    if local.resolve() == remote.resolve():
        raise ValueError("Local staging and remote checkpoint directories must differ")
    marker = remote / "COMPLETE.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous != metadata:
            raise FileExistsError(f"Different checkpoint already published: {remote}")
        for name, item in metadata["files"].items():
            if (remote / name).stat().st_size != item["bytes"]:
                raise ValueError(f"Published checkpoint file size changed: {remote / name}")
        return
    remote.mkdir(parents=True, exist_ok=True)
    for name, item in metadata["files"].items():
        temporary = remote / (name + ".uploading")
        checksum = hashlib.sha256()
        with (local / name).open("rb") as source, temporary.open("wb") as destination:
            for block in iter(lambda: source.read(8 << 20), b""):
                checksum.update(block)
                destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        if temporary.stat().st_size != item["bytes"] or checksum.hexdigest() != item["sha256"]:
            raise ValueError(f"Checkpoint upload differs from snapshot: {name}")
        temporary.replace(remote / name)
    atomic_json(marker, metadata)
    atomic_json(local / "REMOTE_COMPLETE.json", {"remote": str(remote), "step": metadata["step"]})


def _worker(tasks, results):
    # Spawn starts a fresh interpreter. No model, CUDA operation or process
    # group is created here. The parent passes independent shared CPU storage.
    torch.set_num_threads(1)
    while True:
        task = tasks.get()
        if task is None:
            return
        files, metadata, local, remote = task
        step = metadata["step"]
        started = time.monotonic()
        try:
            metadata = write_local(local, files, metadata)
            del files, task
            results.put(
                dict(
                    event="local_complete",
                    step=step,
                    local=str(local),
                    seconds=time.monotonic() - started,
                )
            )
            upload_started = time.monotonic()
            upload_local(local, remote, metadata)
            results.put(
                dict(
                    event="remote_complete",
                    step=step,
                    remote=str(remote),
                    seconds=time.monotonic() - upload_started,
                )
            )
        except Exception:
            results.put(dict(event="error", step=step, error=traceback.format_exc()))
            return


class AsyncCheckpointWriter:
    """Bounded to one immutable checkpoint; failures surface on poll/wait."""

    def __init__(self, status_path):
        if os.environ.get("PYTHON_EXEC"):
            mp.set_executable(os.environ["PYTHON_EXEC"])
        context = mp.get_context("spawn")
        self.tasks = context.Queue(maxsize=1)
        self.results = context.Queue()
        self.worker = context.Process(target=_worker, args=(self.tasks, self.results), daemon=False)
        self.worker.start()
        self.status_path = Path(status_path)
        self.pending = None
        self.snapshot = None
        self.error = None
        self.completed_events = []

    def submit(self, files, metadata, local, remote):
        previous_events = self.wait()
        self.completed_events.extend(previous_events)
        started = time.monotonic()
        required = sum(t.numel() * t.element_size() for t in _tensors(files))
        if Path("/dev/shm").exists() and shutil.disk_usage("/dev/shm").free < required + (64 << 20):
            raise RuntimeError(f"Insufficient /dev/shm for immutable checkpoint ({required} bytes)")
        snapshot = cpu_snapshot(files)
        self.snapshot = snapshot
        self.pending = metadata["step"]
        self.tasks.put((snapshot, copy.deepcopy(metadata), str(local), str(remote)))
        info = dict(
            event="snapshot_submitted",
            step=self.pending,
            local=str(local),
            remote=str(remote),
            snapshot_seconds=time.monotonic() - started,
            snapshot_bytes=required,
            writer_pid=self.worker.pid,
        )
        atomic_json(self.status_path, info)
        return info

    def poll(self):
        events, self.completed_events = self.completed_events, []
        while True:
            try:
                info = self.results.get_nowait()
            except queue.Empty:
                break
            events.append(info)
            if info["event"] == "local_complete":
                self.snapshot = None
            elif info["event"] == "remote_complete":
                self.snapshot = None
                self.pending = None
            elif info["event"] == "error":
                self.error = info["error"]
            atomic_json(self.status_path, info)
        if self.error:
            raise RuntimeError(f"Background checkpoint failed: {self.error}")
        if self.pending is not None and not self.worker.is_alive():
            raise RuntimeError(f"Checkpoint writer exited with code {self.worker.exitcode}")
        return events

    def wait(self, timeout=3600):
        deadline = time.monotonic() + timeout
        events = self.poll()
        while self.pending is not None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Checkpoint {self.pending} upload timed out")
            time.sleep(0.05)
            events.extend(self.poll())
        return events

    def close(self):
        try:
            events = self.wait()
        finally:
            if self.worker.is_alive():
                self.tasks.put(None)
                self.worker.join(timeout=30)
                if self.worker.is_alive():
                    self.worker.terminate()
                    self.worker.join(timeout=5)
            self.tasks.close()
            self.results.close()
        return events

    def abort(self):
        """Release the child after staging fails, without waiting on a bad job."""
        if self.worker.is_alive():
            self.worker.terminate()
        self.worker.join(timeout=5)
        self.tasks.cancel_join_thread()
        self.tasks.close()
        self.results.close()
        self.snapshot = None


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
