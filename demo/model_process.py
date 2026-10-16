"""Local persistent SP worker with bounded startup/request waits."""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

from .settings import ROOT


def _descendant_groups(pid):
    """Snapshot torchrun's separate rank sessions before the launcher exits."""
    found = {}
    pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            continue
        for child in map(int, children):
            try:
                fields = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()
                # Field 22 identifies this process instance even if a PID is reused.
                identity = fields[19]
                group = os.getpgid(child)
            except (OSError, IndexError):
                continue
            if child not in found:
                found[child] = (identity, group)
                pending.append(child)
    return found


def _kill_descendant_groups(groups):
    for pid, (identity, group) in groups.items():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[19] == identity and os.getpgid(pid) == group:
                # Only signal an independent group whose leader we own.
                if group == pid:
                    os.killpg(group, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
        except (OSError, IndexError):
            pass


class ModelProcess:
    def __init__(self, cfg):
        self.process = None
        self.responses = queue.Queue()
        self.timeout = cfg.get("worker_timeout_seconds", 600)
        data = Path(cfg["data_dir"])
        data.mkdir(parents=True, exist_ok=True)
        self.log = (data / "model-worker.log").open("a")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_PORT"):
            environment.pop(key, None)
        size = cfg["sp_size"]
        command = [
            sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={size}", "-m", "demo.model_worker",
            "--run-config", cfg["run_config"], "--checkpoint", cfg["checkpoint"],
            "--sp-size", str(size), "--source", cfg["weights"],
        ]
        if "seed" in cfg:
            command.extend(["--seed", str(cfg["seed"])])
        try:
            self.process = subprocess.Popen(
                command, cwd=ROOT.parent, env=environment, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                start_new_session=True,
            )
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            if self._response().get("ready") is not True:
                raise RuntimeError("Model worker did not become ready")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.log.write(line)
                    self.log.flush()
        finally:
            self.responses.put({"error": "Model worker exited; see model-worker.log"})

    def _response(self):
        try:
            response = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError("Model worker timed out; see model-worker.log") from None
        if not isinstance(response, dict):
            self.close()
            raise RuntimeError("Invalid model worker response")
        if "error" in response:
            self.close()
            raise RuntimeError(response["error"])
        return response

    def request(self, operation, **fields):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Model worker is not running")
        try:
            self.process.stdin.write(json.dumps({"operation": operation, **fields}) + "\n")
            self.process.stdin.flush()
        except OSError:
            self.close()
            raise RuntimeError("Model worker connection closed; see model-worker.log") from None
        return self._response()

    def close(self):
        process = self.process
        if process is not None:
            descendants = _descendant_groups(process.pid)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _kill_descendant_groups(descendants)
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
            _kill_descendant_groups(descendants)
            if hasattr(self, "reader"):
                self.reader.join(timeout=10)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.process = None
        if not self.log.closed:
            self.log.close()
