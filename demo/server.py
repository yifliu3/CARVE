"""Standard-library HTTP server for local preview and opt-in token rollout."""

import argparse
import json
import mimetypes
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "demo"

import numpy as np

from .mesh_assets import FORMATS, load_upload, save_environment
from .recorded_path import plan_path, validate_path
from .settings import ROOT, VERSION, load_config, read, write


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def calibration(out):
    source = read(ROOT / "assets/calibration.json")
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ])
    vertices, colors = [], []
    for box in source["boxes"]:
        x, y, z = box["min"]
        X, Y, Z = box["max"]
        v = np.array([[x, y, z], [X, y, z], [X, Y, z], [x, Y, z],
                      [x, y, Z], [X, y, Z], [X, Y, Z], [x, Y, Z]])
        vertices.extend(v[faces])
        colors.extend([box["color"]] * len(faces))
    return save_environment(
        out, vertices, colors, scale=3, origin=[0, 4, 0],
        K=[[830, 0, 480], [0, 830, 272], [0, 0, 1]],
        metadata={"name": source["name"], "asset_source": source["kind"], "preview_only": True},
    )


def valid_id(sid):
    return len(sid) == 32 and all(c in "0123456789abcdef" for c in sid)


class DemoState:
    def __init__(self, cfg, preview):
        self.cfg, self.preview = cfg, preview
        self.data_dir = Path(cfg["data_dir"])
        self.run_config = None
        if not preview:
            from .token_runtime import inspect_checkpoint, resolve_run_config

            if not cfg.get("checkpoint") or not cfg.get("taeh3_assets"):
                raise ValueError("Generation requires explicit causal checkpoint and pinned TAEH3 assets")
            self.run_config = resolve_run_config(cfg.get("run_config"), sp_size=cfg["sp_size"])
            metadata = inspect_checkpoint(cfg["checkpoint"], self.run_config)
            self.run_config["stage"] = metadata["stage"]
            self.run_config["train"]["seed"] = cfg["seed"]
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.active_scene = None
        self.pose = [0, 0, 0, 0, 0]
        self.runtime = self.encoder = self.renderer = None
        self.decode_history = []
        self.generation_status = "disabled_preview" if preview else "configured_not_loaded"
        self.base_url = None
        calibration(self.data_dir / "scenes/calibration")

    def scene_dir(self, sid):
        if sid != "calibration" and not valid_id(sid):
            raise RequestError(404, "Scene not found")
        root = self.data_dir / "scenes" / sid
        if not (root / "environment/scene.json").is_file():
            raise RequestError(404, "Scene not found")
        return root

    def health(self):
        return {
            "version": VERSION, "preview": self.preview, "active_scene": self.active_scene,
            "committed_pose": self.pose, "busy": self.lock.locked(),
            "generation": self.generation_status,
            "encoding_available": bool(self.cfg.get("taeh3_assets")),
            "reference_preparation_available": not self.preview,
            "configured_static": bool(self.cfg.get("static_condition")),
            "stage": None if self.run_config is None else self.run_config["stage"],
            "weights": self.cfg["weights"],
            "visual_encoder": "taeh3", "visual_decoder": "taeh3",
            "proxy_conditioning": "tae_tokens", "proxy_rgb": [352, 608],
            "encoded_current": [1, 24, 5, 22, 38], "output_size": [34, 60],
        }

    def scene_list(self):
        rows = []
        for path in sorted((self.data_dir / "scenes").glob("*/environment/scene.json")):
            doc = read(path)
            rows.append({"id": path.parents[1].name, "name": doc["name"],
                         "preview_only": doc.get("preview_only", False),
                         "reference_available": (path.parents[1] / "reference.png").is_file(),
                         "static_ready": (path.parents[1] / "static.safetensors").is_file()})
        return {"scenes": rows}

    def prepare_reference(self, sid):
        root = self.scene_dir(sid)
        if self.preview:
            raise RequestError(503, "Start generation mode to prepare a scene reference")
        if not (root / "reference.png").is_file():
            raise ValueError("Upload this scene's appearance reference first")
        self.acquire()
        try:
            if self.active_scene is not None:
                raise RequestError(409, "End the current model session before preparing a reference")
            args = argparse.Namespace(
                reference=[str(root / "reference.png")],
                output=str(root / "static.safetensors"),
                tae_assets=self.cfg["taeh3_assets"], device=self.cfg["encoder_device"],
            )
            from .token_runtime import prepare_static

            self.pool.submit(prepare_static, args, self.run_config).result()
            return {"id": sid, "static_ready": True, "version": VERSION}
        finally:
            self.lock.release()

    def acquire(self):
        if not self.lock.acquire(blocking=False):
            raise RequestError(409, "The asset/model worker is busy")

    def import_asset(self, fields):
        self.acquire()
        try:
            out = self.data_dir / "scenes" / uuid.uuid4().hex
            incoming = out / "upload"
            incoming.mkdir(parents=True)
            options = {"up_axis": "z", "units": "m"}
            path = reference = None
            for name, filename, payload in fields:
                if name in options and filename is None:
                    if len(payload) > 16:
                        raise ValueError("Invalid upload option")
                    options[name] = payload.decode("ascii")
                    continue
                if name not in {"asset", "reference", "static"} or not filename:
                    raise ValueError("Unknown upload field")
                suffix = Path(filename).suffix.lower()
                allowed = {"asset": FORMATS | {".zip"}, "reference": {".png", ".jpg", ".jpeg", ".webp"},
                           "static": {".safetensors"}}[name]
                if suffix not in allowed:
                    raise ValueError("Unsupported upload format")
                target = out / "static.safetensors" if name == "static" else incoming / (name + suffix)
                if target.exists():
                    raise ValueError("Duplicate upload field")
                target.write_bytes(payload)
                if name == "asset":
                    path = target
                elif name == "reference":
                    reference = target
            if path is None:
                raise ValueError("An asset is required")
            scene = load_upload(path, out, reference=reference, progress=lambda *args: None, **options)
            return {"id": out.name, "scene": scene}
        finally:
            self.lock.release()

    def process(self, sid, value):
        root = self.scene_dir(sid)
        operation = value.get("operation")
        if operation not in {"encode", "generate"}:
            raise ValueError("operation must be encode or generate")
        if not self.cfg.get("taeh3_assets"):
            raise RequestError(503, "Pinned TAEH3 assets have not been configured")
        if operation == "generate":
            if self.preview:
                raise RequestError(503, "Preview does not load or impersonate a generation model")
            if read(root / "environment/scene.json").get("preview_only"):
                raise ValueError("Calibration geometry is preview-only, not a semantic training asset")
        self.acquire()
        try:
            if operation == "generate" and self.active_scene not in {None, sid}:
                raise RequestError(409, "End the current model session before switching assets")
            start = self.pose if operation == "generate" and self.active_scene == sid else [0, 0, 0, 0, 0]
            trajectory = validate_path(value.get("trajectory"), start)
            static_path = root / "static.safetensors"
            if operation == "generate" and not static_path.is_file():
                configured = self.cfg.get("static_condition")
                if not configured:
                    raise ValueError("Attach the matching scene static condition or configure static_condition")
                static_path = Path(configured)
            out = self.data_dir / "results" / uuid.uuid4().hex
            out.mkdir(parents=True)
            try:
                blocks = self.pool.submit(self.produce, sid, trajectory, operation, out, static_path).result()
            except BaseException:
                if operation == "generate":
                    self.pool.submit(self.reset).result()
                raise
            return {
                "id": out.name, "frames": len(trajectory["poses"]), "pose": self.pose,
                "video": f"/results/{out.name}/generated.mp4"
                if (out / "generated.mp4").is_file() else None,
                "blocks": [{k: f"/results/{out.name}/{v}" if k in {"tokens", "latents"} else v
                            for k, v in block.items()} for block in blocks],
            }
        finally:
            self.lock.release()

    def produce(self, sid, trajectory, operation, out, static_path):
        """The single executor owns all browser and model objects."""
        import torch
        from .browser_renderer import BrowserRenderer
        from .model_process import ModelProcess
        from .token_runtime import CurrentTokenEncoder, save_current

        if self.renderer is None:
            self.renderer = BrowserRenderer(self.base_url, self.cfg["chromium_executable"])
        if self.encoder is None:
            self.encoder = CurrentTokenEncoder(self.cfg["taeh3_assets"], self.cfg["encoder_device"])
        if operation == "generate" and self.runtime is None:
            self.runtime = ModelProcess(self.cfg)
            self.generation_status = "loaded"
        if operation == "generate" and self.active_scene is None:
            self.runtime.request("start", static=str(static_path),
                                 manifest=self.cfg.get("static_manifest"),
                                 sample_id=self.cfg.get("static_sample_id"))
            self.active_scene = sid
        result = []
        context = list(self.decode_history)
        generated = []
        for index, block in enumerate(plan_path(trajectory, out.name)):
            frames = self.renderer.render(sid, block["poses"])
            pixels = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].contiguous()
            encoded_current = self.encoder.encode_current(pixels)
            token_path = out / f"tokens-{index:05d}.safetensors"
            save_current(token_path, encoded_current)
            item = {"tokens": token_path.name, "visible_frames": block["publication"]["visible_frames"]}
            if operation == "generate":
                latent_path = out / f"latents-{index:05d}.safetensors"
                self.runtime.request("generate", tokens=str(token_path), output=str(latent_path),
                                     steps=self.cfg["denoise_steps"])
                generated.append(latent_path)
                item["latents"] = latent_path.name
                self.pose = block["next_pose"]
            result.append(item)
        decoded = False
        if generated and self.cfg["decode_video"]:
            from safetensors.torch import load_file
            from carve.inference import decode_causal_video

            latents = torch.cat([load_file(str(p))["latents"] for p in context + generated], dim=2)
            decode_causal_video(
                latents, self.cfg["taeh3_assets"], out / "generated.mp4",
                device=torch.device(self.cfg["decoder_device"]),
                skip_frames=len(context) * 17, visible_frames=len(trajectory["poses"]),
            )
            decoded = True
        if generated:
            self.decode_history = context + generated
        write(out / "trajectory.json", trajectory)
        write(out / "result.json", {
            "version": VERSION, "operation": operation, "scene": sid, "blocks": result,
            "frames": len(trajectory["poses"]), "video_decoded": decoded,
            "padding_is_not_published": True,
            "visual_decoder": "taeh3",
            "decoder_tail": "trim_three_prefix_frames_per_chunk",
        })
        return result

    def reset(self):
        if self.runtime is not None:
            try:
                self.runtime.request("reset")
            except (RuntimeError, OSError, TimeoutError):
                self.runtime.close()
                self.runtime = None
                self.generation_status = "configured_not_loaded"
        self.decode_history = []
        self.active_scene = None
        self.pose = [0, 0, 0, 0, 0]

    def stop(self):
        self.acquire()
        try:
            self.pool.submit(self.reset).result()
            return {"reset": True}
        finally:
            self.lock.release()

    def close(self):
        def dispose():
            self.reset()
            if self.renderer is not None:
                self.renderer.close()
            if self.runtime is not None:
                self.runtime.close()
        self.pool.submit(dispose).result()
        self.pool.shutdown(wait=True, cancel_futures=True)


def parse_multipart(content_type, payload):
    if not content_type.startswith("multipart/form-data;") or "\r" in content_type or "\n" in content_type:
        raise ValueError("Use multipart/form-data for asset uploads")
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + payload
    )
    if not message.is_multipart() or message.defects:
        raise ValueError("Invalid multipart upload")
    fields, seen = [], set()
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name or name in seen or part.is_multipart() or part.defects:
            raise ValueError("Invalid or duplicate multipart field")
        seen.add(name)
        fields.append((name, part.get_filename(), part.get_payload(decode=True)))
    return fields


class Handler(BaseHTTPRequestHandler):
    server_version = "CARVEDemo/4"

    def respond(self, status, value):
        payload = json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def file(self, root, name):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise RequestError(404, "File not found")
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with path.open("rb") as source:
            while chunk := source.read(1 << 20):
                self.wfile.write(chunk)

    def body(self, *, upload=False):
        if self.headers.get("Transfer-Encoding"):
            raise RequestError(400, "Chunked request bodies are not supported")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise RequestError(400, "Invalid Content-Length") from None
        limit = self.server.state.cfg["max_upload_mb"] * (1 << 20) if upload else 512 * 1024
        if size < 1 or size > limit:
            raise RequestError(413, "Request body is empty or too large")
        self.connection.settimeout(30)
        payload = self.rfile.read(size)
        if len(payload) != size:
            raise RequestError(400, "Incomplete request body")
        return payload

    def dispatch(self, method):
        state = self.server.state
        path = unquote(urlsplit(self.path).path)
        parts = path.strip("/").split("/")
        if method == "GET":
            if path == "/api/health":
                return self.respond(200, state.health())
            if path == "/api/scenes":
                return self.respond(200, state.scene_list())
            if len(parts) == 4 and parts[0] == "scenes" and parts[2] == "environment":
                if parts[3] not in {"scene.json", "positions.bin", "colors.bin", "geometry.npz"}:
                    raise RequestError(404, "File not found")
                return self.file(state.scene_dir(parts[1]), "/".join(parts[2:]))
            if len(parts) == 3 and parts[0] == "results" and valid_id(parts[1]):
                if Path(parts[2]).suffix not in {".json", ".safetensors", ".mp4"}:
                    raise RequestError(404, "File not found")
                return self.file(state.data_dir / "results" / parts[1], parts[2])
            name = "index.html" if path == "/" else path.lstrip("/")
            if name in {"index.html", "style.css", "app.js", "render.js", "recording.js", "render.html"}:
                return self.file(ROOT / "web", name)
        elif method == "POST":
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise RequestError(403, "Cross-origin requests are not permitted")
            if path == "/api/assets":
                fields = parse_multipart(self.headers.get("Content-Type", ""), self.body(upload=True))
                return self.respond(201, state.import_asset(fields))
            if path == "/api/stop":
                return self.respond(200, state.stop())
            if len(parts) == 4 and parts[:2] == ["api", "scenes"] and parts[3] == "prepare-static":
                return self.respond(200, state.prepare_reference(parts[2]))
            if len(parts) == 4 and parts[:2] == ["api", "scenes"] and parts[3] == "process":
                value = json.loads(self.body())
                if not isinstance(value, dict):
                    raise ValueError("Request must be a JSON object")
                return self.respond(200, state.process(parts[2], value))
        raise RequestError(404, "Route not found")

    def handle_request(self, method):
        try:
            self.dispatch(method)
        except RequestError as error:
            self.respond(error.status, {"error": str(error)})
        except (ValueError, KeyError, OSError) as error:
            self.respond(400, {"error": str(error)})
        except Exception as error:
            self.respond(500, {"error": f"{type(error).__name__}: {error}"})

    def do_GET(self):
        self.handle_request("GET")

    def do_POST(self):
        self.handle_request("POST")

    def log_message(self, format, *args):
        sys.stderr.write(f"{self.log_date_time_string()} {format % args}\n")


class DemoServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, state):
        super().__init__(address, Handler)
        self.state = state
        host, port = self.server_address[:2]
        host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        state.base_url = f"http://{host}:{port}"

    def server_close(self):
        super().server_close()
        self.state.close()


def create_server(cfg=None, *, preview=True):
    cfg = cfg or load_config()
    state = DemoState(cfg, preview)
    try:
        return DemoServer((cfg["host"], cfg["port"]), state)
    except BaseException:
        state.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/demo.json"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--data-dir")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides={
        "host": args.host, "port": args.port,
        "data_dir": str(Path(args.data_dir).resolve()) if args.data_dir else None,
    })
    with create_server(cfg, preview=not args.generate) as server:
        print(f"CARVE v4 {'preview' if not args.generate else 'token runtime'}: {server.state.base_url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
