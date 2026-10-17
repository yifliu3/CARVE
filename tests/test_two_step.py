"""CPU regression checks for the released two-step Stage2 recipe."""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from carve.config import model_contract, resolve
from carve.core import Condition, Core
from carve.evaluation import inference_steps, validate_inference_checkpoint
from carve.streaming import CausalSession
from demo.settings import DENOISE_STEPS, VERSION, load_config
from demo.token_runtime import TokenRuntime, inspect_checkpoint, resolve_run_config

ROOT = Path(__file__).resolve().parents[1]


def config(name="stage2"):
    return resolve(ROOT / "configs" / f"{name}.yaml")


def condition(chunks=2):
    rng = torch.Generator().manual_seed(31)

    def noise(*shape):
        return torch.randn(*shape, generator=rng)

    return Condition(
        noise(1, 4, 5120), torch.tensor([0, 0, 1, 1]),
        [noise(1, 24, 1, 2, 2)], noise(1, 24, 5 * chunks, 2, 2),
        noise(1, 2, 32), noise(1, 24, 5 * chunks, 2, 2), output_size=(2, 2),
    )


class TwoStepConfigTests(unittest.TestCase):
    def test_release_entry_points_use_two_steps_and_prior_stages_keep_budgets(self):
        stage2 = config()
        inference = config("inference")
        demo_run = resolve_run_config(ROOT / "demo/configs/run.yaml")
        for cfg in (stage2, inference, demo_run):
            self.assertEqual(cfg["stage"], "causal_dmd_sgf")
            self.assertEqual(inference_steps(cfg), 2)
            self.assertEqual(cfg["runtime"]["experiment_version"], VERSION)
        self.assertEqual(load_config()["denoise_steps"], DENOISE_STEPS)
        self.assertEqual(inference_steps(config("stage0p5")), 30)
        self.assertEqual(inference_steps(config("stage1")), 4)
        self.assertEqual(stage2["train"]["max_steps"], 1000)
        self.assertEqual(stage2["checkpoint"]["initialize_weights"], "live")
        self.assertEqual(stage2["checkpoint"]["teacher_weights"], "live")
        self.assertEqual(stage2["checkpoint"]["resume"], "")

    def test_two_step_recipe_preserves_visual_contract_for_stage_handoffs(self):
        expected = model_contract(config("stage0p5"))
        for name in ("stage1", "stage2", "inference"):
            self.assertEqual(model_contract(config(name)), expected)

    def test_explicit_solver_override_is_validated(self):
        self.assertEqual(inference_steps(config("inference"), 2), 2)
        self.assertEqual(inference_steps(config("stage1"), 8), 8)
        for steps in (0, -1, 2.0, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                inference_steps(config("inference"), steps)

    def test_demo_rejects_four_step_settings(self):
        for steps in (4, 0, 2.0, True):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "denoise_steps=2"):
                load_config(overrides={"denoise_steps": steps})
        with self.assertRaisesRegex(ValueError, "two denoising steps"):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "run.json"
                cfg = config()
                cfg["train"]["denoise_steps"] = 4
                path.write_text(json.dumps(cfg))
                resolve_run_config(path)

    def test_cli_resolve_and_help_expose_two_step_recipe(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run(
            [sys.executable, "-m", "carve", "resolve", "--config", "configs/inference.yaml"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=True,
        )
        self.assertEqual(json.loads(result.stdout)["train"]["denoise_steps"], 2)
        result = subprocess.run(
            [sys.executable, "-m", "carve", "infer", "--help"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=True,
        )
        self.assertIn("2 for released Stage2", " ".join(result.stdout.split()))


class CheckpointRecipeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config("inference")
        self.meta = dict(schema="cwm-h3.checkpoint.v1", stage="causal_dmd_sgf",
                         anyflow=True, contract=model_contract(self.cfg),
                         config=copy.deepcopy(self.cfg))

    def test_checkpoint_requires_matching_stage_recipe_and_version(self):
        validate_inference_checkpoint(self.cfg, self.meta)
        changes = (
            ("stage", "causal_tf_anyflow"),
            ("steps", 4),
            ("version", "v4-colored-uniform-taeh3-tokens608-frozen-proj"),
        )
        for field, value in changes:
            meta = copy.deepcopy(self.meta)
            if field == "stage":
                meta["stage"] = value
            elif field == "steps":
                meta["config"]["train"]["denoise_steps"] = value
            else:
                meta["config"]["runtime"]["experiment_version"] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_inference_checkpoint(self.cfg, meta)

    def test_demo_accepts_completed_two_step_metadata_and_rejects_old_recipe(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "COMPLETE.json"
            marker.write_text(json.dumps(self.meta))
            self.assertEqual(inspect_checkpoint(folder, self.cfg)["stage"], "causal_dmd_sgf")
            self.meta["config"]["train"]["denoise_steps"] = 4
            marker.write_text(json.dumps(self.meta))
            with self.assertRaisesRegex(ValueError, "denoise_steps"):
                inspect_checkpoint(folder, self.cfg)

    def test_standalone_inference_rejects_four_step_checkpoint_before_model_build(self):
        from carve import inference

        self.meta["config"]["train"]["denoise_steps"] = 4
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(tae_assets="assets", proxy_video="proxy.mp4",
                                   reference_image="reference.png", output=folder,
                                   weights="checkpoint", steps=None)
            with (
                patch.object(inference, "initialize_distributed", return_value=(torch.device("cpu"), 0, 1)),
                patch.object(inference.checkpoint, "read", return_value=(Path("checkpoint"), self.meta)),
                patch.object(inference, "build") as build,
                self.assertRaisesRegex(ValueError, "denoise_steps"),
            ):
                inference.run(self.cfg, args)
            build.assert_not_called()


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_streaming_sgf_has_two_denoising_forwards_then_one_cache_commit(self):
        calls = []

        def forward(c, noisy, t, **kwargs):
            calls.append((kwargs["mode"], float(t), float(kwargs["r"])))
            return torch.zeros_like(noisy)

        core = SimpleNamespace(history=2, shift=12.0, anyflow=True, forward=forward)
        c = condition()
        c.target = None
        session = CausalSession(core, c, seed=42)
        steps = inference_steps(config("inference"))
        clean = session.generate_token_chunk(c.proxy[:, :, :5], steps=steps, sgf=True)
        self.assertTrue(torch.isfinite(clean).all())
        self.assertEqual([x[0] for x in calls], ["rollout", "rollout", "commit"])
        self.assertTrue(all(x[2] == 1 for x in calls))
        self.assertEqual(session.index, 1)

    def test_demo_default_steps_and_explicit_two_reach_session(self):
        runtime = object.__new__(TokenRuntime)
        runtime.cfg = config("inference")
        runtime.meta = {"stage": "causal_dmd_sgf"}
        runtime.device = torch.device("cpu")
        calls = []
        result = torch.zeros(1, 24, 5, 2, 2)
        session = SimpleNamespace(pending=True, commit=lambda: result)
        session.propose_token_chunk = lambda current, **kwargs: calls.append(kwargs)
        runtime.session = session
        current = torch.zeros(1, 24, 5, 22, 38)
        self.assertIs(runtime.generate(current), result)
        self.assertIs(runtime.generate(current, steps=2), result)
        self.assertEqual([x["steps"] for x in calls], [2, 2])
        self.assertTrue(all(x["sgf"] for x in calls))
        with self.assertRaisesRegex(ValueError, "steps=2"):
            runtime.generate(current, steps=4)
        self.assertEqual(len(calls), 2)

    def test_demo_health_reports_two_step_budget(self):
        from demo.server import DemoState

        with tempfile.TemporaryDirectory() as folder:
            settings = load_config(overrides={"data_dir": folder})
            state = DemoState(settings, preview=True)
            try:
                self.assertEqual(state.health()["denoise_steps"], 2)
            finally:
                state.close()

    def test_real_tiny_stage_handoffs_two_step_training_and_exact_resume(self):
        from carve import checkpoint
        from carve.evaluation import generate_test_sample
        from carve.train import Runtime

        def tiny(name):
            cfg = config(name)
            cfg["model"].update(tiny=True, lora_rank=2, lora_alpha=2, history_chunks=2,
                                activation_checkpointing=False, attention_checkpointing=False)
            cfg["distributed"]["sp_size"] = 1
            cfg["runtime"]["device"] = "cpu"
            cfg["train"].update(learning_rate=.002, critic_learning_rate=.001,
                                warmup_steps=0, critic_updates_per_student=1)
            return cfg

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c = condition()
            bid = Runtime(tiny("stage0p5"), dataset=[c])
            self.assertTrue(math.isfinite(bid.train_step()["loss"]))
            checkpoint.save(bid, root / "bid")
            tf_cfg = tiny("stage1")
            tf_cfg["checkpoint"]["initialize"] = str(root / "bid")
            tf = Runtime(tf_cfg, dataset=[c])
            self.assertTrue(math.isfinite(tf.train_step()["loss"]))
            checkpoint.save(tf, root / "tf")
            cfg = tiny("stage2")
            cfg["checkpoint"].update(initialize=str(root / "tf"), teacher=str(root / "bid"))
            runtime = Runtime(cfg, dataset=[c])
            calls = []
            original = Core.rollout

            def rollout(core, *args, **kwargs):
                calls.append((kwargs["steps"], kwargs["sgf"]))
                return original(core, *args, **kwargs)

            with patch.object(Core, "rollout", rollout):
                runtime.train_step()
                result = runtime.train_step()
            self.assertTrue(result["student_updated"])
            self.assertTrue(math.isfinite(result["loss"]))
            self.assertTrue(calls and all(x == (2, True) for x in calls))
            self.assertTrue(runtime.critic_optimizer.state)
            self.assertTrue(all(p.grad is None for p in runtime.teacher.model.parameters()))
            self.assertTrue(all(not p.requires_grad for p in runtime.student.model.proj_in.parameters()))
            sample = copy.deepcopy(c)
            sample.target = None
            forwards = []
            original_forward = runtime.core.forward

            def forward(*args, **kwargs):
                forwards.append((kwargs["mode"], kwargs.get("chunk")))
                return original_forward(*args, **kwargs)

            with patch.object(runtime.core, "forward", forward):
                video, metrics = generate_test_sample(
                    runtime.core, sample, stage=cfg["stage"],
                    steps=inference_steps(cfg), seed=42,
                )
            self.assertEqual(video.shape, sample.video_shape)
            self.assertTrue(torch.isfinite(video).all())
            self.assertEqual(metrics["denoise_steps"], 2)
            self.assertEqual(metrics["sampling_mode"], "causal_sgf")
            self.assertFalse(metrics["target_video_input"])
            self.assertEqual(forwards, [
                ("rollout", 0), ("rollout", 0), ("commit", 0),
                ("rollout", 1), ("rollout", 1), ("commit", 1),
            ])
            checkpoint.save(runtime, root / "stage2")
            expected = runtime.train_step()
            values = checkpoint.values(runtime.student)
            resumed_cfg = copy.deepcopy(cfg)
            resumed_cfg["checkpoint"]["resume"] = str(root / "stage2")
            resumed = Runtime(resumed_cfg, dataset=[c])
            self.assertEqual(resumed.train_step(), expected)
            for key, value in values.items():
                torch.testing.assert_close(resumed.student.trainables[key], value, rtol=0, atol=0)
            changed = copy.deepcopy(resumed_cfg)
            changed["train"]["denoise_steps"] = 4
            with self.assertRaisesRegex(ValueError, "training settings changed"):
                Runtime(changed, dataset=[c])


if __name__ == "__main__":
    unittest.main()
