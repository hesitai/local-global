"""Offline regression checks: image inference/export, metrics and CLI reports.

Run: python test/real_test/check_real_nr.py
No PyIQA installation or pretrained model download is required for these checks.
"""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image
import torch
from skimage import data
from scipy.ndimage import gaussian_filter

from evaluate_real_nr import (main, read_rgb, restore_image, tile_starts,
                              validate_inference_options, build_parser, load_config)
from nr_metrics import MetricSuite, cpbd_score, information_entropy
import nr_model_adapter


class RealEvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        (self.inputs / "场景").mkdir(parents=True)
        self.weights = self.root / "identity.pt"
        torch.jit.script(torch.nn.Identity()).save(str(self.weights))
        rng = np.random.default_rng(43)
        self.rgb = rng.integers(0, 256, (97, 145, 3), dtype=np.uint8)
        Image.fromarray(self.rgb).save(self.inputs / "场景" / "a.png")
        Image.fromarray(self.rgb).save(self.inputs / "场景" / "a.jpg")
        self.config = {
            "input_dir": str(self.inputs), "weights": str(self.weights),
            "output_dir": str(self.root / "results"), "mode": "restore",
            "device": "cpu", "metric_device": "cpu", "recursive": True,
            "metrics": ["LV", "CPBD", "Entropy"], "report_every": 2,
            "model_adapter": str(Path(nr_model_adapter.__file__)),
            "model": {"factory": "torchscript", "pad_multiple": 16,
                      "tile_size": 64, "tile_overlap": 16},
        }
        self.config_path = self.root / "config.json"

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *options):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            return main(["--config", str(self.config_path)] + list(options))

    def test_full_restore_preserves_size_names_pixels_and_true_means(self):
        self.assertEqual(self.run_cli(), 0)
        output = Path(self.config["output_dir"])
        with (output / "per_image_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        expected_lv = []
        for row in rows:
            restored = read_rgb(row["restored_path"])
            np.testing.assert_array_equal(restored, read_rgb(row["input_path"]))
            self.assertEqual(restored.shape, self.rgb.shape)
            self.assertTrue(Path(row["restored_path"]).name.endswith(("a.png.png", "a.jpg.png")))
            expected_lv.append(MetricSuite(["LV"])(restored)["LV"])
            self.assertEqual(row["MOS"], "")
        self.assertNotEqual(rows[0]["restored_path"], rows[1]["restored_path"])
        summary = json.loads((output / "summary.json").read_text())
        self.assertAlmostEqual(summary["metrics"]["LV"]["mean"], float(np.mean(expected_lv)))
        self.assertEqual(summary["metrics"]["MOS"]["valid_count"], 0)
        with (output / "timings.csv").open(encoding="utf-8-sig", newline="") as stream:
            timings = list(csv.DictReader(stream))
        self.assertEqual(len(timings), 2)
        self.assertGreater(float(timings[0]["restoration_seconds"]), 0)
        self.assertGreater(float(timings[0]["CPBD_seconds"]), 0)
        # Existing experiment is never overwritten.
        with redirect_stdout(io.StringIO()):
            from contextlib import redirect_stderr
            with redirect_stderr(io.StringIO()):
                self.assertEqual(self.run_cli(), 1)

    def test_evaluate_records_corrupt_and_tiny_images_without_zero_fill(self):
        self.config["mode"] = "evaluate"
        (self.inputs / "broken.jpg").write_bytes(b"not an image")
        Image.new("RGB", (32, 32)).save(self.inputs / "tiny.png")
        self.assertEqual(self.run_cli(), 2)
        summary = json.loads((Path(self.config["output_dir"]) / "summary.json").read_text())
        self.assertEqual(summary["status_counts"], {"error": 1, "ok": 2, "partial": 1})
        self.assertEqual(summary["metrics"]["CPBD"]["valid_count"], 2)
        self.assertEqual(summary["metrics"]["CPBD"]["missing_count"], 2)
        self.assertEqual(summary["metrics"]["LV"]["valid_count"], 3)

    def test_restore_only_then_evaluate_and_merge_mos(self):
        self.config["mode"] = "restore_only"
        self.assertEqual(self.run_cli(), 0)
        first = Path(self.config["output_dir"])
        self.config["mode"] = "evaluate"
        self.config["input_dir"] = str(first / "restored")
        self.config["output_dir"] = str(self.root / "metrics")
        self.assertEqual(self.run_cli(), 0)
        output = Path(self.config["output_dir"])
        ratings = self.root / "mos.csv"
        ratings.write_text("image_id,rater_id,score\n场景/a.png.png,r1,3\n场景/a.png.png,r2,5\n", encoding="utf-8")
        self.assertEqual(self.run_cli("--summarize-only", "--mos-csv", str(ratings)), 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["metrics"]["MOS"]["mean"], 4.0)
        self.assertEqual(summary["metrics"]["MOS"]["valid_count"], 1)
        self.assertEqual(summary["metrics"]["LV"]["valid_count"], 2)

    def test_constant_entropy_and_blur_sensitivity(self):
        self.assertEqual(information_entropy(np.zeros((64, 64), dtype=np.uint8)), 0.0)
        self.assertEqual(information_entropy(np.arange(256, dtype=np.uint8).reshape(16, 16)), 8.0)
        sharp = data.camera()
        blurry = gaussian_filter(sharp.astype(np.float64), sigma=3).round().astype(np.uint8)
        self.assertGreater(cpbd_score(sharp), cpbd_score(blurry))
        self.assertEqual(cpbd_score(np.zeros((64, 64), dtype=np.uint8)), 0.0)

    def test_model_output_selection_and_strict_checkpoint_loading(self):
        class DictionaryNetwork(torch.nn.Module):
            def forward(self, tensor):
                return {"restored": tensor}

        image = torch.zeros(1, 3, 16, 16)
        with self.assertRaises(ValueError):
            nr_model_adapter.predict(DictionaryNetwork(), image, {})
        self.assertTrue(torch.equal(
            nr_model_adapter.predict(DictionaryNetwork(), image, {"output_key": "restored"}), image))
        model = torch.nn.Conv2d(3, 3, 1)
        checkpoint = self.root / "model.pth"
        torch.save({"state_dict": {"module." + key: value for key, value in model.state_dict().items()}}, checkpoint)
        target = torch.nn.Conv2d(3, 3, 1)
        nr_model_adapter.load_checkpoint_strict(target, checkpoint)
        self.assertTrue(torch.equal(target.weight, model.weight))
        with self.assertRaises(RuntimeError):
            nr_model_adapter.load_checkpoint_strict(torch.nn.Conv2d(3, 4, 1), checkpoint)

    def test_padding_tiling_validation_and_nonfinite_failure(self):
        self.assertEqual(tile_starts(145, 64, 16), [0, 48, 81])
        for invalid in ({"pad_multiple": 0}, {"tile_size": -1},
                        {"tile_size": 64, "tile_overlap": 64},
                        {"tile_batch_size": 0}, {"tile_batch_size": True}, {"amp": "false"}):
            with self.assertRaises(ValueError):
                validate_inference_options(invalid)

        class InvalidNetwork(torch.nn.Module):
            def forward(self, tensor):
                return tensor * float("nan")

        with self.assertRaises(ValueError):
            restore_image(self.rgb, InvalidNetwork(), nr_model_adapter,
                          {"pad_multiple": 16}, torch.device("cpu"))
        for square in (False, True):
            result = restore_image(self.rgb, torch.nn.Identity(), nr_model_adapter,
                                   {"pad_multiple": 128, "square_pad": square}, torch.device("cpu"))
            np.testing.assert_array_equal(result, self.rgb)

    def test_batched_tiles_preserve_results_boundaries_and_final_short_batch(self):
        class PointwiseNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, tensor):
                self.batch_sizes.append(tensor.shape[0])
                return torch.sigmoid(tensor * 0.7 - 0.2)

        for rgb in (self.rgb, self.rgb[:33]):
            options = {"pad_multiple": 16, "tile_size": 64, "tile_overlap": 16,
                       "tile_batch_size": 1}
            reference_model = PointwiseNetwork()
            reference = restore_image(rgb, reference_model, nr_model_adapter, options, torch.device("cpu"))
            batched_model = PointwiseNetwork()
            options["tile_batch_size"] = 4
            stats = {}
            batched = restore_image(rgb, batched_model, nr_model_adapter, options, torch.device("cpu"), stats)
            np.testing.assert_array_equal(batched, reference)
            self.assertEqual(sum(batched_model.batch_sizes), len(reference_model.batch_sizes))
            self.assertEqual(stats["tile_batches"], len(batched_model.batch_sizes))
            self.assertLess(len(batched_model.batch_sizes), len(reference_model.batch_sizes))
        self.assertEqual(batched_model.batch_sizes, [3])

    def test_amp_requires_cuda_and_batch_cli_overrides_config(self):
        with self.assertRaises(ValueError):
            restore_image(self.rgb, torch.nn.Identity(), nr_model_adapter,
                          {"amp": True}, torch.device("cpu"))
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        options = build_parser().parse_args(["--config", str(self.config_path),
            "--tile-batch-size", "4", "--tile-size", "128", "--tile-overlap", "32", "--amp"])
        parsed, _ = load_config(options)
        self.assertEqual(parsed["model"]["tile_batch_size"], 4)
        self.assertEqual(parsed["model"]["tile_size"], 128)
        self.assertEqual(parsed["model"]["tile_overlap"], 32)
        self.assertTrue(parsed["model"]["amp"])

    def test_metric_timings_include_failures_and_do_not_retain_previous_calls(self):
        suite = MetricSuite(["LV", "CPBD", "Entropy"])
        result = suite(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertIsNone(result["CPBD"])
        self.assertEqual(set(suite.last_timings), {"LV", "CPBD", "Entropy"})
        self.assertTrue(all(value >= 0 for value in suite.last_timings.values()))
        suite(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(suite.last_errors, {})


if __name__ == "__main__":
    unittest.main()
