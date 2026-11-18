"""Focused regression checks for MOS validation and report aggregation.

Run with any Python >= 3.8: python test/real_test/check_nr_results.py
"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from nr_results import (
    load_mos, merge_mos_and_resummarize, summarize_rows, write_csv,
    write_mos_template, write_reports,
)


class ResultChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def ratings_file(self, ratings):
        path = self.root / "ratings.csv"
        write_csv(path, ratings, ("image_id", "rater_id", "score"))
        return path

    def test_image_means_are_not_weighted_by_rater_count(self):
        path = self.ratings_file([
            {"image_id": "a.png", "rater_id": "r1", "score": 1},
            {"image_id": "b.png", "rater_id": "r1", "score": 5},
            {"image_id": "b.png", "rater_id": "r2", "score": 5},
            {"image_id": "b.png", "rater_id": "r3", "score": 5},
        ])
        mos = load_mos(path, {"a.png", "b.png"})
        self.assertEqual(mos["b.png"]["MOS_n"], 3)
        result = summarize_rows([mos["a.png"], mos["b.png"]])["MOS"]
        self.assertEqual(result["mean"], 3.0)
        self.assertEqual(result["std"], 2.0)

    def test_bad_ratings_are_rejected(self):
        bad_rows = [
            [{"image_id": "other.png", "rater_id": "r1", "score": 4}],
            [{"image_id": "a.png", "rater_id": "", "score": 4}],
            [{"image_id": "a.png", "rater_id": "r1", "score": "nan"}],
            [{"image_id": "a.png", "rater_id": "r1", "score": 6}],
            [
                {"image_id": "a.png", "rater_id": "r1", "score": 4},
                {"image_id": "a.png", "rater_id": "r1", "score": 5},
            ],
        ]
        for rows in bad_rows:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    load_mos(self.ratings_file(rows), {"a.png"})

    def test_template_is_blank_and_never_overwritten(self):
        path = self.root / "mos.csv"
        self.assertTrue(write_mos_template(path, ["图像/a.png", "b.jpg"]))
        contents = path.read_bytes()
        self.assertTrue(contents.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(load_mos(path, {"图像/a.png", "b.jpg"}), {})
        self.assertFalse(write_mos_template(path, ["other.png"]))
        self.assertEqual(path.read_bytes(), contents)

    def test_missing_nonfinite_and_failed_are_not_zero(self):
        rows = [
            {"status": "ok", "LV": 10, "MOS": 4},
            {"status": "partial", "LV": "20", "MOS": ""},
            {"status": "error", "LV": 1000, "MOS": 5},
            {"status": "ok", "LV": float("nan"), "MOS": None},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["LV"]["mean"], 15)
        self.assertEqual(summary["LV"]["valid_count"], 2)
        self.assertEqual(summary["LV"]["missing_count"], 2)
        self.assertEqual(summary["MOS"]["mean"], 4)
        self.assertIsNone(summary["NIQE"]["mean"])

    def test_resummarize_keeps_objective_scores_and_ignores_failed_mos(self):
        rows = [
            {"image_id": "a.png", "status": "ok", "LV": 12, "Entropy": 4.2},
            {"image_id": "b.png", "status": "error", "error": "model failed"},
        ]
        write_reports(self.root, rows, {"original_run": "retained"})
        path = self.ratings_file([
            {"image_id": "a.png", "rater_id": "r1", "score": 4},
            {"image_id": "a.png", "rater_id": "r2", "score": 2},
            {"image_id": "b.png", "rater_id": "r1", "score": 5},
        ])
        result = merge_mos_and_resummarize(self.root, path)
        self.assertEqual(result["metrics"]["LV"]["mean"], 12)
        self.assertEqual(result["metrics"]["MOS"]["mean"], 3)
        self.assertEqual(result["metrics"]["MOS"]["valid_count"], 1)
        self.assertEqual(result["metadata"]["original_run"], "retained")
        with (self.root / "per_image_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            updated = list(csv.DictReader(stream))
        self.assertEqual(updated[0]["MOS_n"], "2")
        self.assertEqual(updated[1]["MOS"], "")
        stored = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertIsNone(stored["metrics"]["NIQE"]["mean"])


if __name__ == "__main__":
    unittest.main()
