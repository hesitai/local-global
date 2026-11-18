from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import generate_online_2d_epoch as online
import script_uav_global_blur_multisubframe_no_depth as motion


class Online2DEpochTest(unittest.TestCase):
    def test_rendered_patch_is_aligned_and_has_exact_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            height, width = 480, 640
            y_grid, x_grid = np.mgrid[:height, :width]
            source = np.stack(
                (
                    x_grid / float(width - 1),
                    y_grid / float(height - 1),
                    ((x_grid + y_grid) % 97) / 96.0,
                ),
                axis=-1,
            )
            source_path = root / "source.png"
            input_path = root / "input.png"
            target_path = root / "target.png"
            motion.write_image_uint8(source_path, source)
            task = online.PatchTask(
                source_index=0,
                variant_index=0,
                source_path=str(source_path),
                staging_input_path=str(input_path),
                staging_target_path=str(target_path),
                output_name="sample.png",
                epoch=3,
                base_seed=1234,
                patch_size=256,
                cases=("01_heading_linear_uniform",),
                strength_min=1.2,
                strength_max=1.2,
                adaptive_subframes=False,
                fixed_subframes=9,
                interpolation_margin=2,
                crop_trials=256,
            )
            row = online.render_task(task)
            blurry = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
            original = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            top, bottom = int(row["crop_top"]), int(row["crop_bottom"])
            left, right = int(row["crop_left"]), int(row["crop_right"])

            self.assertEqual(blurry.shape, (256, 256, 3))
            self.assertEqual(target.shape, (256, 256, 3))
            np.testing.assert_array_equal(
                target, original[top:bottom, left:right]
            )
            self.assertGreater(float(np.mean(np.abs(
                blurry.astype(np.float32) - target.astype(np.float32)
            ))), 0.0)

    def test_same_task_is_deterministic(self):
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        homographies = np.repeat(np.eye(3)[None, ...], 3, axis=0)
        self.assertEqual(
            online.choose_valid_crop(rng_a, homographies, 480, 640, 256, 2, 8),
            online.choose_valid_crop(rng_b, homographies, 480, 640, 256, 2, 8),
        )

    def test_xy_motion_components_receive_independent_strengths(self):
        trajectory = motion.make_uav_trajectory(
            "01_heading_linear_uniform", 1001, component_seed_base=1
        )
        factors = online.randomize_trajectory_geometry(
            trajectory,
            np.random.default_rng(7),
            1.0,
            1.8,
            direction_deg=0.0,
        )
        x_strength = factors["motion.path_point_1_x"]
        y_strength = factors["motion.path_point_1_y"]
        self.assertGreaterEqual(x_strength, 1.0)
        self.assertLessEqual(x_strength, 1.8)
        self.assertGreaterEqual(y_strength, 1.0)
        self.assertLessEqual(y_strength, 1.8)
        self.assertNotEqual(x_strength, y_strength)
        endpoint = trajectory["path"]["points"][1]
        self.assertAlmostEqual(endpoint[0], 25.2 * x_strength)
        self.assertAlmostEqual(endpoint[1], -9.8 * y_strength)


if __name__ == "__main__":
    unittest.main()
