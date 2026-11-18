from __future__ import annotations

import unittest

import numpy as np

from conv_opt import motion_blur_conv, motion_blur_conv_mirror
from gen_mf import MotionParameters, gen_mf
from script_uav_global_blur_multisubframe_no_depth import (
    build_motion_summary_from_trajectory,
    evaluate_trajectory_at_time as evaluate_uav_trajectory_at_time,
    make_uav_trajectory,
    synthesize_blur_by_multisubframe_v2,
    warp_frame_similarity,
)
from script_uav_global_blur_repro_smooth_flow import make_uav_global_mf_smooth
from script_uav_object_motion_blur_framework_A import (
    evaluate_object_trajectory_at_time,
    make_object_trajectory,
    synthesize_object_motion_blur_A,
)


class CoreTests(unittest.TestCase):
    def test_motion_blur_conv_identity(self) -> None:
        image = np.arange(35, dtype=np.float64).reshape(5, 7) / 34.0
        magnitude = np.ones((5, 7), dtype=np.float64)
        orientation = np.zeros((5, 7), dtype=np.float64)
        self.assertTrue(np.allclose(motion_blur_conv(image, magnitude, orientation), image))
        self.assertTrue(np.allclose(motion_blur_conv_mirror(image, magnitude, orientation), image))

    def test_original_motion_field(self) -> None:
        parameters = MotionParameters(5, -3, 0.2, -0.1, 1e-3, (12, 16), 0.01)
        motion_field = gen_mf(24, 32, parameters)
        self.assertEqual(motion_field.mu.shape, (24, 32))
        self.assertTrue(np.isfinite(motion_field.mu).all())
        self.assertTrue(np.isfinite(motion_field.mv).all())

    def test_matlab_twister_phase_and_smooth_field(self) -> None:
        motion_field, parameters = make_uav_global_mf_smooth(
            321, 481, "03_hover_jitter_nonuniform", 20263109, 1.0, False
        )
        expected_first_phases = np.array([2.53304369, 3.14927432])
        self.assertTrue(
            np.allclose(np.asarray(parameters["phase"])[:2], expected_first_phases, atol=1e-8)
        )
        self.assertEqual(motion_field.mu.shape, (321, 481))

    def test_warp_identity(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.random((20, 25, 3))
        result = warp_frame_similarity(image, 0, 0, 0, 1, 12, 9.5)
        self.assertTrue(np.allclose(result, image, atol=1e-6))

    def test_multisubframe_path_summaries(self) -> None:
        height, width = 20, 30
        tau = np.linspace(-0.5, 0.5, 21)
        center_x, center_y = (width - 1) / 2, (height - 1) / 2
        linear = build_motion_summary_from_trajectory(
            height,
            width,
            tau,
            make_uav_trajectory("01_linear_uniform", 1),
            center_x,
            center_y,
        )
        expected_length = np.sqrt(18.0**2 + 7.0**2)
        self.assertTrue(np.allclose(linear["path_len"], expected_length))

        vibration = build_motion_summary_from_trajectory(
            height,
            width,
            np.linspace(-0.5, 0.5, 101),
            make_uav_trajectory("05_high_frequency_vibration", 1),
            center_x,
            center_y,
        )
        self.assertLess(float(np.max(np.abs(vibration["end_mv"]))), 1e-10)
        self.assertGreater(float(vibration["path_len"].mean()), 1.0)

    def test_global_multisubframe_uses_start_gt(self) -> None:
        image = np.zeros((24, 32, 3), dtype=np.float64)
        image[8:16, 12:20] = 1.0
        trajectory = make_uav_trajectory("02_linear_variable_speed", 1)
        _, _, metadata, _ = synthesize_blur_by_multisubframe_v2(
            image, trajectory, 7, 4
        )
        self.assertEqual(metadata["gt_reference"], "start")
        self.assertEqual(float(metadata["tau_list"][0]), 0.0)
        self.assertEqual(float(metadata["tau_list"][-1]), 1.0)
        start = evaluate_uav_trajectory_at_time(trajectory, 0.0)
        end = evaluate_uav_trajectory_at_time(trajectory, 1.0)
        self.assertTrue(np.allclose(start, (0.0, 0.0, 0.0, 1.0)))
        self.assertAlmostEqual(end[0], trajectory["total_dx"])
        self.assertAlmostEqual(end[1], trajectory["total_dy"])

    def test_object_multisubframe_pipeline(self) -> None:
        background = np.zeros((40, 50, 3), dtype=np.float64)
        mask = np.zeros((40, 50), dtype=bool)
        mask[14:26, 20:31] = True
        foreground = np.zeros_like(background)
        foreground[mask] = 1.0
        trajectory = make_object_trajectory("01_obj_linear_motion", 1, mask)
        blur, metadata, debug = synthesize_object_motion_blur_A(
            foreground,
            background,
            mask.astype(np.float64),
            trajectory,
            7,
            4,
        )
        self.assertEqual(blur.shape, background.shape)
        self.assertEqual(int(metadata["N_subframes"]), 7)
        self.assertEqual(set(debug), {"first", "mid", "last"})

        self.assertTrue(np.allclose(debug["first"], foreground))
        self.assertEqual(metadata["gt_reference"], "start")

        curved = make_object_trajectory("02_obj_curved_motion", 2, mask)
        start = evaluate_object_trajectory_at_time(curved, 0.0)
        middle = evaluate_object_trajectory_at_time(curved, 0.5)
        end = evaluate_object_trajectory_at_time(curved, 1.0)
        self.assertTrue(np.allclose(start, (0.0, 0.0, 0.0, 1.0)))
        self.assertAlmostEqual(end[2] - start[2], curved["total_turn_angle_deg"])
        self.assertEqual(start[3], 1.0)
        self.assertEqual(end[3], 1.0)
        # 圆弧的首、中、末中心不应共线。
        cross = middle[0] * end[1] - middle[1] * end[0]
        self.assertGreater(abs(cross), 1e-6)


if __name__ == "__main__":
    unittest.main()
