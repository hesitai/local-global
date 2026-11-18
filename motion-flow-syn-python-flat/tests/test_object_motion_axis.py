from __future__ import annotations

import unittest

import numpy as np

from script_uav_object_motion_blur_framework_A import (
    estimate_object_front_back_axis,
    evaluate_object_trajectory_at_time,
    make_independent_object_trajectory,
    validate_center_referenced_trajectory,
)


def lateral_fraction(
    direction_xy: np.ndarray, axis_deg: float
) -> float:
    direction = np.asarray(direction_xy, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    angle_rad = np.deg2rad(axis_deg)
    axis = np.asarray([np.cos(angle_rad), np.sin(angle_rad)])
    return abs(
        float(direction[0] * axis[1] - direction[1] * axis[0])
    )


class ObjectMotionAxisTests(unittest.TestCase):
    def test_elongated_top_view_uses_oriented_long_axis(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[4:36, 15:25] = True
        axis_deg, metadata = estimate_object_front_back_axis(mask, 1.5)
        self.assertEqual(
            metadata["front_back_axis_rule"],
            "oriented_bbox_long_axis_for_elongated_top_view",
        )
        self.assertLess(abs(abs(axis_deg) - 90.0), 1.0e-6)

    def test_front_view_near_square_uses_oriented_short_axis(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 8:32] = True
        axis_deg, metadata = estimate_object_front_back_axis(mask, 1.5)
        self.assertEqual(
            metadata["front_back_axis_rule"],
            "oriented_bbox_short_axis_for_front_or_rear_view",
        )
        self.assertLess(abs(abs(axis_deg) - 90.0), 1.0e-6)

    def test_straight_and_curve_center_tangents_follow_front_back_axis(
        self,
    ) -> None:
        mask = np.zeros((50, 50), dtype=bool)
        mask[14:36, 8:42] = True
        for case_name, seed in (
            ("01_obj_linear_motion", 11),
            ("02_obj_curved_motion", 12),
        ):
            trajectory = make_independent_object_trajectory(
                case_name,
                seed,
                mask,
                forced_motion_case=case_name,
                front_back_axis_threshold=1.5,
            )
            validate_center_referenced_trajectory(trajectory)
            epsilon = 1.0e-4
            before = np.asarray(
                evaluate_object_trajectory_at_time(
                    trajectory, -epsilon
                )[:2]
            )
            after = np.asarray(
                evaluate_object_trajectory_at_time(
                    trajectory, epsilon
                )[:2]
            )
            self.assertLess(
                lateral_fraction(
                    after - before,
                    float(trajectory["front_back_axis_deg"]),
                ),
                1.0e-6,
            )


if __name__ == "__main__":
    unittest.main()
