from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from script_uav_global_blur_multisubframe_no_depth import (
    COMPOSITE_MOTION_CONFIGS,
    COMPONENT_SEED_OFFSETS,
    GT_REFERENCE_TAU,
    evaluate_forward_homography_at_time,
    make_uav_trajectory,
    maximum_periodic_frequency,
    prepare_trajectory_for_image,
    run_generation,
    synthesize_blur_by_multisubframe_v2,
    write_image_uint8,
)


class GlobalCompositeMotionTests(unittest.TestCase):
    height = 72
    width = 96
    component_seed_base = 123_000

    def make_composite(self, name: str) -> dict:
        return make_uav_trajectory(
            name,
            999_000,
            component_seed_base=self.component_seed_base,
        )

    def test_all_composites_have_center_identity_and_finite_endpoints(
        self,
    ) -> None:
        for name in COMPOSITE_MOTION_CONFIGS:
            with self.subTest(name=name):
                trajectory = self.make_composite(name)
                prepare_trajectory_for_image(
                    trajectory, self.height, self.width
                )
                center = evaluate_forward_homography_at_time(
                    trajectory,
                    GT_REFERENCE_TAU,
                    self.height,
                    self.width,
                )
                self.assertTrue(
                    np.allclose(center, np.eye(3), atol=1e-9)
                )
                for tau in (0.0, 1.0):
                    homography = evaluate_forward_homography_at_time(
                        trajectory, tau, self.height, self.width
                    )
                    self.assertTrue(np.isfinite(homography).all())

    def test_component_matrix_order_is_explicit_composition(self) -> None:
        trajectory = self.make_composite(
            "10_heading_linear_variable_speed_plus_attitude_micro_vibration"
        )
        tau = 0.23
        expected = np.eye(3, dtype=np.float64)
        for component in trajectory["components"]:
            component_homography = evaluate_forward_homography_at_time(
                component, tau, self.height, self.width
            )
            expected = component_homography @ expected
        expected /= expected[2, 2]
        actual = evaluate_forward_homography_at_time(
            trajectory, tau, self.height, self.width
        )
        self.assertTrue(np.allclose(actual, expected, atol=1e-10))

    def test_shared_components_use_shared_seeds(self) -> None:
        second = make_uav_trajectory(
            "02_heading_linear_variable_speed",
            self.component_seed_base
            + COMPONENT_SEED_OFFSETS[
                "02_heading_linear_variable_speed"
            ],
        )
        ninth = self.make_composite(
            "09_heading_linear_variable_speed_plus_attitude"
        )
        tenth = self.make_composite(
            "10_heading_linear_variable_speed_plus_attitude_micro_vibration"
        )
        self.assertEqual(
            ninth["components"][0]["case_seed"], second["case_seed"]
        )
        self.assertEqual(
            ninth["components"][0]["case_seed"],
            tenth["components"][0]["case_seed"],
        )
        self.assertEqual(
            ninth["components"][1]["case_seed"],
            tenth["components"][1]["case_seed"],
        )

    def test_periodic_safeguard_propagates_from_components(self) -> None:
        nonperiodic = self.make_composite(
            "09_heading_linear_variable_speed_plus_attitude"
        )
        periodic = self.make_composite(
            "10_heading_linear_variable_speed_plus_attitude_micro_vibration"
        )
        self.assertEqual(maximum_periodic_frequency(nonperiodic), 0.0)
        self.assertEqual(maximum_periodic_frequency(periodic), 5.0)
        self.assertEqual(
            nonperiodic["sampling"]["type"], "screen_distance"
        )
        self.assertEqual(
            periodic["sampling"]["type"],
            "screen_distance_or_period",
        )

    def test_composite_synthesis_and_mat_metadata_output(self) -> None:
        image = np.zeros((24, 32, 3), dtype=np.float64)
        image[7:17, 11:21] = (0.9, 0.5, 0.2)
        trajectory = make_uav_trajectory(
            "08_heading_linear_uniform_plus_micro_vibration",
            999_000,
            component_seed_base=self.component_seed_base,
        )
        result = synthesize_blur_by_multisubframe_v2(
            image,
            trajectory,
            5,
            0,
            use_adaptive_sampling=False,
        )
        self.assertEqual(len(result), 4)
        blur, _, metadata, _ = result
        self.assertEqual(blur.shape, image.shape)
        self.assertEqual(metadata["gt_reference"], "center")
        self.assertEqual(metadata["component_count"], 2)
        self.assertTrue(np.isfinite(blur).all())

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            write_image_uint8(input_dir / "sample.png", image)
            run_generation(
                input_dir,
                output_dir,
                (),
                ("08_heading_linear_uniform_plus_micro_vibration",),
                n_subframes=3,
                use_adaptive_sampling=False,
                crop_boundary=False,
                sensor_variant=False,
            )
            prefix = (
                output_dir
                / "sample_08_heading_linear_uniform_plus_micro_vibration"
            )
            self.assertTrue(
                prefix.with_name(prefix.name + "_mf_para.mat").is_file()
            )


if __name__ == "__main__":
    unittest.main()
