from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import uav_sensor_degradation as sensor_model
from script_uav_global_blur_multisubframe_no_depth import (
    make_uav_trajectory,
    run_generation as run_global_generation,
    synthesize_blur_by_multisubframe_v2,
    write_image_uint8,
)
from script_uav_object_motion_blur_framework_A import (
    make_object_trajectory,
    synthesize_multi_object_motion_blur_A,
)
from script_uav_relative_motion_blur_multisubframe import (
    make_centered_platform_trajectory,
    synthesize_relative_motion_blur,
)


class SensorDegradationTests(unittest.TestCase):
    @staticmethod
    def highlight_only_parameters(
        sensor_seed: int = 100
    ) -> sensor_model.SensorSampleParameters:
        config = sensor_model.SensorDegradationConfig(
            saturation_enabled=True,
            saturation_threshold_low=0.90,
            saturation_threshold_high=0.99,
            saturation_gain_range=(0.5, 0.5),
            poisson_gaussian_enabled=False,
        )
        return sensor_model.sample_sensor_parameters(config, sensor_seed)

    def test_disabled_model_is_radiometrically_neutral(self) -> None:
        config = sensor_model.SensorDegradationConfig(
            saturation_enabled=False,
            poisson_gaussian_enabled=False,
        )
        parameters = sensor_model.sample_sensor_parameters(config, 123)
        exposure = np.linspace(0.0, 1.0, 60, dtype=np.float64).reshape(
            4, 5, 3
        )
        restored = sensor_model.restore_saturated_highlights_linear(
            exposure, parameters
        )
        output = sensor_model.finalize_sensor_variant_srgb(
            restored, parameters
        )
        self.assertTrue(np.array_equal(restored, exposure))
        self.assertTrue(
            np.allclose(
                output,
                sensor_model.linear_to_srgb(exposure),
                atol=0.0,
                rtol=0.0,
            )
        )

    def test_highlight_recovery_only_changes_bright_pixels(self) -> None:
        config = sensor_model.SensorDegradationConfig(
            saturation_enabled=True,
            saturation_threshold_low=0.90,
            saturation_threshold_high=0.99,
            saturation_gain_range=(0.5, 0.5),
            poisson_gaussian_enabled=False,
        )
        parameters = sensor_model.sample_sensor_parameters(config, 456)
        frame = np.asarray(
            [[[0.3, 0.2, 0.1], [1.0, 0.9, 0.8]]],
            dtype=np.float64,
        )
        restored = sensor_model.restore_saturated_highlights_linear(
            frame, parameters
        )
        self.assertTrue(np.array_equal(restored[0, 0], frame[0, 0]))
        self.assertTrue(np.all(restored[0, 1] > frame[0, 1]))
        self.assertGreater(float(restored[0, 1, 0]), 1.0)

    def test_poisson_gaussian_noise_is_reproducible(self) -> None:
        config = sensor_model.SensorDegradationConfig(
            saturation_enabled=False,
            poisson_gaussian_enabled=True,
            shot_noise_scale_range=(1.0e-4, 1.0e-4),
            read_noise_sigma_range=(1.0e-5, 1.0e-5),
        )
        parameters = sensor_model.sample_sensor_parameters(config, 789)
        exposure = np.full((32, 32, 3), 0.4, dtype=np.float64)
        first = sensor_model.add_poisson_gaussian_noise_after_exposure(
            exposure, parameters
        )
        second = sensor_model.add_poisson_gaussian_noise_after_exposure(
            exposure, parameters
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, exposure))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_global_sensor_branch_preserves_clean_output(self) -> None:
        image = np.zeros((24, 32, 3), dtype=np.float64)
        image[8:16, 12:20] = 1.0
        trajectory_clean = make_uav_trajectory(
            "01_heading_linear_uniform", 1
        )
        clean_result = synthesize_blur_by_multisubframe_v2(
            image,
            trajectory_clean,
            5,
            0,
            use_adaptive_sampling=False,
        )
        trajectory_sensor = make_uav_trajectory(
            "01_heading_linear_uniform", 1
        )
        sensor_result = synthesize_blur_by_multisubframe_v2(
            image,
            trajectory_sensor,
            5,
            0,
            use_adaptive_sampling=False,
            sensor_parameters=self.highlight_only_parameters(),
        )
        self.assertEqual(len(clean_result), 4)
        self.assertEqual(len(sensor_result), 5)
        self.assertTrue(np.array_equal(clean_result[0], sensor_result[0]))
        self.assertFalse(np.array_equal(sensor_result[0], sensor_result[1]))

    def test_object_and_relative_sensor_branches(self) -> None:
        background = np.zeros((32, 40, 3), dtype=np.float64)
        mask = np.zeros((32, 40), dtype=bool)
        mask[11:21, 15:25] = True
        alpha = mask.astype(np.float64)
        foreground = background.copy()
        foreground[mask] = 1.0
        reference = foreground.copy()
        trajectory = make_object_trajectory(
            "01_obj_linear_motion", 2, mask
        )
        trajectory["motion_strength_ratio"] = float(
            trajectory["total_distance"]
            / trajectory["object_scale"]
        )
        parameters = self.highlight_only_parameters(101)

        object_result = synthesize_multi_object_motion_blur_A(
            [foreground],
            background,
            [alpha],
            [trajectory],
            5,
            0,
            reference_image=reference,
            sensor_parameters=parameters,
        )
        self.assertEqual(len(object_result), 4)
        self.assertEqual(object_result[0].shape, reference.shape)
        self.assertEqual(object_result[1].shape, reference.shape)

        relative_result = synthesize_relative_motion_blur(
            reference,
            background,
            [foreground],
            [alpha],
            [trajectory],
            make_centered_platform_trajectory("00_static", 3),
            5,
            crop_safety_margin=0,
            sensor_parameters=parameters,
        )
        self.assertEqual(len(relative_result), 4)
        self.assertEqual(relative_result[0].shape, reference.shape)
        self.assertEqual(relative_result[1].shape, reference.shape)

    def test_global_generation_writes_sensor_pair_and_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            image = np.zeros((24, 32, 3), dtype=np.float64)
            image[8:16, 12:20] = 1.0
            write_image_uint8(input_dir / "sample.png", image)

            run_global_generation(
                input_dir,
                output_dir,
                (),
                ("01_heading_linear_uniform",),
                n_subframes=3,
                use_adaptive_sampling=False,
                crop_boundary=False,
                sensor_variant=True,
                saturation_highlights=True,
                poisson_gaussian_noise=True,
            )

            prefix = output_dir / "sample_01_heading_linear_uniform"
            expected_suffixes = (
                "_gtimg.png",
                "_blurryimg.png",
                "_blurryimg_sensor.png",
                "_sensor_metadata.npz",
                "_mf_para.mat",
            )
            for suffix in expected_suffixes:
                self.assertTrue(
                    prefix.with_name(prefix.name + suffix).is_file(),
                    suffix,
                )


if __name__ == "__main__":
    unittest.main()
