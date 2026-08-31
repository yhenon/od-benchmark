from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from odbench_outer.hardware import HardwareProfileError, HardwareTarget
from odbench_outer.hardware_verification import (
    HardwareVerificationError,
    cubeprogrammer_download_image,
    diagnose_profile_failure,
    external_memory_images,
    parse_checker_profile,
    parse_generation_report,
    validate_external_memory_image_sizes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class HardwareTargetTests(unittest.TestCase):
    def test_nucleo_target_has_strict_and_buffered_thresholds(self) -> None:
        target = HardwareTarget.load(
            REPO_ROOT / "hardware" / "targets" / "nucleo-n657x0-q.json"
        )
        self.assertEqual(target.allowed_runtime_seconds(final=False), 0.005)
        self.assertEqual(target.allowed_runtime_seconds(final=True), 0.00525)
        self.assertEqual(
            target.allowed_runtime_seconds(final=False, runtime_seconds=0.05), 0.05
        )
        self.assertEqual(
            target.allowed_runtime_seconds(final=True, runtime_seconds=0.05), 0.0525
        )
        self.assertEqual(
            target.public_metadata(runtime_seconds=0.05)["runtime_seconds"], 0.05
        )
        self.assertEqual(target.max_external_image_bytes, 4 * 1024 * 1024)
        self.assertEqual(target.external_flash_timeout_seconds, 900)
        self.assertEqual(target.public_metadata()["quantization_owner"], "agent")

    def test_rejects_excessive_submission_tolerance(self) -> None:
        with self.assertRaisesRegex(HardwareProfileError, "between 0 and 0.5"):
            HardwareTarget.from_document(
                {
                    "schema_version": 1,
                    "id": "target",
                    "kind": "stm32n6",
                    "board": "NUCLEO-N657X0-Q",
                    "description": "Target.",
                    "model_format": "onnx",
                    "runtime_seconds": 0.1,
                    "submission_tolerance_fraction": 0.51,
                    "benchmark_samples": 10,
                    "max_external_image_bytes": 4 * 1024 * 1024,
                    "external_flash_timeout_seconds": 900,
                }
            )

    def test_parses_st_checker_profile(self) -> None:
        profile = parse_checker_profile(
            """
            n_nodes : 56
            duration : 4.209 ms by sample (4.206/4.221/0.004)
            237.60 inf/s
            """
        )
        self.assertEqual(profile["duration_seconds"], 0.004209)
        self.assertEqual(profile["duration_ms"]["max"], 4.221)
        self.assertEqual(profile["inferences_per_second"], 237.60)
        self.assertEqual(profile["nodes"], 56)

    def test_parses_accelerator_coverage_and_memory(self) -> None:
        report = parse_generation_report(
            """
Total: 677.267 kB -- weights: 437.267 kB activations: 240.000 kB
Total number of epochs                               56
>> pure software (SW) epochs                          2
>> hybrid epochs (using both software and hardware)   0
>> pure hardware (HW or EC) epochs                   54
"""
        )
        self.assertEqual(report["total_epochs"], 56)
        self.assertEqual(report["accelerated_epochs"], 54)
        self.assertEqual(report["accelerator_epoch_percent"], 96.429)
        self.assertEqual(report["memory"]["activations_bytes"], 240 * 1024)
        self.assertIn("2 pure software", report["warnings"][0])

    def test_warns_when_compiler_maps_no_accelerator_epochs(self) -> None:
        report = parse_generation_report(
            """
Total number of epochs 4
>> pure software (SW) epochs 4
>> hybrid epochs (using both software and hardware) 0
>> pure hardware (HW or EC) epochs 0
"""
        )
        self.assertEqual(report["accelerator_epoch_percent"], 0.0)
        self.assertIn("0%", report["warnings"][0])

    def test_classifies_timeout_after_inference_starts_as_model_execution(self) -> None:
        failure = diagnose_profile_failure(
            "TimeoutError: STM32 - read timeout 50001.0ms/50000ms",
            'Running c-model "network" with random data (b=1)..',
        )
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertEqual(failure["kind"], "model_execution_timeout")
        self.assertTrue(failure["board_connection_succeeded"])
        self.assertTrue(failure["inference_started"])
        self.assertFalse(failure["retry_unchanged_model"])
        self.assertIn("smaller input", failure["next_action"])

    def test_does_not_misclassify_timeout_before_inference_starts(self) -> None:
        failure = diagnose_profile_failure(
            "TimeoutError: STM32 - read timeout 50001.0ms/50000ms",
            "Creating AiRunner session",
        )
        self.assertIsNone(failure)

    def test_resolves_raw_external_memory_address_from_generated_c(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            network_c = output / "network.c"
            network_c.write_text(
                "/* file postfix=xSPI2 name=octoFlash offset=0x71000000 absolute_mode */"
            )
            raw = output / "network_atonbuf.xSPI2.raw"
            raw.write_bytes(b"weights")
            images = external_memory_images(network_c)
        self.assertEqual(images, [(raw, "0x71000000")])

    def test_copies_raw_image_to_cubeprogrammer_bin_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "network_atonbuf.xSPI2.raw"
            raw.write_bytes(b"weights")
            download = cubeprogrammer_download_image(raw, root)
            self.assertEqual(download.name, "network_atonbuf.xSPI2.raw.bin")
            self.assertEqual(download.read_bytes(), b"weights")

    def test_rejects_oversized_external_memory_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "network_atonbuf.xSPI2.raw"
            image.write_bytes(b"12345")
            with self.assertRaisesRegex(
                HardwareVerificationError, "external-memory image.*5 bytes.*4 bytes"
            ):
                validate_external_memory_image_sizes(
                    [(image, "0x71000000")], max_image_bytes=4
                )


if __name__ == "__main__":
    unittest.main()
