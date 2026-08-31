from __future__ import annotations

import unittest

import numpy as np

from docker.odbench.quantization import (
    DatasetCalibrationReader,
    ModelInput,
    QuantizationError,
)


class QuantizationReaderTests(unittest.TestCase):
    def test_reader_uses_deterministic_training_samples_and_rewinds(self) -> None:
        dataset = [(index, index % 2) for index in range(10)]
        inputs = (ModelInput("input", (1, 1), np.dtype(np.float32)),)
        reader = DatasetCalibrationReader(
            dataset,
            lambda value: np.array([[value]], dtype=np.float32),
            inputs,
            samples=3,
        )
        self.assertEqual(reader.indices, (0, 4, 9))
        first = reader.get_next()
        self.assertEqual(first["input"].tolist(), [[0.0]])
        reader.rewind()
        self.assertEqual(reader.get_next()["input"].tolist(), [[0.0]])

    def test_reader_requires_exact_static_shape(self) -> None:
        reader = DatasetCalibrationReader(
            [(object(), 0)],
            lambda _: np.zeros((3, 32, 32), dtype=np.float32),
            (ModelInput("input", (1, 3, 32, 32), np.dtype(np.float32)),),
            samples=1,
        )
        with self.assertRaisesRegex(QuantizationError, "expected"):
            reader.get_next()

    def test_reader_accepts_named_multi_input_mapping(self) -> None:
        reader = DatasetCalibrationReader(
            [(object(), 0)],
            lambda _: {
                "image": np.ones((1, 3, 2, 2), dtype=np.float32),
                "scale": np.ones((1, 1), dtype=np.float32),
            },
            (
                ModelInput("image", (1, 3, 2, 2), np.dtype(np.float32)),
                ModelInput("scale", (1, 1), np.dtype(np.float32)),
            ),
            samples=1,
        )
        self.assertEqual(set(reader.get_next()), {"image", "scale"})


if __name__ == "__main__":
    unittest.main()
