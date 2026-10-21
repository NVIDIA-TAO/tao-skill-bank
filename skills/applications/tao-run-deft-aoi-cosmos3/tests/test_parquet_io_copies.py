# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""parquet_io.py is vendored into both script directories that read/write mining
parquet (skills import siblings by bare module name and are packaged per skill).
The two copies must stay byte-identical."""
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve()
DEFT_COPY = HERE.parents[1] / "scripts" / "parquet_io.py"
MINING_COPY = HERE.parents[3] / "data" / "tao-mine-aoi-images" / "scripts" / "parquet_io.py"


class ParquetIoCopiesTest(unittest.TestCase):
    def test_copies_are_byte_identical(self) -> None:
        self.assertTrue(DEFT_COPY.is_file() and MINING_COPY.is_file())
        self.assertEqual(DEFT_COPY.read_bytes(), MINING_COPY.read_bytes(),
                         "parquet_io.py drifted between the DEFT and mining skills; copy one over the other")


if __name__ == "__main__":
    unittest.main()
