"""Tests for step timeline metadata generation."""

import unittest
import tempfile
import os
import shutil
from datetime import datetime, timedelta
import sys

# Ensure parent directory is on path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path in (current_dir, parent_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from step_aware_analyzer import build_step_dict
except Exception:  # pragma: no cover - skip if dependencies missing
    build_step_dict = None
from test_utils import MockLogEntry


class TimelineMetadataTest(unittest.TestCase):
    """Verify that build_step_dict returns start and end times."""

    def setUp(self):
        if build_step_dict is None:
            self.skipTest("step_aware_analyzer not available")
        self.temp_dir = tempfile.mkdtemp()
        self.feature_file = os.path.join(self.temp_dir, "test.feature")
        with open(self.feature_file, "w", encoding="utf-8") as f:
            f.write("Feature: Test\n")
            f.write("  Scenario: Example\n")
            f.write("    Given first step\n")
            f.write("    Then second step\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_step_dict_contains_timestamps(self):
        now = datetime.now()
        step_to_logs = {
            1: [MockLogEntry(timestamp=now)],
            2: [MockLogEntry(timestamp=now + timedelta(seconds=5))]
        }

        step_dict = build_step_dict(step_to_logs, self.feature_file)

        for step_num in step_to_logs:
            self.assertIn("start_time", step_dict[step_num])
            self.assertIn("end_time", step_dict[step_num])
            self.assertIsNotNone(step_dict[step_num]["start_time"])
            self.assertIsNotNone(step_dict[step_num]["end_time"])


if __name__ == "__main__":
    unittest.main()
