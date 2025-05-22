import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import json

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from json_utils import serialize_to_json_file


class TestSerializeToJsonFile(unittest.TestCase):
    def test_serialize_to_current_directory(self):
        """serialize_to_json_file should not fail when no directory is provided."""
        data = {"foo": "bar"}
        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                with patch('json_utils.serialize_with_component_awareness') as mock_fn:
                    mock_fn.side_effect = lambda d, f, **kw: json.dump(d, f, indent=kw.get('indent', 2))
                    try:
                        serialize_to_json_file(data, "output.json")
                    except FileNotFoundError as e:
                        self.fail(f"serialize_to_json_file raised FileNotFoundError: {e}")
                self.assertTrue(os.path.exists(os.path.join(temp_dir, "output.json")))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
