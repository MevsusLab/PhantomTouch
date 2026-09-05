import json
import os
import tempfile
import unittest

from phantom_config import PhantomSettings, load_settings, save_settings


class PhantomConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertEqual(PhantomSettings().camera_index, 0)

    def test_round_trip_persistence(self):
        settings = PhantomSettings(cursor_responsiveness=0.8, camera_index=2,
                                   fist_alt_tab=False)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "settings.json")
            save_settings(settings, path)
            self.assertEqual(load_settings(path), settings)

    def test_unknown_keys_are_ignored_for_forward_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "settings.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"camera_index": 3, "future_option": 1}, handle)
            self.assertEqual(load_settings(path).camera_index, 3)

    def test_invalid_range_is_rejected(self):
        with self.assertRaises(ValueError):
            PhantomSettings(scroll_dead_zone=2.0)

    def test_invalid_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "settings.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("not json")
            with self.assertRaises(ValueError):
                load_settings(path)


if __name__ == "__main__":
    unittest.main()
