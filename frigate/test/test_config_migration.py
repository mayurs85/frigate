"""Tests for config file migration functions."""

import unittest

from frigate.util.config import migrate_019_0


class TestMigrate019(unittest.TestCase):
    def test_migrates_model_to_named_models(self):
        config = {
            "version": "0.18-0",
            "mqtt": {"host": "mqtt"},
            "model": {"path": "/config/model.tflite", "width": 320, "height": 320},
        }
        new_config = migrate_019_0(config)
        assert "model" not in new_config
        assert new_config["models"] == {
            "default": {"path": "/config/model.tflite", "width": 320, "height": 320}
        }
        assert new_config["version"] == "0.19-0"

    def test_no_model_defined(self):
        config = {"version": "0.18-0", "mqtt": {"host": "mqtt"}}
        new_config = migrate_019_0(config)
        assert "model" not in new_config
        assert "models" not in new_config
        assert new_config["version"] == "0.19-0"

    def test_existing_models_not_overwritten(self):
        config = {
            "version": "0.18-0",
            "model": {"width": 320},
            "models": {"custom": {"width": 640}},
        }
        new_config = migrate_019_0(config)
        assert "model" not in new_config
        assert new_config["models"] == {"custom": {"width": 640}}
        assert new_config["version"] == "0.19-0"


if __name__ == "__main__":
    unittest.main(verbosity=2)
