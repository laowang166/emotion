import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from classroom_emotion.config import AppConfig, load_config


class ConfigLoaderTests(unittest.TestCase):
    def test_defaults_when_no_path(self) -> None:
        config = load_config(None)

        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.detection.model, "scrfd_2_5g")
        self.assertEqual(config.emotion.batch_buckets, (1, 8, 16, 32, 64))

    def test_reads_known_fields_and_ignores_unknown(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    detection:
                      score_threshold: 0.6
                      unknown_field: 1
                    emotion:
                      batch_buckets: [1, 4, 16]
                    temporal:
                      window_seconds: 5.0
                    """
                ),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertAlmostEqual(config.detection.score_threshold, 0.6)
        self.assertEqual(config.emotion.batch_buckets, (1, 4, 16))
        self.assertAlmostEqual(config.temporal.window_seconds, 5.0)


if __name__ == "__main__":
    unittest.main()
