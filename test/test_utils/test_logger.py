import unittest
from homura.utils import get_logger
import tempfile

class TestLogger(unittest.TestCase):
    def test_stdout(self):
        with self.assertRaises(ValueError):
            get_logger(stdout_filter_level=1)

        with self.assertRaises(ValueError):
            get_logger(stdout_filter_level="debbug")

    def test_logfile(self):
        with self.assertRaises(ValueError):
            with  tempfile.TemporaryDirectory() as fp:
                get_logger(log_file=f"{fp}/test.log", file_filter_level=1)

        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as fp:
                get_logger(log_file=f"{fp}/test.log", file_filter_level="ccritical")
