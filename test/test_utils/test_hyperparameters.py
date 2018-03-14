from pathlib import Path
import unittest

from homura.utils.hyperparameters import HyperParameter


class TestHyperParameters(unittest.TestCase):
    files = Path(__file__).parent / "files"

    def test_load(self):
        pa = HyperParameter(self.files / "test.json",
                            self.files / "test.yaml", lr=0.1)
        self.assertEqual(pa.epochs, 500)
        self.assertEqual(pa.image_size.cifar10, 32)
        self.assertEqual(pa.lr, 0.1)

        with self.assertRaises(AttributeError):
            pa.no_such_method

    def test_register_hp(self):
        pa = HyperParameter()
        pa.register_hp(file_name=self.files / "test.yaml")
        self.assertEqual(pa.epochs, 500)

        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--lr", type=float, default=0.1)
        args = p.parse_args()
        pa.register_hp(args=args)
        self.assertEqual(pa.lr, 0.1)

        with self.assertRaises(Exception):
            pa.register_hp(no_such_method=None)

    def test_unknown_filetype(self):
        pa = HyperParameter()
        with self.assertRaises(Exception):
            pa.register_hp(self.files / "test.dat")

    def test_check_duplicated(self):
        with self.assertRaises(Exception):
            HyperParameter(self.files / "test.yaml", epochs=300)
