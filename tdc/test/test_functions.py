# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
import shutil

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestFunctions(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_Evaluator(self):
        from tdc import Evaluator

        evaluator = Evaluator(name="ROC-AUC")
        print(evaluator([0, 1], [0.5, 0.6]))

    def test_binarize(self):
        from tdc.single_pred import TestSinglePred

        data = TestSinglePred(name="Test_Single_Pred")
        data.binarize(threshold=-5, order="descending")

    def test_convert_to_log(self):
        from tdc.single_pred import TestSinglePred

        data = TestSinglePred(name="Test_Single_Pred")
        data.convert_to_log()

    def test_print_stats(self):
        from tdc.single_pred import TestSinglePred

        data = TestSinglePred(name="Test_Single_Pred")
        data.print_stats()

    def tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        if os.path.exists(os.path.join(os.getcwd(), "oracle")):
            shutil.rmtree(os.path.join(os.getcwd(), "oracle"))


class TestFcdDistanceBackends(unittest.TestCase):
    """fcd_distance returns the same FCD score through either backend (#402)."""

    RAW_FCD = 12.5
    GENERATED = ["CCO", "c1ccccc1"]
    TRAINING = ["CCN", "c1ccncc1"]

    def test_both_backends_give_same_score(self):
        import sys
        import types

        import numpy as np

        from tdc.chem_utils import evaluator

        expected = float(np.exp(-0.2 * self.RAW_FCD))

        tf_fake = types.ModuleType("fcd")
        tf_fake.canonical_smiles = lambda mols: list(mols)
        tf_fake.get_predictions = lambda model, x: np.zeros((len(x), 4))
        tf_fake.calculate_frechet_distance = lambda **kwargs: self.RAW_FCD

        class _FakeFCD:
            def __init__(self, device=None, n_jobs=None):
                pass

            def __call__(self, generated, training):
                return TestFcdDistanceBackends.RAW_FCD

        torch_fake = types.ModuleType("fcd_torch")
        torch_fake.FCD = _FakeFCD

        names = ("tensorflow", "fcd", "torch", "fcd_torch")
        saved = {name: sys.modules.get(name) for name in names}
        saved_chemnet = getattr(evaluator, "chemnet", None)
        saved_fcd = getattr(evaluator, "fcd", None)
        had_chemnet = "chemnet" in vars(evaluator)
        had_fcd = "fcd" in vars(evaluator)
        evaluator.chemnet = object()
        try:
            sys.modules["tensorflow"] = types.ModuleType("tensorflow")
            sys.modules["fcd"] = tf_fake
            tf_score = evaluator.fcd_distance(self.GENERATED, self.TRAINING)
            self.assertAlmostEqual(tf_score, expected)

            sys.modules["tensorflow"] = None
            sys.modules["torch"] = types.ModuleType("torch")
            sys.modules["fcd_torch"] = torch_fake
            torch_score = evaluator.fcd_distance(self.GENERATED, self.TRAINING)
            self.assertAlmostEqual(torch_score, expected)
            self.assertAlmostEqual(torch_score, tf_score)
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod
            if had_chemnet:
                evaluator.chemnet = saved_chemnet
            else:
                vars(evaluator).pop("chemnet", None)
            if had_fcd:
                evaluator.fcd = saved_fcd
            else:
                vars(evaluator).pop("fcd", None)


if __name__ == "__main__":
    unittest.main()
