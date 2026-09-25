# -*- coding: utf-8 -*-
"""Robustness battery for the fork's own changes: fuzzy dataset-name
resolution (issue #256) and AnnData-aware splits (issue #267)."""
from __future__ import division, print_function

import contextlib
import io
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd


class TestFuzzySearchEdges(unittest.TestCase):
    NAMES = ["albendazole", "sars_cov2_3clpro_flexibility", "k562_essential"]

    def _run(self, name, names=None):
        from tdc.utils.misc import fuzzy_search

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = fuzzy_search(name, self.NAMES if names is None else names)
        return result, err.getvalue()

    def test_exact_match_silent(self):
        s, err = self._run("albendazole")
        self.assertEqual(s, "albendazole")
        self.assertNotIn("not an exact match", err)

    def test_tdc_prefix_stripped(self):
        # "tdc.albendazole" should resolve to "albendazole" without warning
        s, err = self._run("tdc.albendazole")
        self.assertEqual(s, "albendazole")

    def test_case_insensitive(self):
        s, err = self._run("ALBENDAZOLE")
        self.assertEqual(s, "albendazole")

    def test_near_miss_warns_and_resolves(self):
        s, err = self._run("sars_cov2_3clpro_flexibilit")
        self.assertEqual(s, "sars_cov2_3clpro_flexibility")
        self.assertIn("not an exact match", err)
        # warning must name the dataset it actually resolved to
        self.assertIn("sars_cov2_3clpro_flexibility", err)

    def test_total_miss_raises(self):
        with self.assertRaises(ValueError):
            self._run("qwerty_nonexistent_dataset_zzz")

    def test_empty_name_list_raises(self):
        with self.assertRaises(ValueError):
            self._run("albendazole", names=[])


class TestAnndataSplitEdges(unittest.TestCase):
    def _loader(self, n=60, obs=None):
        import anndata
        from tdc.multi_pred.anndata_dataset import DataLoader

        rng = np.random.RandomState(0)
        if obs is None:
            obs = pd.DataFrame(
                {"cell_type": [f"ct{i % 6}" for i in range(n)],
                 "pert": [f"p{i % 3}" for i in range(n)]},
                index=[f"cell{i}" for i in range(n)])
        ad = anndata.AnnData(
            X=np.arange(n * 5, dtype=np.float32).reshape(n, 5), obs=obs)
        dl = DataLoader.__new__(DataLoader)
        dl.adata = ad
        return dl

    def test_marker_column_does_not_leak(self):
        # __obs_pos is positional bookkeeping; it must not appear in
        # the returned splits' obs
        dl = self._loader()
        for name, s in dl.get_split_anndata(seed=0).items():
            self.assertNotIn("__obs_pos", s.obs.columns, name)

    def test_x_rows_track_obs_positions(self):
        # X must slice by the SAME positions as obs — a marker-column bug
        # would scramble rows silently
        dl = self._loader(60)
        splits = dl.get_split_anndata(seed=1)
        for s in splits.values():
            idx = [int(n.replace("cell", "")) for n in s.obs_names]
            self.assertTrue(np.allclose(
                s.X, np.asarray(dl.adata.X[idx]).reshape(len(idx), -1)))

    def test_seed_determinism(self):
        a = self._loader(60).get_split_anndata(seed=7)
        b = self._loader(60).get_split_anndata(seed=7)
        for k in a:
            self.assertEqual(list(a[k].obs_names), list(b[k].obs_names))

    def test_cold_split_disjoint_entities(self):
        dl = self._loader(60)
        splits = dl.get_split_anndata(
            method="cold_split", column_name="cell_type", seed=0)
        test_ct = set(splits["test"].obs["cell_type"])
        self.assertTrue(test_ct)
        for k in ("train", "valid"):
            self.assertEqual(len(test_ct & set(splits[k].obs["cell_type"])), 0)

    def test_cold_split_bad_column_raises(self):
        dl = self._loader()
        with self.assertRaises(AttributeError):
            dl.get_split_anndata(method="cold_split", column_name="nope")
        with self.assertRaises(AttributeError):
            dl.get_split_anndata(method="cold_split", column_name=None)

    def test_cold_split_column_list_accepted(self):
        # a single-element list must take the same path as a plain string
        dl = self._loader(60)
        splits = dl.get_split_anndata(
            method="cold_split", column_name=["cell_type"], seed=0)
        test_ct = set(splits["test"].obs["cell_type"])
        self.assertTrue(test_ct)
        for k in ("train", "valid"):
            self.assertEqual(
                len(test_ct & set(splits[k].obs["cell_type"])), 0)

    def test_cold_split_exhausted_entity_space_raises_cleanly(self):
        # two-column entity tuples here leave no disjoint valid/test
        # entities; the contract is a clear ValueError, not a crash
        dl = self._loader(60)
        with self.assertRaises(ValueError):
            dl.get_split_anndata(
                method="cold_split",
                column_name=["cell_type", "pert"], seed=0)

    def test_fractions_partition_all(self):
        dl = self._loader(50)
        splits = dl.get_split_anndata(seed=0, frac=[0.6, 0.2, 0.2])
        self.assertEqual(sum(s.n_obs for s in splits.values()), 50)

    def test_unknown_method_raises(self):
        dl = self._loader()
        with self.assertRaises(AttributeError):
            dl.get_split_anndata(method="scaffold")


if __name__ == "__main__":
    unittest.main()
