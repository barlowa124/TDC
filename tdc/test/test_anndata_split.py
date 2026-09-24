import unittest

import numpy as np
import pandas as pd


class TestAnndataSplit(unittest.TestCase):
    """Issue #267: getter + anndata-aware splits on the AnnData loader."""

    def _loader(self, n=50):
        import anndata

        from tdc.multi_pred.anndata_dataset import DataLoader

        rng = np.random.RandomState(0)
        ad = anndata.AnnData(
            X=rng.rand(n, 10).astype(np.float32),
            obs=pd.DataFrame(
                {
                    "cell_type": [f"ct{i % 10}" for i in range(n)],
                    "perturbation": [f"pert{i % 4}" for i in range(n)],
                },
                index=[f"cell{i}" for i in range(n)],
            ),
        )
        dl = DataLoader.__new__(DataLoader)
        dl.adata = ad
        return dl

    def test_get_anndata(self):
        dl = self._loader()
        self.assertIs(dl.get_anndata(), dl.adata)

    def test_random_split_partitions_obs(self):
        dl = self._loader(50)
        splits = dl.get_split_anndata(seed=0)
        self.assertEqual(set(splits), {"train", "valid", "test"})
        total = sum(s.n_obs for s in splits.values())
        self.assertEqual(total, 50)
        names = [set(s.obs_names) for s in splits.values()]
        self.assertEqual(len(names[0] & names[1]), 0)
        self.assertEqual(len(names[0] & names[2]), 0)
        self.assertEqual(len(names[1] & names[2]), 0)

    def test_cold_split_holds_out_entity(self):
        dl = self._loader(50)
        splits = dl.get_split_anndata(
            method="cold_split", column_name="cell_type", seed=0
        )
        test_ct = set(splits["test"].obs["cell_type"].unique())
        train_ct = set(splits["train"].obs["cell_type"].unique())
        self.assertTrue(test_ct)
        self.assertEqual(len(test_ct & train_ct), 0)

    def test_invalid_method_raises(self):
        dl = self._loader()
        with self.assertRaises(AttributeError):
            dl.get_split_anndata(method="not_a_method")


if __name__ == "__main__":
    unittest.main()
