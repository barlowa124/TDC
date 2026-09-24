import numpy as np
import pandas as pd

from .multi_pred_dataset import DataLoader as DL
from ..dataset_configs.config_map import ConfigMap
from ..feature_generators.anndata_to_dataframe import AnnDataToDataFrame
from ..utils import create_fold, create_fold_setting_cold


class DataLoader(DL):

    def __init__(self,
                 name,
                 path,
                 print_stats=False,
                 dataset_names=None,
                 no_convert=True):
        super(DataLoader, self).__init__(name, path, print_stats, dataset_names)
        self.adata = self.df  # this is in AnnData format
        if no_convert:
            return
        cmap = ConfigMap()
        self.cmap = cmap
        self.config = cmap.get(name)
        if self.config is None:
            # default to converting adata to dataframe as is
            self.df = AnnDataToDataFrame.anndata_to_df(self.adata)
        else:
            cf = self.config()
            self.df = cf.processing_callback(self.adata)

    def get_anndata(self):
        """Return the raw AnnData object backing this loader.

        Exposes ``self.adata`` explicitly; see issue #267 — the attribute
        existed but had no documented getter.
        """
        return self.adata

    def get_split_anndata(
        self,
        method="random",
        seed=42,
        frac=[0.7, 0.1, 0.2],
        column_name=None,
    ):
        """Split the dataset into train/valid/test AnnData objects.

        Mirrors ``get_split`` but returns AnnData subsets aligned with
        ``self.adata.obs``. Supports ``random`` and ``cold_split`` (hold
        out entire obs entities, e.g. a cell type or perturbation).

        Args:
            method (str): 'random' or 'cold_split'.
            seed (int): random seed.
            frac (list): train/valid/test fractions.
            column_name: obs column(s) for cold_split entity holdout.

        Returns:
            dict: {'train', 'valid', 'test'} of AnnData objects.
        """
        # Marker column carries obs positions through the shared fold
        # helpers (they reset_index on output, dropping the index but
        # not columns).
        obs = self.adata.obs.copy()
        obs["__obs_pos"] = np.arange(self.adata.n_obs)

        if method == "random":
            folds = create_fold(obs, seed, frac)
        elif method == "cold_split":
            cols = [column_name] if isinstance(column_name, str) else column_name
            if cols is None or not all(c in self.adata.obs.columns for c in cols):
                raise AttributeError(
                    "For cold_split, please provide one or multiple column "
                    "names that are contained in adata.obs."
                )
            folds = create_fold_setting_cold(obs, seed, frac, cols)
        else:
            raise AttributeError(
                "Please select a splitting strategy from random or cold_split."
            )
        return {
            k: self.adata[v["__obs_pos"].to_numpy()].copy()
            for k, v in folds.items()
        }
