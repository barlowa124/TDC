# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import io
import contextlib

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestFuzzySearch(unittest.TestCase):

    NAMES = [
        "scperturb_gene_normanweissman2019",
        "scperturb_gene_replogleweissman2022_rpe1",
        "scperturb_gene_replogleweissman2022_k562_essential",
    ]

    def test_exact_match_no_warning(self):
        from tdc.utils.misc import fuzzy_search

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            s = fuzzy_search(
                "scperturb_gene_ReplogleWeissman2022_k562_essential",
                self.NAMES)
        self.assertEqual(
            s, "scperturb_gene_replogleweissman2022_k562_essential")
        self.assertNotIn("not an exact match", err.getvalue())

    def test_near_miss_resolves_with_warning(self):
        """Regression test for #256: a near-prefix dataset name (the
        never-uploaded K562_gwps) must not silently load the similar
        k562_essential dataset without a visible warning."""
        from tdc.utils.misc import fuzzy_search

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            s = fuzzy_search(
                "scperturb_gene_ReplogleWeissman2022_K562_gwps", self.NAMES)
        self.assertEqual(
            s, "scperturb_gene_replogleweissman2022_k562_essential")
        warning = err.getvalue()
        self.assertIn("not an exact match", warning)
        self.assertIn("k562_essential", warning)

    def test_unrelated_name_raises(self):
        from tdc.utils.misc import fuzzy_search

        with self.assertRaises(ValueError):
            fuzzy_search("completely_unrelated_name", self.NAMES)


if __name__ == "__main__":
    unittest.main()
