import tempfile
import unittest
from pathlib import Path

import numpy as np

from photoz2.common import features, metrics, read_sealed, save_json, seal
from photoz2.data import clean_photometry, components, select_holdout
from photoz2.experiment import ensure_trainable, make_folds, make_shift_indices
from photoz2.models import make_preprocessor


class ScientificContracts(unittest.TestCase):
    def test_features_never_use_redshift_or_quality_metadata(self):
        mag = np.array([[20, 19, 18, 17, 16], [21, 20, 20, 19, 18]], dtype=np.float32)
        data = {"mag": mag, "err": np.full_like(mag, .1), "y": np.array([1, 2])}
        altered = {**data, "y": np.array([200, -999]), "z_warning": np.array([42, 42])}
        for name, dim in [("mag", 5), ("colors", 4), ("colors_i", 5), ("colors_i_err", 10)]:
            np.testing.assert_array_equal(features(data, name), features(altered, name))
            self.assertEqual(features(data, name).shape[1], dim)

    def test_color_plus_i_is_invertible_for_complete_photometry(self):
        mag = np.random.default_rng(7).normal(size=(30, 5)).astype(np.float32)
        values = features({"mag": mag, "err": np.ones_like(mag)}, "colors_i")
        i = values[:, 4]; r = values[:, 2]+i; g = values[:, 1]+r
        u = values[:, 0]+g; z = i-values[:, 3]
        np.testing.assert_allclose(np.column_stack([u, g, r, i, z]), mag, atol=5e-7)

    def test_invalid_photometry_is_missing_and_requires_three_bands(self):
        mag = np.array([[20, 20, 20, -999, 0], [20, 20, -999, -999, -999]], dtype=float)
        err = np.full_like(mag, .1); err[0, 0] = -1
        corrected, cleaned_error, valid = clean_photometry(mag, err, np.zeros_like(mag))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertTrue(np.isnan(cleaned_error[0, 0]))
        self.assertTrue(np.isnan(corrected[0, -1]))

    def test_identity_bridge_through_ineligible_row_is_excluded(self):
        # Row 1 links old row 0 to candidate row 2 through different identity keys.
        comp = components(np.array(["A", "B", "B", "C", "D"]), np.array([1, 1, 2, 3, 4]))
        chosen, available = select_holdout(np.array([2, 3, 4]), comp, np.array([0]), 2, 9)
        self.assertEqual(available, 2)
        self.assertEqual(set(chosen), {3, 4})

    def test_outer_and_inner_folds_are_disjoint_and_cover_population(self):
        folds = make_folds(1200, 300, 42, .15)
        held_out = []
        for fold in range(3):
            train, val = folds[f"f{fold}_train"], folds[f"f{fold}_eval"]
            it, iv = folds[f"f{fold}_inner_train"], folds[f"f{fold}_inner_val"]
            self.assertFalse(np.intersect1d(train, val).size)
            self.assertFalse(np.intersect1d(it, iv).size)
            np.testing.assert_array_equal(np.sort(np.concatenate([it, iv])), np.sort(train))
            held_out.extend(val)
        np.testing.assert_array_equal(np.sort(held_out), np.sort(folds["cv_population"]))

    def test_imputer_and_scaler_fit_training_only(self):
        prep = make_preprocessor("knn")
        train = np.array([[1., 10.], [3., np.nan], [5., 30.]])
        prep.fit(train)
        before = prep.named_steps["scaler"].mean_.copy()
        prep.transform(np.array([[1e9, -1e9]]))
        np.testing.assert_allclose(prep.named_steps["imputer"].statistics_, [3., 20.])
        np.testing.assert_array_equal(prep.named_steps["scaler"].mean_, before)

    def test_shift_equal_size_and_brightness_constraint(self):
        mag = np.ones((200, 5))*20
        mag[:, 3] = np.linspace(17, 23, 200)
        mag[0, 3] = np.nan
        arms = make_shift_indices({"mag": mag}, 20, 60, 42)
        self.assertEqual(len(arms["bright"]), len(arms["mixed"]))
        self.assertEqual(len(arms["bright"]), 60)
        self.assertTrue(np.all(mag[arms["bright"], 3] < 20))
        self.assertTrue(np.isfinite(mag[arms["mixed"], 3]).all())

    def test_metrics_known_values_and_nonfinite_rejection(self):
        m = metrics([1., 3.], [1.2, 2.2])
        self.assertAlmostEqual(m["mae"], .5)
        self.assertAlmostEqual(m["mae_norm"], .15)
        self.assertEqual(m["outlier_rate_015"], .5)
        with self.assertRaises(ValueError):
            metrics([1], [np.nan])

    def test_test_marker_blocks_new_training_and_seals_detect_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            data, run = Path(temp)/"data", Path(temp)/"run"
            data.mkdir(); run.mkdir()
            ensure_trainable(data, run)
            save_json(data/"TEST_OPENED.json", {"freeze_id": "a"})
            with self.assertRaises(RuntimeError):
                ensure_trainable(data, run)
            path = run/"sealed.json"
            save_json(path, seal({"value": 1}))
            self.assertEqual(read_sealed(path)["value"], 1)
            changed = seal({"value": 1}); changed["value"] = 2
            save_json(path, changed)
            with self.assertRaises(ValueError):
                read_sealed(path)


if __name__ == "__main__":
    unittest.main()
