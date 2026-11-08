"""Small tests for scientific validity and artifact round trips, not performance claims."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from photoz.common import metrics, read_split, verify_data_files
from photoz.data import photometry, prepare, unique_objects
from photoz.models import train_model, predict_model


class DataContracts(unittest.TestCase):
    def test_metrics_known_values(self):
        y = np.array([1., 3.])
        pred = np.array([1.2, 2.2])
        m = metrics(y, pred)
        self.assertAlmostEqual(m['mae'], .5)
        self.assertAlmostEqual(m['rmse'], (.34)**.5)
        self.assertAlmostEqual(m['bias_norm'], -.05)
        self.assertAlmostEqual(m['outlier_rate_015'], .5)
        self.assertAlmostEqual(m['nmad'], 1.4826*.15)

    def test_photometry_and_no_label_feature(self):
        mag = np.array([[20., 19., 18., 17., 16.], [0., 99., np.nan, 20., 21.]])
        err = np.ones((2,5)) * .1
        err[0,0] = -9999
        ext = np.ones((2,5)) * .2
        feat, keep = photometry(mag, err, ext)
        self.assertEqual(feat['rich'].shape, (2,14))
        np.testing.assert_allclose(feat['rich'][0,5:9], np.ones(4), atol=1e-6)
        self.assertTrue(np.isnan(feat['rich'][0,9]))
        self.assertEqual(keep.tolist(), [True, False])

    def test_connected_duplicates_and_invalid_ids(self):
        names = np.array(['A','B','A','C','D'])
        things = np.array([1,2,2,-1,-1])
        np.testing.assert_array_equal(unique_objects(names, things), [0,3,4])

    def test_fits_split_and_saved_model(self):
        rng = np.random.default_rng(7)
        n = 220
        mag = rng.normal(20,1,(n,5)).astype('f4')
        err = np.full((n,5),.1,dtype='f4')
        y = (1.5 + .1*(mag[:,0]-20)).astype('f4')
        names = np.array([f'TEST{i:05}' for i in range(n)])
        names[1] = names[0]
        columns = [fits.Column(name='SDSS_NAME',format='18A',array=names),
                   fits.Column(name='THING_ID',format='K',array=np.arange(n)+1),
                   fits.Column(name='IS_QSO_FINAL',format='I',array=np.ones(n)),
                   fits.Column(name='Z',format='D',array=y)]
        for name, value in [('PSFMAG',mag),('PSFMAGERR',err),('EXTINCTION',np.zeros((n,5)))]:
            columns.append(fits.Column(name=name,format='5E',array=value))
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            fits.BinTableHDU.from_columns(columns).writeto(temp/'input.fits')
            manifest = prepare(temp/'input.fits', temp/'data', 200, technical=True)
            repeat = prepare(temp/'input.fits', temp/'data_repeat', 200, technical=True)
            self.assertEqual(manifest['dataset_id'], repeat['dataset_id'])
            verify_data_files(temp/'data', ['train','val','test'])
            splits = {s: read_split(temp/'data',s) for s in ['train','val','test']}
            for a,b in [('train','val'),('train','test'),('val','test')]:
                self.assertFalse(set(splits[a]['names']) & set(splits[b]['names']))
            self.assertEqual(manifest['n_after_dedup'], n-1)
            train, val = splits['train'], splits['val']
            # Validation outliers must not change the training-fitted imputation/scaling.
            train['rich'][0,0] = np.nan
            val['rich'][:,0] = 1e6
            params = {'hidden':[8],'dropout':0.,'lr':.001,'weight_decay':.0001,'batch_size':32,'epochs':2,'patience':2}
            train_model(temp/'model','mlp','rich',params,train,val,42,'cpu',threads=1)
            import joblib
            prep = joblib.load(temp/'model'/'preprocessor.joblib')
            self.assertAlmostEqual(prep.named_steps['imputer'].statistics_[0], np.nanmedian(train['rich'][:,0]), places=4)
            self.assertLess(abs(prep.named_steps['scaler'].mean_[0]), 100)
            with np.load(temp/'model'/'validation_predictions.npz') as old:
                np.testing.assert_allclose(predict_model(temp/'model',val), old['pred'])
            with self.assertRaises(FileExistsError):
                prepare(temp/'input.fits', temp/'data',200,technical=True)


if __name__ == '__main__':
    unittest.main()
