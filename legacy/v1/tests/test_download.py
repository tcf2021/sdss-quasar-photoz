import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from photoz.data import download


class Response:
    def __init__(self, status, headers, content):
        self.status_code, self.headers, self.content = status, headers, content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, _):
        yield self.content


class DownloadContracts(unittest.TestCase):
    def exercise(self, response, expected, initial=b'abc'):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'file.fits'
            output.with_suffix('.fits.part').write_bytes(initial)
            def validator(path, official):
                if Path(path).read_bytes() != expected:
                    raise ValueError('Incomplete or corrupted body')
            with patch('photoz.data.OFFICIAL_BYTES',len(expected)), patch('photoz.data.validate_fits',validator), patch('requests.get',return_value=response) as get:
                download(output)
            self.assertEqual(get.call_args.kwargs['headers']['Range'], f'bytes={len(initial)}-')
            self.assertEqual(output.read_bytes(), expected)
            self.assertFalse(output.with_suffix('.fits.part').exists())

    def test_resume_valid_range(self):
        self.exercise(Response(206, {'Content-Range':'bytes 3-5/6'}, b'def'), b'abcdef')

    def test_range_ignored_restarts_instead_of_appending(self):
        self.exercise(Response(200, {}, b'abcdef'), b'abcdef')

    def test_bad_range_does_not_destroy_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'file.fits'
            partial=output.with_suffix('.fits.part')
            partial.write_bytes(b'abc')
            with patch('photoz.data.OFFICIAL_BYTES',6), patch('requests.get',return_value=Response(206,{'Content-Range':'bytes 1-5/6'},b'bcdef')):
                with self.assertRaises(RuntimeError):
                    download(output)
            self.assertEqual(partial.read_bytes(), b'abc')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
