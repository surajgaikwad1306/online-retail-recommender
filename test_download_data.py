import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from download_data import download_dataset, validate_dataset


CSV_DATA = (
    b"InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
    b"536365,85123A,WHITE HANGING HEART,6,12/1/2010 8:26,2.55,17850,United Kingdom\n"
)


class FakeResponse(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))}


class DownloadDataTests(unittest.TestCase):
    def test_validate_dataset_rejects_wrong_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retail.csv"
            path.write_text("not,the,retail,dataset\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required"):
                validate_dataset(path, min_size=1)

    @patch("download_data.urllib.request.urlopen")
    def test_download_replaces_invalid_file_after_validation(self, urlopen):
        urlopen.return_value = FakeResponse(CSV_DATA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retail.csv"
            path.write_text("invalid", encoding="utf-8")

            download_dataset("https://example.test/retail.csv", path, min_size=1)

            self.assertEqual(path.read_bytes(), CSV_DATA)
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    @patch("download_data.urllib.request.urlopen", side_effect=URLError("offline"))
    def test_failed_download_preserves_existing_file(self, _urlopen):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retail.csv"
            original = b"existing file"
            path.write_bytes(original)

            with self.assertRaises(SystemExit):
                download_dataset("https://example.test/retail.csv", path, min_size=1)

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    @patch("download_data.urllib.request.urlopen")
    def test_download_rejects_truncated_response(self, urlopen):
        urlopen.return_value = FakeResponse(CSV_DATA)
        urlopen.return_value.headers["Content-Length"] = str(len(CSV_DATA) + 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retail.csv"

            with self.assertRaises(SystemExit):
                download_dataset("https://example.test/retail.csv", path, min_size=1)

            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
