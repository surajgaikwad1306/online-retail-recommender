"""Download the UCI Online Retail dataset (CSV mirror) into data/retail.csv.

Usage: python download_data.py
"""
import csv
import http.client
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/"
       "master/data/retail-data/all/online-retail-dataset.csv")
OUT = Path(__file__).parent / "data" / "retail.csv"
MIN_SIZE = 40_000_000
REQUIRED_COLUMNS = {"InvoiceNo", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "UnitPrice", "CustomerID", "Country"}


def validate_dataset(path, min_size=MIN_SIZE):
    if path.stat().st_size < min_size:
        raise ValueError(f"file is smaller than expected ({path.stat().st_size} bytes)")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, strict=True)
        header = next(reader, None)
        if not header or not REQUIRED_COLUMNS.issubset(header):
            raise ValueError("CSV is missing required Online Retail columns")
        first_row = next(reader, None)
        if first_row is None or len(first_row) != len(header):
            raise ValueError("CSV has no valid data row")


def download_dataset(url=URL, out=OUT, min_size=MIN_SIZE):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            validate_dataset(out, min_size)
            print(f"Already present: {out}")
            return
        except (OSError, ValueError, csv.Error):
            print(f"Existing dataset is incomplete or invalid; downloading again.")

    temporary_path = None
    print(f"Downloading dataset (~45 MB) to {out} ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            expected_size = response.headers.get("Content-Length")
            with tempfile.NamedTemporaryFile(
                    dir=out.parent, prefix=f".{out.name}.", suffix=".part", delete=False) as f:
                temporary_path = Path(f.name)
                shutil.copyfileobj(response, f)
        actual_size = temporary_path.stat().st_size
        if expected_size is not None and actual_size != int(expected_size):
            raise ValueError(f"incomplete download ({actual_size} of {expected_size} bytes)")
        validate_dataset(temporary_path, min_size)
        os.replace(temporary_path, out)
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError,
            OSError, ValueError, csv.Error) as e:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        sys.exit(f"Download failed ({e}). Download 'Online Retail' manually from "
                 "https://archive.ics.uci.edu/dataset/352/online+retail, export it to CSV "
                 f"and save it as {out}")
    print(f"Done: {out.stat().st_size / 1e6:.1f} MB")


def main():
    download_dataset()


if __name__ == "__main__":
    main()
