"""Download the UCI Online Retail dataset (CSV mirror) into data/retail.csv.

Usage: python download_data.py
"""
import sys
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/"
       "master/data/retail-data/all/online-retail-dataset.csv")
OUT = Path(__file__).parent / "data" / "retail.csv"


def main():
    if OUT.exists() and OUT.stat().st_size > 40_000_000:
        print(f"Already present: {OUT}")
        return
    OUT.parent.mkdir(exist_ok=True)
    print(f"Downloading dataset (~45 MB) to {OUT} ...")
    try:
        urllib.request.urlretrieve(URL, OUT)
    except Exception as e:
        sys.exit(f"Download failed ({e}). Download 'Online Retail' manually from "
                 "https://archive.ics.uci.edu/dataset/352/online+retail, export it to CSV "
                 "and save it as data/retail.csv")
    print(f"Done: {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
