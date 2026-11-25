# Data

The dataset is **not committed** to this repo because it is about 45 MB. Download it with:

```bash
python download_data.py
```

This saves `data/retail.csv` (541,909 rows). The downloader validates the CSV structure and checks the transfer size when the server provides it. It downloads to a temporary file before replacing the destination, so an incomplete download won't overwrite an existing dataset. If the existing dataset is invalid, the script downloads a replacement.

**Source:** UCI Machine Learning Repository, *Online Retail*, https://archive.ics.uci.edu/dataset/352/online+retail
(Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33).
It is licensed under CC BY 4.0. The download script uses an identical CSV mirror from the Databricks *Spark: The Definitive Guide* repository.