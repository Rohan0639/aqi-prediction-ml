"""
data_loader.py
==============
Loads the raw AQI CSV file and returns a validated DataFrame.
"""

import logging
import os

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def load_raw_data(cfg: dict) -> pd.DataFrame:
    """
    Load the raw CSV dataset.

    Parameters
    ----------
    cfg : dict
        Project configuration loaded from config.yaml.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all original columns.
    """
    raw_path = cfg["paths"]["raw_data"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path, parse_dates=[cfg["data"]["date_column"]])
    logger.info(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
    logger.info(f"Stations found: {df['Station'].nunique()} unique stations")
    logger.info(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()
    df = load_raw_data(cfg)
    print(df.head())
    print(df.dtypes)
