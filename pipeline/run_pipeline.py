"""
run_pipeline.py
===============
One-command pipeline that runs the full ML workflow:
  1. Load raw data
  2. Clean & preprocess
  3. Engineer features
  4. Save processed data
  5. Train models  (80 / 20 chronological split)
  6. Evaluate models + generate plots
  7. Save best model marker

Usage
-----
    python pipeline/run_pipeline.py
    python pipeline/run_pipeline.py --config config/config.yaml
"""

import argparse
import logging
import os
import sys
import time

# Allow imports from the project root regardless of working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import yaml

from src.data_loader         import load_config, load_raw_data
from src.preprocessing       import clean_data
from src.feature_engineering import engineer_features, get_feature_columns
from src.train_model         import run_training
from src.evaluate_model      import evaluate_models


def setup_logging(cfg: dict):
    level = getattr(logging, cfg["logging"]["level"], logging.INFO)
    logging.basicConfig(
        level=level,
        format=cfg["logging"]["format"],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline_run.log", mode="w"),
        ],
    )


def main(config_path: str):
    # ── Load config ────────────────────────────────────────────
    cfg = load_config(config_path)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    start = time.time()
    logger.info("╔" + "═" * 53 + "╗")
    logger.info("║   AQI PREDICTION PIPELINE – STARTING            ║")
    logger.info("╚" + "═" * 53 + "╝")

    # ── Step 1: Load ───────────────────────────────────────────
    logger.info("\n[1/6] Loading raw data…")
    df_raw = load_raw_data(cfg)

    # ── Step 2: Clean ──────────────────────────────────────────
    logger.info("\n[2/6] Cleaning data…")
    df_clean = clean_data(df_raw)

    # ── Step 3: Feature engineering ────────────────────────────
    logger.info("\n[3/6] Engineering features…")
    df_feat = engineer_features(df_clean, cfg)
    feature_cols = get_feature_columns(cfg)

    # Verify all feature cols exist
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        logger.error(f"Missing feature columns: {missing}")
        sys.exit(1)

    # ── Step 4: Save processed data ────────────────────────────
    logger.info("\n[4/6] Saving processed dataset…")
    proc_path = cfg["paths"]["processed_data"]
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    save_cols = feature_cols + ["Date", "Station", "AQI", "Next_day_AQI",
                                "AQI_Category", "Station_code"]
    save_cols = [c for c in save_cols if c in df_feat.columns]
    df_feat[save_cols].to_csv(proc_path, index=False)
    logger.info(f"  Saved {len(df_feat):,} rows → '{proc_path}'")

    # ── Step 5: Train models ───────────────────────────────────
    logger.info("\n[5/6] Training models…")
    artifacts = run_training(df_feat, feature_cols, cfg)

    # ── Step 6: Evaluate & plot ────────────────────────────────
    logger.info("\n[6/6] Evaluating models…")
    best_model = evaluate_models(artifacts, df_feat, feature_cols, cfg)

    # Save best model name to disk for predict.py to read
    import json
    models_dir = cfg["paths"]["models_dir"]
    best_marker = os.path.join(models_dir, "best_model.json")
    with open(best_marker, "w") as fh:
        json.dump({"best_model": best_model}, fh)
    logger.info(f"  Best model: {best_model}  (saved to {best_marker})")

    elapsed = time.time() - start
    logger.info("\n╔" + "═" * 53 + "╗")
    logger.info(f"║   PIPELINE COMPLETE  ({elapsed:.1f}s)              ║")
    logger.info("╚" + "═" * 53 + "╝")
    logger.info("\n  Run next-day prediction:")
    logger.info("    python src/predict.py")
    logger.info("    python src/predict.py --offline   (no API call)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Prediction Pipeline")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config YAML (default: config/config.yaml)")
    args = parser.parse_args()
    main(args.config)
