"""
train_autogluon5.py  (v5 — v4 + Leak Residual)
===============================================
Same training setup as v4 (best_quality, bagging 5, stacking L1)
but reads from data5/ which includes the residual leak features.
"""

import os
import gc
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.common.features.feature_metadata import FeatureMetadata


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "data5"
MODELS_DIR = "models5/autogluon_v5"
TRAIN_PATH = os.path.join(DATA_DIR, "train_ag.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "test_ag.parquet")

TARGET_COL  = "target"
EVAL_METRIC = "mae"
TIME_LIMIT  = 14400 * 2   # 8h default
SEED        = 42
VAL_FRAC    = 0.10

NN_TORCH_CONFIGS = [
    {
        "num_epochs":    50,
        "learning_rate": 1e-3,
        "dropout_prob":  0.1,
        "weight_decay":  1e-6,
        "batch_size":    512,
        "ag_args": {"name_suffix": "Wide"},
    },
    {
        "num_epochs":    60,
        "learning_rate": 3e-4,
        "dropout_prob":  0.3,
        "weight_decay":  1e-4,
        "batch_size":    256,
        "ag_args": {"name_suffix": "Regularized"},
    },
    {
        "num_epochs":    80,
        "learning_rate": 1e-4,
        "dropout_prob":  0.2,
        "weight_decay":  1e-5,
        "batch_size":    1024,
        "ag_args": {"name_suffix": "Slow"},
    },
]

HYPERPARAMETERS = {
    "GBM":      [{}],
    "XGB":      [{}],
    "CAT":      [{}],
    "NN_TORCH": NN_TORCH_CONFIGS,
    "FASTAI":   [{}],
}


# ──────────────────────────────────────────────────────────────────────────────
# SANITIZE
# ──────────────────────────────────────────────────────────────────────────────

def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("Int"):
            df[c] = df[c].astype("float32")
    for c in ["user_id", "business_id"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("category")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def split_train_val(df: pd.DataFrame):
    train_df, val_df = train_test_split(
        df, test_size=VAL_FRAC, random_state=SEED,
        stratify=df[TARGET_COL].astype(int),
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"\n  Split: {len(train_df):,} train / {len(val_df):,} val")
    for name, part in [("Train", train_df), ("Val", val_df)]:
        dist = part[TARGET_COL].value_counts(normalize=True).sort_index()
        print(f"    {name}: " + "  ".join(f"{k}*={v:.1%}" for k, v in dist.items()))
    return train_df, val_df


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────────────────────

def train(time_limit: int = TIME_LIMIT, presets: str = "best_quality"):
    print("\n  Loading data5/train_ag.parquet...")
    df = pd.read_parquet(TRAIN_PATH)
    print(f"    Shape: {df.shape}")

    # Show new leak columns
    leak_cols = [c for c in df.columns if "leak" in c or "non_train" in c]
    print(f"    Leak features: {leak_cols}")
    print(f"    avg_non_train_biz — mean={df['avg_non_train_biz'].mean():.3f}  "
          f"std={df['avg_non_train_biz'].std():.3f}")

    df.drop(columns=["review_id"], inplace=True, errors="ignore")
    df = sanitize(df)

    train_df, val_df = split_train_val(df)
    del df; gc.collect()

    feature_metadata = FeatureMetadata.from_df(train_df)
    os.makedirs(MODELS_DIR, exist_ok=True)

    predictor = TabularPredictor(
        label        = TARGET_COL,
        eval_metric  = EVAL_METRIC,
        path         = MODELS_DIR,
        problem_type = "regression",
        verbosity    = 2,
    )

    print(f"\n  Training (time_limit={time_limit}s = {time_limit/3600:.1f}h)...")
    t0 = time.time()

    predictor.fit(
        train_data           = TabularDataset(train_df),
        tuning_data          = TabularDataset(val_df),
        hyperparameters      = HYPERPARAMETERS,
        use_bag_holdout      = True,
        dynamic_stacking     = False,
        time_limit           = time_limit,
        presets              = presets,
        num_bag_folds        = 5,
        num_stack_levels     = 1,
        ag_args_fit          = {"num_gpus": 1},
        feature_metadata     = feature_metadata,
        excluded_model_types = ["KNN"],
    )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min")
    return predictor, val_df


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(predictor: TabularPredictor, val_df: pd.DataFrame = None):
    print("\n" + "=" * 65 + "\n  LEADERBOARD\n" + "=" * 65)
    lb = predictor.leaderboard(silent=True)
    with pd.option_context("display.max_rows", 40, "display.width", 120):
        print(lb[["model", "score_val", "fit_time", "pred_time_val"]].to_string())

    print("\n  FEATURE IMPORTANCE — top-30")
    try:
        data_fi = val_df if val_df is not None else TabularDataset(TRAIN_PATH)
        fi = predictor.feature_importance(
            data             = data_fi,
            num_shuffle_sets = 5,
            subsample_size   = min(10_000, len(data_fi)),
        )
        print(fi.head(30).to_string())
        os.makedirs("models5", exist_ok=True)
        fi.to_csv("models5/feature_importance_v5.csv")
        print("  Saved: models5/feature_importance_v5.csv")
    except Exception as e:
        print(f"  Feature importance failed: {e}")


def get_best_model_name(predictor: TabularPredictor) -> str | None:
    best_model = getattr(predictor, "model_best", None)
    if best_model:
        return best_model
    try:
        lb = predictor.leaderboard(silent=True)
        if "model" in lb.columns and not lb.empty:
            return lb.iloc[0]["model"]
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────────────────────────

def predict(predictor: TabularPredictor,
            out_raw:     str = "submissions/submission_v5.csv",
            out_rounded: str = "submissions/submission_v5_rounded.csv"):
    print("\n" + "=" * 65 + "\n  PREDICT TEST\n" + "=" * 65)
    test_df = pd.read_parquet(TEST_PATH)

    review_ids = None
    if "review_id" in test_df.columns:
        review_ids = test_df["review_id"].copy()
        test_df.drop(columns=["review_id"], inplace=True)

    test_df = sanitize(test_df)
    best_model = get_best_model_name(predictor)
    print(f"  Best model: {best_model}")
    preds = predictor.predict(TabularDataset(test_df), model=best_model)
    preds_c = preds.clip(1, 5)
    preds_r = preds_c.round(0).clip(1, 5).astype(int)

    os.makedirs("submissions", exist_ok=True)

    def _save(predictions, path):
        df_out = pd.DataFrame({"review_id": review_ids.values, "stars": predictions.values}) \
                 if review_ids is not None else pd.DataFrame({"stars": predictions})
        df_out.to_csv(path, index=False)
        print(f"  Saved {path}  (mean={predictions.mean():.4f}  std={predictions.std():.4f})")

    _save(preds_c, out_raw)
    _save(preds_r, out_rounded)
    return preds_c, preds_r


def load_predictor() -> TabularPredictor:
    print(f"  Loading predictor from {MODELS_DIR}...")
    return TabularPredictor.load(MODELS_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="full", choices=["train", "evaluate", "predict", "full"])
    parser.add_argument("--time_limit", type=int, default=TIME_LIMIT)
    parser.add_argument("--presets",    default="best_quality")
    args = parser.parse_args()

    print("\n" + "#" * 65)
    print("  AUTOGLUON v5 — v4 + Leak Residual de Medias")
    print("#" * 65)

    val_df = None
    if args.mode in ("train", "full"):
        predictor, val_df = train(time_limit=args.time_limit, presets=args.presets)
    else:
        predictor = load_predictor()

    if args.mode in ("evaluate", "full"):
        evaluate(predictor, val_df)

    if args.mode in ("predict", "full"):
        predict(predictor)

    print("\n  Done.\n")
