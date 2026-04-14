"""
train_autogluon41.py  (v4.1 — baseline v4 afinado)
==================================================
v4.1 entrena sobre data41/ y corrige dos cosas prácticas:

  - elimina la referencia a `categories` como texto, porque en preprocess ya va OHE
  - evita el crash de evaluate() en versiones de AutoGluon sin get_model_best()
"""

import gc
import os
import time

import pandas as pd
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split


DATA_DIR = "data41"
MODELS_DIR = "models41/autogluon_v41"
TRAIN_PATH = os.path.join(DATA_DIR, "train_ag.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test_ag.parquet")

TARGET_COL = "target"
EVAL_METRIC = "mae"
TIME_LIMIT = 14400 * 2
SEED = 42
VAL_FRAC = 0.10
TEXT_FEATURES = []

NN_TORCH_CONFIGS = [
    {
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "dropout_prob": 0.1,
        "weight_decay": 1e-6,
        "batch_size": 512,
        "ag_args": {"name_suffix": "Wide"},
    },
    {
        "num_epochs": 60,
        "learning_rate": 3e-4,
        "dropout_prob": 0.3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "ag_args": {"name_suffix": "Regularized"},
    },
    {
        "num_epochs": 80,
        "learning_rate": 1e-4,
        "dropout_prob": 0.2,
        "weight_decay": 1e-5,
        "batch_size": 1024,
        "ag_args": {"name_suffix": "Slow"},
    },
]

HYPERPARAMETERS = {
    "GBM": [{}],
    "XGB": [{}],
    "CAT": [{}],
    "NN_TORCH": NN_TORCH_CONFIGS,
    "FASTAI": [{}],
}


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if str(df[col].dtype).startswith("Int"):
            df[col] = df[col].astype("float32")

    for col in TEXT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("").astype("string")

    for col in ["user_id", "business_id"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in obj_cols:
        if col not in TEXT_FEATURES + ["user_id", "business_id"]:
            df[col] = df[col].astype("category")

    return df


def split_train_val(df: pd.DataFrame):
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_FRAC,
        random_state=SEED,
        stratify=df[TARGET_COL].astype(int),
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"\n  Split aleatorio estratificado ({100 * (1 - VAL_FRAC):.0f}/{100 * VAL_FRAC:.0f}):")
    print(f"    Train : {len(train_df):,} reviews")
    print(f"    Val   : {len(val_df):,} reviews")

    for name, part in [("Train", train_df), ("Val", val_df)]:
        dist = part[TARGET_COL].value_counts(normalize=True).sort_index()
        print(f"    {name} dist: " + "  ".join(f"{k}*={v:.1%}" for k, v in dist.items()))

    return train_df, val_df


def train(time_limit: int = TIME_LIMIT):
    print("\n  Cargando datos...")
    df = pd.read_parquet(TRAIN_PATH)
    print(f"    Shape bruto: {df.shape}")
    print(f"    Columnas: {list(df.columns[:12])} ...")

    df.drop(columns=["review_id"], inplace=True, errors="ignore")
    df = sanitize(df)

    train_df, val_df = split_train_val(df)
    del df
    gc.collect()

    feature_metadata = FeatureMetadata.from_df(train_df)

    os.makedirs(MODELS_DIR, exist_ok=True)

    predictor = TabularPredictor(
        label=TARGET_COL,
        eval_metric=EVAL_METRIC,
        path=MODELS_DIR,
        problem_type="regression",
        verbosity=2,
    )

    print(f"\n  Iniciando entrenamiento (time_limit={time_limit}s = {time_limit / 3600:.1f}h)...")
    print("  Preset: best_quality  |  Stacking: L1+L2  |  Bagging: 5 folds")
    t0 = time.time()

    predictor.fit(
        train_data=TabularDataset(train_df),
        tuning_data=TabularDataset(val_df),
        hyperparameters=HYPERPARAMETERS,
        use_bag_holdout=True,
        dynamic_stacking=False,
        time_limit=time_limit,
        presets="best_quality",
        num_bag_folds=5,
        num_stack_levels=1,
        ag_args_fit={"num_gpus": 1},
        feature_metadata=feature_metadata,
        excluded_model_types=["KNN"],
    )

    elapsed = time.time() - t0
    print(f"\n  Entrenamiento completado en {elapsed / 60:.1f} min")
    return predictor, val_df


def evaluate(predictor: TabularPredictor, val_df: pd.DataFrame):
    print("\n" + "═" * 65)
    print("  EVALUACION")
    print("═" * 65)

    score = predictor.evaluate(val_df, silent=True)
    lb = predictor.leaderboard(val_df, silent=True)
    with pd.option_context("display.max_rows", 40, "display.width", 140):
        cols = [c for c in ["model", "score_val", "fit_time", "pred_time_val"] if c in lb.columns]
        print(lb[cols].to_string(index=False))

    best_model = getattr(predictor, "model_best", None)
    if not best_model and "model" in lb.columns and not lb.empty:
        best_model = lb.iloc[0]["model"]

    print(f"\n  Mejor modelo : {best_model}")
    print(f"  Validacion   : {score}")


if __name__ == "__main__":
    predictor, val_df = train()
    evaluate(predictor, val_df)
