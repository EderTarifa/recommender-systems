"""
train_autogluon4.py  (v4 — Leaky + IDs + Interacciones + Full Ensemble)
========================================================================
Cambios respecto a versiones anteriores:

  DATOS:
    - Lee de data4/ generado por preprocess_autogluon4.py
    - Mantiene user_id, business_id como 'category' (embeddings NCF)
    - Mantiene average_stars, stars_business (leaky — máxima señal)
    - Incluye features de interacción: delta_stars, naive_pred, weighted_*, etc.

  SPLIT:
    - Aleatorio estratificado 90% train / 10% val (NO temporal)
    - Razonamiento: para competición, queremos que el modelo vea la distribución
      completa. El temporal split infrarepresenta usuarios/negocios bien conocidos.
    - Se usa StratifiedKFold por target entero → preserva distribución de clases.

  ENTRENAMIENTO:
    - Preset: 'best_quality' → activa MultiLayer Stacking + Bagging completo
    - num_bag_folds = 5    → 5 folds de bagging por modelo base
    - num_stack_levels = 1  → un nivel de stacking (L1 base → L2 meta)
    - Modelos activos: LightGBM, XGBoost, CatBoost, NN_TORCH, FastAI
    - KNN excluido (lento, raramente útil en tabular de esta escala)
    - GPU habilitada para NN_TORCH y FastAI
    - NN_TORCH: configuraciones múltiples con distintos dropout/lr/tamaño

  POST-PROCESO:
    - Clip a [1, 5]
    - Se guarda también versión redondeada al entero más cercano
      (potencialmente mejor MAE si el test solo tiene enteros 1–5)

  TIEMPO:
    - Por defecto 8h (ajusta TIME_LIMIT según hardware/deadline)
    - RTX 3060 Ti + 32GB RAM: 8-12h es razonable para best_quality
"""

import os
import gc
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.common.features.feature_metadata import FeatureMetadata


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = "data4"
MODELS_DIR  = "models4/autogluon_v4"
TRAIN_PATH  = os.path.join(DATA_DIR, "train_ag.parquet")
TEST_PATH   = os.path.join(DATA_DIR, "test_ag.parquet")

TARGET_COL  = "target"
EVAL_METRIC = "mae"

# Tiempo de entrenamiento. Guía:
#   RTX 3060 Ti + 32GB RAM, best_quality → 8h mínimo para que el stacking sea útil
#   Si tienes todo el día → 14400 * 3 (12h) es el sweet spot
#   Máximo útil → ~24h (los modelos bases ya habrán convergido)
TIME_LIMIT = 14400 * 2    # 8 horas (cambia a gusto)

SEED = 42
VAL_FRAC = 0.10   # 10% para tuning_data de AutoGluon

# Columnas NLP (se marcan como TEXT para activar módulos NLP de AutoGluon)
TEXT_FEATURES = ["categories"]

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS NN_TORCH
# Varias configuraciones para que el ensemble cubra distinto espacio de hiperparam
# ─────────────────────────────────────────────────────────────────────────────
NN_TORCH_CONFIGS = [
    {
        # Config A: rápida y amplia — buena como base de ensemble
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "dropout_prob": 0.1,
        "weight_decay": 1e-6,
        "batch_size": 512,
        "ag_args": {"name_suffix": "Wide"},
    },
    {
        # Config B: más regularizada — menos overfitting, mejor generalización
        "num_epochs": 60,
        "learning_rate": 3e-4,
        "dropout_prob": 0.3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "ag_args": {"name_suffix": "Regularized"},
    },
    {
        # Config C: LR bajo con más épocas — puede capturar señal sutil de embeddings
        "num_epochs": 80,
        "learning_rate": 1e-4,
        "dropout_prob": 0.2,
        "weight_decay": 1e-5,
        "batch_size": 1024,
        "ag_args": {"name_suffix": "Slow"},
    },
]

HYPERPARAMETERS = {
    "GBM":      [{}],    # LightGBM — líder en tabular con leakage
    "XGB":      [{}],    # XGBoost
    "CAT":      [{}],    # CatBoost — especialmente bueno con categoricals
    "NN_TORCH": NN_TORCH_CONFIGS,
    "FASTAI":   [{}],    # FastAI tabular — complementa NN_TORCH con distintas arquitecturas
    # "RF":     [{}],    # Random Forest — raramente útil cuando hay boosting
    # "KNN":    excluido — O(n^2), inutilizable a esta escala
}


# ─────────────────────────────────────────────────────────────────────────────
# SANITIZE
# ─────────────────────────────────────────────────────────────────────────────

def sanitize(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Prepara el DataFrame para AutoGluon:
      - Nullable Int → float32 (evita warnings numpy)
      - TEXT_FEATURES → string
      - user_id, business_id → category (activan embeddings NCF en NN)
      - Resto de object/string → category
    """
    df = df.copy()

    # Nullable integers → float32
    for c in df.columns:
        if str(df[c].dtype).startswith("Int"):
            df[c] = df[c].astype("float32")

    # Columnas TEXT (NLP)
    for c in TEXT_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna("").astype("string")

    # IDs → category (crítico para embeddings)
    for c in ["user_id", "business_id"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Resto de object/string → category
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        if c not in TEXT_FEATURES + ["user_id", "business_id"]:
            df[c] = df[c].astype("category")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_train_val(df: pd.DataFrame):
    """
    Split aleatorio estratificado por target entero.
    90% train → entrenamiento real
    10% val   → tuning_data de AutoGluon (early stopping, val score)

    Estratificamos por target para que ambas particiones tengan la misma
    distribución de 1-5 estrellas (importante cuando hay clases muy desiguales
    como la 5-estrella que suele dominar en Yelp).
    """
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_FRAC,
        random_state=SEED,
        stratify=df[TARGET_COL].astype(int),  # estratificado por estrella entera
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"\n  Split aleatorio estratificado ({100*(1-VAL_FRAC):.0f}/{100*VAL_FRAC:.0f}):")
    print(f"    Train : {len(train_df):,} reviews")
    print(f"    Val   : {len(val_df):,} reviews")

    # Distribución de targets (check que el split es balanceado)
    for name, part in [("Train", train_df), ("Val", val_df)]:
        dist = part[TARGET_COL].value_counts(normalize=True).sort_index()
        print(f"    {name} dist: " + "  ".join(f"{k}★={v:.1%}" for k, v in dist.items()))

    return train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train(time_limit: int = TIME_LIMIT):
    print("\n  Cargando datos...")
    df = pd.read_parquet(TRAIN_PATH)
    print(f"    Shape bruto: {df.shape}")
    print(f"    Columnas: {list(df.columns[:10])} ...")

    # Eliminar review_id si existe en train (no aporta señal)
    df.drop(columns=["review_id"], inplace=True, errors="ignore")

    df = sanitize(df, is_train=True)

    train_df, val_df = split_train_val(df)
    del df
    gc.collect()

    print(f"\n  Stats target — Train:")
    t = train_df[TARGET_COL]
    print(f"    min={t.min():.0f} max={t.max():.0f} mean={t.mean():.3f} std={t.std():.3f}")

    # Feature metadata: forzamos TEXT para categories
    feature_metadata = FeatureMetadata.from_df(train_df)
    text_cols_present = [c for c in TEXT_FEATURES if c in train_df.columns]
    if text_cols_present:
        feature_metadata = feature_metadata.add_special_types(
            {col: ["text"] for col in text_cols_present}
        )
        print(f"\n  Columnas TEXT (NLP activado): {text_cols_present}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    predictor = TabularPredictor(
        label        = TARGET_COL,
        eval_metric  = EVAL_METRIC,
        path         = MODELS_DIR,
        problem_type = "regression",
        verbosity    = 2,
    )

    print(f"\n  Iniciando entrenamiento (time_limit={time_limit}s = {time_limit/3600:.1f}h)...")
    print(f"  Preset: best_quality  |  Stacking: L1+L2  |  Bagging: 5 folds")
    t0 = time.time()

    predictor.fit(
        train_data        = TabularDataset(train_df),
        tuning_data       = TabularDataset(val_df),
        hyperparameters   = HYPERPARAMETERS,
        use_bag_holdout=True,
        dynamic_stacking=False,
        time_limit        = time_limit,
        presets           = "best_quality",       # MultiLayer Stacking + Bagging completo
        num_bag_folds     = 5,                    # 5-fold bagging para cada modelo base
        num_stack_levels  = 1,                    # L1 (base) + L2 (meta) stacking
        ag_args_fit       = {"num_gpus": 1},      # GPU para NN_TORCH y FastAI
        feature_metadata  = feature_metadata,
        excluded_model_types = ["KNN"],           # KNN es O(n^2), inutilizable aquí
    )

    elapsed = time.time() - t0
    print(f"\n  ✓ Entrenamiento completado en {elapsed/60:.1f} min")
    return predictor, val_df


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(predictor: TabularPredictor, val_df: pd.DataFrame = None):
    print("\n" + "═"*65)
    print("  LEADERBOARD")
    print("═"*65)
    lb = predictor.leaderboard(silent=True)
    with pd.option_context("display.max_rows", 40, "display.width", 120):
        print(lb[["model", "score_val", "fit_time", "pred_time_val"]].to_string())

    best = predictor.get_model_best()
    val_mae = -predictor.get_model_attribute(best, "val_score")
    print(f"\n  Mejor modelo : {best}")
    print(f"  MAE (val)    : {val_mae:.4f}")

    print("\n" + "═"*65)
    print("  FEATURE IMPORTANCE — top-30")
    print("═"*65)
    try:
        data_fi = val_df if val_df is not None else TabularDataset(TRAIN_PATH)
        fi = predictor.feature_importance(
            data             = data_fi,
            num_shuffle_sets = 5,
            subsample_size   = min(10_000, len(data_fi)),
        )
        print(fi.head(30).to_string())
        os.makedirs("models4", exist_ok=True)
        fi.to_csv("models4/feature_importance_v4.csv")
        print("\n  → Guardado en models4/feature_importance_v4.csv")
    except Exception as e:
        print(f"  Feature importance no disponible: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def predict(predictor: TabularPredictor,
            output_path: str = "submissions/submission_v4.csv",
            output_path_rounded: str = "submissions/submission_v4_rounded.csv"):
    print("\n" + "═"*65)
    print("  PREDICCIÓN TEST")
    print("═"*65)

    test_df = pd.read_parquet(TEST_PATH)

    # Extraemos review_id para el CSV de submission antes de sanitizar
    review_ids = None
    if "review_id" in test_df.columns:
        review_ids = test_df["review_id"].copy()
        test_df.drop(columns=["review_id"], inplace=True)

    test_df = sanitize(test_df, is_train=False)

    preds = predictor.predict(TabularDataset(test_df), model=predictor.get_model_best())

    # Clipping a rango válido de Yelp
    preds_clipped  = preds.clip(1, 5)
    # Versión redondeada (potencialmente mejor si el test es siempre entero)
    preds_rounded  = preds_clipped.round(0).clip(1, 5).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def save_submission(predictions, path, ids):
        if ids is not None:
            sub = pd.DataFrame({"review_id": ids.values, "stars": predictions.values})
            sub.to_csv(path, index=False)
        else:
            pd.DataFrame({"stars": predictions}).to_csv(path, index=True, index_label="index")
        print(f"  ✓ Guardado: {path}")
        print(f"    min={predictions.min():.3f}  max={predictions.max():.3f}  "
              f"mean={predictions.mean():.3f}  std={predictions.std():.3f}")

    save_submission(preds_clipped, output_path, review_ids)
    save_submission(preds_rounded, output_path_rounded, review_ids)

    print("\n  Ambas versiones guardadas (clipped y rounded).")
    print("  Prueba ambas en Kaggle — la redondeada puede bajar el MAE si el target es entero.")

    return preds_clipped, preds_rounded


def load_predictor() -> TabularPredictor:
    print(f"  Cargando predictor desde {MODELS_DIR}...")
    return TabularPredictor.load(MODELS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full",
                        choices=["train", "evaluate", "predict", "full"])
    parser.add_argument("--time_limit", type=int, default=TIME_LIMIT)
    args = parser.parse_args()

    print("\n" + "█"*65)
    print("  AUTOGLUON v4 — Leaky + IDs + Interacciones + Full Ensemble")
    print("█"*65)

    val_df = None

    if args.mode in ("train", "full"):
        predictor, val_df = train(time_limit=args.time_limit)
    else:
        predictor = load_predictor()

    if args.mode in ("evaluate", "full"):
        evaluate(predictor, val_df=val_df)

    if args.mode in ("predict", "full"):
        predict(predictor)

    print("\n  ✅ Completado\n")
