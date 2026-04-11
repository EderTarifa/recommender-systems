"""
train_autogluon.py  (v2 — post análisis)
=========================================
Sin cambios estructurales respecto a v1, pero con anotaciones
sobre lo que sabemos ahora del dataset:

  - 1.201 categorías únicas → 'categories' como TEXT activa el
    módulo NLP de AutoGluon (FastAI con embeddings de tokens)
  - 81 atributos → muchos son BusinessParking_* y Ambience_*
    (binarios); AutoGluon los tratará como numéricos
  - 15 atributos categóricos (WiFi, NoiseLevel…) → AutoGluon
    los detecta como 'object' y les aplica label encoding
  - Dataset: ~968K reviews, ~30K negocios
    → num_bag_folds=8 y num_stack_levels=2 son viables en tiempo
"""

import pandas as pd
import numpy as np
import os
import gc
import time

from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.common.features.feature_metadata import FeatureMetadata


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models/autogluon"

TRAIN_PATH  = os.path.join(DATA_DIR, "train_ag.parquet")
TEST_PATH   = os.path.join(DATA_DIR, "test_ag.parquet")

TARGET_COL  = "target"
EVAL_METRIC = "mae"

# Ajustar según hardware:
#   GPU + mucha RAM → 14400 (4h)
#   Solo CPU        → 7200  (2h)
TIME_LIMIT = 14400 * 12 # -> 48 horas
SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE METADATA
# Forzamos 'categories' como TEXT para activar los módulos NLP.
# El resto AutoGluon lo infiere bien a partir de los dtypes del parquet.
# ─────────────────────────────────────────────────────────────────────────────

# Columnas que deben tratarse como TEXT (NLP)
# → Solo 'categories' (string largo, multi-valor separado por comas)
TEXT_FEATURES = ["categories"]

# Las siguientes las infiere AutoGluon como 'category' a partir de dtype object:
# top_category, city, state, postal_code, elite_bucket,
# attr_WiFi, attr_NoiseLevel, attr_RestaurantsAttire, attr_Alcohol, attr_Smoking…


# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

HYPERPARAMETERS = {

    # ── LightGBM ─────────────────────────────────────────────────────────────
    # Maneja NaN nativamente (sin imputar) → ideal para nuestros attr_* con NaN
    # Tres variantes para diversidad en el ensemble
    "GBM": [
        {
            "num_leaves":           256,
            "min_child_samples":    20,
            "learning_rate":        0.05,
            "n_estimators":         2000,
            "feature_fraction":     0.8,
            "bagging_fraction":     0.8,
            "bagging_freq":         1,
            "lambda_l1":            0.1,
            "lambda_l2":            0.1,
            "extra_trees":          False,
            "ag_args": {"name_suffix": "Standard"},
        },
        {
            "num_leaves":           512,
            "min_child_samples":    10,
            "learning_rate":        0.03,
            "n_estimators":         3000,
            "feature_fraction":     0.7,
            "bagging_fraction":     0.7,
            "bagging_freq":         1,
            "lambda_l1":            0.05,
            "lambda_l2":            0.1,
            "ag_args": {"name_suffix": "Large"},
        },
        {
            # ExtraTrees variant: baja varianza, buena diversidad para el ensemble
            "num_leaves":           128,
            "min_child_samples":    5,
            "learning_rate":        0.05,
            "n_estimators":         2000,
            "extra_trees":          True,
            "ag_args": {"name_suffix": "XT"},
        },
    ],

    # ── CatBoost ─────────────────────────────────────────────────────────────
    # Especialmente bueno con las 15 columnas categóricas de atributos
    # (WiFi, NoiseLevel, RestaurantsAttire…) y con top_category / city
    "CAT": [
        {
            "iterations":           3000,
            "learning_rate":        0.05,
            "depth":                8,
            "l2_leaf_reg":          3,
            "random_strength":      1,
            "bagging_temperature":  1,
            "ag_args": {"name_suffix": "Standard"},
        },
        {
            "iterations":           2000,
            "learning_rate":        0.1,
            "depth":                6,
            "l2_leaf_reg":          10,
            "ag_args": {"name_suffix": "Shallow"},
        },
    ],

    # ── XGBoost ──────────────────────────────────────────────────────────────
    "XGB": [
        {
            "n_estimators":         2000,
            "learning_rate":        0.05,
            "max_depth":            8,
            "min_child_weight":     5,
            "subsample":            0.8,
            "colsample_bytree":     0.8,
            "reg_alpha":            0.1,
            "reg_lambda":           1.0,
            "tree_method":          "hist",   # → "gpu_hist" si tienes GPU
            "ag_args": {"name_suffix": "Standard"},
        },
    ],

    # ── Random Forest ─────────────────────────────────────────────────────────
    "RF": [
        {
            "n_estimators":         500,
            "max_features":         "sqrt",
            "min_samples_leaf":     5,
            "ag_args": {"name_suffix": "Standard"},
        },
    ],

    # ── Extra Trees ───────────────────────────────────────────────────────────
    "XT": [
        {
            "n_estimators":         500,
            "max_features":         "sqrt",
            "min_samples_leaf":     3,
            "ag_args": {"name_suffix": "Standard"},
        },
    ],

    # ── Neural Net PyTorch ────────────────────────────────────────────────────
    # Con 'categories' como TEXT, AutoGluon genera embeddings de tokens
    # que se concatenan con las features tabulares antes de las capas dense
    "NN_TORCH": [
        {
            "num_epochs":           50,
            "learning_rate":        1e-3,
            "dropout_prob":         0.1,
            "weight_decay":         1e-6,
            "batch_size":           512,
            "layers":               [512, 256, 128],
            "ag_args": {"name_suffix": "Standard"},
        },
        {
            "num_epochs":           30,
            "learning_rate":        3e-4,
            "dropout_prob":         0.3,
            "weight_decay":         1e-5,
            "batch_size":           1024,
            "layers":               [256, 128, 64],
            "ag_args": {"name_suffix": "Regularized"},
        },
    ],

    # ── FastAI ────────────────────────────────────────────────────────────────
    # Aprende embeddings para top_category (1.201 valores únicos) y city.
    # Es el modelo que más provecho saca de las categóricas de alta cardinalidad.
    "FASTAI": [
        {
            "epochs":               30,
            "lr":                   1e-3,
            "emb_drop":             0.1,
            "ps":                   [0.1, 0.1],
            "layers":               [200, 100],
            "ag_args": {"name_suffix": "Standard"},
        },
    ],
}

from autogluon.common.features.feature_metadata import FeatureMetadata

TEXT_FEATURES = ["categories"]

def sanitize_for_autogluon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Nullable integers de pandas -> float32 para evitar warnings de numpy/AutoGluon
    nullable_int_cols = [
        c for c in df.columns
        if str(df[c].dtype).startswith("Int")
    ]
    for c in nullable_int_cols:
        df[c] = df[c].astype("float32")

    # 2) Mantener el texto como texto
    for c in TEXT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # 3) El resto de object/string -> category para evitar detección errónea de datetime
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        if c not in TEXT_FEATURES:
            df[c] = df[c].astype("category")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train(time_limit: int = TIME_LIMIT):
    print("\n  Cargando datos...")
    train_data = TabularDataset(TRAIN_PATH)
    train_data = sanitize_for_autogluon(train_data)
    print(f"    Shape: {train_data.shape}")
    print(f"    Target — min:{train_data[TARGET_COL].min():.0f}  "
          f"max:{train_data[TARGET_COL].max():.0f}  "
          f"mean:{train_data[TARGET_COL].mean():.3f}  "
          f"std:{train_data[TARGET_COL].std():.3f}")

    # Columnas de texto detectadas
    text_cols = [c for c in TEXT_FEATURES if c in train_data.columns]
    print(f"\n  Columnas TEXT (NLP): {text_cols}")

    # Columnas categóricas (dtype object que AutoGluon inferirá como category)
    obj_cols = train_data.select_dtypes(include=["object"]).columns.tolist()
    non_text  = [c for c in obj_cols if c not in text_cols and c != TARGET_COL]
    print(f"  Columnas categóricas (inferidas): {non_text[:10]}{'...' if len(non_text) > 10 else ''}")

    predictor = TabularPredictor(
        label        = TARGET_COL,
        eval_metric  = EVAL_METRIC,
        path         = MODELS_DIR,
        problem_type = "regression",
        verbosity    = 2,
    )
    feature_metadata = FeatureMetadata.from_df(train_data)
    feature_metadata = feature_metadata.add_special_types(
        {col: ["text"] for col in TEXT_FEATURES if col in train_data.columns}
    )

    if text_cols:
        feature_metadata = feature_metadata.add_special_types(
            {col: ["text"] for col in text_cols}
        )

    print(f"\n  Iniciando entrenamiento (time_limit={time_limit}s = {time_limit/3600:.1f}h)...")
    t0 = time.time()

    predictor.fit(
        train_data   = train_data,
        hyperparameters = HYPERPARAMETERS,
        time_limit   = time_limit,
        presets      = "high_v150",

        # Stacking de 2 niveles:
        #   L1 → modelos base con predicciones OOF (8 folds)
        #   L2 → weighted ensemble aprende a combinar las OOF de L1
        num_stack_levels = 1,
        num_bag_folds    = 5, 
        num_bag_sets     = 1,

        holdout_frac     = 0.1,   # 10% validación interna

        # Pasar 'categories' como TEXT para que AutoGluon active NLP
        feature_metadata = feature_metadata,

        excluded_model_types = ["KNN"],
    )

    elapsed = time.time() - t0
    print(f"\n  ✓ Entrenamiento completado en {elapsed/60:.1f} min")

    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(predictor: TabularPredictor):
    print("\n" + "═"*65)
    print("  LEADERBOARD")
    print("═"*65)
    lb = predictor.leaderboard(silent=True)
    print(lb[["model","score_val","fit_time","pred_time_val"]].to_string())

    print("\n" + "═"*65)
    print("  FEATURE IMPORTANCE — top-30 (puede tardar ~2 min)")
    print("═"*65)
    try:
        fi = predictor.feature_importance(
            data             = TabularDataset(TRAIN_PATH),
            num_shuffle_sets = 3,
            subsample_size   = 5000,
        )
        print(fi.head(30).to_string())

        # Guardar para análisis posterior
        fi.to_csv("models/feature_importance.csv")
        print("\n  → Guardado en models/feature_importance.csv")
    except Exception as e:
        print(f"  Feature importance no disponible: {e}")

    best = predictor.get_model_best()
    val_mae = -predictor.get_model_attribute(best, "val_score")
    print(f"\n  Mejor modelo : {best}")
    print(f"  MAE (val)    : {val_mae:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def predict(predictor: TabularPredictor,
            output_path: str = "submissions/submission.csv"):
    print("\n" + "═"*65)
    print("  PREDICCIÓN TEST")
    print("═"*65)

    test_data = TabularDataset(TEST_PATH)
    test_data = sanitize_for_autogluon(test_data)
    preds = predictor.predict(test_data, model=predictor.get_model_best())

    # Las estrellas de Yelp son siempre [1, 5]
    preds = preds.clip(1, 5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({"stars": preds}).to_csv(output_path, index=True, index_label="index")

    print(f"  ✓ Guardado: {output_path}")
    print(f"  min={preds.min():.3f}  max={preds.max():.3f}  "
          f"mean={preds.mean():.3f}  std={preds.std():.3f}")
    return preds


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
    print("  AUTOGLUON — SISTEMA DE RECOMENDACIÓN YELP (v2)")
    print("█"*65)

    if args.mode in ("train", "full"):
        predictor = train(time_limit=args.time_limit)
    else:
        predictor = load_predictor()

    if args.mode in ("evaluate", "full"):
        evaluate(predictor)

    if args.mode in ("predict", "full"):
        predict(predictor)

    print("\n  ✅ Completado\n")