"""
train_autogluon.py  (v2 — post análisis + splits temporales)
============================================================
Cambios en esta versión:
  - Carga y ordena reviews por date_num (timestamp ascendente)
  - Hace splits temporales: 70% train, 15% val, 15% test
  - NO mezcla índices entre splits (mantiene orden cronológico)
  - Usa tuning_data del split temporal como validación interna

Dataset:
  - 1.201 categorías únicas → 'categories' como TEXT (NLP + embeddings)
  - 81 atributos → muchos son BusinessParking_* y Ambience_* (binarios)
  - 15 atributos categóricos (WiFi, NoiseLevel…) → AutoGluon label-encode
  - ~968K reviews, ~30K negocios
  - num_bag_folds=5 y num_stack_levels=1 configurados para tiempo viable

Parámetros importantes:
  - TIME_LIMIT = 14400 * 12 → 48 horas (para GPU con mucha RAM)
    Ajusta según hardware: GPU → 14400 (4h), CPU solo → 7200 (2h)
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
MODELS_DIR = f"models2/autogluon_{int(time.time())}"

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
    "NN_TORCH": [
        {
            "num_epochs": 50,
            "learning_rate": 1e-3,
            "dropout_prob": 0.1,
            "weight_decay": 1e-6,
            "batch_size": 256,  # <-- ¡Bajamos a 256!
            "ag_args": {"name_suffix": "Standard"},
        },
        {
            "num_epochs": 30,
            "learning_rate": 3e-4,
            "dropout_prob": 0.3,
            "weight_decay": 1e-5,
            "batch_size": 512, # <-- ¡Bajamos a 512!
            "ag_args": {"name_suffix": "Regularized"},
        },
    ],
}

from autogluon.common.features.feature_metadata import FeatureMetadata

TEXT_FEATURES = ["categories"]

def load_and_split_by_date(parquet_path: str, 
                           train_frac: float = 0.70,
                           val_frac: float = 0.15):
    """
    Carga parquet, ordena por date_num (timestamp), y hace splits temporales.
    Retorna splits SIN las columnas de rastreo/temporalidad (excepto date).
    
    Returns:
        dict con keys 'train', 'val', 'test' (dataframes sin date_num/user_id/business_id)
        y 'date_col' (nombre de la columna de fecha usada)
    """
    df = pd.read_parquet(parquet_path)
    
    # Columnas a dropear después del split (solo fecha numérica para el split temporal)
    cols_to_drop = ["date_num"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    # Ordenar por date_num (timestamp)
    if "date_num" in df.columns:
        df = df.sort_values("date_num", ascending=True).reset_index(drop=True)
        date_col = "date_num"
    else:
        print("  ⚠ Columna date_num no encontrada. No se puede hacer split temporal.")
        print("    Usando split aleatorio 70/15/15")
        # Fallback: split aleatorio
        np.random.seed(SEED)
        indices = np.random.permutation(len(df))
        n = len(df)
        train_idx = indices[:int(n * train_frac)]
        val_idx = indices[int(n * train_frac):int(n * (train_frac + val_frac))]
        test_idx = indices[int(n * (train_frac + val_frac)):]
        return {
            "train": df.iloc[train_idx].drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True),
            "val": df.iloc[val_idx].drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True),
            "test": df.iloc[test_idx].drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True),
            "date_col": None
        }
    
    # Split temporal: mantener índices ordenados
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    result = {
        "train": df.iloc[:train_end].drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True),
        "val": df.iloc[train_end:val_end].drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True),
        "test": df.iloc[val_end:].drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True),
        "date_col": date_col
    }
    
    print(f"\n  Splits por fecha (date_num ascendente):")
    print(f"    Train: [0:{train_end}] = {len(result['train']):,} reviews ({100*train_frac:.0f}%)")
    print(f"    Val:   [{train_end}:{val_end}] = {len(result['val']):,} reviews ({100*val_frac:.0f}%)")
    print(f"    Test:  [{val_end}:{n}] = {len(result['test']):,} reviews ({100*(1-train_frac-val_frac):.0f}%)")
    
    return result


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
    print("\n  Cargando datos y haciendo splits temporales...")
    splits = load_and_split_by_date(TRAIN_PATH)
    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]
    
    # Aplicar sanitización
    print("  Sanitizando para AutoGluon...")
    train_data = sanitize_for_autogluon(train_data)
    val_data = sanitize_for_autogluon(val_data)
    test_data = sanitize_for_autogluon(test_data)
    
    print(f"\n  Estadísticas Target:")
    print(f"    Train — min:{train_data[TARGET_COL].min():.0f}  "
          f"max:{train_data[TARGET_COL].max():.0f}  "
          f"mean:{train_data[TARGET_COL].mean():.3f}  "
          f"std:{train_data[TARGET_COL].std():.3f}")
    print(f"    Val   — min:{val_data[TARGET_COL].min():.0f}  "
          f"max:{val_data[TARGET_COL].max():.0f}  "
          f"mean:{val_data[TARGET_COL].mean():.3f}  "
          f"std:{val_data[TARGET_COL].std():.3f}")

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
        tuning_data  = val_data,  # Usar split temporal como validación
        hyperparameters = HYPERPARAMETERS,
        time_limit   = time_limit,
        presets      = "high_v150",

        # Solo NN_TORCH: sin stacking ni bagging
        num_stack_levels = 0,
        num_bag_folds    = 0,

        # Pasar metadata para marcar 'categories' como TEXT
        feature_metadata = feature_metadata,
        ag_args_fit={"num_gpus": 1},
        excluded_model_types = ["KNN"],
    )

    elapsed = time.time() - t0
    print(f"\n  ✓ Entrenamiento completado en {elapsed/60:.1f} min")

    return predictor, test_data


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(predictor: TabularPredictor, val_data: pd.DataFrame = None):
    print("\n" + "═"*65)
    print("  LEADERBOARD")
    print("═"*65)
    lb = predictor.leaderboard(silent=True)
    print(lb[["model","score_val","fit_time","pred_time_val"]].to_string())

    print("\n" + "═"*65)
    print("  FEATURE IMPORTANCE — top-30 (puede tardar ~2 min)")
    print("═"*65)
    try:
        if val_data is not None:
            data_for_fi = val_data
        else:
            data_for_fi = TabularDataset(TRAIN_PATH)
        
        fi = predictor.feature_importance(
            data             = data_for_fi,
            num_shuffle_sets = 3,
            subsample_size   = min(5000, len(data_for_fi)),
        )
        print(fi.head(30).to_string())

        # Guardar para análisis posterior
        fi.to_csv("models2/feature_importance.csv")
        print("\n  → Guardado en models/feature_importance.csv")
    except Exception as e:
        print(f"  Feature importance no disponible: {e}")

    best = predictor.model_best

    lb = predictor.leaderboard(silent=True)
    val_mae = -lb.loc[lb["model"] == best, "score_val"].values[0]
    print(f"\n  Mejor modelo : {best}")
    print(f"  MAE (val)    : {val_mae:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def predict(predictor: TabularPredictor,
            test_data: pd.DataFrame = None,
            output_path: str = "submissions/submission.csv"):
    print("\n" + "═"*65)
    print("  PREDICCIÓN TEST")
    print("═"*65)

    if test_data is None:
        print("  Cargando test_data desde archivo...")
        test_data = TabularDataset(TEST_PATH)
        test_data = sanitize_for_autogluon(test_data)
    
    preds = predictor.predict(test_data, model=predictor.model_best)
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

    test_data = None
    
    if args.mode in ("train", "full"):
        predictor, test_data = train(time_limit=args.time_limit)
    else:
        predictor = load_predictor()

    if args.mode in ("evaluate", "full"):
        evaluate(predictor, val_data=test_data if test_data is not None else None)

    if args.mode in ("predict", "full"):
        predict(predictor, test_data=test_data)

    print("\n  ✅ Completado\n")