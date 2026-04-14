"""
infer_autogluon.py
==================
Script SOLO para inferencia con AutoGluon.

- Carga predictor entrenado
- Carga test data
- Aplica misma sanitización
- Genera predicciones
- Guarda submission.csv

Uso:
    python infer_autogluon.py
"""

import pandas as pd
import os
from autogluon.tabular import TabularPredictor, TabularDataset

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "models/autogluon4"   # ⚠️ cambia si usaste timestamp
TEST_PATH  = "data3/test_ag.parquet"
RAW_TEST_PATH = "data3/test_reviews.csv"      # ⚠️ Ajusta la ruta a tu test_reviews.csv original
OUTPUT_PATH = "submissions/submission.csv"

TARGET_COL = "target"
TEXT_FEATURES = ["categories"]


# ─────────────────────────────────────────────────────────────
# SANITIZE (MISMA QUE TRAIN)
# ─────────────────────────────────────────────────────────────
def sanitize_for_autogluon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Nullable ints → float32
    nullable_int_cols = [
        c for c in df.columns
        if str(df[c].dtype).startswith("Int")
    ]
    for c in nullable_int_cols:
        df[c] = df[c].astype("float32")

    # TEXT features
    for c in TEXT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # Object → category
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        if c not in TEXT_FEATURES:
            df[c] = df[c].astype("category")

    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n🚀 Cargando modelo...")
    predictor = TabularPredictor.load(MODEL_PATH)

    print("📂 Cargando test data...")
    test_data = TabularDataset(TEST_PATH)
    test_data = sanitize_for_autogluon(test_data)

    # Cargar solo los review_id originales para no saturar la RAM
    print("📂 Cargando review_ids originales...")
    original_test_ids = pd.read_csv(RAW_TEST_PATH, usecols=["review_id"])

    # Validar que tengan la misma cantidad de filas
    if len(original_test_ids) != len(test_data):
        print(f"⚠️ ADVERTENCIA: Las filas en {RAW_TEST_PATH} ({len(original_test_ids)}) "
              f"no coinciden con {TEST_PATH} ({len(test_data)})")

    print("🤖 Generando predicciones...")
    preds = predictor.predict(test_data, model=predictor.model_best)

    # Clipping (Yelp: 1–5 estrellas)
    preds = preds.clip(1, 5)

    print("💾 Guardando submission...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Asignar el review_id real y los valores predichos
    submission = pd.DataFrame({
        "review_id": original_test_ids["review_id"],
        "stars": preds.values # .values asegura que se asigne correctamente ignorando índices de pandas
    })
    
    submission.to_csv(OUTPUT_PATH, index=False)

    print("\n✅ DONE")
    print(f"📄 Archivo: {OUTPUT_PATH}")
    print(f"📊 Stats → min={preds.min():.3f} max={preds.max():.3f} "
          f"mean={preds.mean():.3f} std={preds.std():.3f}")


if __name__ == "__main__":
    main()