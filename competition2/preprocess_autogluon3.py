"""
preprocess_autogluon3.py  (v3 — post análisis + splits temporales)
"""

import pandas as pd
import numpy as np
import ast
import re
import gc
import os
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR      = "data2"
NEGOCIOS_PATH = os.path.join(DATA_DIR, "negocios.csv")
USUARIOS_PATH = os.path.join(DATA_DIR, "usuarios.csv")
TRAIN_PATH    = os.path.join(DATA_DIR, "train_reviews.csv")

# Buscar test_final.csv; si no existe, fallback a test_reviews.csv
_test_final = os.path.join(DATA_DIR, "test_final.csv")
_test_reviews = os.path.join(DATA_DIR, "test_reviews.csv")
if os.path.exists(_test_final):
    TEST_PATH = _test_final
elif os.path.exists(_test_reviews):
    TEST_PATH = _test_reviews
    print(f"⚠ AVISO: {_test_final} no existe. Usando fallback: {_test_reviews}")
else:
    raise FileNotFoundError(f"Ni {_test_final} ni {_test_reviews} existen en {DATA_DIR}/")

OUT_TRAIN     = os.path.join("data2", "train_ag.parquet")
OUT_TEST      = os.path.join("data2", "test_ag.parquet")

# Atributos que el análisis identificó como categóricos (string, no 0/1).
# Son los 15 de los 81 totales cuyo tipo == categórico.
KNOWN_CATEGORICAL_ATTRS = {
    "WiFi",              # 'free', 'paid', 'no'
    "NoiseLevel",        # 'quiet', 'average', 'loud', 'very_loud'
    "RestaurantsAttire", # 'casual', 'dressy', 'formal'
    "Alcohol",           # 'none', 'beer_and_wine', 'full_bar'
    "Smoking",           # 'no', 'outdoor', 'yes'
    "BYOBCorkage",
    "AgesAllowed",
}


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(s: str) -> str:
    """Elimina prefijos u' que rompen ast.literal_eval."""
    return re.sub(r"\bu(['\"])", r"\1", s)


def _to_scalar(v, is_categorical: bool = False):
    """
    Convierte valor crudo a tipo limpio.
      - bool           → 0/1
      - 'True'/'False' → 0/1
      - is_categorical → string limpio (sin comillas, sin u')
      - numérico       → int o float
      - None/null      → None
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v

    s = str(v).strip()
    s = re.sub(r"^u?['\"](.+)['\"]$", r"\1", s).strip()

    if s.lower() in ("none", "null", ""):
        return None
    if s.lower() == "true":
        return 1
    if s.lower() == "false":
        return 0

    if is_categorical:
        return s.lower()

    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass

    return s.lower()   # fallback: string categórico inesperado


def _parse_subdict(raw) -> dict:
    """Parsea un valor que puede ser un sub-dict almacenado como string."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        result = ast.literal_eval(_clean(raw))
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def parse_all_attributes(attr_str) -> dict:
    """
    Parsea el campo 'attributes' y devuelve un dict plano con TODOS
    los atributos. No filtra por top-N (son solo 81 en total).

    Sub-dicts (BusinessParking, Ambience, Music, GoodForMeal, BestNights…)
    se aplanan: attr_BusinessParking_street, attr_Ambience_casual, etc.
    Los 15 atributos categóricos se devuelven como string limpio.
    Los 66 binarios/numéricos se devuelven como int/float.
    """
    out = {}

    if pd.isna(attr_str) or str(attr_str).strip() in ("", "nan"):
        return out

    try:
        outer = ast.literal_eval(_clean(str(attr_str)))
    except Exception:
        return out

    if not isinstance(outer, dict):
        return out

    for key, raw_val in outer.items():
        is_cat = key in KNOWN_CATEGORICAL_ATTRS

        # Sub-dict almacenado como string → aplanar
        if isinstance(raw_val, str) and raw_val.strip().startswith("{"):
            sub = _parse_subdict(raw_val)
            if sub:
                for sub_key, sub_val in sub.items():
                    v = _to_scalar(sub_val, is_categorical=False)
                    if v is not None:
                        out[f"attr_{key}_{sub_key}"] = v
                continue

        v = _to_scalar(raw_val, is_categorical=is_cat)
        if v is not None:
            out[f"attr_{key}"] = v

    return out


# ─────────────────────────────────────────────────────────────────────────────
# HOURS
# ─────────────────────────────────────────────────────────────────────────────

def extract_hours(hours_str) -> dict:
    feats = {
        "hours_days_open"     : np.nan,
        "hours_avg_open_hour" : np.nan,
        "hours_avg_close_hour": np.nan,
        "hours_avg_daily_hrs" : np.nan,
        "hours_open_weekend"  : np.nan,
        "hours_open_weekday"  : np.nan,
    }
    if pd.isna(hours_str) or str(hours_str).strip() in ("", "nan"):
        return feats
    try:
        d = ast.literal_eval(_clean(str(hours_str)))
    except Exception:
        return feats
    if not isinstance(d, dict) or not d:
        return feats

    WEEKEND = {"Saturday", "Sunday"}
    opens, closes, durations = [], [], []
    has_weekend, has_weekday = 0, 0

    for day, rng in d.items():
        try:
            o_s, c_s = str(rng).split("-")
            oh, om = map(int, o_s.split(":"))
            ch, cm = map(int, c_s.split(":"))
            o, c = oh + om / 60, ch + cm / 60
            opens.append(o)
            closes.append(c)
            durations.append((c - o) % 24)
            if day in WEEKEND:
                has_weekend = 1
            else:
                has_weekday = 1
        except Exception:
            continue

    if opens:
        feats["hours_days_open"]      = len(opens)
        feats["hours_avg_open_hour"]  = round(np.mean(opens), 2)
        feats["hours_avg_close_hour"] = round(np.mean(closes), 2)
        feats["hours_avg_daily_hrs"]  = round(np.mean(durations), 2)
        feats["hours_open_weekend"]   = has_weekend
        feats["hours_open_weekday"]   = has_weekday

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# PROCESO NEGOCIOS
# ─────────────────────────────────────────────────────────────────────────────

def process_negocios(path: str) -> pd.DataFrame:
    print("\n" + "═"*65)
    print("  NEGOCIOS")
    print("═"*65)

    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape original: {df.shape}")

    df.drop(columns=["name", "address"], inplace=True, errors="ignore")

    # ── Postal code: imputar por moda de ciudad ───────────────────────────────
    mode_map = (
        df.dropna(subset=["postal_code"])
          .groupby("city")["postal_code"]
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    mask = df["postal_code"].isna()
    df.loc[mask, "postal_code"] = df.loc[mask, "city"].map(mode_map)
    df["postal_code"] = (df["postal_code"]
                         .astype(str)
                         .str.replace(r"\.0$", "", regex=True)
                         .replace("nan", "UNKNOWN"))

    # ── Numéricas básicas ─────────────────────────────────────────────────────
    df["stars"]        = pd.to_numeric(df["stars"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    df["is_open"]      = pd.to_numeric(df["is_open"], errors="coerce").fillna(0).astype(np.int8)
    df["latitude"]     = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"]    = pd.to_numeric(df["longitude"], errors="coerce")
    df.rename(columns={
        "stars":        "stars_business",
        "review_count": "review_count_business",
    }, inplace=True)

    # ── Categories ────────────────────────────────────────────────────────────
    # Solo 1.201 categorías únicas → no hace falta filtrar nada
    # Top-8 ya cubre el 90% de reviews, top-11 el 95%
    df["categories"] = df["categories"].fillna("").astype(str)

    df["top_category"] = df["categories"].apply(
        lambda s: s.split(",")[0].strip() if s.strip() else "Unknown"
    )
    df["n_categories"] = df["categories"].apply(
        lambda s: len([x for x in s.split(",") if x.strip()]) if s.strip() else 0
    ).astype(np.int8)

    # ── Hours ─────────────────────────────────────────────────────────────────
    print("  Extrayendo features de horario...")
    hours_feats = df["hours"].apply(extract_hours)
    df_hours    = pd.DataFrame(list(hours_feats), index=df.index)
    df = pd.concat([df.drop(columns=["hours"]), df_hours], axis=1)
    del hours_feats, df_hours
    gc.collect()

    # ── Attributes: parsear TODOS los 81 ─────────────────────────────────────
    # No filtramos porque son solo 81; 41 tienen >5% cobertura de negocios.
    # Los NaN se dejan: AutoGluon imputa por modelo (LightGBM los maneja nativamente,
    # CatBoost también, NN los rellena con la media del fold).
    print("  Parseando attributes (todos los 81 atributos)...")
    attrs_parsed = df["attributes"].apply(parse_all_attributes)
    attr_df = pd.DataFrame(list(attrs_parsed), index=df.index)
    print(f"    → {attr_df.shape[1]} columnas generadas tras aplanar sub-dicts")

    # Asignar tipos eficientes según si el atributo es categórico o numérico
    for col in attr_df.columns:
        # ¿Corresponde a un atributo categórico conocido?
        base_key = col.replace("attr_", "")
        is_known_cat = base_key in KNOWN_CATEGORICAL_ATTRS

        sample = attr_df[col].dropna().head(300)
        is_mostly_str = (
            len(sample) > 0 and
            sample.apply(lambda x: isinstance(x, str)).mean() > 0.5
        )

        if is_known_cat or is_mostly_str:
            # Mantener como object/string → AutoGluon lo trata como categoría
            attr_df[col] = attr_df[col].astype(object)
        else:
            # Binario o numérico → float32
            attr_df[col] = pd.to_numeric(attr_df[col], errors="coerce").astype("float32")

    df = pd.concat([df.drop(columns=["attributes"]), attr_df], axis=1)
    del attrs_parsed, attr_df
    gc.collect()

    # ── City / state / postal_code → strings ─────────────────────────────────
    for col in ["city", "state", "postal_code"]:
        df[col] = df[col].astype(str).replace("nan", "Unknown")

    attr_cols  = [c for c in df.columns if c.startswith("attr_")]
    hours_cols = [c for c in df.columns if c.startswith("hours_")]
    print(f"\n  ✓ Shape final       : {df.shape}")
    print(f"    attr_* generadas  : {len(attr_cols)}")
    print(f"    hours_* generadas : {len(hours_cols)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PROCESO USUARIOS
# ─────────────────────────────────────────────────────────────────────────────

def process_usuarios(path: str) -> pd.DataFrame:
    print("\n" + "═"*65)
    print("  USUARIOS")
    print("═"*65)

    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape original: {df.shape}")

    df.drop(columns=["name"], inplace=True, errors="ignore")

    # ── Elite ─────────────────────────────────────────────────────────────────
    def elite_features(s):
        if pd.isna(s) or str(s).strip() in ("", "nan"):
            return 0, 0, 0, []
        try:
            years = [int(y) for y in str(s).split(",") if y.strip().isdigit()]
            return (len(years), min(years) if years else 0, len(years), years)
        except Exception:
            return 0, 0, 0, []

    parsed = df["elite"].apply(elite_features)
    df["elite_count"]     = [x[0] for x in parsed]
    df["elite_from_year"] = [x[1] for x in parsed]
    df["years_as_elite"]  = [x[2] for x in parsed]
    df["elite_years_list"] = [x[3] for x in parsed]
    df["elite_bucket"] = pd.cut(
        df["elite_count"],
        bins=[-1, 0, 2, 5, 10, 999],
        labels=["none", "1-2", "3-5", "6-10", "10+"]
    ).astype(str)
    df.drop(columns=["elite"], inplace=True)
    del parsed

    # ── Friends ───────────────────────────────────────────────────────────────
    df["friend_count"] = df["friends"].apply(
        lambda s: 0 if pd.isna(s) or str(s).strip() == ""
                  else len(str(s).split(","))
    ).astype(np.int32)
    df["has_friends"] = (df["friend_count"] > 0).astype(np.int8)
    df.drop(columns=["friends"], inplace=True)

    # ── Yelping since ─────────────────────────────────────────────────────────
    # SE MANTIENE para calcular user_seniority_days en build_dataset
    df["yelping_since"] = pd.to_datetime(df["yelping_since"], errors="coerce")

    # ── Numéricas ─────────────────────────────────────────────────────────────
    for col in ["useful", "funny", "cool", "review_count", "fans", "average_stars"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["useful_per_review"] = (df["useful"] / (df["review_count"] + 1)).round(3)
    df["fans_per_review"]   = (df["fans"]   / (df["review_count"] + 1)).round(3)

    # ── Compliments (todos separados) ─────────────────────────────────────────
    comp_cols = [c for c in df.columns if c.startswith("compliment_")]
    for col in comp_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)
    df["compliment_total"] = df[comp_cols].sum(axis=1).astype(np.int32)

    print(f"  ✓ Shape final: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(reviews_path: str,
                  usuarios_df: pd.DataFrame,
                  negocios_df: pd.DataFrame,
                  output_path: str,
                  is_train: bool = True,
                  train_stats: dict = None) -> tuple:

    print(f"\n  Cargando reviews: {reviews_path}")
    reviews = pd.read_csv(reviews_path, low_memory=False)
    print(f"    Shape: {reviews.shape}")

    reviews.drop(columns=["review_id"], inplace=True, errors="ignore")

    if is_train:
        reviews.rename(columns={"stars": "target"}, inplace=True)
        reviews["target"] = pd.to_numeric(reviews["target"], errors="coerce")

    for col in ["useful", "funny", "cool"]:
        if col in reviews.columns:
            reviews.rename(columns={col: f"review_{col}"}, inplace=True)
            reviews[f"review_{col}"] = (pd.to_numeric(reviews[f"review_{col}"], errors="coerce")
                                         .fillna(0).astype(np.int32))

    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews["date_num"] = reviews["date"].astype(np.int64) // 10**9
    reviews["review_year"]       = reviews["date"].dt.year.astype("Int16")
    reviews["review_month"]      = reviews["date"].dt.month.astype("Int8")
    reviews["review_dow"]        = reviews["date"].dt.dayofweek.astype("Int8")
    reviews["review_is_weekend"] = (reviews["review_dow"] >= 5).astype(np.int8)

    print("    Mergeando usuarios...")
    df = reviews.merge(usuarios_df, on="user_id", how="left")
    del reviews
    gc.collect()

    print("    Mergeando negocios...")
    df = df.merge(negocios_df, on="business_id", how="left")
    gc.collect()

    # ─── USUARIOS: user_seniority_days y was_elite_at_review ────────────────
    print("    Calculando user_seniority_days y was_elite_at_review...")
    df["user_seniority_days"] = (df["date"] - df["yelping_since"]).dt.days.clip(lower=0).fillna(0).astype(np.int32)
    df["was_elite_at_review"] = df.apply(
        lambda row: 1 if (isinstance(row["elite_years_list"], list) and 
                          row["review_year"] in row["elite_years_list"]) 
                   else 0,
        axis=1
    ).astype(np.int8)
    df.drop(columns=["yelping_since", "elite_years_list"], inplace=True, errors="ignore")

    # ─── ORDENAR POR FECHA ───────────────────────────────────────────────────
    df = df.sort_values("date_num", ascending=True).reset_index(drop=True)

    if is_train:
        print("    Calculando estadísticas temporales (expanding windows)...")
        # Máscaras de cold start ANTES de imputar
        cold_user_mask = df.groupby("user_id").cumcount() == 0
        cold_biz_mask  = df.groupby("business_id").cumcount() == 0

        # Expanding windows por usuario
        user_stats = df.groupby("user_id", sort=False).apply(
            lambda g: g[["target"]].expanding(min_periods=1).mean().shift(1)
        ).reset_index(level=0, drop=True)
        df["user_avg_stars_at_time"] = user_stats["target"]
        df["user_review_count_at_time"] = (
            df.groupby("user_id", sort=False)["target"]
            .expanding(min_periods=1).count().shift(1)
            .reset_index(level=0, drop=True)
        )

        # Expanding windows por negocio
        biz_stats = df.groupby("business_id", sort=False).apply(
            lambda g: g[["target"]].expanding(min_periods=1).mean().shift(1)
        ).reset_index(level=0, drop=True)
        df["biz_avg_stars_at_time"] = biz_stats["target"]
        df["biz_review_count_at_time"] = (
            df.groupby("business_id", sort=False)["target"]
            .expanding(min_periods=1).count().shift(1)
            .reset_index(level=0, drop=True)
        )

        # Media global de target
        global_mean = df["target"].mean()

        # Imputar NaN con media global
        df["user_avg_stars_at_time"] = df["user_avg_stars_at_time"].fillna(global_mean).astype("float32")
        df["user_review_count_at_time"] = df["user_review_count_at_time"].fillna(0).astype(np.int32)
        df["biz_avg_stars_at_time"] = df["biz_avg_stars_at_time"].fillna(global_mean).astype("float32")
        df["biz_review_count_at_time"] = df["biz_review_count_at_time"].fillna(0).astype(np.int32)

        # Cold start flags
        df["is_cold_user"] = cold_user_mask.astype(np.int8)
        df["is_cold_biz"] = cold_biz_mask.astype(np.int8)

        # Delta de estrellas
        df["delta_stars"] = (df["user_avg_stars_at_time"] - df["biz_avg_stars_at_time"]).round(2).astype("float32")

        # Construir train_stats
        train_stats = {
            "user_stats": df.groupby("user_id")[["user_avg_stars_at_time", "user_review_count_at_time"]].last(),
            "biz_stats": df.groupby("business_id")[["biz_avg_stars_at_time", "biz_review_count_at_time"]].last(),
            "global_mean": global_mean
        }

    else:
        print("    Usando estadísticas de entrenamiento...")
        if train_stats is None:
            raise ValueError("train_stats requerido para dataset de test")

        # Lookup desde train_stats
        user_lookup = train_stats["user_stats"]
        biz_lookup = train_stats["biz_stats"]
        global_mean = train_stats["global_mean"]

        df["user_avg_stars_at_time"] = df["user_id"].map(user_lookup["user_avg_stars_at_time"]).fillna(global_mean).astype("float32")
        df["user_review_count_at_time"] = df["user_id"].map(user_lookup["user_review_count_at_time"]).fillna(0).astype(np.int32)
        df["biz_avg_stars_at_time"] = df["business_id"].map(biz_lookup["biz_avg_stars_at_time"]).fillna(global_mean).astype("float32")
        df["biz_review_count_at_time"] = df["business_id"].map(biz_lookup["biz_review_count_at_time"]).fillna(0).astype(np.int32)

        # Cold start (usuarios/negocios nuevos)
        df["is_cold_user"] = (~df["user_id"].isin(user_lookup.index)).astype(np.int8)
        df["is_cold_biz"] = (~df["business_id"].isin(biz_lookup.index)).astype(np.int8)

        # Delta
        df["delta_stars"] = (df["user_avg_stars_at_time"] - df["biz_avg_stars_at_time"]).round(2).astype("float32")

    # ─── PRIORIDAD DE COLUMNAS ───────────────────────────────────────────────
    priority_cols = []
    if is_train and "target" in df.columns:
        priority_cols.append("target")
    for col in ["user_id", "business_id", "date_num"]:
        if col in df.columns:
            priority_cols.append(col)
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]

    # ─── DROPEAR COLUMNAS ANTES DE GUARDAR ───────────────────────────────────
    cols_to_drop = ["review_useful", "review_funny", "review_cool", 
                    "average_stars", "stars_business", "review_count", 
                    "review_count_business", "date",
                    "days_yelping", "years_yelping"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    cat_cols_obj = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols     = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n    ✓ Shape final          : {df.shape}")
    print(f"    Columnas numéricas     : {len(num_cols)}")
    print(f"    Columnas string/object : {len(cat_cols_obj)}")
    print(f"      → {cat_cols_obj[:12]}{'...' if len(cat_cols_obj) > 12 else ''}")
    print(f"    NaN totales            : {df.isna().sum().sum():,}")

    print(f"\n    Guardando {output_path}...")
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"    ✓ Guardado. Tamaño: {size_mb:.1f} MB")

    return df, train_stats


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█"*65)
    print("  PREPROCESADO PARA AUTOGLUON  (v2 — post análisis)")
    print("█"*65)
    print("""
  Decisiones basadas en analyze_cats_attrs.py:

  CATEGORIES (1.201 únicas, top-8 = 90% reviews):
    → texto completo 'categories' (modelos NLP de AutoGluon)
    → 'top_category' string (árboles)
    → 'n_categories' int (feature adicional)
    → SIN OHE ni top-N

  ATTRIBUTES (81 únicos, 41 con >5% cobertura):
    → Todos los 81 atributos (no merece filtrar)
    → Sub-dicts aplanados (BusinessParking_*, Ambience_*, Music_*, …)
    → 66 numéricos/binarios → float32
    → 15 categóricos → string (AutoGluon label-encode/embeddings)
    → NaN → se dejan (LightGBM y CatBoost los manejan nativamente)
  """)

    os.makedirs("data2", exist_ok=True)

    negocios_df = process_negocios(NEGOCIOS_PATH)
    usuarios_df = process_usuarios(USUARIOS_PATH)

    print("\n" + "═"*65)
    print("  GENERANDO TRAIN")
    print("═"*65)
    train_df, train_stats = build_dataset(TRAIN_PATH, usuarios_df, negocios_df,
                                           OUT_TRAIN, is_train=True)
    del train_df
    gc.collect()

    print("\n" + "═"*65)
    print("  GENERANDO TEST")
    print("═"*65)
    test_df, _ = build_dataset(TEST_PATH, usuarios_df, negocios_df,
                               OUT_TEST, is_train=False, train_stats=train_stats)
    del test_df
    gc.collect()

    print("\n" + "█"*65)
    print("  ✅ DONE")
    print(f"  → {OUT_TRAIN}")
    print(f"  → {OUT_TEST}")
    print("█"*65 + "\n")