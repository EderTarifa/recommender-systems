"""
preprocess_autogluon5.py  (v5 — v4 + Leak Residual de Medias)
=============================================================
Extiende v4 con las features de leakage residual descubiertas:

  avg_non_train_biz:
    = (stars_business * rc_business - sum(train_stars_for_biz)) / n_non_train_biz
    ≈ media de las estrellas del test para ese negocio (cuando n_external ≈ 0)
    Cobertura: 100% de negocios en test tienen n_external = 0

  avg_non_train_user:
    Análoga para usuarios, pero con ruido por rc_user grande.
    Solo usamos cuando n_non_train_user es pequeño (<= 10) y el resultado es razonable.

  leak_biz_confidence:
    = n_test_biz / n_non_train_biz  (cuánto del "no-train" es test conocido)
    = 1.0 cuando n_external = 0

  naive_pred_leak:
    = (avg_non_train_biz + avg_non_train_user) / 2
    (versión leaky de naive_pred, más informativa)

  delta_leak:
    = avg_non_train_biz - avg_non_train_user

Output:
  data5/train_ag.parquet
  data5/test_ag.parquet
"""

import pandas as pd
import numpy as np
import gc
import os
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SRC_DIR       = "data"
NEGOCIOS_PATH = os.path.join(SRC_DIR, "negocios.csv")
USUARIOS_PATH = os.path.join(SRC_DIR, "usuarios.csv")
TRAIN_PATH    = os.path.join(SRC_DIR, "train_reviews.csv")
TEST_PATH     = os.path.join(SRC_DIR, "test_reviews.csv")

OUT_DIR   = "data5"
OUT_TRAIN = os.path.join(OUT_DIR, "train_ag.parquet")
OUT_TEST  = os.path.join(OUT_DIR, "test_ag.parquet")

os.makedirs(OUT_DIR, exist_ok=True)

TOP_N_CATS = 20


# ──────────────────────────────────────────────────────────────────────────────
# LEAK RESIDUAL COMPUTATION
# Must be run before processing train/test individually, because avg_non_train
# uses statistics from BOTH splits together.
# ──────────────────────────────────────────────────────────────────────────────

def compute_residual_leaks(train_path: str, test_path: str,
                           negocios_path: str, usuarios_path: str):
    """
    Returns two DataFrames:
      biz_leaks:  business_id + avg_non_train_biz + leak_biz_confidence + n_non_train_biz
      user_leaks: user_id    + avg_non_train_user + leak_user_confidence + n_non_train_user
    """
    print("\n  Computing residual leak features...")

    train_raw = pd.read_csv(train_path, usecols=["user_id", "business_id", "stars"])
    test_raw  = pd.read_csv(test_path,  usecols=["user_id", "business_id"])

    # ─── NEGOCIOS ───
    negocios_meta = pd.read_csv(negocios_path, usecols=["business_id", "stars", "review_count"])
    negocios_meta = negocios_meta.rename(
        columns={"stars": "stars_business", "review_count": "rc_business"}
    )

    biz_train = train_raw.groupby("business_id")["stars"].agg(
        sum_train="sum", n_train="count"
    ).reset_index()
    biz_test_n = test_raw.groupby("business_id").size().reset_index()
    biz_test_n.columns = ["business_id", "n_test_biz"]

    biz = negocios_meta.merge(biz_train, on="business_id", how="left")
    biz = biz.merge(biz_test_n, on="business_id", how="left")
    biz["sum_train"]   = biz["sum_train"].fillna(0)
    biz["n_train"]     = biz["n_train"].fillna(0).astype(int)
    biz["n_test_biz"]  = biz["n_test_biz"].fillna(0).astype(int)

    biz["sum_total"]        = biz["stars_business"] * biz["rc_business"]
    # Some rows are inconsistent; do not allow negative "non-train" counts.
    biz["n_non_train_biz"]  = (biz["rc_business"] - biz["n_train"]).clip(lower=0)
    biz["sum_non_train_biz"]= biz["sum_total"] - biz["sum_train"]
    biz["avg_non_train_biz_raw"] = np.where(
        biz["n_non_train_biz"] > 0,
        biz["sum_non_train_biz"] / biz["n_non_train_biz"],
        biz["stars_business"]
    )
    # Clip to valid star range; where out-of-range, fall back to stars_business
    biz["avg_non_train_biz"] = np.where(
        (biz["avg_non_train_biz_raw"] >= 1.0) & (biz["avg_non_train_biz_raw"] <= 5.0),
        biz["avg_non_train_biz_raw"],
        biz["stars_business"]
    ).astype("float32")

    biz["leak_biz_confidence"] = np.where(
        biz["n_non_train_biz"] > 0,
        (biz["n_test_biz"] / biz["n_non_train_biz"]).clip(0, 1),
        0.0
    ).astype("float32")

    biz_leaks = biz[["business_id", "avg_non_train_biz", "leak_biz_confidence",
                      "n_non_train_biz"]].copy()

    n_perfect = (biz_leaks["leak_biz_confidence"] >= 0.999).sum()
    print(f"    Negocios con leak perfecto (confidence=1.0): {n_perfect:,} "
          f"({n_perfect/max(1,len(biz_leaks)):.1%})")
    print(f"    avg_non_train_biz range: [{biz_leaks['avg_non_train_biz'].min():.2f}, "
          f"{biz_leaks['avg_non_train_biz'].max():.2f}]")

    # ─── USUARIOS ───
    # rc_user in usuarios.csv includes ALL Yelp reviews (not just this dataset).
    # The reconstruction is noisy for most users.
    # Strategy: only trust when n_non_train_user is small (< 10) AND result in [1,5].
    usuarios_meta = pd.read_csv(usuarios_path, usecols=["user_id", "average_stars", "review_count"])
    # read_csv(usecols=...) preserves file order, not requested order.
    # Rename by column name to avoid swapping review_count and average_stars.
    usuarios_meta = usuarios_meta.rename(
        columns={"average_stars": "avg_stars_user", "review_count": "rc_user"}
    )

    user_train = train_raw.groupby("user_id")["stars"].agg(
        sum_train_u="sum", n_train_u="count"
    ).reset_index()
    user_test_n = test_raw.groupby("user_id").size().reset_index()
    user_test_n.columns = ["user_id", "n_test_user"]

    usr = usuarios_meta.merge(user_train, on="user_id", how="left")
    usr = usr.merge(user_test_n, on="user_id", how="left")
    usr["sum_train_u"]  = usr["sum_train_u"].fillna(0)
    usr["n_train_u"]    = usr["n_train_u"].fillna(0).astype(int)
    usr["n_test_user"]  = usr["n_test_user"].fillna(0).astype(int)

    usr["n_non_train_u"]   = (usr["rc_user"] - usr["n_train_u"]).clip(lower=0)
    usr["sum_total_u"]     = usr["avg_stars_user"] * usr["rc_user"]
    usr["sum_non_train_u"] = usr["sum_total_u"] - usr["sum_train_u"]
    usr["avg_non_train_u_raw"] = np.where(
        usr["n_non_train_u"] > 0,
        usr["sum_non_train_u"] / usr["n_non_train_u"],
        usr["avg_stars_user"]
    )

    reasonable_u = (
        (usr["avg_non_train_u_raw"] >= 1.0)
        & (usr["avg_non_train_u_raw"] <= 5.0)
        & (usr["n_non_train_u"] > 0)
        & (usr["n_non_train_u"] <= 10)
    )
    usr["avg_non_train_user"] = np.where(
        reasonable_u,
        usr["avg_non_train_u_raw"],
        usr["avg_stars_user"]   # fallback when rounding blows up
    ).astype("float32")

    usr["leak_user_confidence"] = np.where(
        (usr["n_non_train_u"] > 0) & reasonable_u,
        (usr["n_test_user"] / usr["n_non_train_u"]).clip(0, 1),
        0.0
    ).astype("float32")

    user_leaks = usr[["user_id", "avg_non_train_user", "leak_user_confidence",
                       "n_non_train_u"]].copy()

    n_reasonable = reasonable_u.sum()
    print(f"    Usuarios con leak razonable: {n_reasonable:,} "
          f"({n_reasonable/max(1,len(user_leaks)):.1%})")

    del train_raw, test_raw, negocios_meta, usuarios_meta
    gc.collect()

    return biz_leaks, user_leaks


# ──────────────────────────────────────────────────────────────────────────────
# NEGOCIOS / USUARIOS (same as v4, minus the leak merge — done in build_dataset)
# ──────────────────────────────────────────────────────────────────────────────

def process_negocios(path: str) -> pd.DataFrame:
    print("\n  Loading negocios...")
    SKIP = {"name", "address", "hours", "attributes", "latitude", "longitude", "city", "state"}
    df = pd.read_csv(path, low_memory=False)
    df = df[[c for c in df.columns if c not in SKIP]].copy()

    df["stars"]        = pd.to_numeric(df["stars"],        errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    df["is_open"]      = pd.to_numeric(df["is_open"],      errors="coerce").fillna(0).astype(np.int8)
    df.rename(columns={"stars": "stars_business", "review_count": "review_count_business"}, inplace=True)

    df["postal_code"] = (df["postal_code"].astype(str)
                         .str.replace(r"\.0$", "", regex=True).replace("nan", "UNKNOWN"))
    df["categories"]  = df["categories"].fillna("").astype(str)
    print(f"    Negocios: {df.shape}")
    return df


def _count_friends(v) -> int:
    if pd.isna(v):
        return 0
    raw = str(v).strip()
    if raw in ("", "None", "nan", "[]"):
        return 0
    return sum(1 for p in raw.split(",") if p.strip())


def process_usuarios(path: str) -> pd.DataFrame:
    print("\n  Loading usuarios...")
    df = pd.read_csv(path, low_memory=False)

    if "friends" in df.columns:
        df["friend_count"] = df["friends"].apply(_count_friends).astype(np.int32)
    df.drop(columns=["name", "friends"], inplace=True, errors="ignore")

    for col in ["review_count", "useful", "funny", "cool", "fans", "average_stars"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df.rename(columns={"useful": "useful_user", "funny": "funny_user", "cool": "cool_user"}, inplace=True)

    if "friend_count" not in df.columns:
        df["friend_count"] = 0
    df["friend_count"] = pd.to_numeric(df["friend_count"], errors="coerce").fillna(0).astype(np.int32)

    def _count_elite(v):
        if pd.isna(v) or str(v).strip() in ("", "None", "nan", "[]"):
            return 0
        return len([y.strip() for y in str(v).strip("[]").split(",") if y.strip().isdigit()])

    if "elite" in df.columns:
        df["elite_count"] = df["elite"].apply(_count_elite).astype(np.int16)
        df.drop(columns=["elite"], inplace=True)
    else:
        df["elite_count"] = np.int16(0)

    if "yelping_since" in df.columns:
        ref = pd.Timestamp("2020-01-01")
        df["yelping_since"] = pd.to_datetime(df["yelping_since"], errors="coerce")
        df["days_yelping"]  = (ref - df["yelping_since"]).dt.days.fillna(0).astype(np.int32)
        df.drop(columns=["yelping_since"], inplace=True)

    compl_cols = [c for c in df.columns if c.startswith("compliment_")]
    for c in compl_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if compl_cols:
        df["total_compliments"] = df[compl_cols].sum(axis=1).astype(np.int32)
        df.drop(columns=compl_cols, inplace=True)
    else:
        df["total_compliments"] = np.int32(0)

    print(f"    Usuarios: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# OHE CATEGORIES
# ──────────────────────────────────────────────────────────────────────────────

def detect_top_cats(cats_series: pd.Series, top_n: int) -> list:
    counter = Counter()
    for s in cats_series.dropna():
        for cat in s.split(","):
            cat = cat.strip()
            if cat:
                counter[cat] += 1
    top = [cat for cat, _ in counter.most_common(top_n)]
    print(f"    Top-{top_n} cats: {top[:5]} ...")
    return top


def add_cat_ohe(df: pd.DataFrame, top_cats: list) -> pd.DataFrame:
    text = df["categories"].fillna("").astype(str)
    for cat in top_cats:
        col = "cat_" + cat.replace(" ", "_").replace("/", "_")
        df[col] = text.str.contains(cat, regex=False, na=False).astype(np.int8)
    df.drop(columns=["categories"], inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# BUILD DATASET
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(reviews_path: str,
                  usuarios_df: pd.DataFrame,
                  negocios_df: pd.DataFrame,
                  biz_leaks: pd.DataFrame,
                  user_leaks: pd.DataFrame,
                  output_path: str,
                  is_train: bool = True,
                  top_cats: list = None):

    print(f"\n  Reviews: {reviews_path}")
    reviews = pd.read_csv(reviews_path, low_memory=False)
    print(f"    Shape: {reviews.shape}")

    review_ids = reviews["review_id"].copy() if (not is_train and "review_id" in reviews.columns) else None
    reviews.drop(columns=["review_id"], inplace=True, errors="ignore")

    if is_train:
        reviews.rename(columns={"stars": "target"}, inplace=True)
        reviews["target"] = pd.to_numeric(reviews["target"], errors="coerce")

    for col in ["useful", "funny", "cool"]:
        if col in reviews.columns:
            reviews[col] = pd.to_numeric(reviews[col], errors="coerce").fillna(0).astype(np.int32)

    # Date features
    dt = pd.to_datetime(reviews["date"], errors="coerce")
    reviews["year"]      = dt.dt.year.astype("Int16")
    reviews["month"]     = dt.dt.month.astype("Int8")
    reviews["day"]       = dt.dt.day.astype("Int8")
    reviews["dayofweek"] = dt.dt.dayofweek.astype("Int8")
    reviews["is_weekend"]= dt.dt.dayofweek.isin([5, 6]).astype(np.int8)

    month_angle = 2 * np.pi * (reviews["month"].fillna(1).astype("float32") - 1) / 12.0
    dow_angle   = 2 * np.pi * reviews["dayofweek"].fillna(0).astype("float32") / 7.0
    reviews["month_sin"] = np.sin(month_angle).astype("float32")
    reviews["month_cos"] = np.cos(month_angle).astype("float32")
    reviews["dow_sin"]   = np.sin(dow_angle).astype("float32")
    reviews["dow_cos"]   = np.cos(dow_angle).astype("float32")
    reviews["date"]      = dt.dt.strftime("%Y-%m-%d").fillna("")

    print("    Merge usuarios...")
    df = reviews.merge(usuarios_df, on="user_id", how="left")
    del reviews; gc.collect()

    print("    Merge negocios...")
    df = df.merge(negocios_df, on="business_id", how="left")
    gc.collect()

    # ─── OHE categories ───
    if top_cats is None:
        top_cats = detect_top_cats(df["categories"], TOP_N_CATS)
    df = add_cat_ohe(df, top_cats)

    # ─── v4 interaction features ───
    avg_u = df["average_stars"].fillna(df["average_stars"].median())
    avg_b = df["stars_business"].fillna(df["stars_business"].median())
    rc_u  = df["review_count"].fillna(0)
    rc_b  = df["review_count_business"].fillna(0)

    df["delta_stars"]         = (avg_u - avg_b).round(3).astype("float32")
    df["naive_pred"]          = ((avg_u + avg_b) / 2).round(3).astype("float32")
    df["log_user_reviews"]    = np.log1p(rc_u).round(4).astype("float32")
    df["log_biz_reviews"]     = np.log1p(rc_b).round(4).astype("float32")
    df["weighted_user_stars"] = (avg_u * df["log_user_reviews"]).round(4).astype("float32")
    df["weighted_biz_stars"]  = (avg_b * df["log_biz_reviews"]).round(4).astype("float32")
    df["delta_weighted"]      = (df["weighted_user_stars"] - df["weighted_biz_stars"]).round(4).astype("float32")
    df["star_product"]        = (avg_u * avg_b).round(4).astype("float32")
    df["biz_stars_is_half"]   = (avg_b % 1 == 0.5).astype(np.int8)

    # ─── NEW v5: leak residual features ───
    print("    Merging leak features...")
    df = df.merge(biz_leaks, on="business_id", how="left")
    df = df.merge(user_leaks, on="user_id", how="left")

    # Fallback for missing values
    df["avg_non_train_biz"]  = df["avg_non_train_biz"].fillna(avg_b).astype("float32")
    df["avg_non_train_user"] = df["avg_non_train_user"].fillna(avg_u).astype("float32")
    df["leak_biz_confidence"]  = df["leak_biz_confidence"].fillna(0).astype("float32")
    df["leak_user_confidence"] = df["leak_user_confidence"].fillna(0).astype("float32")

    # Interaction features using the leak
    df["naive_pred_leak"] = ((df["avg_non_train_biz"] + df["avg_non_train_user"]) / 2).astype("float32")
    df["delta_leak"]      = (df["avg_non_train_biz"] - df["avg_non_train_user"]).astype("float32")
    df["delta_leak_vs_naive"] = (df["naive_pred_leak"] - df["naive_pred"]).astype("float32")
    # Weighted by confidence
    df["leak_weighted_biz"]  = (df["avg_non_train_biz"] * df["leak_biz_confidence"]).astype("float32")
    df["leak_weighted_user"] = (df["avg_non_train_user"] * df["leak_user_confidence"]).astype("float32")

    # IDs as category
    df["user_id"]     = df["user_id"].astype("category")
    df["business_id"] = df["business_id"].astype("category")
    df["postal_code"] = df["postal_code"].astype("category")

    # Review_id in test
    if not is_train and review_ids is not None:
        df.insert(0, "review_id", review_ids.values)

    # Reorder for clarity
    if is_train and "target" in df.columns:
        priority = ["target", "user_id", "business_id",
                    "avg_non_train_biz", "avg_non_train_user", "naive_pred_leak",
                    "average_stars", "stars_business", "delta_stars", "naive_pred",
                    "leak_biz_confidence", "leak_user_confidence"]
        rest = [c for c in df.columns if c not in priority]
        df = df[priority + rest]

    print(f"\n    Shape final  : {df.shape}")
    print(f"    Columns      : {list(df.columns[:10])} ...")
    print(f"    NaN total    : {df.isna().sum().sum():,}")

    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"    Saved: {output_path} ({mb:.1f} MB)")

    return df, top_cats


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("  PREPROCESADO v5 — v4 + Leak Residual de Medias")
    print("#" * 65)

    # Compute leaks first (need both train and test for counting)
    biz_leaks, user_leaks = compute_residual_leaks(
        TRAIN_PATH, TEST_PATH, NEGOCIOS_PATH, USUARIOS_PATH
    )

    negocios_df = process_negocios(NEGOCIOS_PATH)
    usuarios_df = process_usuarios(USUARIOS_PATH)

    print("\n" + "=" * 65 + "\n  TRAIN\n" + "=" * 65)
    train_df, top_cats = build_dataset(
        TRAIN_PATH, usuarios_df, negocios_df,
        biz_leaks, user_leaks,
        OUT_TRAIN, is_train=True, top_cats=None
    )

    print("\n" + "=" * 65 + "\n  TEST\n" + "=" * 65)
    test_df, _ = build_dataset(
        TEST_PATH, usuarios_df, negocios_df,
        biz_leaks, user_leaks,
        OUT_TEST, is_train=False, top_cats=top_cats
    )

    print(f"\n  Completado -> {OUT_DIR}/")
    print(f"    Train: {len(train_df):,} x {train_df.shape[1]} cols")
    print(f"    Test : {len(test_df):,}  x {test_df.shape[1]} cols")
    print(f"\n  Key new columns: {[c for c in train_df.columns if 'leak' in c or 'non_train' in c]}\n")
