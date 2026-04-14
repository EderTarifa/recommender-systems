"""
preprocess_autogluon_definitive.py
==================================
Preprocessing definitivo para AutoGluon:

  - base de v4 (features leaky fuertes)
  - correccion de friend_count
  - fecha con senal temporal explicita
  - cf_predict con SVD++ (surprise)
      * OOF en train para evitar leakage fila a fila
      * fit completo para predecir test

Outputs:
  data_definitive_cf/train_ag.parquet
  data_definitive_cf/test_ag.parquet
"""

import gc
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SRC_DIR = "data"
NEGOCIOS_PATH = os.path.join(SRC_DIR, "negocios.csv")
USUARIOS_PATH = os.path.join(SRC_DIR, "usuarios.csv")
TRAIN_PATH = os.path.join(SRC_DIR, "train_reviews.csv")
TEST_PATH = os.path.join(SRC_DIR, "test_reviews.csv")

OUT_DIR = "data_definitive_cf"
OUT_TRAIN = os.path.join(OUT_DIR, "train_ag.parquet")
OUT_TEST = os.path.join(OUT_DIR, "test_ag.parquet")

os.makedirs(OUT_DIR, exist_ok=True)

TOP_N_CATS = 20
CF_N_SPLITS = 5
CF_RANDOM_STATE = 42
CF_PARAMS = {
    "n_factors": 48,
    "n_epochs": 12,
    "lr_all": 0.007,
    "reg_all": 0.02,
    "random_state": CF_RANDOM_STATE,
    "verbose": True,
}


def process_negocios(path: str) -> pd.DataFrame:
    print("\n  Cargando negocios...")
    skip_cols = {"name", "address", "hours", "attributes", "latitude", "longitude", "city", "state"}
    df_raw = pd.read_csv(path, low_memory=False)
    keep = [c for c in df_raw.columns if c not in skip_cols]
    df = df_raw[keep].copy()
    del df_raw
    gc.collect()

    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    df["is_open"] = pd.to_numeric(df["is_open"], errors="coerce").fillna(0).astype(np.int8)
    df.rename(columns={"stars": "stars_business", "review_count": "review_count_business"}, inplace=True)

    df["postal_code"] = (
        df["postal_code"].astype(str).str.replace(r"\.0$", "", regex=True).replace("nan", "UNKNOWN")
    )
    df["categories"] = df["categories"].fillna("").astype(str)

    print(f"    Negocios: {df.shape}")
    return df


def _count_friends(value) -> int:
    if pd.isna(value):
        return 0
    raw = str(value).strip()
    if raw in ("", "None", "nan", "[]"):
        return 0
    return sum(1 for part in raw.split(",") if part.strip())


def process_usuarios(path: str) -> pd.DataFrame:
    print("\n  Cargando usuarios...")
    df = pd.read_csv(path, low_memory=False)
    print(f"    Shape raw: {df.shape}")

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

    def count_elite(value) -> int:
        if pd.isna(value) or str(value).strip() in ("", "None", "nan", "[]"):
            return 0
        return len([y.strip() for y in str(value).strip("[]").split(",") if y.strip().isdigit()])

    if "elite" in df.columns:
        df["elite_count"] = df["elite"].apply(count_elite).astype(np.int16)
        df.drop(columns=["elite"], inplace=True)
    else:
        df["elite_count"] = np.int16(0)

    if "yelping_since" in df.columns:
        ref = pd.Timestamp("2020-01-01")
        df["yelping_since"] = pd.to_datetime(df["yelping_since"], errors="coerce")
        df["days_yelping"] = (ref - df["yelping_since"]).dt.days.fillna(0).astype(np.int32)
        df.drop(columns=["yelping_since"], inplace=True)

    compliment_cols = [c for c in df.columns if c.startswith("compliment_")]
    for col in compliment_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if compliment_cols:
        df["total_compliments"] = df[compliment_cols].sum(axis=1).astype(np.int32)
        df.drop(columns=compliment_cols, inplace=True)
    else:
        df["total_compliments"] = np.int32(0)

    print(f"    Usuarios: {df.shape}")
    return df


def detect_top_cats(cats_series: pd.Series, top_n: int) -> list:
    counter = Counter()
    for value in cats_series.dropna():
        for cat in value.split(","):
            cat = cat.strip()
            if cat:
                counter[cat] += 1
    top = [cat for cat, _ in counter.most_common(top_n)]
    print(f"    Top-{top_n} cats: {top[:5]} ... (total {len(top)})")
    return top


def add_cat_ohe(df: pd.DataFrame, top_cats: list) -> pd.DataFrame:
    text = df["categories"].fillna("").astype(str)
    for cat in top_cats:
        col = "cat_" + cat.replace(" ", "_").replace("/", "_")
        df[col] = text.str.contains(cat, regex=False, na=False).astype(np.int8)
    df.drop(columns=["categories"], inplace=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    review_date = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = review_date.dt.year.astype("Int16")
    df["month"] = review_date.dt.month.astype("Int8")
    df["day"] = review_date.dt.day.astype("Int8")
    df["dayofweek"] = review_date.dt.dayofweek.astype("Int8")
    df["is_weekend"] = review_date.dt.dayofweek.isin([5, 6]).astype(np.int8)

    month_angle = 2 * np.pi * (df["month"].fillna(1).astype("float32") - 1) / 12.0
    dow_angle = 2 * np.pi * df["dayofweek"].fillna(0).astype("float32") / 7.0
    df["month_sin"] = np.sin(month_angle).astype("float32")
    df["month_cos"] = np.cos(month_angle).astype("float32")
    df["dow_sin"] = np.sin(dow_angle).astype("float32")
    df["dow_cos"] = np.cos(dow_angle).astype("float32")

    df["date"] = review_date.dt.strftime("%Y-%m-%d").fillna("")
    return df


def _fit_surprise_svdpp(ratings: pd.DataFrame):
    try:
        from surprise import Dataset, Reader, SVDpp
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar surprise. Ejecuta este preprocess en el entorno del notebook donde tienes "
            "instalado scikit-surprise."
        ) from exc

    reader = Reader(rating_scale=(1, 5))
    surprise_df = ratings[["user_id", "business_id", "target"]].copy()
    surprise_df["user_id"] = surprise_df["user_id"].astype(str)
    surprise_df["business_id"] = surprise_df["business_id"].astype(str)
    data = Dataset.load_from_df(surprise_df, reader)
    trainset = data.build_full_trainset()
    algo = SVDpp(**CF_PARAMS)
    algo.fit(trainset)
    return algo


def _predict_surprise(algo, pairs_df: pd.DataFrame) -> np.ndarray:
    preds = np.empty(len(pairs_df), dtype="float32")
    for idx, row in enumerate(pairs_df.itertuples(index=False), start=0):
        preds[idx] = algo.predict(str(row.user_id), str(row.business_id)).est
    return preds


def build_cf_features(train_reviews: pd.DataFrame, test_reviews: pd.DataFrame):
    print("\n  Construyendo cf_predict con SVD++...")
    train_cf = train_reviews[["review_id", "user_id", "business_id", "target"]].copy()
    test_cf = test_reviews[["review_id", "user_id", "business_id"]].copy()

    known_users = set(train_cf["user_id"].astype(str).unique())
    known_businesses = set(train_cf["business_id"].astype(str).unique())

    oof_pred = np.zeros(len(train_cf), dtype="float32")
    oof_known_user = np.zeros(len(train_cf), dtype="int8")
    oof_known_business = np.zeros(len(train_cf), dtype="int8")
    skf = StratifiedKFold(n_splits=CF_N_SPLITS, shuffle=True, random_state=CF_RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_cf, train_cf["target"].astype(int)), start=1):
        print(f"    Fold {fold}/{CF_N_SPLITS}...")
        tr = train_cf.iloc[tr_idx][["user_id", "business_id", "target"]]
        va = train_cf.iloc[va_idx][["user_id", "business_id"]]
        fold_users = set(tr["user_id"].astype(str).unique())
        fold_businesses = set(tr["business_id"].astype(str).unique())
        algo = _fit_surprise_svdpp(tr)
        oof_pred[va_idx] = _predict_surprise(algo, va)
        oof_known_user[va_idx] = va["user_id"].astype(str).isin(fold_users).astype("int8")
        oof_known_business[va_idx] = va["business_id"].astype(str).isin(fold_businesses).astype("int8")

    print("    Fit completo para test...")
    full_algo = _fit_surprise_svdpp(train_cf[["user_id", "business_id", "target"]])
    test_pred = _predict_surprise(full_algo, test_cf[["user_id", "business_id"]])

    train_out = pd.DataFrame(
        {
            "review_id": train_cf["review_id"].values,
            "cf_predict": np.clip(oof_pred, 1, 5).astype("float32"),
            "cf_known_user": oof_known_user,
            "cf_known_business": oof_known_business,
        }
    )
    test_out = pd.DataFrame(
        {
            "review_id": test_cf["review_id"].values,
            "cf_predict": np.clip(test_pred, 1, 5).astype("float32"),
            "cf_known_user": test_cf["user_id"].astype(str).isin(known_users).astype("int8"),
            "cf_known_business": test_cf["business_id"].astype(str).isin(known_businesses).astype("int8"),
        }
    )

    print(
        "    Cobertura CF test | "
        f"user={test_out['cf_known_user'].mean():.4%} | "
        f"business={test_out['cf_known_business'].mean():.4%}"
    )
    return train_out, test_out


def build_dataset(
    reviews_path: str,
    usuarios_df: pd.DataFrame,
    negocios_df: pd.DataFrame,
    output_path: str,
    cf_df: pd.DataFrame,
    is_train: bool = True,
    top_cats: list = None,
):
    print(f"\n  Reviews: {reviews_path}")
    reviews = pd.read_csv(reviews_path, low_memory=False)
    print(f"    Shape: {reviews.shape}")

    if is_train:
        reviews.rename(columns={"stars": "target"}, inplace=True)
        reviews["target"] = pd.to_numeric(reviews["target"], errors="coerce")

    for col in ["useful", "funny", "cool"]:
        if col in reviews.columns:
            reviews[col] = pd.to_numeric(reviews[col], errors="coerce").fillna(0).astype(np.int32)

    reviews = add_time_features(reviews)
    reviews = reviews.merge(cf_df, on="review_id", how="left")

    print("    Merge usuarios...")
    df = reviews.merge(usuarios_df, on="user_id", how="left")
    del reviews
    gc.collect()

    print("    Merge negocios...")
    df = df.merge(negocios_df, on="business_id", how="left")
    gc.collect()

    if top_cats is None:
        top_cats = detect_top_cats(df["categories"], TOP_N_CATS)
    df = add_cat_ohe(df, top_cats)

    avg_u = df["average_stars"].fillna(df["average_stars"].median())
    avg_b = df["stars_business"].fillna(df["stars_business"].median())
    rc_u = df["review_count"].fillna(0)
    rc_b = df["review_count_business"].fillna(0)

    df["delta_stars"] = (avg_u - avg_b).round(3).astype("float32")
    df["naive_pred"] = ((avg_u + avg_b) / 2).round(3).astype("float32")
    df["log_user_reviews"] = np.log1p(rc_u).round(4).astype("float32")
    df["log_biz_reviews"] = np.log1p(rc_b).round(4).astype("float32")
    df["weighted_user_stars"] = (avg_u * df["log_user_reviews"]).round(4).astype("float32")
    df["weighted_biz_stars"] = (avg_b * df["log_biz_reviews"]).round(4).astype("float32")
    df["delta_weighted"] = (df["weighted_user_stars"] - df["weighted_biz_stars"]).round(4).astype("float32")
    df["star_product"] = (avg_u * avg_b).round(4).astype("float32")
    df["biz_stars_is_half"] = (avg_b % 1 == 0.5).astype(np.int8)
    df["user_biz_review_ratio"] = (df["log_user_reviews"] - df["log_biz_reviews"]).astype("float32")
    df["cf_minus_naive"] = (df["cf_predict"] - df["naive_pred"]).astype("float32")
    df["cf_times_naive"] = (df["cf_predict"] * df["naive_pred"]).astype("float32")

    df["user_id"] = df["user_id"].astype("category")
    df["business_id"] = df["business_id"].astype("category")
    df["postal_code"] = df["postal_code"].astype("category")

    if is_train and "target" in df.columns:
        priority = [
            "target",
            "user_id",
            "business_id",
            "average_stars",
            "stars_business",
            "naive_pred",
            "cf_predict",
            "cf_minus_naive",
            "friend_count",
            "dayofweek",
            "is_weekend",
        ]
        rest = [c for c in df.columns if c not in priority]
        df = df[priority + rest]

    print(f"\n    Shape final    : {df.shape}")
    print(f"    Columnas       : {list(df.columns[:14])} ...")
    print(f"    NaN totales    : {df.isna().sum().sum():,}")

    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"    Guardado: {output_path} ({mb:.1f} MB)")

    return df, top_cats


if __name__ == "__main__":
    print("\n" + "█" * 65)
    print("  PREPROCESADO DEFINITIVO — v4 + tiempo + friend_count + cf_predict")
    print("█" * 65)

    train_reviews = pd.read_csv(TRAIN_PATH, usecols=["review_id", "user_id", "business_id", "stars"])
    train_reviews.rename(columns={"stars": "target"}, inplace=True)
    test_reviews = pd.read_csv(TEST_PATH, usecols=["review_id", "user_id", "business_id"])
    train_cf, test_cf = build_cf_features(train_reviews, test_reviews)
    del train_reviews, test_reviews
    gc.collect()

    negocios_df = process_negocios(NEGOCIOS_PATH)
    usuarios_df = process_usuarios(USUARIOS_PATH)

    print("\n" + "═" * 65 + "\n  TRAIN\n" + "═" * 65)
    train_df, top_cats = build_dataset(
        TRAIN_PATH,
        usuarios_df,
        negocios_df,
        OUT_TRAIN,
        cf_df=train_cf,
        is_train=True,
        top_cats=None,
    )

    print("\n" + "═" * 65 + "\n  TEST\n" + "═" * 65)
    test_df, _ = build_dataset(
        TEST_PATH,
        usuarios_df,
        negocios_df,
        OUT_TEST,
        cf_df=test_cf,
        is_train=False,
        top_cats=top_cats,
    )

    print(f"\n  Completado -> {OUT_DIR}/")
    print(f"     Train: {len(train_df):,} x {train_df.shape[1]} cols")
    print(f"     Test : {len(test_df):,} x {test_df.shape[1]} cols")
    print(f"\n  Columnas: {list(train_df.columns)}\n")
