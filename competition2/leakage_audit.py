"""
Leakage audit for the Yelp competition setup.

This script does not try to engineer new leakage features.
It quantifies where shortcut-like signal could realistically come from:

1. Metadata coverage:
   - Are test users/businesses present in usuarios.csv / negocios.csv?
2. Labeled interaction coverage:
   - Are test users/businesses present in train_reviews.csv?
   - How many test rows have known users, known businesses, or seen pairs?
3. Memorization readiness:
   - How many test rows belong to users/businesses with enough train history?
4. Duplicate / overlap checks:
   - Exact key overlap between train/test on plausible review identifiers.
5. Interaction repetition:
   - How often the same (user_id, business_id) pair repeats in train.

Usage:
    python competition2/leakage_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def metadata_coverage(
    train_reviews: pd.DataFrame,
    test_reviews: pd.DataFrame,
    users: pd.DataFrame,
    businesses: pd.DataFrame,
) -> None:
    section("1. Metadata Coverage")

    train_users_profile = set(users["user_id"].unique())
    train_biz_profile = set(businesses["business_id"].unique())
    test_users = set(test_reviews["user_id"].unique())
    test_biz = set(test_reviews["business_id"].unique())

    row_user_profile = test_reviews["user_id"].isin(train_users_profile)
    row_biz_profile = test_reviews["business_id"].isin(train_biz_profile)

    print("This is merge coverage against usuarios.csv / negocios.csv.")
    print("Useful for feature availability, not enough for collaborative filtering.")
    print()
    print(
        f"Test unique users present in usuarios.csv: "
        f"{pct(len(test_users & train_users_profile) / len(test_users))}"
    )
    print(f"New test users missing from usuarios.csv: {len(test_users - train_users_profile):,}")
    print(
        f"Test unique businesses present in negocios.csv: "
        f"{pct(len(test_biz & train_biz_profile) / len(test_biz))}"
    )
    print(f"New test businesses missing from negocios.csv: {len(test_biz - train_biz_profile):,}")
    print()
    print(f"Test rows with user metadata available: {pct(row_user_profile.mean())}")
    print(f"Test rows with business metadata available: {pct(row_biz_profile.mean())}")
    print(f"Test rows with both metadata blocks available: {pct((row_user_profile & row_biz_profile).mean())}")


def labeled_interaction_coverage(train_reviews: pd.DataFrame, test_reviews: pd.DataFrame) -> None:
    section("2. Labeled Interaction Coverage")

    train_users = set(train_reviews["user_id"].unique())
    train_biz = set(train_reviews["business_id"].unique())
    test_users = set(test_reviews["user_id"].unique())
    test_biz = set(test_reviews["business_id"].unique())

    row_known_user = test_reviews["user_id"].isin(train_users)
    row_known_biz = test_reviews["business_id"].isin(train_biz)
    row_both_known = row_known_user & row_known_biz

    train_pairs = set(zip(train_reviews["user_id"], train_reviews["business_id"]))
    test_pairs = pd.Series(list(zip(test_reviews["user_id"], test_reviews["business_id"])))
    row_seen_pair = test_pairs.isin(train_pairs)

    print("This is the relevant view for CF / SVD++ / ID embeddings.")
    print()
    print(
        f"Test unique users seen in train_reviews.csv: "
        f"{pct(len(test_users & train_users) / len(test_users))}"
    )
    print(f"New test users vs labeled train interactions: {len(test_users - train_users):,}")
    print(
        f"Test unique businesses seen in train_reviews.csv: "
        f"{pct(len(test_biz & train_biz) / len(test_biz))}"
    )
    print(f"New test businesses vs labeled train interactions: {len(test_biz - train_biz):,}")
    print()
    print(f"Test rows with known user: {pct(row_known_user.mean())}")
    print(f"Test rows with known business: {pct(row_known_biz.mean())}")
    print(f"Test rows with both user and business known: {pct(row_both_known.mean())}")
    print(f"Test rows where exact (user_id, business_id) pair was already seen: {pct(row_seen_pair.mean())}")


def memorization_readiness(train_reviews: pd.DataFrame, test_reviews: pd.DataFrame) -> None:
    section("3. Memorization Readiness")

    user_counts = train_reviews["user_id"].value_counts()
    biz_counts = train_reviews["business_id"].value_counts()

    print("Share of test rows attached to users/businesses with enough labeled history in train.")
    print()

    for threshold in [1, 2, 3, 5, 10]:
        mask = test_reviews["user_id"].map(user_counts).fillna(0) >= threshold
        print(f"Test rows with user having >= {threshold:>2} train reviews: {pct(mask.mean())}")

    print()

    for threshold in [1, 2, 3, 5, 10, 20, 50]:
        mask = test_reviews["business_id"].map(biz_counts).fillna(0) >= threshold
        print(f"Test rows with business having >= {threshold:>2} train reviews: {pct(mask.mean())}")


def interaction_repetition(train_reviews: pd.DataFrame) -> None:
    section("4. Repeated Interactions Inside Train")

    pair_counts = train_reviews.groupby(["user_id", "business_id"]).size()
    repeated_pairs = pair_counts[pair_counts > 1]

    print(f"Train rows: {len(train_reviews):,}")
    print(f"Unique (user_id, business_id) pairs in train: {pair_counts.shape[0]:,}")
    print(f"Repeated train pairs: {len(repeated_pairs):,}")
    print(f"Train rows belonging to repeated pairs: {int(repeated_pairs.sum()):,}")

    if len(repeated_pairs) > 0:
        print()
        print("Top repeated train pairs:")
        for (user_id, business_id), count in repeated_pairs.sort_values(ascending=False).head(10).items():
            print(f"  user={user_id}  business={business_id}  count={count}")


def smoothed_mean(sum_series: pd.Series, count_series: pd.Series, global_mean: float, alpha: float) -> pd.Series:
    return (sum_series + alpha * global_mean) / (count_series + alpha)


def assign_pair_values(
    frame: pd.DataFrame,
    pair_series: pd.Series,
    default: pd.Series,
) -> pd.Series:
    pair_index = pd.MultiIndex.from_frame(frame[["user_id", "business_id"]])
    values = pair_series.reindex(pair_index)
    return values.to_numpy(dtype=float, na_value=np.nan) if hasattr(values, "to_numpy") else values.fillna(default).to_numpy()


def oof_entity_benchmarks(train_reviews: pd.DataFrame, n_splits: int, random_state: int) -> None:
    section("5. OOF Entity-Memory Benchmarks")

    df = train_reviews[["user_id", "business_id", "stars"]].copy()
    y = df["stars"].astype(float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    preds = {
        "global_mean": np.zeros(len(df), dtype=float),
        "user_mean": np.zeros(len(df), dtype=float),
        "business_mean": np.zeros(len(df), dtype=float),
        "pair_mean": np.zeros(len(df), dtype=float),
        "user_business_blend": np.zeros(len(df), dtype=float),
    }
    seen_flags = {
        "user_mean": np.zeros(len(df), dtype=bool),
        "business_mean": np.zeros(len(df), dtype=bool),
        "pair_mean": np.zeros(len(df), dtype=bool),
    }

    alpha = 10.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y.astype(int)), start=1):
        tr = df.iloc[train_idx]
        va = df.iloc[val_idx]

        global_mean = float(tr["stars"].mean())

        user_agg = tr.groupby("user_id")["stars"].agg(["sum", "count"])
        biz_agg = tr.groupby("business_id")["stars"].agg(["sum", "count"])
        pair_agg = tr.groupby(["user_id", "business_id"])["stars"].agg(["sum", "count"])

        user_mean = smoothed_mean(user_agg["sum"], user_agg["count"], global_mean, alpha)
        biz_mean = smoothed_mean(biz_agg["sum"], biz_agg["count"], global_mean, alpha)
        pair_mean = smoothed_mean(pair_agg["sum"], pair_agg["count"], global_mean, alpha)

        user_pred = va["user_id"].map(user_mean)
        biz_pred = va["business_id"].map(biz_mean)

        pair_index = pd.MultiIndex.from_frame(va[["user_id", "business_id"]])
        pair_pred = pair_mean.reindex(pair_index)

        user_count = va["user_id"].map(user_agg["count"]).fillna(0.0)
        biz_count = va["business_id"].map(biz_agg["count"]).fillna(0.0)

        user_filled = user_pred.fillna(global_mean)
        biz_filled = biz_pred.fillna(global_mean)
        pair_filled = pair_pred.fillna(np.nan)

        blend_weight_user = np.log1p(user_count)
        blend_weight_biz = np.log1p(biz_count)
        blend_den = blend_weight_user + blend_weight_biz
        blend_num = (
            user_filled.to_numpy() * blend_weight_user.to_numpy()
            + biz_filled.to_numpy() * blend_weight_biz.to_numpy()
        )
        blend_base = np.full(len(va), global_mean, dtype=float)
        positive_den = blend_den.to_numpy() > 0
        blend_base[positive_den] = blend_num[positive_den] / blend_den.to_numpy()[positive_den]
        blend_pred = np.where(~pd.isna(pair_filled), pair_filled.to_numpy(), blend_base)

        preds["global_mean"][val_idx] = global_mean
        preds["user_mean"][val_idx] = user_filled.to_numpy()
        preds["business_mean"][val_idx] = biz_filled.to_numpy()
        preds["pair_mean"][val_idx] = np.where(~pd.isna(pair_filled), pair_filled.to_numpy(), biz_filled.to_numpy())
        preds["user_business_blend"][val_idx] = blend_pred

        seen_flags["user_mean"][val_idx] = user_pred.notna().to_numpy()
        seen_flags["business_mean"][val_idx] = biz_pred.notna().to_numpy()
        seen_flags["pair_mean"][val_idx] = (~pd.isna(pair_filled)).to_numpy()

        print(
            f"Fold {fold}/{n_splits}: "
            f"global={global_mean:.4f}  "
            f"user_seen={pct(user_pred.notna().mean())}  "
            f"biz_seen={pct(biz_pred.notna().mean())}  "
            f"pair_seen={pct((~pd.isna(pair_filled)).mean())}"
        )

    print()
    print("OOF MAE on train_reviews.csv")
    print("These are diagnostic baselines, not leaderboard estimates.")
    print()

    for name, pred in preds.items():
        mae = mean_absolute_error(y, pred)
        line = f"{name:20s}  MAE={mae:.5f}"
        if name in seen_flags:
            line += f"  seen_coverage={pct(seen_flags[name].mean())}"
        print(line)


def duplicate_overlap_checks(train_reviews: pd.DataFrame, test_reviews: pd.DataFrame, max_samples: int) -> None:
    section("6. Train/Test Overlap Checks")

    key_sets = [
        ["user_id", "business_id", "date"],
        ["user_id", "business_id", "useful", "funny", "cool", "date"],
        ["business_id", "date", "useful", "funny", "cool"],
    ]

    print(f"Exact raw duplicate rows in train: {int(train_reviews.duplicated().sum()):,}")
    print(f"Exact raw duplicate rows in test : {int(test_reviews.duplicated().sum()):,}")
    print()

    for keys in key_sets:
        train_keys = train_reviews[keys].drop_duplicates()
        test_keys = test_reviews[keys].drop_duplicates()
        overlap = test_keys.merge(train_keys, on=keys, how="inner")
        print(f"Overlap on keys {keys}: {len(overlap):,} unique matches")

        if len(overlap) and max_samples > 0:
            print("Sample matches:")
            print(overlap.head(max_samples).to_string(index=False))
            print()

    suspicious_keys = ["business_id", "date", "useful", "funny", "cool"]
    suspicious = test_reviews.merge(
        train_reviews,
        on=suspicious_keys,
        how="inner",
        suffixes=("_test", "_train"),
    )

    if not suspicious.empty:
        print("Rows matching on business/date/votes but not on user_id can indicate partial duplication or coincidence:")
        cols = [
            "review_id_test",
            "user_id_test",
            "review_id_train",
            "user_id_train",
            *suspicious_keys,
        ]
        print(suspicious[cols].head(max_samples).to_string(index=False))


def interpretation_notes() -> None:
    section("7. Interpretation Notes")

    print("What these diagnostics mean:")
    print("- Metadata availability is not leakage by itself. It only means merges are possible.")
    print("- Known users/businesses in train_reviews matter for CF, ID embeddings, and target encodings.")
    print("- Seen (user_id, business_id) pairs are the closest thing to pair memorization.")
    print("- Exact overlaps on review-like keys are more suspicious than broad entity overlap.")
    print("- If you build target-derived features, use OOF / expanding logic to avoid row leakage.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit potential leakage pathways.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing train_reviews.csv, test_reviews.csv, usuarios.csv, negocios.csv",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="How many suspicious overlap samples to print",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of OOF folds for entity-memory benchmarks",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for OOF folds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent / args.data_dir

    train_path = base_dir / "train_reviews.csv"
    test_path = base_dir / "test_reviews.csv"
    users_path = base_dir / "usuarios.csv"
    biz_path = base_dir / "negocios.csv"

    for path in [train_path, test_path, users_path, biz_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    train_reviews = read_csv(train_path)
    test_reviews = read_csv(test_path)
    users = read_csv(users_path, usecols=["user_id"])
    businesses = read_csv(biz_path, usecols=["business_id"])

    section("0. Dataset Summary")
    print(f"Train reviews: {len(train_reviews):,}")
    print(f"Test reviews : {len(test_reviews):,}")
    print(f"Users table   : {len(users):,}")
    print(f"Business table: {len(businesses):,}")

    metadata_coverage(train_reviews, test_reviews, users, businesses)
    labeled_interaction_coverage(train_reviews, test_reviews)
    memorization_readiness(train_reviews, test_reviews)
    interaction_repetition(train_reviews)
    oof_entity_benchmarks(train_reviews, n_splits=args.n_splits, random_state=args.random_state)
    duplicate_overlap_checks(train_reviews, test_reviews, max_samples=args.max_samples)
    interpretation_notes()


if __name__ == "__main__":
    main()
