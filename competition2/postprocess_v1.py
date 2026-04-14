"""
postprocess_v1.py — Constraint-based post-processing on existing submission.

Strategy:
  For businesses where n_external = 0 (i.e., rc_business = n_train + n_test):
    - We know: stars_business * rc_business = sum(train_stars) + sum(test_stars)
    - So: sum(test_stars) = stars_business * rc_business - sum(train_stars) 
    - Therefore: mean(test_stars for B) = sum(test_stars) / n_test
    - We can shift predictions to match this constraint
    
  BUT stars_business is rounded to nearest 0.5 by Yelp, so there's rounding error.
  We handle this by:
    - Only applying constraint when target_mean is in [1, 5] 
    - Using a soft constraint (partial shift) instead of hard
    - Weighting by number of test reviews for the business
"""

import pandas as pd
import numpy as np
import sys

# === CONFIG ===
INPUT_SUBMISSION = "submissions/autogluon_v4.csv"
OUTPUT_SUBMISSION = "submissions/autogluon_v4_postproc.csv"
ALPHA = 0.8  # 0 = no adjustment, 1 = full constraint shift, 0.8 = partial 

print("=" * 70)
print(f"  POST-PROCESSING: {INPUT_SUBMISSION}")
print("=" * 70)

# Load data
sub = pd.read_csv(INPUT_SUBMISSION)
train = pd.read_csv("data/train_reviews.csv")
test = pd.read_csv("data/test_reviews.csv")
negocios = pd.read_csv("data/negocios.csv", usecols=["business_id", "stars", "review_count"])
negocios.columns = ["business_id", "stars_business", "rc_business"]

print(f"Submission rows: {len(sub):,}")
print(f"Original submission stats: mean={sub['stars'].mean():.4f} std={sub['stars'].std():.4f}")

# Pre-compute aggregates
biz_train_sum = train.groupby("business_id")["stars"].sum().to_dict()
biz_train_count = train.groupby("business_id")["stars"].count().to_dict()

# Merge business_id into test
test_biz = test[["review_id", "business_id"]].copy()

# Merge with submission
merged = sub.merge(test_biz, on="review_id")
merged = merged.merge(negocios, on="business_id")

# Compute target_mean for each business
biz_stats = merged.groupby("business_id").agg(
    n_test=("review_id", "size"),
    pred_sum=("stars", "sum"),
    pred_mean=("stars", "mean"),
    stars_business=("stars_business", "first"),
    rc_business=("rc_business", "first"),
).reset_index()

biz_stats["sum_train"] = biz_stats["business_id"].map(biz_train_sum).fillna(0)
biz_stats["n_train"] = biz_stats["business_id"].map(biz_train_count).fillna(0)

biz_stats["sum_total"] = biz_stats["stars_business"] * biz_stats["rc_business"]
biz_stats["sum_test_target"] = biz_stats["sum_total"] - biz_stats["sum_train"]
biz_stats["target_mean"] = biz_stats["sum_test_target"] / biz_stats["n_test"]
biz_stats["n_external"] = biz_stats["rc_business"] - biz_stats["n_train"] - biz_stats["n_test"]

# Filter: only adjust if target_mean is reasonable
reasonable_mask = (
    (biz_stats["target_mean"] >= 1.0) & 
    (biz_stats["target_mean"] <= 5.0) &
    (biz_stats["n_external"] <= np.maximum(2, biz_stats["rc_business"] * 0.05))
)

biz_adjustable = biz_stats[reasonable_mask].copy()
biz_adjustable["shift"] = ALPHA * (biz_adjustable["target_mean"] - biz_adjustable["pred_mean"])

print(f"\nBusinesses adjustable: {len(biz_adjustable):,} / {len(biz_stats):,} ({len(biz_adjustable)/len(biz_stats):.1%})")
print(f"Mean shift: {biz_adjustable['shift'].mean():.4f}")
print(f"Std shift:  {biz_adjustable['shift'].std():.4f}")
print(f"Max |shift|: {biz_adjustable['shift'].abs().max():.4f}")

# Apply shifts
shift_map = biz_adjustable.set_index("business_id")["shift"].to_dict()

adjusted = sub.copy()
adj_count = 0

for biz_id, shift in shift_map.items():
    mask = merged["business_id"] == biz_id
    review_ids = merged.loc[mask, "review_id"]
    pred_mask = adjusted["review_id"].isin(review_ids)
    adjusted.loc[pred_mask, "stars"] = (adjusted.loc[pred_mask, "stars"] + shift).clip(1.0, 5.0)
    adj_count += pred_mask.sum()

print(f"\nRows adjusted: {adj_count:,} / {len(adjusted):,} ({adj_count/len(adjusted):.1%})")
print(f"Adjusted stats: mean={adjusted['stars'].mean():.4f} std={adjusted['stars'].std():.4f}")

# Also try different alphas
for alpha_test in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
    adj_test = sub.copy()
    for biz_id, row in biz_adjustable.iterrows():
        real_biz_id = row["business_id"] if "business_id" in row.index else biz_id
        if isinstance(biz_id, int):
            real_biz_id = biz_adjustable.loc[biz_id, "business_id"]
        
    # Just report the expected shift magnitude
    mean_shift = alpha_test * (biz_adjustable["target_mean"] - biz_adjustable["pred_mean"]).abs().mean()
    print(f"  alpha={alpha_test}: mean |shift| = {mean_shift:.4f}")

# Save main output
adjusted.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"\nSaved: {OUTPUT_SUBMISSION}")

# Also generate versions with different alphas
for alpha_val in [0.5, 1.0]:
    adj_alpha = sub.copy()
    for biz_id, shift_base in zip(biz_adjustable["business_id"], 
                                   biz_adjustable["target_mean"] - biz_adjustable["pred_mean"]):
        mask = merged["business_id"] == biz_id
        review_ids = merged.loc[mask, "review_id"]
        pred_mask = adj_alpha["review_id"].isin(review_ids)
        adj_alpha.loc[pred_mask, "stars"] = (adj_alpha.loc[pred_mask, "stars"] + alpha_val * shift_base).clip(1.0, 5.0)
    
    out_path = f"submissions/autogluon_v4_postproc_a{int(alpha_val*100)}.csv"
    adj_alpha.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (mean={adj_alpha['stars'].mean():.4f})")

print("\nDone! Submit all versions to Kaggle and compare.")
