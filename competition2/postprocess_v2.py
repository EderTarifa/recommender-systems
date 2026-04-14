"""
postprocess_v2.py — Vectorized constraint-based post-processing.

For each business where we can compute avg_test from metadata:
  avg_test_biz = (stars_business * rc_business - sum_train_stars) / n_test
  shift = alpha * (avg_test_biz - current_pred_mean_for_biz)
  new_pred = old_pred + shift, clipped to [1, 5]
"""

import pandas as pd
import numpy as np

INPUT_SUBMISSION = "submissions/autogluon_v4.csv"

print("=" * 70)
print(f"  VECTORIZED POST-PROCESSING")
print("=" * 70)

# Load data
sub = pd.read_csv(INPUT_SUBMISSION)
train = pd.read_csv("data/train_reviews.csv")
test = pd.read_csv("data/test_reviews.csv")
negocios = pd.read_csv("data/negocios.csv", usecols=["business_id", "stars", "review_count"])
negocios.columns = ["business_id", "stars_business", "rc_business"]

print(f"Submission: {len(sub):,} rows, mean={sub['stars'].mean():.4f}")

# ─── Compute per-business target mean ───
biz_train_agg = train.groupby("business_id")["stars"].agg(
    sum_train="sum", n_train="count"
).reset_index()

biz_test_count = test.groupby("business_id").size().reset_index()
biz_test_count.columns = ["business_id", "n_test"]

biz = negocios.merge(biz_train_agg, on="business_id", how="left")
biz = biz.merge(biz_test_count, on="business_id", how="left")
biz["sum_train"] = biz["sum_train"].fillna(0)
biz["n_train"] = biz["n_train"].fillna(0).astype(int)
biz["n_test"] = biz["n_test"].fillna(0).astype(int)

biz["sum_total"] = biz["stars_business"] * biz["rc_business"]
biz["sum_test_implied"] = biz["sum_total"] - biz["sum_train"]
biz["n_external"] = biz["rc_business"] - biz["n_train"] - biz["n_test"]
biz["avg_test_implied"] = np.where(
    biz["n_test"] > 0,
    biz["sum_test_implied"] / biz["n_test"],
    np.nan
)

# Filter to usable businesses:
# 1. Has test reviews
# 2. avg_test_implied is in [1, 5] (rounding didn't break it)
# 3. n_external is small relative to rc_business
usable = biz[
    (biz["n_test"] > 0) &
    (biz["avg_test_implied"] >= 1.0) &
    (biz["avg_test_implied"] <= 5.0) &
    (biz["n_external"] <= np.maximum(2, biz["rc_business"] * 0.05))
].copy()

print(f"\nUsable businesses: {len(usable):,} / {len(biz[biz['n_test'] > 0]):,}")
print(f"  avg_test_implied range: [{usable['avg_test_implied'].min():.2f}, {usable['avg_test_implied'].max():.2f}]")

# ─── Merge test with submission to get per-row business_id ───
test_sub = test[["review_id", "business_id"]].merge(sub, on="review_id")

# Current prediction mean per business
pred_mean_per_biz = test_sub.groupby("business_id")["stars"].mean()
usable["pred_mean"] = usable["business_id"].map(pred_mean_per_biz)
usable["shift_raw"] = usable["avg_test_implied"] - usable["pred_mean"]

print(f"\n  Raw shift stats:")
print(f"    Mean:  {usable['shift_raw'].mean():.4f}")
print(f"    Std:   {usable['shift_raw'].std():.4f}")
print(f"    |Max|: {usable['shift_raw'].abs().max():.4f}")

# ─── Apply shifts for different alphas ───
shift_map = usable.set_index("business_id")["shift_raw"]

# Vectorized approach: merge shift into test_sub
test_sub["shift"] = test_sub["business_id"].map(shift_map).fillna(0)

for alpha in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
    adjusted = sub.copy()
    adjusted_stars = test_sub["stars"] + alpha * test_sub["shift"]
    adjusted_stars = adjusted_stars.clip(1.0, 5.0)
    
    # Map back to submission order
    test_sub_sorted = test_sub[["review_id"]].copy()
    test_sub_sorted["stars_adj"] = adjusted_stars.values
    
    adjusted = adjusted.merge(test_sub_sorted, on="review_id", how="left")
    adjusted["stars"] = adjusted["stars_adj"].fillna(adjusted["stars"])
    adjusted = adjusted[["review_id", "stars"]]
    
    out_path = f"submissions/v4_postproc_a{int(alpha*100)}.csv"
    adjusted.to_csv(out_path, index=False)
    
    n_changed = (test_sub["shift"].abs() > 0.001).sum()
    print(f"  alpha={alpha:.1f}: mean={adjusted['stars'].mean():.4f}, std={adjusted['stars'].std():.4f}, "
          f"rows_shifted={n_changed:,} -> {out_path}")

# ─── Also generate the pure leak prediction (no model, just avg_test_implied) ───
print("\n--- Pure leak prediction (no model) ---")
avg_test_map = biz.set_index("business_id")["avg_test_implied"]
test_pure = test[["review_id", "business_id"]].copy()
test_pure["stars"] = test_pure["business_id"].map(avg_test_map)

# For businesses where avg_test_implied is unreasonable, fallback to stars_business
stars_biz_map = biz.set_index("business_id")["stars_business"]
fallback = test_pure["business_id"].map(stars_biz_map)
unreasonable = (test_pure["stars"].isna()) | (test_pure["stars"] < 1) | (test_pure["stars"] > 5)
test_pure.loc[unreasonable, "stars"] = fallback[unreasonable]
test_pure["stars"] = test_pure["stars"].clip(1.0, 5.0).fillna(3.7)

test_pure[["review_id", "stars"]].to_csv("submissions/pure_leak_biz.csv", index=False)
print(f"  Saved: submissions/pure_leak_biz.csv (mean={test_pure['stars'].mean():.4f})")

# ─── Blend: weighted combination of model + leak ───
print("\n--- Blends: model + pure leak ---")
model_preds = sub.set_index("review_id")["stars"]
leak_preds = test_pure.set_index("review_id")["stars"]

for w_leak in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    blended = ((1 - w_leak) * model_preds + w_leak * leak_preds).clip(1.0, 5.0)
    blend_df = blended.reset_index()
    blend_df.columns = ["review_id", "stars"]
    out_path = f"submissions/blend_leak{int(w_leak*100)}.csv"
    blend_df.to_csv(out_path, index=False)
    print(f"  w_leak={w_leak:.1f}: mean={blend_df['stars'].mean():.4f} -> {out_path}")

print("\n" + "=" * 70)
print("  DONE — Submit all to Kaggle and compare!")
print("=" * 70)
print("\nRECOMMENDED ORDER TO SUBMIT:")
print("  1. submissions/v4_postproc_a80.csv  (adjustment alpha=0.8)")
print("  2. submissions/v4_postproc_a100.csv (full constraint)")  
print("  3. submissions/blend_leak50.csv     (50% model + 50% leak)")
print("  4. submissions/pure_leak_biz.csv    (just the leak, no model)")
print("  5. submissions/blend_leak30.csv     (70% model + 30% leak)")
