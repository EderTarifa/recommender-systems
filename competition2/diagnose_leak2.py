"""
diagnose_leak2.py — Deeper investigation of the residual mean leak.

Focus:
  1. Verify the exact arithmetic for businesses (where n_external=0 is 100%)
  2. Understand why user avg_non_train has crazy values (likely rounding in average_stars)
  3. Compute what MAE we'd get if we just predicted avg_non_train_biz for test
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("  DEEP DIAGNOSTIC - LEAK RESIDUAL")
print("=" * 70)

train = pd.read_csv("data/train_reviews.csv")
test = pd.read_csv("data/test_reviews.csv")
negocios = pd.read_csv("data/negocios.csv", usecols=["business_id", "stars", "review_count"])
usuarios = pd.read_csv("data/usuarios.csv", usecols=["user_id", "average_stars", "review_count"])

# ===== NEGOCIOS =====
print("\n=== NEGOCIOS: Arithmetic check ===\n")

negocios.columns = ["business_id", "stars_business", "rc_business"]

# Compute train aggregates
biz_train = train.groupby("business_id")["stars"].agg(["sum", "count"]).reset_index()
biz_train.columns = ["business_id", "sum_train", "n_train"]

# Compute test counts per business
biz_test = test.groupby("business_id").size().reset_index()
biz_test.columns = ["business_id", "n_test"]

biz = negocios.merge(biz_train, on="business_id", how="left")
biz = biz.merge(biz_test, on="business_id", how="left")
biz["sum_train"] = biz["sum_train"].fillna(0)
biz["n_train"] = biz["n_train"].fillna(0).astype(int)
biz["n_test"] = biz["n_test"].fillna(0).astype(int)

# Focus on businesses in test
biz = biz[biz["n_test"] > 0].copy()

# Total sum from metadata
biz["sum_total_meta"] = biz["stars_business"] * biz["rc_business"]

# Non-train sum and count
biz["n_non_train"] = biz["rc_business"] - biz["n_train"]
biz["sum_non_train"] = biz["sum_total_meta"] - biz["sum_train"]

# avg_non_train (this is approximately avg_test since n_external ~ 0)
biz["avg_non_train"] = np.where(
    biz["n_non_train"] > 0,
    biz["sum_non_train"] / biz["n_non_train"],
    biz["stars_business"]
)

# Check: is avg_non_train reasonable (between 1 and 5)?
print(f"Negocios en test: {len(biz):,}")
print(f"avg_non_train_biz stats:")
print(f"  [1, 5] range: {((biz['avg_non_train'] >= 1) & (biz['avg_non_train'] <= 5)).mean():.1%}")
print(f"  < 1: {(biz['avg_non_train'] < 1).sum():,} ({(biz['avg_non_train'] < 1).mean():.1%})")
print(f"  > 5: {(biz['avg_non_train'] > 5).sum():,} ({(biz['avg_non_train'] > 5).mean():.1%})")

# The issue might be rounding in stars_business (Yelp rounds to 0.5)
# Let's see how much sum_total_meta differs from the true integer sum
biz["sum_train_check"] = biz["sum_train"]
biz["expected_sum_total"] = biz["sum_train"]  # we don't know test stars, but check consistency

# For the problematic businesses, check:
problematic = biz[(biz["avg_non_train"] < 1) | (biz["avg_non_train"] > 5)]
print(f"\nProblematic businesses (avg_non_train outside [1,5]): {len(problematic):,}")
if len(problematic) > 0:
    print("  Sample problematic:")
    cols = ["business_id", "stars_business", "rc_business", "n_train", "n_test", 
            "sum_train", "sum_total_meta", "sum_non_train", "n_non_train", "avg_non_train"]
    print(problematic[cols].head(10).to_string())

# The rounding issue: stars_business is rounded to nearest 0.5
# sum_total = round(true_mean, 0.5) * rc
# This introduces error = (round(true_mean, 0.5) - true_mean) * rc
# For rc=100, rounding error in sum could be up to 25 (0.25 * 100)

# Let's clip avg_non_train to [1, 5] and see what we get
biz["avg_non_train_clipped"] = biz["avg_non_train"].clip(1, 5)

# ===== ESTIMATE MAE IF WE JUST PREDICT avg_non_train_biz =====
print("\n=== MAE ESTIMATE: What if we predict avg_non_train_biz for each test row? ===\n")

# We can't compute real MAE (don't have test labels), but we can estimate
# by looking at train with OOF logic
print("Using OOF on train to estimate the value of avg_non_train_biz as a predictor:")

from sklearn.model_selection import KFold

y_train = train["stars"].values
oof_pred_biz_mean = np.full(len(train), np.nan)
oof_pred_residual = np.full(len(train), np.nan)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
    tr = train.iloc[tr_idx]
    va = train.iloc[va_idx]
    
    # Business mean from fold train
    biz_mean_fold = tr.groupby("business_id")["stars"].mean()
    oof_pred_biz_mean[va_idx] = va["business_id"].map(biz_mean_fold).fillna(tr["stars"].mean()).values
    
    # Residual approach: compute what "avg_non_fold" would be
    # This simulates what avg_non_train_biz does for test
    biz_fold_agg = tr.groupby("business_id")["stars"].agg(["sum", "count"])
    
    # For each business in val, compute avg using metadata minus fold-train
    va_biz = va[["business_id"]].drop_duplicates().merge(
        negocios[["business_id", "stars_business", "rc_business"]], on="business_id", how="left"
    )
    va_biz = va_biz.merge(
        biz_fold_agg.reset_index().rename(columns={"sum": "sum_fold", "count": "n_fold"}),
        on="business_id", how="left"
    )
    va_biz["sum_fold"] = va_biz["sum_fold"].fillna(0)
    va_biz["n_fold"] = va_biz["n_fold"].fillna(0)
    
    va_biz["sum_total"] = va_biz["stars_business"] * va_biz["rc_business"]
    va_biz["sum_non_fold"] = va_biz["sum_total"] - va_biz["sum_fold"] 
    va_biz["n_non_fold"] = va_biz["rc_business"] - va_biz["n_fold"]
    va_biz["avg_non_fold"] = np.where(
        va_biz["n_non_fold"] > 0,
        va_biz["sum_non_fold"] / va_biz["n_non_fold"],
        va_biz["stars_business"]
    )
    va_biz["avg_non_fold_clipped"] = va_biz["avg_non_fold"].clip(1, 5)
    
    residual_map = va_biz.set_index("business_id")["avg_non_fold_clipped"]
    oof_pred_residual[va_idx] = va["business_id"].map(residual_map).fillna(tr["stars"].mean()).values

from sklearn.metrics import mean_absolute_error

global_mean = train["stars"].mean()
mae_global = mean_absolute_error(y_train, np.full(len(y_train), global_mean))
mae_biz_mean = mean_absolute_error(y_train, oof_pred_biz_mean)
mae_residual = mean_absolute_error(y_train, oof_pred_residual)

print(f"  global_mean predictor:     MAE = {mae_global:.5f}")
print(f"  business_mean OOF:         MAE = {mae_biz_mean:.5f}")
print(f"  avg_non_fold (residual):   MAE = {mae_residual:.5f}")

# ===== USUARIOS =====
print("\n=== USUARIOS: Investigating the crazy values ===\n")

usuarios.columns = ["user_id", "avg_stars_user", "rc_user"]

# Check: is average_stars rounded?
print("average_stars decimal distribution (sample):")
sample_avg = usuarios["avg_stars_user"].dropna().head(10000)
n_decimals = sample_avg.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
print(f"  Mean decimal places: {n_decimals.mean():.1f}")
print(f"  Max decimal places:  {n_decimals.max()}")

# The issue: average_stars has 2 decimal places typically
# sum_total = average_stars * review_count can accumulate rounding error
# For users with high review_count, this error grows

user_train = train.groupby("user_id")["stars"].agg(["sum", "count"]).reset_index()
user_train.columns = ["user_id", "sum_train", "n_train"]

usr = usuarios.merge(user_train, on="user_id", how="left")
usr["sum_train"] = usr["sum_train"].fillna(0)
usr["n_train"] = usr["n_train"].fillna(0).astype(int)

# Filter to test users
test_users = set(test["user_id"].unique())
usr_test = usr[usr["user_id"].isin(test_users)].copy()

usr_test["sum_total"] = usr_test["avg_stars_user"] * usr_test["rc_user"]
usr_test["n_non_train"] = usr_test["rc_user"] - usr_test["n_train"]
usr_test["sum_non_train"] = usr_test["sum_total"] - usr_test["sum_train"]
usr_test["avg_non_train"] = np.where(
    usr_test["n_non_train"] > 0,
    usr_test["sum_non_train"] / usr_test["n_non_train"],
    usr_test["avg_stars_user"]
)

# Check reasonability
in_range = ((usr_test["avg_non_train"] >= 0.5) & (usr_test["avg_non_train"] <= 5.5))
print(f"\nUsuarios en test: {len(usr_test):,}")
print(f"avg_non_train_user in [0.5, 5.5]: {in_range.mean():.1%}")
print(f"avg_non_train_user < 0: {(usr_test['avg_non_train'] < 0).sum():,}")
print(f"avg_non_train_user > 6: {(usr_test['avg_non_train'] > 6).sum():,}")

# For users with really bad values, show examples
bad_users = usr_test[(usr_test["avg_non_train"] < 0) | (usr_test["avg_non_train"] > 6)].head(5)
if len(bad_users) > 0:
    print("\nSample bad users:")
    print(bad_users[["user_id", "avg_stars_user", "rc_user", "n_train", "n_non_train", 
                      "sum_total", "sum_train", "sum_non_train", "avg_non_train"]].to_string())

# The answer: average_stars is rounded to 2 decimals
# For a user with rc=1000 and avg_stars=3.12:
#   sum_total = 3120.00 (exactly)
#   But true sum might be 3124 (avg 3.124 rounded to 3.12)
#   If n_train=990, sum_train=3090
#   sum_non_train = 3120 - 3090 = 30
#   n_non_train = 10
#   avg_non_train = 3.0 ← still reasonable
# But with more extreme rounding and small n_non_train, it can blow up

# Let's clip and still use it
usr_test["avg_non_train_clipped"] = usr_test["avg_non_train"].clip(1, 5)

# For users where avg_non_train is reasonable, let's see the distribution
reasonable = usr_test[in_range]
print(f"\nReasonable avg_non_train_user stats ({len(reasonable):,} users):")
print(f"  Mean:   {reasonable['avg_non_train'].mean():.4f}")
print(f"  Median: {reasonable['avg_non_train'].median():.4f}")
print(f"  Std:    {reasonable['avg_non_train'].std():.4f}")

print("\n=== COMBINED NAIVE PREDICTOR ===\n")

# Best possible naive predictor combining both leaks
# For test, merge avg_non_train_biz and avg_non_train_user
test_enriched = test[["review_id", "user_id", "business_id"]].copy()

biz_leak = biz[["business_id", "avg_non_train_clipped"]].rename(
    columns={"avg_non_train_clipped": "leak_biz"}
)
usr_leak = usr_test[["user_id", "avg_non_train_clipped"]].rename(
    columns={"avg_non_train_clipped": "leak_user"}
)

test_enriched = test_enriched.merge(biz_leak, on="business_id", how="left")
test_enriched = test_enriched.merge(usr_leak, on="user_id", how="left")

# Fill missing user leak with stars_business (from metadata)
biz_meta = negocios.set_index("business_id")["stars_business"]
test_enriched["leak_user"] = test_enriched["leak_user"].fillna(
    test_enriched["business_id"].map(biz_meta)
)
test_enriched["leak_biz"] = test_enriched["leak_biz"].fillna(
    test_enriched["business_id"].map(biz_meta)
)

# Combined prediction
test_enriched["leak_combined"] = (test_enriched["leak_biz"] + test_enriched["leak_user"]) / 2

print("Test enriched with leaks - coverage:")
print(f"  leak_biz not-null:  {test_enriched['leak_biz'].notna().mean():.1%}")
print(f"  leak_user not-null: {test_enriched['leak_user'].notna().mean():.1%}")
print(f"  leak_combined stats:")
print(f"    Mean:   {test_enriched['leak_combined'].mean():.4f}")
print(f"    Median: {test_enriched['leak_combined'].median():.4f}")
print(f"    Std:    {test_enriched['leak_combined'].std():.4f}")

# Save a naive submission based on just the leak
naive_sub = test_enriched[["review_id"]].copy()
naive_sub["stars"] = test_enriched["leak_biz"].clip(1, 5).values
naive_sub.to_csv("submissions/naive_leak_biz_only.csv", index=False)
print("\nSaved submissions/naive_leak_biz_only.csv (just avg_non_train_biz)")

naive_sub2 = test_enriched[["review_id"]].copy()
naive_sub2["stars"] = test_enriched["leak_combined"].clip(1, 5).values
naive_sub2.to_csv("submissions/naive_leak_combined.csv", index=False)
print("Saved submissions/naive_leak_combined.csv (avg of biz+user leaks)")

print("\nDone!")
