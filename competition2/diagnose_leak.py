"""
diagnose_leak.py — Diagnostic script to measure the quality of residual mean leakage.

Checks:
  1. For businesses: what fraction of review_count is covered by train+test?
  2. For users: same.
  3. When coverage is high, avg_non_train is essentially avg_test → direct target leak.
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("  DIAGNÓSTICO DE LEAK RESIDUAL")
print("=" * 70)

# ─── Load raw data ───
train = pd.read_csv("data/train_reviews.csv")
test = pd.read_csv("data/test_reviews.csv")
negocios = pd.read_csv("data/negocios.csv", usecols=["business_id", "stars", "review_count"])
usuarios = pd.read_csv("data/usuarios.csv", usecols=["user_id", "average_stars", "review_count"])

print(f"\nTrain reviews: {len(train):,}")
print(f"Test reviews:  {len(test):,}")
print(f"Negocios:      {len(negocios):,}")
print(f"Usuarios:      {len(usuarios):,}")

# ─── NEGOCIOS ───
print("\n" + "═" * 70)
print("  NEGOCIOS — Cobertura train+test vs review_count")
print("═" * 70)

negocios.columns = ["business_id", "stars_business", "rc_business"]

biz_train_agg = train.groupby("business_id")["stars"].agg(["sum", "count"]).reset_index()
biz_train_agg.columns = ["business_id", "sum_train_biz", "n_train_biz"]

n_test_biz = test.groupby("business_id").size().reset_index()
n_test_biz.columns = ["business_id", "n_test_biz"]

biz = negocios.merge(biz_train_agg, on="business_id", how="left")
biz = biz.merge(n_test_biz, on="business_id", how="left")
biz["sum_train_biz"] = biz["sum_train_biz"].fillna(0)
biz["n_train_biz"] = biz["n_train_biz"].fillna(0).astype(int)
biz["n_test_biz"] = biz["n_test_biz"].fillna(0).astype(int)

biz["n_covered"] = biz["n_train_biz"] + biz["n_test_biz"]
biz["n_external"] = biz["rc_business"] - biz["n_covered"]
biz["frac_covered"] = biz["n_covered"] / biz["rc_business"].clip(lower=1)

# Filter to businesses that appear in test (we only care about those)
biz_in_test = biz[biz["n_test_biz"] > 0].copy()

print(f"\nNegocios totales en metadata: {len(biz):,}")
print(f"Negocios que aparecen en test: {len(biz_in_test):,}")

print(f"\n--- Cobertura (solo negocios presentes en test) ---")
for thresh in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
    frac = (biz_in_test["frac_covered"] >= thresh).mean()
    count = (biz_in_test["frac_covered"] >= thresh).sum()
    print(f"  frac_covered >= {thresh:.2f}: {frac:.1%} ({count:,} negocios)")

print(f"\n--- Negocios con n_external <= K (solo test-relevant) ---")
for k in [0, 1, 2, 5, 10]:
    frac = (biz_in_test["n_external"] <= k).mean()
    count = (biz_in_test["n_external"] <= k).sum()
    # Weighted by test rows
    test_rows = test.merge(biz_in_test[biz_in_test["n_external"] <= k][["business_id"]], on="business_id")
    frac_rows = len(test_rows) / len(test)
    print(f"  n_external <= {k:>2}: {frac:.1%} negocios ({count:,}), cubre {frac_rows:.1%} filas test")

# ─── Calcular avg_non_train para negocios ───
biz_in_test["sum_total"] = biz_in_test["stars_business"] * biz_in_test["rc_business"]
biz_in_test["sum_non_train"] = biz_in_test["sum_total"] - biz_in_test["sum_train_biz"]
biz_in_test["n_non_train"] = biz_in_test["rc_business"] - biz_in_test["n_train_biz"]
biz_in_test["avg_non_train_biz"] = np.where(
    biz_in_test["n_non_train"] > 0,
    biz_in_test["sum_non_train"] / biz_in_test["n_non_train"],
    biz_in_test["stars_business"]
)

print(f"\n--- Stats de avg_non_train_biz (negocios en test) ---")
print(f"  Mean:   {biz_in_test['avg_non_train_biz'].mean():.4f}")
print(f"  Median: {biz_in_test['avg_non_train_biz'].median():.4f}")
print(f"  Std:    {biz_in_test['avg_non_train_biz'].std():.4f}")
print(f"  Min:    {biz_in_test['avg_non_train_biz'].min():.4f}")
print(f"  Max:    {biz_in_test['avg_non_train_biz'].max():.4f}")

# Sanity check: how different is avg_non_train from stars_business?
diff = (biz_in_test["avg_non_train_biz"] - biz_in_test["stars_business"]).abs()
print(f"\n--- |avg_non_train_biz - stars_business| ---")
print(f"  Mean diff: {diff.mean():.4f}")
print(f"  Max diff:  {diff.max():.4f}")
print(f"  Casos con diff > 0.1: {(diff > 0.1).sum():,} ({(diff > 0.1).mean():.1%})")
print(f"  Casos con diff > 0.5: {(diff > 0.5).sum():,} ({(diff > 0.5).mean():.1%})")

# ─── USUARIOS ───
print("\n" + "═" * 70)
print("  USUARIOS — Cobertura train+test vs review_count")
print("═" * 70)

usuarios.columns = ["user_id", "avg_stars_user", "rc_user"]

user_train_agg = train.groupby("user_id")["stars"].agg(["sum", "count"]).reset_index()
user_train_agg.columns = ["user_id", "sum_train_user", "n_train_user"]

n_test_user = test.groupby("user_id").size().reset_index()
n_test_user.columns = ["user_id", "n_test_user"]

usr = usuarios.merge(user_train_agg, on="user_id", how="left")
usr = usr.merge(n_test_user, on="user_id", how="left")
usr["sum_train_user"] = usr["sum_train_user"].fillna(0)
usr["n_train_user"] = usr["n_train_user"].fillna(0).astype(int)
usr["n_test_user"] = usr["n_test_user"].fillna(0).astype(int)

usr["n_covered"] = usr["n_train_user"] + usr["n_test_user"]
usr["n_external"] = usr["rc_user"] - usr["n_covered"]
usr["frac_covered"] = usr["n_covered"] / usr["rc_user"].clip(lower=1)

# Filter to users that appear in test
usr_in_test = usr[usr["n_test_user"] > 0].copy()

print(f"\nUsuarios totales en metadata: {len(usr):,}")
print(f"Usuarios que aparecen en test: {len(usr_in_test):,}")

print(f"\n--- Cobertura (solo usuarios presentes en test) ---")
for thresh in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
    frac = (usr_in_test["frac_covered"] >= thresh).mean()
    count = (usr_in_test["frac_covered"] >= thresh).sum()
    print(f"  frac_covered >= {thresh:.2f}: {frac:.1%} ({count:,} usuarios)")

print(f"\n--- Usuarios con n_external <= K (solo test-relevant) ---")
for k in [0, 1, 2, 5, 10]:
    frac = (usr_in_test["n_external"] <= k).mean()
    count = (usr_in_test["n_external"] <= k).sum()
    test_rows = test.merge(usr_in_test[usr_in_test["n_external"] <= k][["user_id"]], on="user_id")
    frac_rows = len(test_rows) / len(test)
    print(f"  n_external <= {k:>2}: {frac:.1%} usuarios ({count:,}), cubre {frac_rows:.1%} filas test")

# ─── Calcular avg_non_train para usuarios ───
usr_in_test["sum_total_u"] = usr_in_test["avg_stars_user"] * usr_in_test["rc_user"]
usr_in_test["sum_non_train_u"] = usr_in_test["sum_total_u"] - usr_in_test["sum_train_user"]
usr_in_test["n_non_train_u"] = usr_in_test["rc_user"] - usr_in_test["n_train_user"]
usr_in_test["avg_non_train_user"] = np.where(
    usr_in_test["n_non_train_u"] > 0,
    usr_in_test["sum_non_train_u"] / usr_in_test["n_non_train_u"],
    usr_in_test["avg_stars_user"],
)

print(f"\n--- Stats de avg_non_train_user (usuarios en test) ---")
print(f"  Mean:   {usr_in_test['avg_non_train_user'].mean():.4f}")
print(f"  Median: {usr_in_test['avg_non_train_user'].median():.4f}")
print(f"  Std:    {usr_in_test['avg_non_train_user'].std():.4f}")

diff_u = (usr_in_test["avg_non_train_user"] - usr_in_test["avg_stars_user"]).abs()
print(f"\n--- |avg_non_train_user - avg_stars_user| ---")
print(f"  Mean diff: {diff_u.mean():.4f}")
print(f"  Casos con diff > 0.1: {(diff_u > 0.1).sum():,} ({(diff_u > 0.1).mean():.1%})")

# ─── RESUMEN FINAL ───
print("\n" + "█" * 70)
print("  RESUMEN Y RECOMENDACIÓN")
print("█" * 70)

tight_biz = biz_in_test[biz_in_test["n_external"] <= 2]
tight_biz_rows = test.merge(tight_biz[["business_id"]], on="business_id")
print(f"\nNegocios con constraint tight (n_external <= 2): {len(tight_biz):,}")
print(f"  → cubren {len(tight_biz_rows):,} filas de test ({len(tight_biz_rows)/len(test):.1%})")

tight_usr = usr_in_test[usr_in_test["n_external"] <= 2]
tight_usr_rows = test.merge(tight_usr[["user_id"]], on="user_id")
print(f"\nUsuarios con constraint tight (n_external <= 2): {len(tight_usr):,}")
print(f"  → cubren {len(tight_usr_rows):,} filas de test ({len(tight_usr_rows)/len(test):.1%})")

if (biz_in_test["n_external"] <= 2).mean() > 0.3:
    print("\n🟢 HAY LEAK FUERTE POR NEGOCIO. Procede con la Hipótesis A inmediatamente.")
elif (biz_in_test["n_external"] <= 2).mean() > 0.1:
    print("\n🟡 HAY LEAK MODERADO POR NEGOCIO. Vale la pena explotar.")
else:
    print("\n🔴 LEAK POR NEGOCIO ES DÉBIL. Busca otra hipótesis.")

if (usr_in_test["n_external"] <= 2).mean() > 0.3:
    print("🟢 HAY LEAK FUERTE POR USUARIO. Procede con la Hipótesis A para usuarios.")
elif (usr_in_test["n_external"] <= 2).mean() > 0.1:
    print("🟡 HAY LEAK MODERADO POR USUARIO. Vale la pena explotar.")
else:
    print("🔴 LEAK POR USUARIO ES DÉBIL. Centra el esfuerzo en negocios.")

print()
