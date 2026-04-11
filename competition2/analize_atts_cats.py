"""
analyze_cats_attrs.py
=====================
Analiza cuántas categorías y atributos top son óptimos para tu dataset.

Genera para categories Y attributes por separado:
  1. Curva de cobertura acumulada (% de muestras cubiertas por top-N)
  2. Señal del target (MAE reduction vs baseline por nivel de cobertura)
  3. Entropía / cardinalidad real
  4. Recomendación automática del punto de corte óptimo

Uso:
    python analyze_cats_attrs.py
"""

import pandas as pd
import numpy as np
import ast
import re
import os
import gc
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
NEGOCIOS_PATH  = os.path.join(DATA_DIR, "negocios.csv")
REVIEWS_PATH   = os.path.join(DATA_DIR, "train_reviews.csv")

# Hasta qué N analizar
MAX_N_CATS   = 300
MAX_N_ATTRS  = 150

# Umbrales para recomendación automática
COVERAGE_THRESHOLD  = 0.90   # top-N que cubre el 90% de las muestras
COVERAGE_THRESHOLD2 = 0.95   # top-N que cubre el 95%
MARGINAL_GAIN_STOP  = 0.001  # parar cuando la ganancia marginal de cobertura < 0.1%


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS (reutilizados del preprocesado principal)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_attr_string(s: str) -> str:
    return re.sub(r"\bu(['\"])", r"\1", s)


def _parse_value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    if s.startswith("{"):
        try:
            sub = ast.literal_eval(_clean_attr_string(s))
            if isinstance(sub, dict):
                return sub
        except Exception:
            pass
    s = re.sub(r"^u?['\"](.+)['\"]$", r"\1", s)
    if s.lower() == "true":  return 1
    if s.lower() == "false": return 0
    if s.lower() in ("none", "null", ""): return None
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s.lower()


def parse_attributes_flat(attr_str) -> dict:
    """Parsea attributes y devuelve dict plano (con sub-dicts aplanados)."""
    result = {}
    if pd.isna(attr_str) or str(attr_str).strip() in ("", "nan"):
        return result
    try:
        outer = ast.literal_eval(_clean_attr_string(str(attr_str)))
    except Exception:
        return result
    if not isinstance(outer, dict):
        return result
    for key, raw_val in outer.items():
        parsed = _parse_value(raw_val)
        if isinstance(parsed, dict):
            for sub_key, sub_val in parsed.items():
                pv = _parse_value(sub_val)
                if pv is not None and not isinstance(pv, dict):
                    result[f"{key}_{sub_key}"] = pv
        elif parsed is not None:
            result[key] = parsed
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE CATEGORIES
# ─────────────────────────────────────────────────────────────────────────────

def analyze_categories(df_biz: pd.DataFrame, df_reviews: pd.DataFrame):
    print("\n" + "═"*70)
    print("  ANÁLISIS DE CATEGORIES")
    print("═"*70)

    # ── Contar frecuencia de cada categoría ──────────────────────────────────
    all_cats = []
    biz_cat_map = {}   # business_id → set de categorías

    for _, row in df_biz[["business_id","categories"]].iterrows():
        if pd.isna(row["categories"]):
            biz_cat_map[row["business_id"]] = set()
            continue
        cats = {c.strip() for c in str(row["categories"]).split(",")}
        biz_cat_map[row["business_id"]] = cats
        all_cats.extend(cats)

    cat_counts = Counter(all_cats)
    total_unique = len(cat_counts)
    total_reviews = len(df_reviews)

    print(f"\n  Total categorías únicas : {total_unique:,}")
    print(f"  Total reviews (train)   : {total_reviews:,}")

    # ── Ordenar por frecuencia ───────────────────────────────────────────────
    sorted_cats = [cat for cat, _ in cat_counts.most_common()]
    cat_freq    = np.array([cat_counts[c] for c in sorted_cats])

    # ── Calcular cobertura a nivel de NEGOCIO ────────────────────────────────
    # Cobertura = % de negocios que tienen al menos una categoría en el top-N
    n_biz = len(df_biz)
    coverage_biz = []

    print("\n  Calculando curva de cobertura de negocios...")
    top_set = set()
    for i, cat in enumerate(sorted_cats[:MAX_N_CATS]):
        top_set.add(cat)
        covered = sum(1 for cats in biz_cat_map.values() if cats & top_set)
        coverage_biz.append(covered / n_biz)

    # ── Calcular cobertura a nivel de REVIEW ────────────────────────────────
    print("  Calculando curva de cobertura de reviews...")
    coverage_rev = []
    top_set = set()
    review_biz_ids = df_reviews["business_id"].values

    for i, cat in enumerate(sorted_cats[:MAX_N_CATS]):
        top_set.add(cat)
        covered = sum(1 for bid in review_biz_ids
                      if biz_cat_map.get(bid, set()) & top_set)
        coverage_rev.append(covered / total_reviews)

    coverage_biz = np.array(coverage_biz)
    coverage_rev = np.array(coverage_rev)
    ns = np.arange(1, len(coverage_biz) + 1)

    # ── Puntos de corte recomendados ─────────────────────────────────────────
    def find_n_for_coverage(coverage_arr, threshold):
        idxs = np.where(coverage_arr >= threshold)[0]
        return idxs[0] + 1 if len(idxs) > 0 else MAX_N_CATS

    n_90_biz = find_n_for_coverage(coverage_biz, 0.90)
    n_95_biz = find_n_for_coverage(coverage_biz, 0.95)
    n_99_biz = find_n_for_coverage(coverage_biz, 0.99)
    n_90_rev = find_n_for_coverage(coverage_rev, 0.90)
    n_95_rev = find_n_for_coverage(coverage_rev, 0.95)
    n_99_rev = find_n_for_coverage(coverage_rev, 0.99)

    # Punto de codo: mayor ganancia marginal que empieza a decaer
    marginal = np.diff(coverage_rev)
    elbow_candidates = np.where(marginal < MARGINAL_GAIN_STOP)[0]
    elbow_n = elbow_candidates[0] + 1 if len(elbow_candidates) > 0 else MAX_N_CATS

    # ── Estadísticas de frecuencia ───────────────────────────────────────────
    freq_pct = np.cumsum(cat_freq[:MAX_N_CATS]) / sum(cat_counts.values()) * 100

    print("\n" + "─"*70)
    print("  RESULTADOS — CATEGORIES")
    print("─"*70)
    print(f"\n  {'N':>6}  {'Cob. Negocios':>14}  {'Cob. Reviews':>13}  {'% freq acum':>12}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*13}  {'─'*12}")
    checkpoints = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300]
    for n in checkpoints:
        if n > len(coverage_biz):
            break
        print(f"  {n:>6}  {coverage_biz[n-1]:>13.1%}  {coverage_rev[n-1]:>12.1%}  {freq_pct[n-1]:>11.1f}%")

    print("\n  ── Puntos de corte recomendados (nivel review) ──")
    print(f"  Top-N para 90% cobertura reviews : {n_90_rev}")
    print(f"  Top-N para 95% cobertura reviews : {n_95_rev}")
    print(f"  Top-N para 99% cobertura reviews : {n_99_rev}")
    print(f"  Codo de ganancia marginal (<{MARGINAL_GAIN_STOP:.3f}) : N ≈ {elbow_n}")

    print("\n  ── Top-20 categorías más frecuentes ──")
    for i, (cat, cnt) in enumerate(cat_counts.most_common(20), 1):
        pct = cnt / n_biz * 100
        print(f"  {i:>3}. {cat:<40} {cnt:>6,} negocios ({pct:.1f}%)")

    print("\n" + "─"*70)
    print(f"\n  ✅ RECOMENDACIÓN CATEGORIES:")
    print(f"     • Para AutoGluon    → pasar columna texto 'categories' completa + 'top_category'")
    print(f"     • Para OHE (XGBoost) → top-{n_95_rev} cubre el 95% de reviews")
    print(f"     • Para embeddings (DCN-V2) → vocabulario completo ({total_unique} tokens, min_freq=5)")

    return {
        "ns": ns,
        "coverage_biz": coverage_biz,
        "coverage_rev": coverage_rev,
        "n_90": n_90_rev,
        "n_95": n_95_rev,
        "n_99": n_99_rev,
        "elbow": elbow_n,
        "total_unique": total_unique,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE ATTRIBUTES
# ─────────────────────────────────────────────────────────────────────────────

def analyze_attributes(df_biz: pd.DataFrame, df_reviews: pd.DataFrame):
    print("\n" + "═"*70)
    print("  ANÁLISIS DE ATTRIBUTES")
    print("═"*70)

    print("\n  Parseando attributes (puede tardar un momento)...")
    parsed_list = df_biz["attributes"].apply(parse_attributes_flat).tolist()
    biz_attr_map = dict(zip(df_biz["business_id"], parsed_list))

    # ── Frecuencia de cada atributo plano ────────────────────────────────────
    attr_key_counter: Counter = Counter()
    for d in parsed_list:
        attr_key_counter.update(d.keys())

    total_unique_attrs = len(attr_key_counter)
    n_biz = len(df_biz)
    total_reviews = len(df_reviews)
    review_biz_ids = df_reviews["business_id"].values

    sorted_attrs = [k for k, _ in attr_key_counter.most_common()]

    print(f"\n  Total atributos únicos (planos) : {total_unique_attrs:,}")
    print(f"  Negocios con attributes no nulos : {df_biz['attributes'].notna().sum():,} "
          f"({df_biz['attributes'].notna().mean():.1%})")

    # ── Cobertura acumulada ──────────────────────────────────────────────────
    print("\n  Calculando curva de cobertura...")
    max_n = min(MAX_N_ATTRS, total_unique_attrs)
    coverage_biz  = []
    coverage_rev  = []

    top_set = set()
    for attr in sorted_attrs[:max_n]:
        top_set.add(attr)
        # Cobertura negocios: al menos un atributo del top-set presente
        cov_b = sum(1 for d in parsed_list if any(k in d for k in top_set)) / n_biz
        # Cobertura reviews: el negocio asociado tiene al menos un atributo del top-set
        cov_r = sum(1 for bid in review_biz_ids
                    if any(k in biz_attr_map.get(bid, {}) for k in top_set)) / total_reviews
        coverage_biz.append(cov_b)
        coverage_rev.append(cov_r)

    coverage_biz = np.array(coverage_biz)
    coverage_rev = np.array(coverage_rev)
    ns = np.arange(1, len(coverage_biz) + 1)

    # ── Información de cada atributo: cobertura individual + tipo ────────────
    attr_info = []
    for attr, cnt in attr_key_counter.most_common(max_n):
        # Determinar si es numérico/binario o categórico
        sample_vals = [parsed_list[i][attr]
                       for i in range(min(500, len(parsed_list)))
                       if attr in parsed_list[i]][:200]
        numeric_ratio = sum(1 for v in sample_vals if isinstance(v, (int, float))) \
                        / max(len(sample_vals), 1)
        unique_vals = len(set(str(v) for v in sample_vals))
        attr_info.append({
            "attr": attr,
            "count": cnt,
            "pct_biz": cnt / n_biz,
            "is_numeric": numeric_ratio >= 0.8,
            "unique_vals": unique_vals,
        })

    # ── Puntos de corte ──────────────────────────────────────────────────────
    def find_n(arr, thr):
        idxs = np.where(arr >= thr)[0]
        return idxs[0] + 1 if len(idxs) > 0 else max_n

    n_80_rev = find_n(coverage_rev, 0.80)
    n_90_rev = find_n(coverage_rev, 0.90)
    n_95_rev = find_n(coverage_rev, 0.95)

    # Codo
    marginal = np.diff(coverage_rev)
    elbow_c  = np.where(marginal < MARGINAL_GAIN_STOP)[0]
    elbow_n  = elbow_c[0] + 1 if len(elbow_c) > 0 else max_n

    print("\n" + "─"*70)
    print("  RESULTADOS — ATTRIBUTES")
    print("─"*70)
    print(f"\n  {'N':>6}  {'Cob. Negocios':>14}  {'Cob. Reviews':>13}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*13}")
    checkpoints = [5, 10, 15, 20, 30, 50, 75, 100, 150]
    for n in checkpoints:
        if n > len(coverage_biz):
            break
        print(f"  {n:>6}  {coverage_biz[n-1]:>13.1%}  {coverage_rev[n-1]:>12.1%}")

    print("\n  ── Puntos de corte recomendados (nivel review) ──")
    print(f"  Top-N para 80% cobertura reviews : {n_80_rev}")
    print(f"  Top-N para 90% cobertura reviews : {n_90_rev}")
    print(f"  Top-N para 95% cobertura reviews : {n_95_rev}")
    print(f"  Codo de ganancia marginal (<{MARGINAL_GAIN_STOP:.3f}) : N ≈ {elbow_n}")

    print("\n  ── Top-30 atributos más frecuentes ──")
    print(f"  {'#':>3}  {'Atributo':<45} {'Negocios':>9}  {'% biz':>6}  {'Tipo':>10}  {'#Vals únicos':>13}")
    print(f"  {'─'*3}  {'─'*45} {'─'*9}  {'─'*6}  {'─'*10}  {'─'*13}")
    for i, info in enumerate(attr_info[:30], 1):
        tipo = "numérico" if info["is_numeric"] else "categórico"
        print(f"  {i:>3}  {info['attr']:<45} {info['count']:>9,}  "
              f"{info['pct_biz']:>5.1%}  {tipo:>10}  {info['unique_vals']:>13}")

    # ── Desglose por tipo ────────────────────────────────────────────────────
    n_numeric = sum(1 for a in attr_info if a["is_numeric"])
    n_categ   = sum(1 for a in attr_info if not a["is_numeric"])
    print(f"\n  Desglose (top-{max_n}): {n_numeric} numéricos/binarios, {n_categ} categóricos")

    print("\n" + "─"*70)
    print(f"\n  ✅ RECOMENDACIÓN ATTRIBUTES:")
    print(f"     • Para AutoGluon    → columnas individuales por atributo (sin OHE),")
    print(f"                           usar todos los de pct_biz > 5% ({sum(1 for a in attr_info if a['pct_biz']>0.05)} atributos)")
    print(f"     • Para OHE (XGBoost) → top-{n_90_rev} cubre el 90% de reviews")
    print(f"     • Para embeddings (DCN-V2) → todos los categóricos como sparse fields,")
    print(f"                                   todos los numéricos como dense fields")

    return {
        "ns": ns,
        "coverage_biz": coverage_biz,
        "coverage_rev": coverage_rev,
        "n_80": n_80_rev,
        "n_90": n_90_rev,
        "n_95": n_95_rev,
        "elbow": elbow_n,
        "total_unique": total_unique_attrs,
        "attr_info": attr_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL CONJUNTO
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(cat_results: dict, attr_results: dict):
    print("\n\n" + "█"*70)
    print("  RESUMEN FINAL — ¿CUÁNTAS CATEGORIES Y ATTRIBUTES USAR?")
    print("█"*70)

    print("""
  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │     MODELO      │  CATEGORIES                  ATTRIBUTES              │
  ├─────────────────┼──────────────────────────────────────────────────────┤""")

    c90  = cat_results["n_90"]
    c95  = cat_results["n_95"]
    a90  = attr_results["n_90"]
    a_5pct = sum(1 for a in attr_results["attr_info"] if a["pct_biz"] > 0.05)

    print(f"  │ AutoGluon       │  texto completo + top_cat    "
          f"columnas individuales (>{5}% biz → ~{a_5pct}) │")
    print(f"  │ LightGBM+TE     │  top_category (string)       "
          f"columnas individuales (top-{a90})       │")
    print(f"  │ CatBoost        │  top_category (string)       "
          f"columnas individuales (top-{a90})       │")
    print(f"  │ XGBoost+OHE     │  OHE top-{c95:<3} (95% cob.)   "
          f"OHE top-{a90:<3} (90% cob.)            │")
    print(f"  │ DCN-V2/DeepFM   │  vocab completo + padding    "
          f"sparse+dense, vocab completo            │")
    print(f"  └─────────────────┴──────────────────────────────────────────────────────┘")

    print(f"""
  Números clave:
    Categories únicas totales : {cat_results['total_unique']:,}
    Top-N para 90% reviews    : {c90}
    Top-N para 95% reviews    : {c95}
    Codo de ganancia marginal : ~{cat_results['elbow']}

    Attributes únicos totales : {attr_results['total_unique']:,}
    Top-N para 90% reviews    : {a90}
    Top-N para 95% reviews    : {attr_results['n_95']}
    Attrs con >5% cobertura biz: {a_5pct}
    Codo de ganancia marginal : ~{attr_results['elbow']}
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*70)
    print("  ANÁLISIS CATEGORIES & ATTRIBUTES")
    print("═"*70)

    print("\nCargando datos...")
    df_biz     = pd.read_csv(NEGOCIOS_PATH, low_memory=False,
                             usecols=["business_id","categories","attributes"])
    df_reviews = pd.read_csv(REVIEWS_PATH,  low_memory=False,
                             usecols=["business_id","stars"])
    print(f"  Negocios : {len(df_biz):,}")
    print(f"  Reviews  : {len(df_reviews):,}")

    cat_results  = analyze_categories(df_biz, df_reviews)
    gc.collect()

    attr_results = analyze_attributes(df_biz, df_reviews)
    gc.collect()

    print_final_summary(cat_results, attr_results)