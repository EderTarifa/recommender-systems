# preprocess_dcn.py

import json
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════
# PASO 1: Construir vocabulario global (solo con train)
# ══════════════════════════════════════════════════════════════════════════

class VocabBuilder:
    """
    Construye un vocabulario global: (field, valor) → índice entero.
    Índice 0 reservado para OOV/padding.
    """
    def __init__(self):
        self.vocabs = defaultdict(lambda: {"<PAD>": 0, "<OOV>": 1})
        self.counters = defaultdict(Counter)

    def fit_field(self, field: str, series: pd.Series, min_freq: int = 3):
        for val in series.dropna():
            self.counters[field][str(val)] += 1
        for val, cnt in self.counters[field].items():
            if cnt >= min_freq and val not in self.vocabs[field]:
                self.vocabs[field][val] = len(self.vocabs[field])

    def fit_multivalue_field(self, field: str, series: pd.Series,
                             sep: str = ",", min_freq: int = 5):
        """Para campos como categories que tienen múltiples valores."""
        for s in series.dropna():
            for val in str(s).split(sep):
                self.counters[field][val.strip()] += 1
        for val, cnt in self.counters[field].items():
            if cnt >= min_freq and val not in self.vocabs[field]:
                self.vocabs[field][val] = len(self.vocabs[field])

    def transform_field(self, field: str, series: pd.Series) -> pd.Series:
        vocab = self.vocabs[field]
        return series.apply(lambda x: vocab.get(str(x), vocab["<OOV>"]) if pd.notna(x) else 0)

    def transform_multivalue(self, field: str, series: pd.Series,
                              max_len: int = 10, sep: str = ",") -> np.ndarray:
        """Devuelve array (N, max_len) con padding a 0."""
        vocab = self.vocabs[field]
        oov = vocab["<OOV>"]
        result = []
        for s in series:
            if pd.isna(s):
                result.append([0] * max_len)
                continue
            ids = [vocab.get(v.strip(), oov) for v in str(s).split(sep)][:max_len]
            ids += [0] * (max_len - len(ids))  # padding
            result.append(ids)
        return np.array(result, dtype=np.int32)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dict(self.vocabs), f)

    def load(self, path: str):
        with open(path) as f:
            raw = json.load(f)
        self.vocabs = defaultdict(lambda: {"<PAD>": 0, "<OOV>": 1}, raw)


# ══════════════════════════════════════════════════════════════════════════
# PASO 2: Definir fields para DCN-V2
# ══════════════════════════════════════════════════════════════════════════

# Sparse fields (un solo valor por muestra)
SPARSE_FIELDS = [
    "state", "city", "top_category",
    "WiFi", "NoiseLevel", "Alcohol",
    "RestaurantsPriceRange2",
    "elite_bucket",      # feature engineered: 0/1-2/3-5/6+
    "friend_bucket",     # feature engineered: bins de friend_count
]

# Multi-value fields (varios valores por muestra, con padding)
MULTI_FIELDS = {
    "categories": 10,   # max 10 categorías por negocio
}

# Dense fields (numéricas continuas, NO pasan por embedding)
DENSE_FIELDS = [
    "stars_business", "review_count_business", "is_open",
    "days_yelping", "useful", "funny", "cool",
    "friend_count", "fans", "average_stars",
    "elite_count", "years_as_elite",
    "hours_days_open", "hours_avg_daily_hrs",
    "hours_open_weekend", "hours_open_weekday",
    "review_year", "review_month", "review_dow",
    # compliments
    "compliment_hot", "compliment_more", "compliment_profile",
    "compliment_cute", "compliment_list", "compliment_note",
    "compliment_plain", "compliment_cool", "compliment_funny",
    "compliment_writer", "compliment_photos",
    # atributos numéricos
    "attr_RestaurantsPriceRange2", "attr_GoodForKids",
    "attr_OutdoorSeating", "attr_HasTV",
    "attr_BusinessParking_street", "attr_BusinessParking_lot",
]


# ══════════════════════════════════════════════════════════════════════════
# PASO 3: Dataset PyTorch
# ══════════════════════════════════════════════════════════════════════════

import torch
from torch.utils.data import Dataset

class YelpDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: VocabBuilder,
                 sparse_fields, multi_fields, dense_fields,
                 target_col="target"):

        self.target = torch.FloatTensor(df[target_col].values) \
                      if target_col in df.columns else None

        # Sparse → (N,) int tensors
        self.sparse = {
            f: torch.LongTensor(vocab.transform_field(f, df[f]).values)
            for f in sparse_fields if f in df.columns
        }

        # Multi-value → (N, max_len) int tensors
        self.multi = {
            f: torch.LongTensor(vocab.transform_multivalue(f, df[f], max_len=max_len))
            for f, max_len in multi_fields.items() if f in df.columns
        }

        # Dense → (N, D) float tensor
        dense_cols = [f for f in dense_fields if f in df.columns]
        dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
        # Normalizar
        self.dense_mean = dense_data.mean(axis=0)
        self.dense_std  = dense_data.std(axis=0) + 1e-8
        self.dense = torch.FloatTensor(
            (dense_data - self.dense_mean) / self.dense_std
        )

    def __len__(self):
        return self.dense.shape[0]

    def __getitem__(self, idx):
        sample = {
            "sparse": {k: v[idx] for k, v in self.sparse.items()},
            "multi":  {k: v[idx] for k, v in self.multi.items()},
            "dense":  self.dense[idx],
        }
        if self.target is not None:
            sample["target"] = self.target[idx]
        return sample


# ══════════════════════════════════════════════════════════════════════════
# PASO 4: DCN-V2 con DeepCTR-Torch
# ══════════════════════════════════════════════════════════════════════════

# pip install deepctr-torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DCN

def build_dcn_feature_columns(vocab: VocabBuilder,
                               sparse_fields, multi_fields, dense_fields,
                               embedding_dim=16):
    feature_columns = []

    for f in sparse_fields:
        vocab_size = len(vocab.vocabs.get(f, {})) + 1
        feature_columns.append(
            SparseFeat(f, vocabulary_size=vocab_size, embedding_dim=embedding_dim)
        )

    for f, max_len in multi_fields.items():
        vocab_size = len(vocab.vocabs.get(f, {})) + 1
        feature_columns.append(
            VarLenSparseFeat(
                SparseFeat(f, vocabulary_size=vocab_size, embedding_dim=embedding_dim),
                maxlen=max_len,
                combiner="mean"   # mean pooling de los embeddings de categorías
            )
        )

    feature_columns.append(DenseFeat("dense", len(dense_fields)))

    return feature_columns

# Entrenamiento
def train_dcn(feature_columns, X_train_dict, y_train, X_val_dict, y_val):
    model = DCN(
        linear_feature_columns=feature_columns,
        dnn_feature_columns=feature_columns,
        cross_num=3,           # capas cross
        dnn_hidden_units=(512, 256, 128),
        l2_reg_embedding=1e-5,
        l2_reg_cross=1e-5,
        l2_reg_dnn=1e-5,
        task="regression",
    )
    model.compile(
        optimizer="adam",
        loss="mae",           # MAE directamente
        metrics=["mae"]
    )
    model.fit(
        X_train_dict, y_train,
        batch_size=4096,
        epochs=30,
        validation_data=(X_val_dict, y_val),
        verbose=2
    )
    return model