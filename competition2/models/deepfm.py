import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# 1) Encoder tabular
# ---------------------------------------------------------
class TabularFeatureEncoder:
    """
    Convierte un DataFrame con columnas categóricas y numéricas
    en tensores para modelos tipo DeepFM / xDeepFM.
    """

    def __init__(self, categorical_cols: List[str], numerical_cols: List[str]):
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.cat_maps: Dict[str, Dict[object, int]] = {}
        self.num_means: Dict[str, float] = {}
        self.num_stds: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame):
        for col in self.categorical_cols:
            series = df[col].fillna("__nan__").astype(str)
            uniq = series.unique().tolist()
            # 0 reservado para unknown/padding
            self.cat_maps[col] = {val: idx + 1 for idx, val in enumerate(uniq)}

        for col in self.numerical_cols:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
            mean = float(s.mean())
            std = float(s.std(ddof=0))
            if std < 1e-8:
                std = 1.0
            self.num_means[col] = mean
            self.num_stds[col] = std

        return self

    @property
    def cat_cardinalities(self) -> List[int]:
        return [len(self.cat_maps[col]) for col in self.categorical_cols]

    def transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cat_arrays = []
        for col in self.categorical_cols:
            mapping = self.cat_maps[col]
            vals = df[col].fillna("__nan__").astype(str).map(mapping).fillna(0).astype(np.int64).values
            cat_arrays.append(vals)

        if len(cat_arrays) > 0:
            cat_x = np.stack(cat_arrays, axis=1)
        else:
            cat_x = np.zeros((len(df), 0), dtype=np.int64)

        num_arrays = []
        for col in self.numerical_cols:
            s = pd.to_numeric(df[col], errors="coerce").fillna(self.num_means[col]).astype(float)
            s = (s - self.num_means[col]) / self.num_stds[col]
            num_arrays.append(s.values.astype(np.float32))

        if len(num_arrays) > 0:
            num_x = np.stack(num_arrays, axis=1)
        else:
            num_x = np.zeros((len(df), 0), dtype=np.float32)

        y = None
        if target_col is not None:
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).values.astype(np.float32)

        cat_x_t = torch.tensor(cat_x, dtype=torch.long)
        num_x_t = torch.tensor(num_x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32) if y is not None else None
        return cat_x_t, num_x_t, y_t


class _TabularDataset(Dataset):
    def __init__(self, cat_x: torch.Tensor, num_x: torch.Tensor, y: torch.Tensor):
        self.cat_x = cat_x
        self.num_x = num_x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.cat_x[idx], self.num_x[idx], self.y[idx]


# ---------------------------------------------------------
# 2) Bloque base
# ---------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _BaseDeepRecRegressor:
    def __init__(
        self,
        emb_dim: int = 16,
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        epochs: int = 20,
        device: Optional[str] = None,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.emb_dim = emb_dim
        self.mlp_hidden_dims = mlp_hidden_dims or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder: Optional[TabularFeatureEncoder] = None
        self.model: Optional[nn.Module] = None

        torch.manual_seed(seed)
        np.random.seed(seed)

    def _build(self, cat_cardinalities: List[int], num_dense: int):
        raise NotImplementedError

    def _predict_raw(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(cat_x.to(self.device), num_x.to(self.device)).cpu()

    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        numerical_cols: List[str],
        target_col: str = "stars",
        val_df: Optional[pd.DataFrame] = None,
    ):
        self.encoder = TabularFeatureEncoder(categorical_cols, numerical_cols).fit(df)
        cat_x, num_x, y = self.encoder.transform(df, target_col=target_col)

        self.model = self._build(self.encoder.cat_cardinalities, len(numerical_cols)).to(self.device)

        train_ds = _TabularDataset(cat_x, num_x, y)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        val_loader = None
        if val_df is not None:
            cat_v, num_v, y_v = self.encoder.transform(val_df, target_col=target_col)
            val_loader = DataLoader(_TabularDataset(cat_v, num_v, y_v), batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_state = None
        best_val = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0

            for cat_b, num_b, y_b in train_loader:
                cat_b = cat_b.to(self.device)
                num_b = num_b.to(self.device)
                y_b = y_b.to(self.device)

                pred = self.model(cat_b, num_b)
                loss = loss_fn(pred, y_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_b)

            train_rmse = math.sqrt(total_loss / len(train_ds))

            if val_loader is not None:
                self.model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for cat_b, num_b, y_b in val_loader:
                        pred = self.model(cat_b.to(self.device), num_b.to(self.device))
                        preds.append(pred.cpu())
                        trues.append(y_b)
                preds = torch.cat(preds)
                trues = torch.cat(trues)
                val_rmse = torch.sqrt(torch.mean((preds - trues) ** 2)).item()

                if self.verbose:
                    print(f"Epoch {epoch:02d} | train_rmse={train_rmse:.4f} | val_rmse={val_rmse:.4f}")

                if val_rmse < best_val:
                    best_val = val_rmse
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                if self.verbose:
                    print(f"Epoch {epoch:02d} | train_rmse={train_rmse:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.encoder is None or self.model is None:
            raise RuntimeError("Primero llama a fit().")

        cat_x, num_x, _ = self.encoder.transform(df, target_col=None)
        self.model.eval()
        preds = []
        loader = DataLoader(
            _TabularDataset(cat_x, num_x, torch.zeros(len(df), dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for cat_b, num_b, _ in loader:
                out = self.model(cat_b.to(self.device), num_b.to(self.device))
                preds.append(out.cpu())

        preds = torch.cat(preds).numpy()
        return np.clip(preds, 1.0, 5.0)


# ---------------------------------------------------------
# 3) DeepFM
# ---------------------------------------------------------
class _DeepFMNet(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],
        num_dense: int,
        emb_dim: int,
        mlp_hidden_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.num_fields = len(cat_cardinalities)
        self.num_dense = num_dense

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card + 1, emb_dim, padding_idx=0) for card in cat_cardinalities]
        )
        self.cat_linear = nn.ModuleList(
            [nn.Embedding(card + 1, 1, padding_idx=0) for card in cat_cardinalities]
        )

        self.dense_linear = nn.Linear(num_dense, 1) if num_dense > 0 else None

        deep_in_dim = self.num_fields * emb_dim + num_dense
        self.mlp = _MLP(deep_in_dim, mlp_hidden_dims, dropout)

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor):
        # Embeddings [B, F, D]
        if self.num_fields > 0:
            emb_list = [emb(cat_x[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            embs = torch.stack(emb_list, dim=1)
        else:
            embs = None

        # Linear term
        linear_logit = 0.0
        if self.num_fields > 0:
            linear_logit = sum(self.cat_linear[i](cat_x[:, i]) for i in range(self.num_fields))
        if self.dense_linear is not None:
            linear_logit = linear_logit + self.dense_linear(num_x)

        # FM second order
        fm_logit = 0.0
        if self.num_fields > 0:
            sum_emb = torch.sum(embs, dim=1)                 # [B, D]
            sum_emb_sq = sum_emb * sum_emb
            sq_emb = embs * embs
            sq_sum_emb = torch.sum(sq_emb, dim=1)            # [B, D]
            fm_vec = 0.5 * (sum_emb_sq - sq_sum_emb)         # [B, D]
            fm_logit = torch.sum(fm_vec, dim=1, keepdim=True) # [B, 1]

        # Deep part
        if self.num_fields > 0 and self.num_dense > 0:
            deep_in = torch.cat([embs.flatten(1), num_x], dim=1)
        elif self.num_fields > 0:
            deep_in = embs.flatten(1)
        else:
            deep_in = num_x

        deep_logit = self.mlp(deep_in)

        logit = linear_logit + fm_logit + deep_logit
        rating = 1.0 + 4.0 * torch.sigmoid(logit)
        return rating.squeeze(-1)


class DeepFMRegressor(_BaseDeepRecRegressor):
    def _build(self, cat_cardinalities: List[int], num_dense: int):
        return _DeepFMNet(
            cat_cardinalities=cat_cardinalities,
            num_dense=num_dense,
            emb_dim=self.emb_dim,
            mlp_hidden_dims=self.mlp_hidden_dims,
            dropout=self.dropout,
        )


# ---------------------------------------------------------
# 4) xDeepFM
# ---------------------------------------------------------
class _xDeepFMNet(nn.Module):
    """
    Versión práctica de xDeepFM:
    - linear
    - CIN
    - deep MLP
    """

    def __init__(
        self,
        cat_cardinalities: List[int],
        num_dense: int,
        emb_dim: int,
        mlp_hidden_dims: List[int],
        cin_layer_sizes: List[int],
        dropout: float,
    ):
        super().__init__()
        self.num_fields = len(cat_cardinalities)
        self.num_dense = num_dense
        self.emb_dim = emb_dim
        self.cin_layer_sizes = cin_layer_sizes

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card + 1, emb_dim, padding_idx=0) for card in cat_cardinalities]
        )
        self.cat_linear = nn.ModuleList(
            [nn.Embedding(card + 1, 1, padding_idx=0) for card in cat_cardinalities]
        )
        self.dense_linear = nn.Linear(num_dense, 1) if num_dense > 0 else None

        # CIN layers: input channels = prev_fields * num_fields
        cin_layers = []
        prev_fields = self.num_fields if self.num_fields > 0 else 1
        for out_channels in cin_layer_sizes:
            cin_layers.append(nn.Conv1d(prev_fields * self.num_fields, out_channels, kernel_size=1))
            prev_fields = out_channels
        self.cin_layers = nn.ModuleList(cin_layers)
        self.cin_linear = nn.Linear(sum(cin_layer_sizes), 1)

        deep_in_dim = self.num_fields * emb_dim + num_dense
        self.mlp = _MLP(deep_in_dim, mlp_hidden_dims, dropout)

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor):
        if self.num_fields > 0:
            emb_list = [emb(cat_x[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            x0 = torch.stack(emb_list, dim=1)  # [B, F, D]
        else:
            x0 = None

        # linear
        linear_logit = 0.0
        if self.num_fields > 0:
            linear_logit = sum(self.cat_linear[i](cat_x[:, i]) for i in range(self.num_fields))
        if self.dense_linear is not None:
            linear_logit = linear_logit + self.dense_linear(num_x)

        # CIN
        cin_outs = []
        if self.num_fields > 0:
            hidden = x0
            for conv in self.cin_layers:
                # [B, H, F, D] -> [B, H*F, D]
                z = torch.einsum("bfd,bhd->bfhd", x0, hidden).reshape(
                    x0.size(0), hidden.size(1) * x0.size(1), x0.size(2)
                )
                z = conv(z)               # [B, out_channels, D]
                z = torch.relu(z)
                cin_outs.append(z.sum(dim=-1))  # [B, out_channels]
                hidden = z                  # next hidden: [B, out_channels, D]

            cin_cat = torch.cat(cin_outs, dim=1)  # [B, sum(cin_layers)]
            cin_logit = self.cin_linear(cin_cat)
        else:
            cin_logit = 0.0

        # deep
        if self.num_fields > 0 and self.num_dense > 0:
            deep_in = torch.cat([x0.flatten(1), num_x], dim=1)
        elif self.num_fields > 0:
            deep_in = x0.flatten(1)
        else:
            deep_in = num_x

        deep_logit = self.mlp(deep_in)

        logit = linear_logit + cin_logit + deep_logit
        rating = 1.0 + 4.0 * torch.sigmoid(logit)
        return rating.squeeze(-1)


class xDeepFMRegressor(_BaseDeepRecRegressor):
    def __init__(
        self,
        emb_dim: int = 16,
        mlp_hidden_dims: Optional[List[int]] = None,
        cin_layer_sizes: Optional[List[int]] = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        epochs: int = 20,
        device: Optional[str] = None,
        verbose: bool = True,
        seed: int = 42,
    ):
        super().__init__(
            emb_dim=emb_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            verbose=verbose,
            seed=seed,
        )
        self.cin_layer_sizes = cin_layer_sizes or [64, 64, 64]

    def _build(self, cat_cardinalities: List[int], num_dense: int):
        return _xDeepFMNet(
            cat_cardinalities=cat_cardinalities,
            num_dense=num_dense,
            emb_dim=self.emb_dim,
            mlp_hidden_dims=self.mlp_hidden_dims,
            cin_layer_sizes=self.cin_layer_sizes,
            dropout=self.dropout,
        )