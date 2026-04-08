from numba import njit
import pandas as pd
import numpy as np

    
def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def evaluar(nombre, predict_fn, eval_df, verbose=True):
    y_true, y_pred = [], []
    for row in eval_df.itertuples():
        pred = float(np.clip(predict_fn(row.user, row.item), 1, 10))
        y_true.append(row.rating)
        y_pred.append(pred)
    score = mae(y_true, y_pred)
    if verbose:
        print(f'  {nombre:<8}  MAE = {score:.4f}')
    return {'modelo': nombre, 'MAE': score}

@njit
def sgd_epoch(users, items, ratings, U, V, bu, bi, mu, lr, reg, reg_bias):
    idx = np.random.permutation(len(users))
    total_err = 0.0
    for i in idx:
        u, v, r = users[i], items[i], ratings[i]
        pred = mu + bu[u] + bi[v] + np.dot(U[u], V[v])
        err  = r - pred
        total_err += abs(err)
        bu[u] += lr * (err - reg_bias * bu[u])
        bi[v] += lr * (err - reg_bias * bi[v])
        U[u]  += lr * (err * V[v] - reg * U[u])
        V[v]  += lr * (err * U[u] - reg * V[v])
    return total_err / len(users)


class PMF:

    def __init__(self, n_factors=50, n_epochs=100, lr=0.005,
                 reg=0.02, reg_bias=0.005, patience=5):
        self.n_factors = n_factors; self.n_epochs = n_epochs
        self.lr = lr; self.reg = reg
        self.reg_bias = reg_bias; self.patience = patience
        self.train_mae = []; self.val_mae = []

    def fit(self, train_df, val_df=None):
        full_df = train_df if val_df is None else pd.concat([train_df, val_df])
        n_u = full_df['user'].max() + 1
        n_i = full_df['item'].max() + 1

        self.U  = np.random.normal(0, 0.01, (n_u, self.n_factors))
        self.V  = np.random.normal(0, 0.01, (n_i, self.n_factors))
        self.bu = np.zeros(n_u)
        self.bi = np.zeros(n_i)
        self.mu = train_df['rating'].mean()

        users   = train_df['user'].values.astype(np.int64)
        items   = train_df['item'].values.astype(np.int64)
        ratings = train_df['rating'].values.astype(np.float64)

        best_val          = float('inf')
        epochs_no_improve = 0
        best_state        = None

        print('  Compilando Numba (solo primera vez)...')
        for epoch in range(self.n_epochs):
            train_mae = sgd_epoch(
                users, items, ratings,
                self.U, self.V, self.bu, self.bi, self.mu,
                self.lr, self.reg, self.reg_bias
            )
            self.train_mae.append(train_mae)

            if val_df is not None:
                val_mae = evaluar('', self.predict, val_df, verbose=False)['MAE']
                self.val_mae.append(val_mae)

                if val_mae < best_val:
                    best_val          = val_mae
                    epochs_no_improve = 0
                    best_state = {
                        'U': self.U.copy(), 'V': self.V.copy(),
                        'bu': self.bu.copy(), 'bi': self.bi.copy()
                    }
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f'  Early stopping en época {epoch+1} | mejor val MAE: {best_val:.4f}')
                        self.U  = best_state['U']
                        self.V  = best_state['V']
                        self.bu = best_state['bu']
                        self.bi = best_state['bi']
                        break

            if (epoch + 1) % 10 == 0:
                msg = f'  Epoch {epoch+1:>3}/{self.n_epochs} | train: {self.train_mae[-1]:.4f}'
                if val_df is not None: msg += f' | val: {self.val_mae[-1]:.4f}'
                print(msg)

        return self

    def predict(self, user, item):
        u_known = user < self.U.shape[0]
        i_known = item < self.V.shape[0]
        bu = self.bu[user] if u_known else 0.0
        bi = self.bi[item] if i_known else 0.0
        if not u_known and not i_known:
            return self.mu
        if not u_known:
            return self.mu + bi
        if not i_known:
            return self.mu + bu
        return self.mu + bu + bi + self.U[user] @ self.V[item]

