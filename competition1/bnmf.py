import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class BNMF:
    def __init__(self, n_factors=800, lambda_U=0.1, lambda_V=0.1,
                 lambda_bu=0.001, lambda_bi=0.001,
                 lr=0.016, n_epochs=200, batch_size=256,
                 clip_min=1.0, clip_max=10.0, random_state=42,
                 optimizer='sgd', huber_delta=1.0, log_every=1,
                 use_biases=True):
        self.n_factors    = n_factors
        self.lambda_U     = lambda_U
        self.lambda_V     = lambda_V
        self.lambda_bu    = lambda_bu
        self.lambda_bi    = lambda_bi
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.clip_min     = clip_min
        self.clip_max     = clip_max
        self.random_state = random_state
        self.optimizer    = optimizer
        self.huber_delta  = huber_delta
        self.log_every    = log_every
        self.use_biases   = use_biases
        self.U = self.V = None
        self.b_u = self.b_i = None
        self.mu  = None
        self.history = []
        self.sparsity_U = []
        self.sparsity_V = []

    def _init_factors(self, n_users, n_items, global_mean):
        rng   = np.random.default_rng(self.random_state)
        scale = 1 / np.sqrt(self.n_factors)
        # Inicializar desde la distribución prior Exp(lambda)
        self.U   = rng.exponential(scale, (n_users, self.n_factors))
        self.V   = rng.exponential(scale, (n_items, self.n_factors))
        self.mu  = global_mean
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        if self.optimizer == 'adam':
            self.mU  = np.zeros_like(self.U);   self.vU  = np.zeros_like(self.U)
            self.mV  = np.zeros_like(self.V);   self.vV  = np.zeros_like(self.V)
            self.mbu = np.zeros_like(self.b_u); self.vbu = np.zeros_like(self.b_u)
            self.mbi = np.zeros_like(self.b_i); self.vbi = np.zeros_like(self.b_i)
            self._adam_t = 0

    def _predict_batch(self, u_idx, i_idx):
        dot = np.sum(self.U[u_idx] * self.V[i_idx], axis=1)
        if self.use_biases:
            return self.mu + self.b_u[u_idx] + self.b_i[i_idx] + dot
        return self.mu + dot

    def _huber_grad(self, errors):
        d = self.huber_delta
        return np.where(np.abs(errors) <= d, errors, d * np.sign(errors))

    def _metrics(self, u_idx, i_idx, ratings):
        preds  = np.clip(self._predict_batch(u_idx, i_idx), self.clip_min, self.clip_max)
        errors = ratings - preds
        huber  = np.where(
            np.abs(errors) <= self.huber_delta,
            0.5 * errors ** 2,
            self.huber_delta * (np.abs(errors) - 0.5 * self.huber_delta)
        ).mean()
        return {
            'huber': float(huber),
            'mae'  : float(np.mean(np.abs(errors))),
            'rmse' : float(np.sqrt(np.mean(errors ** 2)))
        }

    def _adam_update(self, param, m, v, grad, beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_t += 1
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * grad ** 2
        param -= self.lr * (m / (1 - beta1 ** self._adam_t)) / \
                           (np.sqrt(v / (1 - beta2 ** self._adam_t)) + eps)

    def _save_state(self):
        state = {'U': self.U.copy(), 'V': self.V.copy()}
        if self.use_biases:
            state.update({'b_u': self.b_u.copy(), 'b_i': self.b_i.copy()})
        return state

    def _load_state(self, state):
        self.U, self.V = state['U'], state['V']
        if self.use_biases:
            self.b_u, self.b_i = state['b_u'], state['b_i']

    def fit(self, x_train, y_train, x_val, y_val,
            user_map, item_map, patience=3):

        train_users = x_train['user'].map(user_map).values
        train_items = x_train['item'].map(item_map).values
        train_r     = y_train.values.astype(np.float64)

        val_mask  = x_val['user'].isin(user_map) & x_val['item'].isin(item_map)
        val_users = x_val.loc[val_mask, 'user'].map(user_map).values
        val_items = x_val.loc[val_mask, 'item'].map(item_map).values
        val_r     = y_val[val_mask].values.astype(np.float64)

        global_mean = float(train_r.mean())
        self._init_factors(len(user_map), len(item_map), global_mean)

        n_samples  = len(train_r)
        rng        = np.random.default_rng(self.random_state)
        best_val   = np.inf
        best_state = self._save_state()
        no_improve = 0

        print(f"{'Epoch':>6}  {'Tr Huber':>9}  {'Tr MAE':>8}  {'Tr RMSE':>9}  "
              f"{'Va Huber':>9}  {'Va MAE':>8}  {'Va RMSE':>9}  "
              f"{'SpU':>6}  {'SpV':>6}  {'Status':>12}")
        print("─" * 100)

        for epoch in range(self.n_epochs):
            perm   = rng.permutation(n_samples)
            u_shuf = train_users[perm]
            i_shuf = train_items[perm]
            r_shuf = train_r[perm]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                u_b = u_shuf[start:end]
                i_b = i_shuf[start:end]
                r_b = r_shuf[start:end]

                U_b  = self.U[u_b]
                V_b  = self.V[i_b]
                dot  = np.sum(U_b * V_b, axis=1)

                preds  = self.mu + self.b_u[u_b] + self.b_i[i_b] + dot \
                         if self.use_biases else self.mu + dot
                errors = r_b - preds
                dgrad  = self._huber_grad(errors)

                grad_U = -dgrad[:, None] * V_b + self.lambda_U
                grad_V = -dgrad[:, None] * U_b + self.lambda_V

                if self.use_biases:
                    grad_bu = -dgrad + self.lambda_bu * self.b_u[u_b]
                    grad_bi = -dgrad + self.lambda_bi * self.b_i[i_b]

                if self.optimizer == 'adam':
                    for idx in np.unique(u_b):
                        mask = u_b == idx
                        self._adam_update(self.U[idx], self.mU[idx], self.vU[idx],
                                          grad_U[mask].mean(axis=0))
                        if self.use_biases:
                            self._adam_update(self.b_u[idx:idx+1], self.mbu[idx:idx+1],
                                              self.vbu[idx:idx+1],
                                              np.array([grad_bu[mask].mean()]))
                    for idx in np.unique(i_b):
                        mask = i_b == idx
                        self._adam_update(self.V[idx], self.mV[idx], self.vV[idx],
                                          grad_V[mask].mean(axis=0))
                        if self.use_biases:
                            self._adam_update(self.b_i[idx:idx+1], self.mbi[idx:idx+1],
                                              self.vbi[idx:idx+1],
                                              np.array([grad_bi[mask].mean()]))
                else:
                    for idx in np.unique(u_b):
                        mask = u_b == idx
                        self.U[idx] -= self.lr * grad_U[mask].mean(axis=0)
                        if self.use_biases:
                            self.b_u[idx] -= self.lr * grad_bu[mask].mean()
                    for idx in np.unique(i_b):
                        mask = i_b == idx
                        self.V[idx] -= self.lr * grad_V[mask].mean(axis=0)
                        if self.use_biases:
                            self.b_i[idx] -= self.lr * grad_bi[mask].mean()

                np.maximum(self.U, 0, out=self.U)
                np.maximum(self.V, 0, out=self.V)

            tr  = self._metrics(train_users, train_items, train_r)
            vl  = self._metrics(val_users,   val_items,   val_r)
            spU = float(np.mean(self.U == 0))
            spV = float(np.mean(self.V == 0))
            self.sparsity_U.append(spU)
            self.sparsity_V.append(spV)

            if vl['mae'] < best_val:
                best_val = vl['mae']; best_state = self._save_state()
                no_improve = 0; status = '✔ best'
            else:
                no_improve += 1; status = f'no imp {no_improve}/{patience}'

            self.history.append({'epoch': epoch + 1,
                                  **{f'train_{k}': v for k, v in tr.items()},
                                  **{f'val_{k}':   v for k, v in vl.items()},
                                  'sparsity_U': spU, 'sparsity_V': spV})

            if (epoch + 1) % self.log_every == 0:
                print(f"{epoch+1:>6}  {tr['huber']:>9.4f}  {tr['mae']:>8.4f}  "
                      f"{tr['rmse']:>9.4f}  {vl['huber']:>9.4f}  {vl['mae']:>8.4f}  "
                      f"{vl['rmse']:>9.4f}  {spU:>6.2%}  {spV:>6.2%}  {status:>12}")

            if no_improve >= patience:
                print(f"\nEarly stopping época {epoch+1} | best val MAE: {best_val:.4f}")
                break

        self._load_state(best_state)
        return self

    def plot_loss(self):
        df = pd.DataFrame(self.history).set_index('epoch')
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, m, title in zip(axes[:3], ['huber', 'mae', 'rmse'],
                                            ['Huber Loss', 'MAE', 'RMSE']):
            ax.plot(df[f'train_{m}'], label='Train', linewidth=2)
            ax.plot(df[f'val_{m}'],   label='Val',   linewidth=2)
            best_ep = int(df[f'val_{m}'].idxmin())
            ax.axvline(best_ep, color='red', linestyle='--',
                       label=f'Best={best_ep} ({df[f"val_{m}"].min():.4f})')
            ax.set_title(title); ax.set_xlabel('Época')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
        axes[3].plot(df['sparsity_U'], label='Sparsity U', linewidth=2, color='steelblue')
        axes[3].plot(df['sparsity_V'], label='Sparsity V', linewidth=2, color='darkorange')
        axes[3].set_title('Sparsity de factores (fracción = 0)')
        axes[3].set_xlabel('Época'); axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.4)
        axes[3].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        fig.suptitle('BNMF — Training History', fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

    def predict_test(self, x_test, user_map, item_map,
                     user_means, item_means, global_mean,
                     cold_start_unknown_user='global_mean',
                     cold_start_unknown_item='user_mean',
                     cold_start_both='const_7'):

        def _cold(strat, user_idx=None, item_idx=None):
            if strat == 'item_mean':
                v = item_means[item_idx] if item_idx is not None else np.nan
            elif strat == 'user_mean':
                v = user_means[user_idx] if user_idx is not None else np.nan
            elif strat == 'global_mean':
                v = global_mean
            elif strat.startswith('const_'):
                v = float(strat.split('_')[1])
            else:
                v = np.nan
            return global_mean if np.isnan(v) else v

        ids, preds = [], []
        for _, row in tqdm(x_test.iterrows(), total=len(x_test), desc='BNMF Predicting'):
            row_id, user_id, item_id = row.iloc[0], row.iloc[1], row.iloc[2]
            ids.append(row_id)
            u_known = user_id in user_map
            i_known = item_id in item_map
            if not u_known and not i_known:
                preds.append(_cold(cold_start_both))
            elif not u_known:
                preds.append(_cold(cold_start_unknown_user, item_idx=item_map[item_id]))
            elif not i_known:
                preds.append(_cold(cold_start_unknown_item, user_idx=user_map[user_id]))
            else:
                u = user_map[user_id]; i = item_map[item_id]
                p = float(self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.U[u], self.V[i]))
                preds.append(np.clip(p, self.clip_min, self.clip_max))
        return pd.DataFrame({'id': ids, 'rating': preds})