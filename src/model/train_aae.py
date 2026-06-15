import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score, average_precision_score,
)
sys.path.insert(0, os.path.dirname(__file__))
from aae import AAE


class DatasetConfig:
    def __init__(self, data_path, target_column, id_columns=None, merge_files=None):
        self.data_path = data_path
        self.target_column = target_column
        self.id_columns = id_columns or []
        self.merge_files = merge_files or []


class FraudDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class FraudDetector:
    def __init__(self, datasets, name='model', test_size=0.2,
                 latent_dim=32, batch_size=256, epochs=100,
                 lr_recon=1e-3, lr_disc=3e-4,
                 threshold_percentile=99,
                 score_weight=0.5,
                 scaler_type='standard',
                 random_state=42,
                 max_train_samples=None):
        self.datasets = datasets
        self.name = name
        self.test_size = test_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_recon = lr_recon
        self.lr_disc = lr_disc
        self.threshold_percentile = threshold_percentile
        self.score_weight = score_weight
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if scaler_type == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal',
                                              n_quantiles=1000,
                                              random_state=random_state)
        else:
            self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.score_stats = None
        self.cols_to_keep = None
        self.medians = None
        self.indicator_cols = None

    def _encode_categoricals(self, X):
        for col in X.select_dtypes(include=['object', 'str']).columns:
            X[col] = X[col].factorize()[0]
            X[col] = X[col].replace(-1, np.nan)
        return X

    def _add_missing_indicators(self, X, fit=False):
        if fit:
            missing_pct = X.isnull().mean()
            self.indicator_cols = missing_pct[missing_pct > 0.05].index.tolist()
        cols = [col for col in self.indicator_cols if col in X.columns]
        if cols:
            indicators = pd.DataFrame(
                {f'{col}_missing': X[col].isnull().astype(float) for col in cols},
                index=X.index,
            )
            X = pd.concat([X, indicators], axis=1)
        return X

    def _fill_missing(self, X_values):
        for i in range(X_values.shape[1]):
            mask = np.isnan(X_values[:, i])
            fill = 0.0 if np.isnan(self.medians[i]) else self.medians[i]
            X_values[mask, i] = fill
        return X_values

    def _load_single(self, cfg):
        df = pd.read_csv(cfg.data_path)
        for m in cfg.merge_files:
            df = df.merge(pd.read_csv(m['path']), on=m['on'], how='left')
        y = df[cfg.target_column].values
        X = df.drop(cfg.target_column, axis=1)
        for col in cfg.id_columns:
            if col in X.columns:
                X = X.drop(col, axis=1)
        X = self._encode_categoricals(X)
        return X, y

    def _preprocess(self, X_all, y_all, fit=True):
        X_all = self._add_missing_indicators(X_all, fit=fit)
        if fit:
            missing_pct = X_all.isnull().sum() / len(X_all)
            self.cols_to_keep = missing_pct[missing_pct <= 0.8].index
        X_all = X_all[self.cols_to_keep]

        X_train, X_test_df, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_all,
        )

        X_train_normal_df = X_train[y_train == 0]
        if self.max_train_samples and len(X_train_normal_df) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_train_normal_df), self.max_train_samples, replace=False)
            X_train_normal_df = X_train_normal_df.iloc[idx]
        X_train_normal = X_train_normal_df.to_numpy(dtype=np.float32)
        X_test_values = X_test_df.to_numpy(dtype=np.float32)

        if fit:
            self.medians = np.nanmedian(X_train_normal, axis=0).astype(np.float32)

        X_train_normal = self._fill_missing(X_train_normal)
        X_test_values = self._fill_missing(X_test_values)

        if fit:
            X_train_normal = self.scaler.fit_transform(X_train_normal).astype(np.float32)
        else:
            X_train_normal = self.scaler.transform(X_train_normal).astype(np.float32)
        X_test_values = self.scaler.transform(X_test_values).astype(np.float32)

        X_train_normal = np.nan_to_num(X_train_normal, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_values = np.nan_to_num(X_test_values, nan=0.0, posinf=0.0, neginf=0.0)

        return X_train_normal, X_test_values, y_test

    def _raw_scores(self, X_np):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_np).to(self.device)
            X_t = torch.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
            X_t = torch.clamp(X_t, -10.0, 10.0)
            mu, logvar = self.model.encoder(X_t)
            x_r = self.model.decode(mu)
            recon = (X_t - x_r).abs().mean(dim=1).cpu().numpy()
            uncertainty = logvar.clamp(-7, 7).mean(dim=1).cpu().numpy()
            disc = self.model.discriminate(mu).squeeze(1).cpu().numpy()
        return recon + uncertainty, disc

    def _anomaly_score(self, X_np):
        recon, disc = self._raw_scores(X_np)
        r_min, r_max, d_min, d_max = self.score_stats
        recon_norm = (recon - r_min) / (r_max - r_min + 1e-8)
        disc_norm = (disc - d_min) / (d_max - d_min + 1e-8)
        return self.score_weight * recon_norm + (1 - self.score_weight) * disc_norm

    def load_data(self):
        X_all, y_all = self._load_single(self.datasets[0])
        return self._preprocess(X_all, y_all, fit=True)

    def train(self):
        torch.manual_seed(self.random_state)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.random_state)
        print(f"Device: {self.device}")
        X_train_normal, X_test, y_test = self.load_data()

        input_dim = X_train_normal.shape[1]
        print(f"Input dim: {input_dim} | Normal training samples: {len(X_train_normal)}")

        self.model = AAE(input_dim, self.latent_dim).to(self.device)

        enc_dec_params = (list(self.model.encoder.parameters()) +
                          list(self.model.decoder.parameters()))
        opt_recon = optim.Adam(enc_dec_params, lr=self.lr_recon, weight_decay=1e-5)
        opt_disc = optim.Adam(self.model.discriminator.parameters(),
                              lr=self.lr_disc, betas=(0.5, 0.999))
        opt_gen = optim.Adam(self.model.encoder.parameters(),
                             lr=self.lr_disc, betas=(0.5, 0.999))

        sched_recon = optim.lr_scheduler.ReduceLROnPlateau(
            opt_recon, patience=5, factor=0.5, min_lr=1e-6
        )

        pin = self.device.type == 'cuda'
        loader = DataLoader(FraudDataset(X_train_normal), batch_size=self.batch_size,
                            shuffle=True, pin_memory=pin)

        for epoch in range(self.epochs):
            self.model.train()
            total_recon = total_disc = 0.0

            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                batch_x = torch.nan_to_num(batch_x, nan=0.0, posinf=0.0, neginf=0.0)
                batch_x = torch.clamp(batch_x, -10.0, 10.0)
                n = batch_x.size(0)
                batch_x = batch_x + 0.05 * torch.randn_like(batch_x)

                # Phase 1: reconstruction
                opt_recon.zero_grad()
                mu1, logvar1 = self.model.encoder(batch_x)
                z = mu1 + torch.randn_like(mu1) * torch.exp(0.5 * logvar1.clamp(-7, 7))
                x_recon = self.model.decode(z)
                recon_loss = F.huber_loss(x_recon, batch_x, reduction='mean', delta=1.0)
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_dec_params, max_norm=1.0)
                opt_recon.step()

                # Phase 2: discriminator — N(0,I) = REAL, encoder = FAKE
                with torch.no_grad():
                    mu2, logvar2 = self.model.encoder(batch_x)
                    z_enc = mu2 + torch.randn_like(mu2) * torch.exp(0.5 * logvar2.clamp(-7, 7))
                z_prior = torch.randn(n, self.latent_dim, device=self.device)
                d_real = self.model.discriminate(z_prior)
                d_fake = self.model.discriminate(z_enc)
                disc_loss = (F.binary_cross_entropy(d_real, torch.full_like(d_real, 0.9)) +
                             F.binary_cross_entropy(d_fake, torch.full_like(d_fake, 0.1)))
                opt_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
                opt_disc.step()

                # Phase 3: encoder adversarial update
                mu3, logvar3 = self.model.encoder(batch_x)
                z_gen = mu3 + torch.randn_like(mu3) * torch.exp(0.5 * logvar3.clamp(-7, 7))
                d_gen = self.model.discriminate(z_gen)
                gen_loss = F.binary_cross_entropy(d_gen, torch.full_like(d_gen, 0.9))
                opt_gen.zero_grad()
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), max_norm=1.0)
                opt_gen.step()

                total_recon += recon_loss.item()
                total_disc += disc_loss.item()

            n_batches = len(loader)
            sched_recon.step(total_recon / n_batches)

            if (epoch + 1) % 10 == 0:
                lr_now = opt_recon.param_groups[0]['lr']
                print(f"Epoch [{epoch+1:3d}/{self.epochs}]  "
                      f"recon={total_recon/n_batches:.4f}  "
                      f"disc={total_disc/n_batches:.4f}  "
                      f"lr={lr_now:.6f}")

        recon_tr, disc_tr = self._raw_scores(X_train_normal)
        self.score_stats = (recon_tr.min(), recon_tr.max(), disc_tr.min(), disc_tr.max())
        scores_train = self._anomaly_score(X_train_normal)
        self.threshold = np.percentile(scores_train, self.threshold_percentile)
        print(f"Threshold ({self.threshold_percentile}th pct): {self.threshold:.6f}")
        return X_test, y_test

    def evaluate(self, X_test, y_test):
        if y_test is None:
            print("No labels available — skipping evaluation.")
            return

        anomaly_scores = self._anomaly_score(X_test)
        y_pred = (anomaly_scores > self.threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, anomaly_scores)
        pr_auc = average_precision_score(y_test, anomaly_scores)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        os.makedirs('src/model/results', exist_ok=True)
        out_path = f'src/model/results/aae_results_{self.name}.txt'
        with open(out_path, 'w') as f:
            f.write(f"AAE Results — {self.name}\n\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1:        {f1:.4f}\n")
            f.write(f"ROC-AUC:   {roc_auc:.4f}\n")
            f.write(f"PR-AUC:    {pr_auc:.4f}\n\n")
            f.write(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}\n")
            f.write(f"Threshold ({self.threshold_percentile}th pct): {self.threshold:.6f}\n")
            f.write(f"Latent dim: {self.latent_dim} | Epochs: {self.epochs}\n")

        print(f"Results -> {out_path}")
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    def save(self, path, X_test, y_test):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state':    self.model.state_dict(),
            'scaler':         self.scaler,
            'score_stats':    self.score_stats,
            'threshold':      self.threshold,
            'cols_to_keep':   self.cols_to_keep,
            'medians':        self.medians,
            'indicator_cols': self.indicator_cols,
            'input_dim':      self.model.encoder.shared[0].in_features,
            'latent_dim':     self.latent_dim,
            'X_test':         X_test,
            'y_test':         y_test,
        }, path)
        print(f"Checkpoint saved -> {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.scaler         = ckpt['scaler']
        self.score_stats    = ckpt['score_stats']
        self.threshold      = ckpt['threshold']
        self.cols_to_keep   = ckpt['cols_to_keep']
        self.medians        = ckpt['medians']
        self.indicator_cols = ckpt['indicator_cols']
        self.model = AAE(ckpt['input_dim'], ckpt['latent_dim']).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        print(f"Checkpoint loaded <- {path}")
        return ckpt['X_test'], ckpt['y_test']


if __name__ == "__main__":
    cc = FraudDetector(
        datasets=[DatasetConfig(data_path='Datasets/creditcard.csv', target_column='Class')],
        name='creditcard',
        latent_dim=32, batch_size=128, epochs=150,
        threshold_percentile=99, score_weight=0.5,
    )
    Xc, yc = cc.train()
    cc.save('src/model/checkpoints/creditcard.pt', Xc, yc)
    cc.evaluate(Xc, yc)

    ieee = FraudDetector(
        datasets=[DatasetConfig(
            data_path='Datasets/train_transaction.csv',
            target_column='isFraud',
            id_columns=['TransactionID'],
            merge_files=[{'path': 'Datasets/train_identity.csv', 'on': 'TransactionID'}],
        )],
        name='ieee',
        latent_dim=64, batch_size=256, epochs=150,
        threshold_percentile=99, score_weight=0.5,
        scaler_type='quantile',
        max_train_samples=200_000,
    )
    Xi, yi = ieee.train()
    ieee.save('src/model/checkpoints/ieee.pt', Xi, yi)
    ieee.evaluate(Xi, yi)
