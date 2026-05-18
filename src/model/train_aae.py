import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score, average_precision_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from aae import AAE

# Discriminator labels (flipped vs standard AAE):
# REAL = encoder outputs from training normals  → discriminator learns "what normal codes look like"
# FAKE = random prior samples                   → discriminator learns to reject these as "not normal"
# At test time: 1 - discriminator(z) is HIGH for fraud (unrecognised latent codes)
#               and LOW for normals (recognised as training-normal codes).
_REAL = 0.9
_FAKE = 0.1


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
                 lr_recon=1e-3, lr_disc=1e-4,
                 threshold_percentile=99, kl_weight=1.0, latent_weight=0.5,
                 random_state=42):
        self.datasets = datasets
        self.name = name
        self.test_size = test_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_recon = lr_recon
        self.lr_disc = lr_disc
        self.threshold_percentile = threshold_percentile
        self.kl_weight = kl_weight
        # Weight of the discriminator anomaly signal relative to reconstruction error.
        # Combined score = recon_error + latent_weight * (1 - discriminator(mu))
        self.latent_weight = latent_weight
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
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

    def _combine_datasets(self):
        parts_X, parts_y = [], []
        for cfg in self.datasets:
            X, y = self._load_single(cfg)
            parts_X.append(X)
            parts_y.append(y)
        X_all = pd.concat(parts_X, axis=0, ignore_index=True, sort=False)
        y_all = np.concatenate(parts_y)
        return X_all, y_all

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

        X_train_normal = X_train[y_train == 0].values
        X_test_values = X_test_df.values

        if fit:
            self.medians = np.nanmedian(X_train_normal, axis=0)

        X_train_normal = self._fill_missing(X_train_normal)
        X_test_values = self._fill_missing(X_test_values)

        if fit:
            X_train_normal = self.scaler.fit_transform(X_train_normal)
        else:
            X_train_normal = self.scaler.transform(X_train_normal)
        X_test_values = self.scaler.transform(X_test_values)

        X_train_normal = np.nan_to_num(X_train_normal, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_values = np.nan_to_num(X_test_values, nan=0.0, posinf=0.0, neginf=0.0)

        return X_train_normal, X_test_values, y_test

    def _anomaly_score(self, X_np):
        """
        Combined score: reconstruction error from the deterministic mu path
        PLUS a discriminator signal.

        The discriminator was trained to output HIGH (~0.9) for latent codes
        that came from training normal transactions, and LOW (~0.1) for random
        prior samples.  At test time, fraud latent codes are unfamiliar to the
        discriminator → output is low → (1 - output) is high.

        This adds a second independent signal on top of reconstruction error,
        which is what the VAE baseline only has.
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_np).to(self.device)
            mu    = self.model.encode(X_t)           # deterministic
            x_r   = self.model.decode(mu)
            recon = torch.mean((X_t - x_r) ** 2, dim=1)
            disc  = self.model.discriminate(mu).squeeze(1)
            # disc ≈ 0.9 for normals, ≈ 0.1 for non-normal-like codes
            anomaly_disc = 1.0 - disc
        return (recon + self.latent_weight * anomaly_disc).cpu().numpy()

    def load_data(self):
        X_all, y_all = self._combine_datasets()
        return self._preprocess(X_all, y_all, fit=True)

    def train(self):
        print(f"Device: {self.device}")
        X_train_normal, X_test, y_test = self.load_data()

        input_dim = X_train_normal.shape[1]
        print(f"Input dim: {input_dim} | Normal training samples: {len(X_train_normal)}")
        print(f"kl_weight={self.kl_weight} | latent_weight={self.latent_weight}")

        self.model = AAE(input_dim, self.latent_dim).to(self.device)

        enc_dec_params = (list(self.model.encoder.parameters()) +
                          list(self.model.decoder.parameters()))
        opt_recon = optim.Adam(enc_dec_params, lr=self.lr_recon, weight_decay=1e-5)
        opt_disc  = optim.Adam(self.model.discriminator.parameters(),
                               lr=self.lr_disc, betas=(0.5, 0.999))

        sched_recon = optim.lr_scheduler.ReduceLROnPlateau(
            opt_recon, patience=5, factor=0.5, min_lr=1e-6
        )

        pin = self.device.type == 'cuda'
        loader = DataLoader(FraudDataset(X_train_normal), batch_size=self.batch_size,
                            shuffle=True, pin_memory=pin)

        for epoch in range(self.epochs):
            self.model.train()
            total_recon = total_kl = total_disc = 0.0

            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                n = batch_x.size(0)

                # --- Phase 1: reconstruction + KL ---
                # Stochastic encode so KL gradients flow through the encoder.
                # KL forces each normal sample's code toward N(0,I) analytically,
                # giving the tight normal manifold that drives good PR-AUC.
                opt_recon.zero_grad()
                mu, logvar, z = self.model.encode_stochastic(batch_x)
                x_recon    = self.model.decode(z)
                recon_loss = F.mse_loss(x_recon, batch_x, reduction='mean')
                kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                (recon_loss + self.kl_weight * kl_loss).backward()
                torch.nn.utils.clip_grad_norm_(enc_dec_params, max_norm=1.0)
                opt_recon.step()

                # --- Phase 2: discriminator as normal-code detector ---
                # REAL label = encoder mu from training normals (what normal codes look like)
                # FAKE label = random prior samples
                # No generator phase: the encoder is NOT trained to fool the discriminator.
                # This means at test time the discriminator can flag fraud codes that the
                # encoder produces from unseen fraud inputs.
                opt_disc.zero_grad()
                with torch.no_grad():
                    z_real = self.model.encode(batch_x)          # mu, detached
                z_fake   = torch.randn(n, self.latent_dim, device=self.device)
                d_real   = self.model.discriminate(z_real)
                d_fake   = self.model.discriminate(z_fake)
                disc_loss = (F.binary_cross_entropy(d_real, torch.full_like(d_real, _REAL)) +
                             F.binary_cross_entropy(d_fake, torch.full_like(d_fake, _FAKE)))
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
                opt_disc.step()

                total_recon += recon_loss.item()
                total_kl    += kl_loss.item()
                total_disc  += disc_loss.item()

            n_batches = len(loader)
            sched_recon.step(total_recon / n_batches)

            if (epoch + 1) % 10 == 0:
                lr_now = opt_recon.param_groups[0]['lr']
                print(f"Epoch [{epoch+1:3d}/{self.epochs}]  "
                      f"recon={total_recon/n_batches:.4f}  "
                      f"kl={total_kl/n_batches:.4f}  "
                      f"disc={total_disc/n_batches:.4f}  "
                      f"lr={lr_now:.6f}")

        scores_train = self._anomaly_score(X_train_normal)
        self.threshold = np.percentile(scores_train, self.threshold_percentile)
        print(f"Threshold ({self.threshold_percentile}th pct): {self.threshold:.6f}")
        return X_test, y_test

    def evaluate(self, X_test, y_test, output_name=None):
        if output_name is None:
            output_name = f'aae_results_{self.name}'

        if y_test is None:
            print("No labels available — skipping evaluation.")
            return

        anomaly_scores = self._anomaly_score(X_test)
        y_pred = (anomaly_scores > self.threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        accuracy  = accuracy_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, anomaly_scores)
        pr_auc    = average_precision_score(y_test, anomaly_scores)
        cm        = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        os.makedirs('src/model/results', exist_ok=True)
        out_path = f'src/model/results/{output_name}.txt'
        with open(out_path, 'w') as f:
            f.write(f"AAE Results for {output_name}:\n\n")
            f.write("Classification Metrics:\n")
            f.write(f"Accuracy:           {accuracy:.4f}\n")
            f.write(f"Precision:          {precision:.4f}\n")
            f.write(f"Recall:             {recall:.4f}\n")
            f.write(f"F1-Score:           {f1:.4f}\n\n")
            f.write("Anomaly Scoring:\n")
            f.write(f"ROC-AUC:            {roc_auc:.4f}\n")
            f.write(f"PR-AUC:             {pr_auc:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"TN:     {tn}\n")
            f.write(f"FP:     {fp}\n")
            f.write(f"FN:     {fn}\n")
            f.write(f"TP:     {tp}\n\n")
            f.write(f"Threshold ({self.threshold_percentile}th pct): {self.threshold:.6f}\n")
            f.write(f"kl_weight:     {self.kl_weight}\n")
            f.write(f"latent_weight: {self.latent_weight}\n")
            f.write(f"Latent dim:    {self.latent_dim}\n")
            f.write(f"Epochs:        {self.epochs}\n")

        print(f"\nResults -> {out_path}")
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    detector_cc = FraudDetector(
        datasets=[DatasetConfig(
            data_path='Datasets/creditcard.csv',
            target_column='Class',
        )],
        name='creditcard',
        latent_dim=32,
        batch_size=128,
        epochs=150,
        threshold_percentile=99,
        kl_weight=1.0,
        latent_weight=2.5,
    )
    X_test_cc, y_test_cc = detector_cc.train()
    detector_cc.evaluate(X_test_cc, y_test_cc)

    detector_ieee = FraudDetector(
        datasets=[DatasetConfig(
            data_path='Datasets/train_transaction.csv',
            target_column='isFraud',
            id_columns=['TransactionID'],
            merge_files=[{'path': 'Datasets/train_identity.csv', 'on': 'TransactionID'}],
        )],
        name='ieee',
        latent_dim=64,
        batch_size=128,
        epochs=200,
        threshold_percentile=99,
        kl_weight=1.0,
        latent_weight=2.5,
    )
    X_test_ieee, y_test_ieee = detector_ieee.train()
    detector_ieee.evaluate(X_test_ieee, y_test_ieee)
