import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score, average_precision_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from aae import AAE

_REAL = 0.9
_FAKE = 0.1


class FraudDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class FraudDetector:
    def __init__(self, data_path, target_column, id_columns=None, merge_files=None,
                 test_data_path=None, test_merge_files=None, test_size=0.2,
                 latent_dim=32, batch_size=256, epochs=100,
                 lr_recon=1e-3, lr_disc=1e-4, lr_gen=1e-4,
                 threshold_percentile=95, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.id_columns = id_columns or []
        self.merge_files = merge_files or []
        self.test_data_path = test_data_path
        self.test_merge_files = test_merge_files or []
        self.test_size = test_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_recon = lr_recon
        self.lr_disc = lr_disc
        self.lr_gen = lr_gen
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = RobustScaler()
        self.model = None
        self.threshold = None
        self.cols_to_keep = None
        self.medians = None
        self.indicator_cols = None

    def _encode_categoricals(self, X):
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].factorize()[0]
            X[col] = X[col].replace(-1, np.nan)
        return X

    def _add_missing_indicators(self, X, fit=False):
        if fit:
            missing_pct = X.isnull().mean()
            self.indicator_cols = missing_pct[missing_pct > 0.05].index.tolist()
        for col in self.indicator_cols:
            if col in X.columns:
                X[f'{col}_missing'] = X[col].isnull().astype(float)
        return X

    def _fill_missing(self, X_values):
        for i in range(X_values.shape[1]):
            mask = np.isnan(X_values[:, i])
            fill = 0.0 if np.isnan(self.medians[i]) else self.medians[i]
            X_values[mask, i] = fill
        return X_values

    def load_data(self):
        df = pd.read_csv(self.data_path)
        for m in self.merge_files:
            df = df.merge(pd.read_csv(m['path']), on=m['on'], how='left')

        y = df[self.target_column].values
        X = df.drop([self.target_column], axis=1)
        for col in self.id_columns:
            if col in X.columns:
                X = X.drop(col, axis=1)

        X = self._encode_categoricals(X)
        X = self._add_missing_indicators(X, fit=True)
        missing_pct = X.isnull().sum() / len(X)
        self.cols_to_keep = missing_pct[missing_pct <= 0.8].index
        X = X[self.cols_to_keep]

        if self.test_data_path is not None:
            df_test = pd.read_csv(self.test_data_path)
            for m in self.test_merge_files:
                df_test = df_test.merge(pd.read_csv(m['path']), on=m['on'], how='left')

            if self.target_column in df_test.columns:
                y_test = df_test[self.target_column].values
                X_test = df_test.drop([self.target_column], axis=1)
            else:
                y_test = None
                X_test = df_test.copy()

            for col in self.id_columns:
                if col in X_test.columns:
                    X_test = X_test.drop(col, axis=1)
            X_test = self._encode_categoricals(X_test)
            X_test = self._add_missing_indicators(X_test, fit=False)
            X_test = X_test[self.cols_to_keep]

            X_train_normal = X[y == 0].values
            X_test_values = X_test.values
        else:
            X_train, X_test_df, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            X_train_normal = X_train[y_train == 0].values
            X_test_values = X_test_df.values

        self.medians = np.nanmedian(X_train_normal, axis=0)
        X_train_normal = self._fill_missing(X_train_normal)
        X_test_values = self._fill_missing(X_test_values)

        X_train_normal = self.scaler.fit_transform(X_train_normal)
        X_test_values = self.scaler.transform(X_test_values)

        X_train_normal = np.nan_to_num(X_train_normal, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_values = np.nan_to_num(X_test_values, nan=0.0, posinf=0.0, neginf=0.0)

        return X_train_normal, X_test_values, y_test

    def train(self):
        print(f"Device: {self.device}")
        X_train_normal, X_test, y_test = self.load_data()

        input_dim = X_train_normal.shape[1]
        print(f"Input dim: {input_dim} | Normal training samples: {len(X_train_normal)}")

        self.model = AAE(input_dim, self.latent_dim).to(self.device)

        enc_dec_params = (list(self.model.encoder.parameters()) +
                          list(self.model.decoder.parameters()))
        opt_recon = optim.Adam(enc_dec_params, lr=self.lr_recon, weight_decay=1e-5)
        opt_disc  = optim.Adam(self.model.discriminator.parameters(),
                               lr=self.lr_disc, betas=(0.5, 0.999))
        opt_gen   = optim.Adam(self.model.encoder.parameters(),
                               lr=self.lr_gen, betas=(0.5, 0.999))

        sched_recon = optim.lr_scheduler.ReduceLROnPlateau(
            opt_recon, patience=5, factor=0.5, min_lr=1e-6
        )

        pin = self.device.type == 'cuda'
        loader = DataLoader(FraudDataset(X_train_normal), batch_size=self.batch_size,
                            shuffle=True, pin_memory=pin)

        for epoch in range(self.epochs):
            self.model.train()
            total_recon = total_disc = total_gen = 0.0

            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                n = batch_x.size(0)

                # --- Phase 1: Reconstruction ---
                opt_recon.zero_grad()
                z = self.model.encode(batch_x)
                x_recon = self.model.decode(z)
                recon_loss = F.mse_loss(x_recon, batch_x, reduction='mean')
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_dec_params, max_norm=1.0)
                opt_recon.step()

                # --- Phase 2a: Discriminator ---
                opt_disc.zero_grad()
                z_prior = torch.randn(n, self.latent_dim, device=self.device)
                z_fake  = self.model.encode(batch_x).detach()

                d_real = self.model.discriminate(z_prior)
                d_fake = self.model.discriminate(z_fake)
                disc_loss = (F.binary_cross_entropy(d_real, torch.full_like(d_real, _REAL)) +
                             F.binary_cross_entropy(d_fake, torch.full_like(d_fake, _FAKE)))
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
                opt_disc.step()

                # --- Phase 2b: Generator (encoder adversarial) ---
                opt_gen.zero_grad()
                z_fake  = self.model.encode(batch_x)
                d_fake  = self.model.discriminate(z_fake)
                gen_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), max_norm=1.0)
                opt_gen.step()

                total_recon += recon_loss.item()
                total_disc  += disc_loss.item()
                total_gen   += gen_loss.item()

            n_batches = len(loader)
            sched_recon.step(total_recon / n_batches)

            if (epoch + 1) % 10 == 0:
                lr_now = opt_recon.param_groups[0]['lr']
                print(f"Epoch [{epoch+1:3d}/{self.epochs}]  "
                      f"recon={total_recon/n_batches:.4f}  "
                      f"disc={total_disc/n_batches:.4f}  "
                      f"gen={total_gen/n_batches:.4f}  "
                      f"lr={lr_now:.6f}")

        # Threshold: percentile of reconstruction errors on normal training data
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_train_normal).to(self.device)
            x_recon  = self.model.reconstruct(X_tensor)
            recon_errors = torch.mean((X_tensor - x_recon) ** 2, dim=1).cpu().numpy()
            self.threshold = np.percentile(recon_errors, self.threshold_percentile)

        print(f"Threshold ({self.threshold_percentile}th pct): {self.threshold:.6f}")

        dataset_name = os.path.basename(self.data_path).replace('.csv', '')
        self.save_checkpoint(f'src/model/checkpoints/aae_{dataset_name}.pt')
        return X_test, y_test

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state':         self.model.state_dict(),
            'input_dim':           self.model.encoder.net[0].in_features,
            'latent_dim':          self.latent_dim,
            'scaler':              self.scaler,
            'medians':             self.medians,
            'cols_to_keep':        list(self.cols_to_keep),
            'indicator_cols':      self.indicator_cols,
            'threshold':           self.threshold,
            'threshold_percentile': self.threshold_percentile,
        }, path)
        print(f"Checkpoint saved -> {path}")

    def load_full_data(self):
        """All samples (both classes) through the already-fitted preprocessing pipeline.
        Must be called after train() so scaler/medians are ready."""
        df = pd.read_csv(self.data_path)
        for m in self.merge_files:
            df = df.merge(pd.read_csv(m['path']), on=m['on'], how='left')

        y = df[self.target_column].values
        X = df.drop([self.target_column], axis=1)
        for col in self.id_columns:
            if col in X.columns:
                X = X.drop(col, axis=1)

        X = self._encode_categoricals(X)
        X = self._add_missing_indicators(X, fit=False)
        X = X[self.cols_to_keep]
        X_values = X.values
        X_values = self._fill_missing(X_values)
        X_values = self.scaler.transform(X_values)
        X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)
        return X_values, y

    def evaluate(self, X_test, y_test, output_name=None):
        if y_test is None:
            print("No labels available — skipping evaluation.")
            return

        if output_name is None:
            dataset_name = os.path.basename(self.data_path).replace('.csv', '')
            output_name = f'aae_{dataset_name}'

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            x_recon  = self.model.reconstruct(X_tensor)
            recon_errors = torch.mean((X_tensor - x_recon) ** 2, dim=1).cpu().numpy()

        y_pred = (recon_errors > self.threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        accuracy  = accuracy_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, recon_errors)
        pr_auc    = average_precision_score(y_test, recon_errors)
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
            f.write(f"Latent dim:  {self.latent_dim}\n")
            f.write(f"Epochs:      {self.epochs}\n")

        print(f"\nResults -> {out_path}")
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    aae_cc = FraudDetector(
        data_path='Datasets/creditcard.csv',
        target_column='Class',
        test_size=0.2,
        latent_dim=16,
        epochs=100,
    )
    X_test_cc, y_test_cc = aae_cc.train()
    aae_cc.evaluate(X_test_cc, y_test_cc)

    aae_ieee = FraudDetector(
        data_path='Datasets/train_transaction.csv',
        target_column='isFraud',
        id_columns=['TransactionID'],
        merge_files=[
            {'path': 'Datasets/train_identity.csv', 'on': 'TransactionID'}
        ],
        test_size=0.2,
        latent_dim=64,
        epochs=100,
    )
    X_test_ieee, y_test_ieee = aae_ieee.train()
    aae_ieee.evaluate(X_test_ieee, y_test_ieee)
