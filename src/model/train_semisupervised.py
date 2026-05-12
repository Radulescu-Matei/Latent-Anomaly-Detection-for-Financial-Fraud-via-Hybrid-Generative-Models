import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score, average_precision_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from train_aae import FraudDetector


class LabeledDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LatentClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        hidden = max(64, latent_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(1)


class SemiSupervisedDetector:
    def __init__(self, detector: FraudDetector):
        self.detector = detector
        self.device = detector.device
        self.classifier = None
        self._Z_test = None
        self._y_test = None

    def _encode_all(self):
        X_all, y_all = self.detector.load_full_data()

        self.detector.model.eval()
        latent_vectors = []
        with torch.no_grad():
            for i in range(0, len(X_all), 1024):
                batch = torch.FloatTensor(X_all[i:i + 1024]).to(self.device)
                z = self.detector.model.encode(batch)
                latent_vectors.append(z.cpu().numpy())

        return np.vstack(latent_vectors), y_all

    def train_classifier(self, epochs=50, lr=1e-3, batch_size=256):
        print("\n--- Semi-supervised: training latent classifier ---")
        Z_all, y_all = self._encode_all()

        Z_train, Z_test, y_train, y_test = train_test_split(
            Z_all, y_all,
            test_size=self.detector.test_size,
            random_state=self.detector.random_state,
            stratify=y_all,
        )

        n_normal = (y_train == 0).sum()
        n_fraud  = (y_train == 1).sum()
        pos_weight = torch.tensor([n_normal / n_fraud], device=self.device)
        print(f"Train: {n_normal} normal, {n_fraud} fraud | pos_weight={pos_weight.item():.1f}")

        latent_dim = Z_train.shape[1]
        self.classifier = LatentClassifier(latent_dim).to(self.device)
        optimizer = optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        pin = self.device.type == 'cuda'
        loader = DataLoader(LabeledDataset(Z_train, y_train),
                            batch_size=batch_size, shuffle=True, pin_memory=pin)

        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0.0
            for z_batch, y_batch in loader:
                z_batch = z_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.classifier(z_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}]  loss={total_loss/len(loader):.4f}")

        self._Z_test = Z_test
        self._y_test = y_test
        print("Classifier training complete.")

    def evaluate(self, output_name=None):
        if output_name is None:
            dataset_name = os.path.basename(self.detector.data_path).replace('.csv', '')
            output_name = f'aae_semisupervised_{dataset_name}'

        self.classifier.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(self._Z_test).to(self.device)
            scores = torch.sigmoid(self.classifier(Z_tensor)).cpu().numpy()

        y_pred = (scores > 0.5).astype(int)
        y_test = self._y_test

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        accuracy  = accuracy_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, scores)
        pr_auc    = average_precision_score(y_test, scores)
        cm        = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        os.makedirs('src/model/results', exist_ok=True)
        out_path = f'src/model/results/{output_name}.txt'
        with open(out_path, 'w') as f:
            f.write(f"AAE Semi-supervised Results for {output_name}:\n\n")
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
            f.write(f"Latent dim:  {self.detector.latent_dim}\n")
            f.write(f"AAE Epochs:  {self.detector.epochs}\n")
            f.write(f"Clf Epochs:  50\n")

        print(f"\nResults -> {out_path}")
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    # --- Credit card ---
    detector_cc = FraudDetector(
        data_path='Datasets/creditcard.csv',
        target_column='Class',
        test_size=0.2,
        latent_dim=16,
        epochs=100,
    )
    X_test_cc, y_test_cc = detector_cc.train()
    detector_cc.evaluate(X_test_cc, y_test_cc)

    ss_cc = SemiSupervisedDetector(detector_cc)
    ss_cc.train_classifier(epochs=50)
    ss_cc.evaluate()

    # --- IEEE-CIS ---
    detector_ieee = FraudDetector(
        data_path='Datasets/train_transaction.csv',
        target_column='isFraud',
        id_columns=['TransactionID'],
        merge_files=[{'path': 'Datasets/train_identity.csv', 'on': 'TransactionID'}],
        test_size=0.2,
        latent_dim=64,
        epochs=100,
    )
    X_test_ieee, y_test_ieee = detector_ieee.train()
    detector_ieee.evaluate(X_test_ieee, y_test_ieee)

    ss_ieee = SemiSupervisedDetector(detector_ieee)
    ss_ieee.train_classifier(epochs=50)
    ss_ieee.evaluate()
