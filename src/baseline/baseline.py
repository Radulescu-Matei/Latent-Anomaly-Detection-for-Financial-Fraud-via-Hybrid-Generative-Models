import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, average_precision_score
import os

## Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    ##Encoder
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    ## Decoder
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

## Loss function
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

## Independent of dataset
class Fraud_Dataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

## Find if it's fraud or not
class Fraud_Classifier:
    def __init__(self, data_path, target_column, id_columns=None, merge_files=None,
                 test_data_path=None, test_merge_files=None, test_size=0.2,
                 latent_dim=32, batch_size=128, epochs=50, lr=0.001, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.id_columns = id_columns if id_columns is not None else []
        self.merge_files = merge_files if merge_files is not None else []
        self.test_data_path = test_data_path
        self.test_merge_files = test_merge_files if test_merge_files is not None else []
        self.test_size = test_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        
    ## Load dataset from multiple files / diffrent class and so on for generalization
    def load_data(self):
        df = pd.read_csv(self.data_path)
        
        for merge_info in self.merge_files:
            df_merge = pd.read_csv(merge_info['path'])
            df = df.merge(df_merge, on=merge_info['on'], how='left')
        
        y = df[self.target_column].values
        X = df.drop([self.target_column], axis=1)
        
        for id_col in self.id_columns:
            if id_col in X.columns:
                X = X.drop([id_col], axis=1)
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].factorize()[0]
        
        X = X.fillna(0).values
        
        if self.test_data_path is not None:
            df_test = pd.read_csv(self.test_data_path)
            
            for merge_info in self.test_merge_files:
                df_test_merge = pd.read_csv(merge_info['path'])
                df_test = df_test.merge(df_test_merge, on=merge_info['on'], how='left')
            
            if self.target_column in df_test.columns:
                y_test = df_test[self.target_column].values
                X_test = df_test.drop([self.target_column], axis=1)
            else:
                y_test = None
                X_test = df_test
            
            for id_col in self.id_columns:
                if id_col in X_test.columns:
                    X_test = X_test.drop([id_col], axis=1)
            
            for col in X_test.select_dtypes(include=['object']).columns:
                X_test[col] = X_test[col].factorize()[0]
            
            X_test = X_test.fillna(0).values
            
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
        
        X_train_normal = X_train[y_train == 0]
        
        X_train_normal = self.scaler.fit_transform(X_train_normal)
        X_test = self.scaler.transform(X_test)
        
        X_train_normal = np.nan_to_num(X_train_normal, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_train_normal, X_test, y_test
    
    ## Train the baseline
    def train(self):
        X_train_normal, X_test, y_test = self.load_data()
        
        input_dim = X_train_normal.shape[1]
        self.model = VAE(input_dim, self.latent_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        train_dataset = Fraud_Dataset(X_train_normal)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                x_recon, mu, logvar = self.model(batch_x)
                loss = vae_loss(x_recon, batch_x, mu, logvar)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        self.model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train_normal).to(self.device)
            x_recon, _, _ = self.model(X_train_tensor)
            recon_errors = torch.mean((X_train_tensor - x_recon) ** 2, dim=1).cpu().numpy()
            self.threshold = np.percentile(recon_errors, 95)
        
        return X_test, y_test
    

    ## Construct statistics
    def evaluate(self, X_test, y_test, output_name=None):
        if y_test is None:
            return
        
        if output_name is None:
            dataset_name = os.path.basename(self.data_path).replace('.csv', '')
            output_name = f'vae_baseline_{dataset_name}'
        
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            x_recon, _, _ = self.model(X_test_tensor)
            recon_errors = torch.mean((X_test_tensor - x_recon) ** 2, dim=1).cpu().numpy()
        
        recon_errors = np.nan_to_num(recon_errors, nan=1e10, posinf=1e10, neginf=0.0)
        
        y_pred = (recon_errors > self.threshold).astype(int)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, recon_errors)
        pr_auc = average_precision_score(y_test, recon_errors)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        os.makedirs('src/baseline/results', exist_ok=True)
        
        ## Statistics for documentation
        with open(f'src/baseline/results/{output_name}.txt', 'w') as f:
            f.write(f"Baseline for {output_name} file:\n\n")
            
            ## Classic
            f.write("Classification Metrics:\n")
            f.write(f"Accuracy:           {accuracy:.4f}\n")
            f.write(f"Precision:          {precision:.4f}\n")
            f.write(f"Recall:             {recall:.4f}\n")
            f.write(f"F1-Score:           {f1:.4f}\n\n")
            
            ## Anomaly
            f.write("Anomaly scoring:\n")
            f.write(f"ROC-AUC:            {roc_auc:.4f}\n")
            f.write(f"PR-AUC:             {pr_auc:.4f}\n\n")
            
            ## Confusion Matrix
            f.write("Confusion Matrix:\n")
            f.write(f"TN:     {tn}\n")
            f.write(f"FP:    {fp}\n")
            f.write(f"FN:    {fn}\n")
            f.write(f"TP:     {tp}\n\n")
            

            ## If error > than it's probably fraud
            f.write(f"Threshold:{self.threshold:.6f}")