This project explores a fraud detection system that combines two types of neural networks to better spot suspicious transactions. One part of the system learns how normal transactions look and tries to recreate them, while another part checks whether those reconstructions still match real-world transaction patterns. If a transaction is hard to recreate or looks unrealistic compared to normal data, it gets flagged as potentially fraudulent.

By mixing these two approaches, the system is designed to handle highly imbalanced data (where fraud is rare) and adapt as transaction behavior changes over time. This makes it a flexible starting point for building more reliable fraud detection models.

The project will be developed step by step. It will start with simple baseline models and autoencoders trained on public fraud datasets. From there, the combined model will be added and improved, followed by work on better anomaly scoring and evaluation. Later stages will focus on improving stability, robustness, and understanding what the model learns internally.

FROM PARENT DIRECTORY :
RUN EDA:
python3 src/eda/class_imbalance.py
python3 src/eda/feature_correlation.py
python3 src/eda/mising_values.py - This is on IEEE dataset only ast the credit card dataset has no missing values.
python3 src/eda/outliers.py
python3 src/eda/time_patterns.py
python3 src/eda/transaction_amount_analysis
TRAIN BASELINE:
src/baseline/baseline.py - has the actual model which by changing parameters is evaluated on diffrent dataset.
python3 src/baseline/train_baseline.py
