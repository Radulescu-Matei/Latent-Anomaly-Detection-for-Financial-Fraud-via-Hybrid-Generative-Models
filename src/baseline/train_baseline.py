from baseline import Fraud_Classifier

if __name__ == "__main__":
        vae_cc = Fraud_Classifier(
                data_path='Datasets/creditcard.csv',
                target_column='Class',
                test_size=0.2,
                epochs=50
            )
        X_test_cc, y_test_cc = vae_cc.train()
        vae_cc.evaluate(X_test_cc, y_test_cc)

        vae_ieee = Fraud_Classifier(
            data_path='Datasets/train_transaction.csv',
            target_column='isFraud',
            id_columns=['TransactionID'],
            merge_files=[
                {'path': 'Datasets/train_identity.csv', 'on': 'TransactionID'}
            ],
            test_size=0.2,
            epochs=50
        )
        X_test_ieee, y_test_ieee = vae_ieee.train()
        vae_ieee.evaluate(X_test_ieee, y_test_ieee)
