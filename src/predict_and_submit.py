import pandas as pd
import joblib
from advanced_preprocess import advanced_preprocess

if __name__ == '__main__':
    # Load the trained models and scaler
    rf_model = joblib.load('../models/rf_model.pkl')
    gb_model = joblib.load('../models/gb_model.pkl')
    lr_model = joblib.load('../models/lr_model.pkl')
    svm_model = joblib.load('../models/svm_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')

    # Load original test data to get PassengerId
    test_df_original = pd.read_csv('../data/test.csv')

    # Preprocess test data using the advanced preprocessor
    test_df_processed = advanced_preprocess(test_df_original.copy())

    # Load train_advanced.csv to ensure consistent columns
    train_df_for_cols = pd.read_csv('../data/train_advanced.csv')
    
    # Align columns of test_df_processed with train_df_for_cols
    # Get columns from training data (excluding 'PassengerId' and 'Survived')
    train_cols_for_prediction = [col for col in train_df_for_cols.columns if col not in ['PassengerId', 'Survived']]
    
    # Add missing columns to test set and fill with 0
    for col in train_cols_for_prediction:
        if col not in test_df_processed.columns:
            test_df_processed[col] = 0
            
    # Ensure the order of columns is the same as in training data
    test_df_processed = test_df_processed[train_cols_for_prediction]

    # Scale the test data for LR and SVM
    test_df_scaled = scaler.transform(test_df_processed)

    # Get probabilities from each model
    rf_proba = rf_model.predict_proba(test_df_processed)[:, 1]
    gb_proba = gb_model.predict_proba(test_df_processed)[:, 1]
    lr_proba = lr_model.predict_proba(test_df_scaled)[:, 1]
    svm_proba = svm_model.predict_proba(test_df_scaled)[:, 1]

    # Weighted average for ensemble prediction
    ensemble_proba = (0.3 * rf_proba + 0.3 * gb_proba + 0.2 * lr_proba + 0.2 * svm_proba)
    predictions = (ensemble_proba > 0.5).astype(int)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'PassengerId': test_df_original['PassengerId'],
        'Survived': predictions
    })

    # Save submission file
    submission_df.to_csv('../submission.csv', index=False)

    print('Submission file created successfully at submission.csv')


