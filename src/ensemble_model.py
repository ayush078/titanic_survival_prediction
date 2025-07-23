import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

def train_ensemble_model():
    # Load advanced preprocessed data
    train_df = pd.read_csv("../data/train_advanced.csv")
    
    # Separate features and target
    X = train_df.drop(["PassengerId", "Survived"], axis=1)
    y = train_df["Survived"]
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define individual models
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    lr = LogisticRegression(
        C=1.0,
        random_state=42,
        max_iter=1000
    )
    
    svm = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=42
    )
    
    # Train individual models
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_pred)
    print(f"Random Forest Validation Accuracy: {rf_acc:.4f}")
    
    print("Training Gradient Boosting...")
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_val)
    gb_acc = accuracy_score(y_val, gb_pred)
    print(f"Gradient Boosting Validation Accuracy: {gb_acc:.4f}")
    
    print("Training Logistic Regression...")
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_val_scaled)
    lr_acc = accuracy_score(y_val, lr_pred)
    print(f"Logistic Regression Validation Accuracy: {lr_acc:.4f}")
    
    print("Training SVM...")
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_val_scaled)
    svm_acc = accuracy_score(y_val, svm_pred)
    print(f"SVM Validation Accuracy: {svm_acc:.4f}")
    
    # Create ensemble with voting
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr),
            ('svm', svm)
        ],
        voting='soft'
    )
    
    # For ensemble, we need to handle scaling for some models
    # We'll create a custom ensemble approach
    
    # Get probabilities from each model
    rf_proba = rf.predict_proba(X_val)[:, 1]
    gb_proba = gb.predict_proba(X_val)[:, 1]
    lr_proba = lr.predict_proba(X_val_scaled)[:, 1]
    svm_proba = svm.predict_proba(X_val_scaled)[:, 1]
    
    # Weighted average (you can adjust weights based on individual performance)
    ensemble_proba = (0.3 * rf_proba + 0.3 * gb_proba + 0.2 * lr_proba + 0.2 * svm_proba)
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    ensemble_acc = accuracy_score(y_val, ensemble_pred)
    print(f"Ensemble Validation Accuracy: {ensemble_acc:.4f}")
    
    print("\nEnsemble Classification Report:")
    print(classification_report(y_val, ensemble_pred))
    
    # Cross-validation for ensemble
    # For simplicity, we'll use the Random Forest for CV (best single model typically)
    cv_scores = cross_val_score(rf, X, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Feature Importances:")
    print(feature_importance.head(15))
    
    # Save models
    joblib.dump(rf, "../models/rf_model.pkl")
    joblib.dump(gb, "../models/gb_model.pkl")
    joblib.dump(lr, "../models/lr_model.pkl")
    joblib.dump(svm, "../models/svm_model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    
    print("\nModels saved successfully!")
    
    return rf, gb, lr, svm, scaler

if __name__ == "__main__":
    models = train_ensemble_model()

