import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    # Load preprocessed data
    train_df = pd.read_csv("../data/train_processed.csv")
    
    # Separate features and target
    X = train_df.drop(["PassengerId", "Survived"], axis=1)
    y = train_df["Survived"]
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    rf = RandomForestClassifier(random_state=42)
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 5, 6, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_rf_model = grid_search.best_estimator_
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Validate best model on validation set
    y_pred = best_rf_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Validation Accuracy with best model: {accuracy:.4f}")
    print("\nClassification Report with best model:")
    print(classification_report(y_val, y_pred))
    
    # Cross-validation with best model
    cv_scores = cross_val_score(best_rf_model, X, y, cv=5)
    print(f"\nCross-validation scores with best model: {cv_scores}")
    print(f"Mean CV accuracy with best model: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance with best model:")
    print(feature_importance)
    
    # Save the best model
    joblib.dump(best_rf_model, "../models/titanic_model.pkl")
    print("\nBest model saved to models/titanic_model.pkl")
    
    return best_rf_model

if __name__ == "__main__":
    model = train_model()

