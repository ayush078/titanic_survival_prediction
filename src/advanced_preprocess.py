import pandas as pd
import numpy as np
import re

def extract_title(name):
    """Extract title from name"""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def advanced_preprocess(df):
    """Advanced preprocessing with more feature engineering"""
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Extract titles from names
    df['Title'] = df['Name'].apply(extract_title)
    
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Map titles to numbers
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    
    # Handle missing values more intelligently
    # Age: Fill based on title and class
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Embarked: Fill with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Fare: Fill based on class
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[1, 2, 3, 4, 5])
    df['AgeGroup'] = df['AgeGroup'].astype(int)
    
    # Fare groups
    df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=[1, 2, 3, 4])
    df['FareGroup'] = df['FareGroup'].astype(int)
    
    # Family size groups
    df['FamilySizeGroup'] = 0
    df.loc[df['FamilySize'] == 1, 'FamilySizeGroup'] = 1  # Alone
    df.loc[(df['FamilySize'] >= 2) & (df['FamilySize'] <= 4), 'FamilySizeGroup'] = 2  # Small
    df.loc[df['FamilySize'] >= 5, 'FamilySizeGroup'] = 3  # Large
    
    # Interaction features
    df['Sex_Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
    df['Age_Pclass'] = df['AgeGroup'].astype(str) + '_' + df['Pclass'].astype(str)
    
    # Convert categorical features to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # One-hot encode interaction features
    df = pd.get_dummies(df, columns=['Sex_Pclass', 'Age_Pclass'], prefix=['SexPclass', 'AgePclass'])
    
    # Drop unnecessary features
    df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)
    
    return df

if __name__ == '__main__':
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    train_df_processed = advanced_preprocess(train_df.copy())
    test_df_processed = advanced_preprocess(test_df.copy())

    # Ensure both datasets have the same columns
    train_cols = set(train_df_processed.columns)
    test_cols = set(test_df_processed.columns)
    
    # Add missing columns to test set
    for col in train_cols - test_cols:
        if col != 'Survived':  # Don't add target column to test set
            test_df_processed[col] = 0
    
    # Remove extra columns from test set
    for col in test_cols - train_cols:
        test_df_processed.drop(col, axis=1, inplace=True)
    
    # Reorder columns to match
    if 'Survived' in train_df_processed.columns:
        cols = ['PassengerId', 'Survived'] + [col for col in train_df_processed.columns if col not in ['PassengerId', 'Survived']]
        train_df_processed = train_df_processed[cols]
    
    test_cols_ordered = ['PassengerId'] + [col for col in test_df_processed.columns if col != 'PassengerId']
    test_df_processed = test_df_processed[test_cols_ordered]

    train_df_processed.to_csv('../data/train_advanced.csv', index=False)
    test_df_processed.to_csv('../data/test_advanced.csv', index=False)

    print('Advanced data preprocessing complete.')
    print(f'Training data shape: {train_df_processed.shape}')
    print(f'Test data shape: {test_df_processed.shape}')

