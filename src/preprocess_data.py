
import pandas as pd

def preprocess(df):
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Convert categorical features to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Drop unnecessary features
    df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

    return df

if __name__ == '__main__':
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    train_df_processed = preprocess(train_df.copy())
    test_df_processed = preprocess(test_df.copy())

    train_df_processed.to_csv('../data/train_processed.csv', index=False)
    test_df_processed.to_csv('../data/test_processed.csv', index=False)

    print('Data preprocessing complete. Processed files saved to data/train_processed.csv and data/test_processed.csv')


