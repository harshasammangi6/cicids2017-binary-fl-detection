import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and scaling features.
    """
    # Handle missing values (if any)
    df = df.dropna()

    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def train_model(X_train, y_train):
    """
    Train a logistic regression model with hyperparameter tuning.
    """
    # Define the model
    model = LogisticRegression()

    # Define hyperparameters for tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Return the best model
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print metrics.
    """
    preds = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, preds)}')
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

def run():
    # Load the dataset
    df = pd.read_csv('data/sample.csv')

    # Preprocess the data
    X, y = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the trained model
    dump(model, 'logistic_regression_model.joblib')
    print("\nModel saved as 'logistic_regression_model.joblib'")

if __name__ == '__main__':
    run()