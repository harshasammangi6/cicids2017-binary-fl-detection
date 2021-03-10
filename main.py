import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run():
    df = pd.read_csv('data/sample.csv')
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, preds)}')

if __name__ == '__main__':
    run()
