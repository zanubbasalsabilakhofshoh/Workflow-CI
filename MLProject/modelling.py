import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('../dataset_preprocessed/heart_disease_preprocessing.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", accuracy)

with open('outputs/model.pkl', 'wb') as f:
    pickle.dump(model, f)

mlflow.log_artifact('outputs/model.pkl')

print(f"Accuracy: {accuracy}")