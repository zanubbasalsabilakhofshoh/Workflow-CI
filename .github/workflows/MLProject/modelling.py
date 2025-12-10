
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset setelah preprocessing
df = pd.read_csv("namadataset_preprocessing/dataset_clean.csv")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="model_autolog"):
    mlflow.sklearn.autolog()
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

print("Training selesai: modelling.py")
