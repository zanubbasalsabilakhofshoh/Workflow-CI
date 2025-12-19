import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ci-training")

    DATA_PATH = "../MLProject/dataset_preprocessed/heart_disease_preprocessing.csv"

    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="rf_ci_run"):
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy_manual", acc)

        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

    os.makedirs("outputs", exist_ok=True)

    run_artifacts = os.path.join("mlruns", "0", run_id, "artifacts")
    if os.path.exists(run_artifacts):
        shutil.make_archive("outputs/mlflow_artifacts", "zip", run_artifacts)

    print("Training selesai (CI). Run ID:", run_id)

if __name__ == "__main__":
    main()
