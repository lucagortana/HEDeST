from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_confusion_matrices(conf_matrices, labels, out_dir):
    with pd.ExcelWriter(out_dir) as writer:
        for i, cm in enumerate(conf_matrices):
            df_cm = pd.DataFrame(cm, index=labels, columns=labels)
            df_cm.to_excel(writer, sheet_name=f"Seed_{42+i}")
    logger.info(f"Confusion matrices saved to {out_dir}.")


def load_data(features_file, ground_truth_file):
    features_dict = torch.load(features_file)
    features = {k: v.numpy() for k, v in features_dict.items()}

    df = pd.read_csv(ground_truth_file)
    df.set_index("nucleus_id", inplace=True)
    df.index = df.index.astype(str)

    labels = df.idxmax(axis=1)

    X = np.array([features[str(i)] for i in list(features.keys())])
    y = np.array([labels[str(i)] for i in list(features.keys())])

    return X, y, sorted(df.columns.tolist())


def train_xgboost(X, y, balance, rs):
    classes = np.unique(y)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y_encoded = np.array([class_to_idx[label] for label in y])
    logger.info("Data encoded.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=rs
    )
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=rs)
    logger.info("Data split into train, validation, and test sets.")

    if balance:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    else:
        sample_weights = None

    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(classes),
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=10,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=1,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train, sample_weight=sample_weights, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True
    )

    return model, y_train, y_test, model.predict(X_train), model.predict(X_test)


def train_logistic_regression(X, y, balance, rs, penalty, C):
    classes = np.unique(y)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y_encoded = np.array([class_to_idx[label] for label in y])
    logger.info("Data encoded.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=rs
    )
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=rs)
    logger.info("Data split into train, validation, and test sets.")

    solver = "saga" if penalty in ["l1", "elasticnet"] else "lbfgs"

    model = LogisticRegression(
        multi_class="multinomial",
        penalty=penalty,
        C=C,
        solver=solver,
        l1_ratio=0.5 if penalty == "elasticnet" else None,
        class_weight="balanced" if balance else None,
        max_iter=5000,
        random_state=rs,
        verbose=1,
    )

    model.fit(X_train, y_train)
    logger.info("Logistic Regression model trained.")
    return model, y_train, y_test, model.predict(X_train), model.predict(X_test)


def evaluate_model(y_true, y_pred):
    metrics = {
        "Global Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Weighted F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "Weighted Precision": precision_score(y_true, y_pred, average="weighted"),
        "Weighted Recall": recall_score(y_true, y_pred, average="weighted"),
    }
    return metrics


def compute_statistics(metrics_list):
    """
    Compute mean and confidence intervals for a list of metrics.

    Args:
        metrics_list (List[Dict[str, float]]): List of dictionaries containing metrics from each run.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Tuple containing the mean and confidence intervals for each metric.
    """

    df_metrics = pd.DataFrame(metrics_list)
    mean_values = df_metrics.mean().to_dict()
    std_values = df_metrics.std()
    count_values = df_metrics.count()
    se_values = std_values / np.sqrt(count_values)
    ci_values = {f"{key} ci": 1.96 * se for key, se in se_values.items()}

    return mean_values, ci_values


def main():
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier on image features.")
    parser.add_argument("--features_file", type=str, required=True, help="Path to the feature dictionary")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth CSV file")
    parser.add_argument("--out_path", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--classifier", type=str, choices=["xgboost", "logistic"], default="xgboost")
    parser.add_argument("--balance", action="store_true", help="Enable class balancing")
    parser.add_argument(
        "--penalty",
        type=str,
        choices=["l1", "l2", "elasticnet"],
        default="l2",
        help="Regularization type for logistic regression",
    )
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for logistic regression")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to use")

    args = parser.parse_args()

    X, y, labels = load_data(args.features_file, args.ground_truth_file)
    logger.info("Data loaded.")

    metrics_train_list = []
    metrics_test_list = []
    conf_matrices = []

    for i in range(args.num_seeds):
        logger.info(f"Running seed {i+1}/{args.num_seeds}...")

        if args.classifier == "xgboost":
            model, y_train, y_test, y_pred_on_train, y_pred_on_test = train_xgboost(X, y, args.balance, rs=42 + i)
        else:
            model, y_train, y_test, y_pred_on_train, y_pred_on_test = train_logistic_regression(
                X, y, args.balance, rs=42 + i, penalty=args.penalty, C=args.C
            )

        metrics_train_list.append(evaluate_model(y_train, y_pred_on_train))
        metrics_test_list.append(evaluate_model(y_test, y_pred_on_test))
        conf_matrices.append(confusion_matrix(y_test, y_pred_on_test))

    temp_metrics_train, temp_metrics_train_ci = compute_statistics(metrics_train_list)
    temp_metrics_test, temp_metrics_test_ci = compute_statistics(metrics_test_list)
    final_metrics_train = {**temp_metrics_train, **temp_metrics_train_ci}
    final_metrics_test = {**temp_metrics_test, **temp_metrics_test_ci}

    if args.classifier == "xgboost":
        cm_file = f"{args.out_path}/cm_{args.classifier}_balance_{args.balance}.xlsx"
        metrics_train_file = f"{args.out_path}/metrics_train_{args.classifier}_balance_{args.balance}.json"
        metrics_test_file = f"{args.out_path}/metrics_test_{args.classifier}_balance_{args.balance}.json"
    else:
        cm_file = f"{args.out_path}/cm_{args.classifier}_balance_{args.balance}_penalty_{args.penalty}_C_{args.C}.xlsx"
        metrics_train_file1 = f"{args.out_path}/metrics_train_{args.classifier}_"
        metrics_train_file2 = f"balance_{args.balance}_penalty_{args.penalty}_C_{args.C}.json"
        metrics_train_file = metrics_train_file1 + metrics_train_file2

        metrics_test_file1 = f"{args.out_path}/metrics_test_{args.classifier}_"
        metrics_test_file2 = f"balance_{args.balance}_penalty_{args.penalty}_C_{args.C}.json"
        metrics_test_file = metrics_test_file1 + metrics_test_file2

    save_confusion_matrices(conf_matrices, labels, cm_file)

    with open(metrics_train_file, "w") as f:
        json.dump(final_metrics_train, f, indent=4)
    with open(metrics_test_file, "w") as f:
        json.dump(final_metrics_test, f, indent=4)
    logger.info(f"Metrics saved to {metrics_train_file} and {metrics_test_file}.")


if __name__ == "__main__":
    main()
