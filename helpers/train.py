from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    matthews_corrcoef, classification_report, confusion_matrix
)
import pandas as pd
import numpy as np
import time

def train_logistic_regression(X, y) -> tuple[pd.DataFrame, str, np.ndarray]:
    """
    Train and evaluate only a Logistic Regression model.
    
    Args:
        X (np.ndarray or pd.DataFrame): Feature matrix.
        y (np.ndarray or pd.Series): Target labels.
    
    Returns:
        tuple: DataFrame with metrics, classification report string, confusion matrix.
    """
    random_state = 271828
    logistic_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            solver='lbfgs',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    start_time = time.time()
    
    # Cross-validated predictions
    y_pred = cross_val_predict(logistic_pipeline, X, y, cv=cv, method="predict", n_jobs=-1)
    
    # Evaluation
    f1 = f1_score(y, y_pred, average='micro')
    balanced_acc = balanced_accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    class_report = classification_report(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    elapsed = time.time() - start_time
    
    print(f'F1 Score: {f1:.4f}')
    print(f'Balanced Accuracy: {balanced_acc:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f'Matthews Corr Coef: {mcc:.4f}')
    print(f'Elapsed Time: {elapsed:.2f}s\n')
    print(class_report)
    
    return conf_matrix
