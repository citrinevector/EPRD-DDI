import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score


def metric(y_true, y_pred, empty=-1):
    y_pred = np.argmax(y_pred, axis=1)

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return {"precision": precision, "f1_macro": f1_macro, "recall": recall, "acc": acc, "kappa": kappa}
