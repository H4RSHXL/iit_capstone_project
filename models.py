"""Shared model code for Parts 2 and 4 (logistic regression, KNN, metrics).
Importing this file never runs a Part."""
import math

import numpy as np

from common import load_data, stratified_split


def prepare():
    """Task 2.1: stratified split + standardise with TRAIN mean / sample SD."""
    X, outcome, value = load_data()
    tr, te = stratified_split(outcome)
    Xtr, Xte = X[tr], X[te]
    mean = Xtr.mean(axis=0)
    sd = np.sqrt(((Xtr - mean) ** 2).sum(axis=0) / (len(Xtr) - 1))
    return {
        "X": X, "outcome": outcome, "value": value,
        "tr": tr, "te": te,
        "Xtr_raw": Xtr, "Xte_raw": Xte,
        "Xtr": (Xtr - mean) / sd, "Xte": (Xte - mean) / sd,
        "ytr": outcome[tr], "yte": outcome[te],
    }


# ---------------- logistic regression ----------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(X, y, w, b):
    """Mean binary cross-entropy, computed stably via logaddexp."""
    z = X @ w + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def train_logistic(X, y, eta, epochs, record=()):
    """Batch gradient descent from zeros. Returns w, b, {epoch: loss}."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(X, y, w, b)
    for t in range(1, epochs + 1):
        err = sigmoid(X @ w + b) - y
        w = w - eta * (X.T @ err) / n
        b = b - eta * float(err.sum()) / n
        if t in record:
            losses[t] = bce(X, y, w, b)
    return w, b, losses


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


# ---------------- metrics ----------------
def confusion(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return tp, fp, tn, fn


def _div(a, b):
    return a / b if b else 0.0


def rates(tp, fp, tn, fn):
    acc = _div(tp + tn, tp + fp + tn + fn)
    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    f1 = _div(2 * prec * rec, prec + rec)
    spec = _div(tn, tn + fp)
    return acc, prec, rec, f1, spec


# ---------------- K-nearest neighbours ----------------
def knn_positive_counts(ref_X, ref_y, query_X, ks):
    """For each K in ks, the number of positive neighbours of every query row.
    Neighbours are ranked by (distance, file-order position): stable argsort."""
    d2 = ((query_X[:, None, :] - ref_X[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(d2, axis=1, kind="stable")
    sorted_labels = ref_y[order]                    # labels in neighbour order
    csum = np.cumsum(sorted_labels, axis=1)
    return {k: csum[:, k - 1] for k in ks}


def knn_predict(counts, k):
    return counts >= (k + 1) / 2.0


def choose_k(Xtr, ytr, ks=tuple(range(1, 22, 2))):
    """Task 2.7: 5-fold CV (fold = position in train set mod 5), mean held-out F1.
    Returns (list of (K, mean_cv_f1), best K). Ties go to the smaller K."""
    ks = list(ks)
    pos = np.arange(len(ytr))
    scores = {k: [] for k in ks}
    for fold in range(5):
        held = pos % 5 == fold
        counts = knn_positive_counts(Xtr[~held], ytr[~held], Xtr[held], ks)
        for k in ks:
            tp, fp, tn, fn = confusion(ytr[held], knn_predict(counts[k], k))
            scores[k].append(rates(tp, fp, tn, fn)[3])
    rows = [(k, float(np.mean(scores[k]))) for k in ks]
    best = rows[0]
    for r in rows:
        if r[1] > best[1]:            # strict '>' keeps the smaller K on ties
            best = r
    return rows, best[0]
