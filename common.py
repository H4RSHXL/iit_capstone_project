"""Shared helpers: data loading, output formatting, and the pinned Part 2 split.
Importing this file never runs a Part."""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "records.csv")
OUT_DIR = os.path.join(ROOT, "outputs")


def load_data():
    """Return (X, outcome, value). X has 7 columns X1..X7 in file order."""
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = list(reader)
    arr = np.array([[float(v) for v in r[1:]] for r in rows])  # drop record_id
    X = arr[:, 0:7]
    outcome = arr[:, 7].astype(int)
    value = arr[:, 8]
    return X, outcome, value


def fmt(v):
    """Pinned printing: 4-decimal floats, plain integers, yes/no booleans."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%.4f" % float(v)
    return str(v)


class Results:
    """Collects `key = value` lines and writes them to outputs/<name>."""

    def __init__(self):
        self.items = []

    def add(self, key, value):
        self.items.append((key, value))

    def write(self, name):
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, name), "w") as f:
            for k, v in self.items:
                f.write("%s = %s\n" % (k, fmt(v)))


def write_csv(name, header, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def stratified_split(outcome):
    """Task 2.1: per class, first floor(7*count/10) rows (file order) -> train."""
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = [i for i in range(len(outcome)) if outcome[i] == cls]
        cut = (7 * len(idx)) // 10
        train_idx += idx[:cut]
        test_idx += idx[cut:]
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))
