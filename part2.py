"""Part 2 - classification and evaluation (logistic regression, KNN, cost-based threshold)."""
import numpy as np

from common import Results, write_csv
from models import (bce, choose_k, confusion, knn_positive_counts, knn_predict,
                    predict_proba, prepare, rates, train_logistic)

D = prepare()
Xtr, Xte, ytr, yte = D["Xtr"], D["Xte"], D["ytr"], D["yte"]
R = Results()

# ---- 2.1 split
R.add("train_rows", len(ytr))
R.add("train_positives", int(ytr.sum()))
R.add("test_rows", len(yte))
R.add("test_positives", int(yte.sum()))

# ---- 2.2 logistic regression from scratch (eta 0.1, 1000 updates)
w, b, losses = train_logistic(Xtr, ytr, 0.1, 1000, record=(0, 1, 10, 100, 1000))
for e in (0, 1, 10, 100, 1000):
    R.add("loss_epoch_%d" % e, losses[e])
for j in range(7):
    R.add("weight_X%d" % (j + 1), float(w[j]))
R.add("bias", b)
R.add("test_bce", bce(Xte, yte, w, b))

# ---- 2.3 learning-rate effect (200 updates)
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
    R.add("loss_epoch_200_eta_%s" % eta, l[200])

# ---- 2.4 odds ratios
for j in range(7):
    R.add("odds_ratio_X%d" % (j + 1), float(np.exp(w[j])))

# ---- 2.5 two thresholds on the test rows
p_te = predict_proba(Xte, w, b)
p_tr = predict_proba(Xtr, w, b)
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(yte, p_te >= tau)
    acc, prec, rec, f1, spec = rates(tp, fp, tn, fn)
    for name, val in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", acc),
                      ("precision", prec), ("recall", rec), ("f1", f1), ("specificity", spec)):
        R.add("tau_%s_%s" % (tau, name), val)
R.add("blind_accuracy", float(np.mean(yte == 0)))

# ---- 2.6 threshold sweep
rows = []
for k in range(1, 10):
    tau = k / 10.0
    tp, fp, tn, fn = confusion(yte, p_te >= tau)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    rows.append([tau, tp, fp, tn, fn, tpr, fpr])
write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# ---- 2.7 KNN with 5-fold CV (fold = position in train set mod 5)
cv_rows, best_k = choose_k(Xtr, ytr)
write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [list(r) for r in cv_rows])
R.add("knn_best_k", best_k)
counts_te = knn_positive_counts(Xtr, ytr, Xte, [best_k])
tp, fp, tn, fn = confusion(yte, knn_predict(counts_te[best_k], best_k))
acc, prec, rec, f1, _ = rates(tp, fp, tn, fn)
for name, val in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", acc),
                  ("precision", prec), ("recall", rec), ("f1", f1)):
    R.add("knn_" + name, val)


# ---- 2.8 choose the threshold with money (train rows only)
def cost(y, pred):
    tp, fp, tn, fn = confusion(y, pred)
    return 50 * fp + 400 * fn


best_tau, best_cost = None, None
for k in range(1, 20):
    tau = k / 20.0
    c = cost(ytr, p_tr >= tau)
    if best_cost is None or c < best_cost:      # strict '<': ties go to smaller tau
        best_tau, best_cost = tau, c
R.add("cost_tau", best_tau)
R.add("cost_train_at_tau", best_cost)
R.add("cost_test_at_tau", cost(yte, p_te >= best_tau))
R.add("cost_test_at_0.5", cost(yte, p_te >= 0.5))
R.add("cost_test_flag_nobody", cost(yte, np.zeros(len(yte), dtype=bool)))

R.write("part2_results.txt")
