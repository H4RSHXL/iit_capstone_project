"""Part 4 - model choice, AdaBoost by hand, and neural-network sizing."""
import math

import numpy as np

from common import Results, write_csv
from models import (choose_k, confusion, knn_positive_counts, knn_predict, predict_proba,
                    prepare, rates, train_logistic)

D = prepare()
ytr, yte = D["ytr"], D["yte"]
Xtr_raw, Xte_raw = D["Xtr_raw"], D["Xte_raw"]
R = Results()


# ---------------- 4.1 which split is purest? ----------------
def entropy(p_, n_):
    if n_ == 0:
        return 0.0
    q = p_ / n_
    return sum(-v * math.log2(v) for v in (q, 1 - q) if v > 0)


def gini(p_, n_):
    if n_ == 0:
        return 0.0
    q = p_ / n_
    return 1 - q * q - (1 - q) * (1 - q)


# yes-side conditions on ORIGINAL units; columns: X1=0, X3=2, X6=5, X7=6
SPLITS = {
    "S1": lambda A: A[:, 5] == 1,
    "S2": lambda A: A[:, 6] == 1,
    "S3": lambda A: A[:, 2] > 0.5,
    "S4": lambda A: A[:, 0] <= 24,
}
N_TR, P_TR = len(ytr), int(ytr.sum())
root_e, root_g = entropy(P_TR, N_TR), gini(P_TR, N_TR)
R.add("root_entropy", root_e)
R.add("root_gini", root_g)

split_rows, ig, gg = [], {}, {}
for name, cond in SPLITS.items():
    yes = cond(Xtr_raw)
    n_yes, n_no = int(yes.sum()), int((~yes).sum())
    p_yes, p_no = int(ytr[yes].sum()), int(ytr[~yes].sum())
    ig[name] = root_e - (n_yes / N_TR * entropy(p_yes, n_yes) + n_no / N_TR * entropy(p_no, n_no))
    gg[name] = root_g - (n_yes / N_TR * gini(p_yes, n_yes) + n_no / N_TR * gini(p_no, n_no))
    split_rows.append([name, n_yes, p_yes, n_no, p_no, ig[name], gg[name]])
write_csv("part4_splits.csv",
          ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
          split_rows)
best_ig = max(SPLITS, key=lambda s: (ig[s], -list(SPLITS).index(s)))
best_gg = max(SPLITS, key=lambda s: (gg[s], -list(SPLITS).index(s)))
R.add("best_split_by_info_gain", best_ig)
R.add("best_split_by_gini_gain", best_gg)

# ---------------- 4.2 stump + voting ----------------
yes_tr = SPLITS[best_ig](Xtr_raw)
rate_yes = float(ytr[yes_tr].mean())
rate_no = float(ytr[~yes_tr].mean())
R.add("stump_rate_yes", rate_yes)
R.add("stump_rate_no", rate_no)

# logistic (Task 2.2 model), KNN (Task 2.7 K), stump -- all on the 240 test rows
w, b, _ = train_logistic(D["Xtr"], ytr, 0.1, 1000)
p_log = predict_proba(D["Xte"], w, b)
_, best_k = choose_k(D["Xtr"], ytr)
cnt = knn_positive_counts(D["Xtr"], ytr, D["Xte"], [best_k])[best_k]
p_knn = cnt / best_k
pred_knn = knn_predict(cnt, best_k)
p_stump = np.where(SPLITS[best_ig](Xte_raw), rate_yes, rate_no)

pred_log = p_log >= 0.3
pred_stump = p_stump >= 0.3
votes = pred_log.astype(int) + pred_knn.astype(int) + pred_stump.astype(int)
pred_hard = votes >= 2
pred_soft = (p_log + p_knn + p_stump) / 3.0 >= 0.3

vote_rows = []
for name, pred in (("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
                   ("hard_vote", pred_hard), ("soft_vote", pred_soft)):
    tp, fp, tn, fn = confusion(yte, pred)
    _, prec, rec, f1, _ = rates(tp, fp, tn, fn)
    vote_rows.append([name, tp, fp, tn, fn, prec, rec, f1])
write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
          vote_rows)

# ---------------- 4.3 two rounds of AdaBoost by hand ----------------
y_ab = np.array([+1, +1, +1, -1, -1, -1, +1, -1], dtype=float)
h1 = np.array([+1, +1, -1, -1, -1, -1, -1, -1], dtype=float)
h2 = np.array([-1, +1, +1, -1, +1, -1, +1, -1], dtype=float)
wt = np.full(8, 1 / 8)
ada_rows, alphas, errors = [], [], []
for rnd, h in ((1, h1), (2, h2)):
    eps = float(wt[h != y_ab].sum())
    alpha = 0.5 * math.log((1 - eps) / eps)
    unnorm = wt * np.exp(-alpha * y_ab * h)
    new = unnorm / unnorm.sum()
    for pt in range(8):
        ada_rows.append([rnd, pt + 1, float(wt[pt]), float(unnorm[pt]), float(new[pt])])
    if rnd == 1:
        misclassified_weight = float(new[h1 != y_ab].sum())
    errors.append(eps)
    alphas.append(alpha)
    wt = new
R.add("round1_error", errors[0])
R.add("round1_alpha", alphas[0])
R.add("round2_error", errors[1])
R.add("round2_alpha", alphas[1])
write_csv("part4_adaboost.csv",
          ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)
ens = np.sign(alphas[0] * h1 + alphas[1] * h2)
R.add("ensemble_accuracy", float(np.mean(ens == y_ab)))

# ---------------- 4.4 out-of-bag share ----------------
R.add("oob_fraction", (1 - 1 / 560) ** 560)


# ---------------- 4.5 neural-network sizing (counting only; nothing is trained) ----------------
def dense(n_in, n_out):
    return n_in * n_out + n_out


def conv_params(k, c_in, filters):
    return k * k * c_in * filters + filters


def out_side(size, k, pad, stride):
    return (size - k + 2 * pad) // stride + 1


# Option 1: MLP 784 -> 16 -> 16 -> 10
mlp = {"mlp_hidden1": dense(784, 16), "mlp_hidden2": dense(16, 16), "mlp_output": dense(16, 10)}
# Option 2: CNN
s = 28
conv1 = conv_params(5, 1, 6)
s1 = out_side(s, 5, 2, 1)           # conv 5x5, pad 2
p1 = out_side(s1, 2, 0, 2)          # max-pool 2x2 stride 2
conv2 = conv_params(5, 6, 16)
s2 = out_side(p1, 5, 0, 1)          # conv 5x5, pad 0
p2 = out_side(s2, 2, 0, 2)
flat = p2 * p2 * 16
cnn = {"cnn_conv1": conv1, "cnn_conv2": conv2, "cnn_dense1": dense(flat, 32),
       "cnn_dense2": dense(32, 10)}
arch_rows = [[k, v] for k, v in {**mlp, **cnn}.items()]
write_csv("part4_architecture.csv", ["layer", "parameters"], arch_rows)
R.add("mlp_total", sum(mlp.values()))
R.add("cnn_total", sum(cnn.values()))
R.add("fc_layer_params", dense(128 * 128 * 3, 1024))
R.add("conv_layer_params", conv_params(3, 3, 64))
# Model B: simple RNN
rnn_total = 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16   # W_xh, W_hh, W_hy, b_h, b_y
R.add("rnn_total", rnn_total)

# extra (not required): numbers the README quotes
R.add("round1_misclassified_weight_total", misclassified_weight)
R.add("cnn_shape_input", "28x28x1")
R.add("cnn_shape_conv1", "%dx%dx6" % (s1, s1))
R.add("cnn_shape_pool1", "%dx%dx6" % (p1, p1))
R.add("cnn_shape_conv2", "%dx%dx16" % (s2, s2))
R.add("cnn_shape_pool2", "%dx%dx16" % (p2, p2))
R.add("cnn_shape_flatten", "%d" % flat)
R.add("cnn_best_k_used", best_k)

R.write("part4_results.txt")
