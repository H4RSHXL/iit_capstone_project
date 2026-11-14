"""Part 3 - segmentation (K-means, hierarchical) and PCA. All 800 rows, X1..X5."""
import numpy as np

from common import Results, load_data, write_csv

X7, outcome, value = load_data()
X = X7[:, :5]
n = len(X)
R = Results()

# standardise X1..X5 over all 800 rows (sample SD)
mean = X.mean(axis=0)
sd = np.sqrt(((X - mean) ** 2).sum(axis=0) / (n - 1))
Z = (X - mean) / sd


# ---------------- 3.1 K-means from scratch ----------------
def kmeans(Z, K):
    step = len(Z) // K
    cent = np.array([Z[i * step] for i in range(K)], dtype=float)
    labels = np.full(len(Z), -1)
    passes = 0
    while True:
        passes += 1
        d2 = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)            # argmin -> ties go to lower index
        if np.array_equal(new, labels):
            break                              # no row changed: stop
        labels = new
        for k in range(K):
            members = Z[labels == k]
            if len(members):                   # empty cluster stays where it is
                cent[k] = members.mean(axis=0)
    wcss = float(((Z - cent[labels]) ** 2).sum())
    return labels, cent, wcss, passes


wcss, labels_by_k = {}, {}
for K in (2, 3, 4, 5, 6):
    lab, cent, w, passes = kmeans(Z, K)
    wcss[K], labels_by_k[K] = w, lab
    R.add("wcss_k%d" % K, w)

# ---------------- 3.2 elbow ----------------
elbow_k = None
for K in (3, 4, 5):
    if wcss[K] - wcss[K + 1] < 0.5 * (wcss[K - 1] - wcss[K]):
        elbow_k = K
        break
R.add("elbow_k", elbow_k if elbow_k is not None else "none")

# ---------------- 3.3 profile the K=3 segments ----------------
lab3 = labels_by_k[3]
order = sorted(range(3), key=lambda k: X[lab3 == k, 2].mean())   # ascending mean X3
seg_of_cluster = {c: "ABC"[i] for i, c in enumerate(order)}
seg_labels = np.array([seg_of_cluster[c] for c in lab3])
seg_rows = []
for s in "ABC":
    m = seg_labels == s
    seg_rows.append([s, int(m.sum())] + [float(X[m, j].mean()) for j in range(5)]
                    + [float(outcome[m].mean()), float(value[m].mean())])
    R.add("size_segment_%s" % s, int(m.sum()))
write_csv("part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
           "outcome_rate", "mean_value"], seg_rows)

# ---------------- 3.4 hierarchical clustering on the supplied table ----------------
names = ["E1", "E2", "E3", "E4", "E5", "E6"]
DM = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def linkage_dist(a, b, rule):
    ds = [DM[i, j] for i in a for j in b]
    return min(ds) if rule == "single" else (max(ds) if rule == "complete" else sum(ds) / len(ds))


def label(c):
    return "+".join(names[i] for i in sorted(c))


def agglomerate(rule):
    clusters = [(i,) for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = linkage_dist(clusters[i], clusters[j], rule)
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        a, b = clusters[i], clusters[j]
        if min(a) > min(b):
            a, b = b, a                       # cluster_a has the smaller first member
        merges.append((a, b, d))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [tuple(sorted(a + b))]
    return merges


hier_rows, cut_info = [], {}
for rule in ("single", "complete", "average"):
    merges = agglomerate(rule)
    for no, (a, b, d) in enumerate(merges, 1):
        hier_rows.append([rule, no, label(a), label(b), d])
    heights = [m[2] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    below = int(np.argmax(gaps)) + 1          # merges below the cut (largest gap)
    cut_info[rule] = (6 - below, merges[:below])
write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
          hier_rows)
for rule in ("single", "complete", "average"):
    R.add("%s_clusters_at_cut" % rule, cut_info[rule][0])


def members_at_cut(merges_below):
    clusters = [(i,) for i in range(6)]
    for a, b, _ in merges_below:
        clusters = [c for c in clusters if c not in (a, b)] + [tuple(sorted(a + b))]
    return " | ".join(label(c) for c in sorted(clusters))


# ---------------- 3.5 PCA by the seven steps ----------------
mu = Z.mean(axis=0)                               # (1) mean vector (zeros after standardising)
Xc = Z - mu                                       # (2) centre
S = (Xc.T @ Xc) / (n - 1)                         # (3) covariance
vals, vecs = np.linalg.eigh(S)                    # (4) eigen-decomposition
idx = np.argsort(-vals)                           # (5) largest first
vals, vecs = vals[idx], vecs[:, idx]
for j in range(5):                                # sign convention
    if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
        vecs[:, j] = -vecs[:, j]
W = vecs[:, :2]                                   # (6) top two
proj = Xc @ W                                     # (7) project

R.add("eigenvalue_sum", float(vals.sum()))
for j in range(5):
    R.add("eigenvalue_%d" % (j + 1), float(vals[j]))
ratio = vals / vals.sum()
cum = np.cumsum(ratio)
for j in range(5):
    R.add("explained_ratio_%d" % (j + 1), float(ratio[j]))
for j in range(5):
    R.add("cumulative_ratio_%d" % (j + 1), float(cum[j]))
R.add("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1)
for j in range(5):
    R.add("pc1_loading_X%d" % (j + 1), float(vecs[j, 0]))
for j in range(5):
    R.add("pc2_loading_X%d" % (j + 1), float(vecs[j, 1]))
for r in range(3):
    R.add("row%d_pc1" % (r + 1), float(proj[r, 0]))
    R.add("row%d_pc2" % (r + 1), float(proj[r, 1]))
for s in "ABC":
    R.add("mean_pc1_segment_%s" % s, float(proj[seg_labels == s, 0].mean()))

# extra (not required): which cohorts sit together at each dendrogram cut
for rule in ("single", "complete", "average"):
    R.add("%s_cut_members" % rule, members_at_cut(cut_info[rule][1]))

R.write("part3_results.txt")
