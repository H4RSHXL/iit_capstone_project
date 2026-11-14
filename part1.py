"""Part 1 - estimation, inference and a regression baseline. All 800 rows."""
import math

import numpy as np

from common import Results, load_data

X, outcome, y = load_data()
x4 = X[:, 3]          # avg_order_value_inr_hundreds
n = len(y)
R = Results()


def sample_sd(a):
    return math.sqrt(float(np.sum((a - a.mean()) ** 2)) / (len(a) - 1))


# ---- 1.1 estimate with an interval
mean_value = float(y.mean())
sd_value = sample_sd(y)
se_value = sd_value / math.sqrt(n)
R.add("n", n)
R.add("mean_value", mean_value)
R.add("sd_value", sd_value)
R.add("se_value", se_value)
R.add("ci95_low", mean_value - 1.96 * se_value)
R.add("ci95_high", mean_value + 1.96 * se_value)

# ---- 1.2 test H0: mu = 8.0
z_stat = (mean_value - 8.0) / se_value
R.add("z_stat", z_stat)
R.add("reject_h0", abs(z_stat) > 1.96)

# ---- 1.3 sample size for margin of error 0.5
E = 0.5
n_required = int(math.ceil((1.96 * sd_value / E) ** 2))
R.add("n_required", n_required)


# ---- 1.4 closed-form simple regression
def simple_ols(x, t):
    """Slope = cov/var (both n-1), intercept = mean(t) - slope*mean(x)."""
    cov = float(np.sum((x - x.mean()) * (t - t.mean()))) / (len(x) - 1)
    var = float(np.sum((x - x.mean()) ** 2)) / (len(x) - 1)
    slope = cov / var
    return slope, float(t.mean()) - slope * float(x.mean())


reg_slope, reg_intercept = simple_ols(x4, y)
fitted = reg_intercept + reg_slope * x4
tss = float(np.sum((y - y.mean()) ** 2))
ess = float(np.sum((fitted - y.mean()) ** 2))
rss = float(np.sum((y - fitted) ** 2))
R.add("reg_slope", reg_slope)
R.add("reg_intercept", reg_intercept)
R.add("tss", tss)
R.add("ess", ess)
R.add("rss", rss)
R.add("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
R.add("r2", 1 - rss / tss)
R.add("rmse", math.sqrt(rss / n))
R.add("mae", float(np.mean(np.abs(y - fitted))))

# ---- 1.5 gradient descent on the standardised predictor
mx, sx = float(x4.mean()), sample_sd(x4)
z = (x4 - mx) / sx
m, c, eta = 0.0, 0.0, 0.1


def cost(m_, c_):
    e_ = y - (m_ * z + c_)
    return float(np.sum(e_ ** 2)) / n


R.add("gd_J0", cost(m, c))
for step in range(1, 501):
    e = y - (m * z + c)
    grad_m = -2.0 / n * float(np.sum(z * e))
    grad_c = -2.0 / n * float(np.sum(e))
    m -= eta * grad_m
    c -= eta * grad_c
    if step in (1, 10, 100, 500):
        R.add("gd_J%d" % step, cost(m, c))
gd_slope = m / sx
gd_intercept = c - m * mx / sx
R.add("gd_slope", gd_slope)
R.add("gd_intercept", gd_intercept)
R.add("gd_slope_gap", abs(gd_slope - reg_slope))


# ---- 1.6 funnel check, then the log remedy
def half_sds(fit, resid):
    order = np.argsort(fit, kind="stable")   # stable: ties keep row order
    lower, upper = resid[order[:400]], resid[order[400:]]
    return sample_sd(lower), sample_sd(upper)


lo, up = half_sds(fitted, y - fitted)
R.add("resid_sd_lower_raw", lo)
R.add("resid_sd_upper_raw", up)
R.add("resid_sd_ratio_raw", up / lo)

ly, lx = np.log(y), np.log(x4)
ll_slope, ll_intercept = simple_ols(lx, ly)
ll_fit = ll_intercept + ll_slope * lx
ll_rss = float(np.sum((ly - ll_fit) ** 2))
ll_tss = float(np.sum((ly - ly.mean()) ** 2))
ll_r2 = 1 - ll_rss / ll_tss
R.add("loglog_intercept", ll_intercept)
R.add("loglog_slope", ll_slope)
R.add("loglog_r2", ll_r2)
R.add("loglog_rmse", math.sqrt(ll_rss / n))
lo2, up2 = half_sds(ll_fit, ly - ll_fit)
R.add("resid_sd_lower_log", lo2)
R.add("resid_sd_upper_log", up2)
R.add("resid_sd_ratio_log", up2 / lo2)

# ---- 1.7 multiple regression by the normal equation
# columns: 1, ln(X4), X1, X2, X3, X5, X6, X7
D = np.column_stack([np.ones(n), lx, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(D.T @ D, D.T @ ly)
mr_fit = D @ beta
mr_rss = float(np.sum((ly - mr_fit) ** 2))
mr_r2 = 1 - mr_rss / ll_tss
p = 7
mr_adj = 1 - (1 - mr_r2) * (n - 1) / (n - p - 1)
for j in range(8):
    R.add("mr_beta_%d" % j, float(beta[j]))
R.add("mr_r2", mr_r2)
R.add("mr_adj_r2", mr_adj)
R.add("loglog_adj_r2", 1 - (1 - ll_r2) * (n - 1) / (n - 1 - 1))

R.write("part1_results.txt")
