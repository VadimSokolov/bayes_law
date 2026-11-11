"""Figures for Bayes and Law (Polson and Sokolov).
All data are either analytic or simulated for illustration; simulated
figures are labelled as such in their captions."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "axes.titleweight": "normal", "axes.labelweight": "normal", "font.weight": "normal",
})
NB = "#1c2e5c"; GOLD = "#d6b43e"; GREY = "#7a7a7a"; RED = "#9b3d3d"; GREEN = "#3d7a5a"
W, H = 4.3, 2.6
rng = np.random.default_rng(2026)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(name + ".pdf")
    plt.close(fig)

# 1 posterior probability against prior for several likelihood ratios
fig, ax = plt.subplots(figsize=(W, H))
p0 = np.logspace(-7, -0.01, 400)
for lr, c in [(10, GREY), (100, GOLD), (1e4, NB), (1e6, RED)]:
    o = p0 / (1 - p0) * lr
    ax.plot(p0, o / (1 + o), color=c, lw=1.6, label=f"LR = {lr:g}")
ax.set_xscale("log"); ax.set_xlabel("prior probability of guilt"); ax.set_ylabel("posterior probability")
ax.axhline(0.5, color=GREY, ls=":", lw=0.8)
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_posterior_odds")

# 2 database search and population size: probability that the matched person is the source
fig, ax = plt.subplots(figsize=(W, H))
N = np.logspace(2, 8, 300)
for rmp, c in [(1e-4, GREY), (1e-6, GOLD), (1e-9, NB)]:
    post = 1 / (1 + (N - 1) * rmp)
    ax.plot(N, post, color=c, lw=1.6, label=f"match probability {rmp:g}")
ax.set_xscale("log"); ax.set_xlabel("size of plausible suspect population N")
ax.set_ylabel("P(source | match)"); ax.legend(frameon=False, loc="lower left")
save(fig, "fig_population")

# 3 decision threshold for conviction
fig, ax = plt.subplots(figsize=(W, H))
p = np.linspace(0, 1, 400)
L_fc, L_fa = 10.0, 1.0  # loss of false conviction, loss of false acquittal
ax.plot(p, (1 - p) * L_fc, color=RED, lw=1.6, label="expected loss if convict")
ax.plot(p, p * L_fa, color=NB, lw=1.6, label="expected loss if acquit")
ps = L_fc / (L_fc + L_fa)
ax.axvline(ps, color=GOLD, ls="--", lw=1)
ax.text(ps - 0.02, 7.5, f"threshold {ps:.3f}", ha="right", color=GREY)
ax.set_xlabel("posterior probability of guilt p"); ax.set_ylabel("expected loss")
ax.legend(frameon=False, loc="upper center")
save(fig, "fig_threshold")

# 4 Lindley paradox: Bayes factor at fixed z = 2 as n grows
fig, ax = plt.subplots(figsize=(W, H))
n = np.logspace(0, 6, 300)
z = 2.0
for tau, c in [(0.5, GOLD), (1.0, NB), (2.0, RED)]:
    # point null versus N(0, tau^2) alternative on the effect, unit variance data
    r = n * tau**2
    bf01 = np.sqrt(1 + r) * np.exp(-0.5 * z**2 * r / (1 + r))
    ax.plot(n, bf01, color=c, lw=1.6, label=f"prior scale {tau}")
ax.axhline(1, color=GREY, ls=":", lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("sample size n (z = 2 throughout)"); ax.set_ylabel("Bayes factor for the null")
ax.legend(frameon=False)
save(fig, "fig_lindley")

# 5 Hand formula: total expected cost of care
fig, ax = plt.subplots(figsize=(W, H))
x = np.linspace(0, 10, 400)
burden = x
expected_harm = 100 * 0.3 * np.exp(-0.45 * x)
ax.plot(x, burden, color=NB, lw=1.6, label="burden of care B(x)")
ax.plot(x, expected_harm, color=RED, lw=1.6, label="expected harm P(x) L")
ax.plot(x, burden + expected_harm, color=GOLD, lw=2, label="total social cost")
xs = np.log(100 * 0.3 * 0.45) / 0.45
ax.axvline(xs, color=GREY, ls="--", lw=0.8)
ax.set_xlabel("level of care x"); ax.set_ylabel("cost"); ax.legend(frameon=False)
save(fig, "fig_hand")

# 6 loss of chance damages
fig, ax = plt.subplots(figsize=(W, H))
q = np.linspace(0, 1, 400)
ax.plot(q, (q > 0.5).astype(float), color=RED, lw=1.6, label="all or nothing (balance of probabilities)")
ax.plot(q, q, color=NB, lw=1.6, label="proportional to lost chance")
ax.set_xlabel("probability the negligence caused the harm"); ax.set_ylabel("fraction of loss compensated")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_loss_chance")

# 7 probability of causation and relative risk
fig, ax = plt.subplots(figsize=(W, H))
rr = np.linspace(1, 6, 400)
ax.plot(rr, 1 - 1 / rr, color=NB, lw=1.8)
ax.axvline(2, color=GOLD, ls="--", lw=1); ax.axhline(0.5, color=GREY, ls=":", lw=0.8)
ax.text(2.1, 0.2, "doubling of risk", color=GREY)
ax.set_xlabel("relative risk RR"); ax.set_ylabel("probability of causation 1 - 1/RR")
save(fig, "fig_doubling")

# 8 Greensill style receivables: concentration and portfolio losses (simulated)
fig, ax = plt.subplots(figsize=(W, H))
def losses(weights, pd, rho, sims=40000):
    k = len(weights)
    zc = rng.standard_normal(sims)[:, None]
    zi = rng.standard_normal((sims, k))
    lat = np.sqrt(rho) * zc + np.sqrt(1 - rho) * zi
    thr = stats.norm.ppf(pd)
    return ((lat < thr) * weights).sum(axis=1)
w_div = np.ones(200) / 200
w_conc = np.r_[0.5, np.ones(199) * 0.5 / 199]
for w, c, lab in [(w_div, NB, "diversified 200 obligors"), (w_conc, RED, "one obligor at half the book")]:
    L = losses(w, 0.03, 0.2)
    ax.hist(L, bins=60, range=(0, 0.7), density=True, histtype="step", color=c, lw=1.5, label=lab)
ax.set_yscale("log"); ax.set_xlabel("fraction of fund lost"); ax.set_ylabel("density (log scale)")
ax.legend(frameon=False)
save(fig, "fig_greensill_loss")

# 9 posterior for an obligor default probability after missed payments (Beta model)
fig, ax = plt.subplots(figsize=(W, H))
th = np.linspace(0, 0.6, 500)
for (a, b), c, lab in [((1, 30), GREY, "prior: favourable view"),
                       ((1 + 2, 30 + 10), GOLD, "after 2 late of 12 settlements"),
                       ((1 + 6, 30 + 18), RED, "after 6 late of 24 settlements")]:
    ax.plot(th, stats.beta.pdf(th, a, b), color=c, lw=1.6, label=lab)
ax.set_xlabel("probability that a settlement is late or rolled over"); ax.set_ylabel("density"); ax.legend(frameon=False)
save(fig, "fig_greensill_beta")

# 10 low template DNA: likelihood ratio and contamination probability
fig, ax = plt.subplots(figsize=(W, H))
c = np.logspace(-4, -0.3, 300)
rmp = 1e-9
for d, col in [(0.0, NB), (0.2, GOLD), (0.5, RED)]:
    # P(profile | source) = 1 - d ; P(profile | not source) = c + (1 - c) rmp
    lr = (1 - d) * (1) / (c + (1 - c) * rmp) + 0
    ax.plot(c, lr, color=col, lw=1.6, label=f"dropout {d}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("probability of contamination or transfer"); ax.set_ylabel("likelihood ratio")
ax.legend(frameon=False)
save(fig, "fig_lowtemplate")

# 11 skill share in a game: fraction of outcome variance explained by skill as hands accumulate
fig, ax = plt.subplots(figsize=(W, H))
m = np.logspace(0, 5, 300)
for s2, col in [(0.0005, GREY), (0.002, GOLD), (0.01, NB)]:
    # mean result per hand = skill + noise / sqrt(m); share = s2 / (s2 + 1/m)
    ax.plot(m, s2 / (s2 + 1 / m), color=col, lw=1.6, label=f"skill variance {s2}")
ax.axhline(0.5, color=GREY, ls=":", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("number of hands played"); ax.set_ylabel("share of variance due to skill")
ax.legend(frameon=False)
save(fig, "fig_skill")

# 12 jury representation: binomial test and Bayes factor
fig, ax = plt.subplots(figsize=(W, H))
nj = 870; p_pop = 0.791
k = np.arange(300, 760)
pmf = stats.binom.pmf(k, nj, p_pop)
ax.plot(k, pmf, color=NB, lw=1.4)
ax.axvline(339, color=RED, ls="--", lw=1)
ax.text(345, pmf.max() * 0.8, "observed 339", color=RED)
ax.set_xlabel("Mexican American grand jurors among 870"); ax.set_ylabel("probability under random selection")
save(fig, "fig_castaneda")

# 13 hierarchical shrinkage of legal origin effects (simulated)
fig, ax = plt.subplots(figsize=(W, H))
J = 24
origin = np.r_[np.zeros(12), np.ones(12)]
true = 0.3 + 0.25 * origin + rng.normal(0, 0.12, J)
se = rng.uniform(0.08, 0.3, J)
y = true + rng.normal(0, se)
# empirical Bayes shrinkage within origin groups
est = np.empty(J)
for g in (0, 1):
    idx = origin == g
    mu = np.average(y[idx], weights=1 / se[idx] ** 2)
    tau2 = max(np.var(y[idx]) - np.mean(se[idx] ** 2), 0.005)
    B = se[idx] ** 2 / (se[idx] ** 2 + tau2)
    est[idx] = B * mu + (1 - B) * y[idx]
for j in range(J):
    col = NB if origin[j] == 0 else RED
    ax.plot([0, 1], [y[j], est[j]], color=col, lw=0.8, alpha=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["raw country estimate", "partially pooled"])
ax.set_ylabel("index of creditor protection")
ax.text(1.02, est[origin == 0].mean(), "civil law", color=NB, va="center")
ax.text(1.02, est[origin == 1].mean(), "common law", color=RED, va="center")
save(fig, "fig_shrinkage")

# 14 Indian backlog dynamics (stylised queue)
fig, ax = plt.subplots(figsize=(W, H))
years = np.arange(0, 21)
B0 = 5.6  # crore pending
inflow = 2.4
for out, col, lab in [(2.35, RED, "disposal 2.35 crore a year"), (2.5, GOLD, "disposal 2.50"), (2.8, NB, "disposal 2.80")]:
    B = np.maximum(B0 + (inflow - out) * years, 0)
    ax.plot(years, B, color=col, lw=1.6, label=lab)
ax.set_xlabel("years from 2026"); ax.set_ylabel("pending cases (crore)")
ax.legend(frameon=False)
save(fig, "fig_india_backlog")

# 15 vendetta: probability of retaliation versus solidarity displayed (stylised logistic)
fig, ax = plt.subplots(figsize=(W, H))
s = np.linspace(0, 5, 300)
for base, col, lab in [(-3.0, NB, "single assailant"), (-1.5, GOLD, "two kin collaborated"), (-0.3, RED, "group attack on bystanders")]:
    ax.plot(s, 1 / (1 + np.exp(-(base + 0.6 * s))), color=col, lw=1.6, label=lab)
ax.set_xlabel("size of victim's kin group"); ax.set_ylabel("probability of revenge")
ax.legend(frameon=False)
save(fig, "fig_vendetta")

# 16 Becker: expected penalty iso curves in (probability, fine) space
fig, ax = plt.subplots(figsize=(W, H))
pp = np.linspace(0.01, 1, 300)
for epen, col in [(1, GREY), (2, GOLD), (4, NB)]:
    ax.plot(pp, epen / pp, color=col, lw=1.6, label=f"expected sanction {epen}")
ax.set_ylim(0, 40); ax.set_xlabel("probability of conviction p"); ax.set_ylabel("sanction f")
ax.legend(frameon=False)
save(fig, "fig_becker")

# 17 Becker Murphy Grossman: enforcement and elasticity
fig, ax = plt.subplots(figsize=(W, H))
e = np.linspace(0.1, 2.0, 300)
# spending on producers' costs relative to value of output reduction; sign flips at unit elasticity
ax.plot(e, (1 - e), color=NB, lw=1.8)
ax.axhline(0, color=GREY, lw=0.8); ax.axvline(1, color=GOLD, ls="--", lw=1)
ax.text(1.05, 0.6, "unit elasticity", color=GREY)
ax.set_xlabel("absolute demand elasticity"); ax.set_ylabel("change in spending per unit price rise")
save(fig, "fig_bmg")

# 18 Priest and Klein selection: plaintiff win rate among tried cases
# Model: case quality Y ~ N(0.5, 1); plaintiff wins at trial if Y > 0. Each party observes
# Y plus independent noise with sd s and forms P_i = Phi(estimate / s). The case is tried
# when P_p - P_d exceeds (C_p + C_d) / J = 0.3; otherwise it settles.
fig, ax = plt.subplots(figsize=(W, H))
sd_err = np.linspace(0.05, 2.0, 40)
ygrid = np.linspace(-6, 7, 1300)  # grid avoids Y = 0 exactly
prior_w = stats.norm.pdf(ygrid, 0.5, 1.0)
za = rng.standard_normal(300); zb = rng.standard_normal(300)
e1 = np.r_[za, -zb]; e2 = np.r_[zb, -za]  # antithetic pairs keep the tried region symmetric
win = []
for s_err in sd_err:
    pp = stats.norm.cdf((ygrid[:, None] + s_err * e1[None, :]) / s_err)
    pdd = stats.norm.cdf((ygrid[:, None] + s_err * e2[None, :]) / s_err)
    tried = ((pp - pdd) > 0.3).mean(axis=1)
    win.append(np.sum(prior_w * tried * (ygrid > 0)) / np.sum(prior_w * tried))
ax.plot(sd_err, win, color=NB, lw=1.8, label="tried cases")
ax.axhline(stats.norm.sf(0, 0.5, 1.0), color=RED, ls=":", lw=1.2, label="all disputes")
ax.axhline(0.5, color=GOLD, ls="--", lw=1)
ax.set_ylim(0.4, 0.8)
ax.set_xlabel("noise in the parties' forecasts"); ax.set_ylabel("plaintiff win rate")
ax.legend(frameon=False, loc="center right")
save(fig, "fig_priest_klein")

# 19 evidence accumulation in decibans (sequential test)
fig, ax = plt.subplots(figsize=(W, H))
for path in range(6):
    steps = rng.normal(0.8 if path < 3 else -0.6, 3.0, 40)
    ax.plot(np.r_[0, np.cumsum(steps)], color=NB if path < 3 else RED, lw=1, alpha=0.9)
ax.axhline(20, color=GOLD, ls="--", lw=1); ax.axhline(-20, color=GOLD, ls="--", lw=1)
ax.text(1, 21.5, "stop: 100 to 1 for", color=GREY); ax.text(1, -24.5, "stop: 100 to 1 against", color=GREY)
ax.set_xlabel("items of evidence"); ax.set_ylabel("cumulative weight (decibans)")
save(fig, "fig_decibans")

# 20 calibration of a risk tool (simulated)
fig, ax = plt.subplots(figsize=(W, H))
pr = rng.beta(2, 5, 6000)
truth = rng.uniform(size=pr.size) < np.clip(0.05 + 0.75 * pr, 0, 1)
bins = np.linspace(0, 0.8, 9)
mid = 0.5 * (bins[1:] + bins[:-1])
obs = [truth[(pr >= a) & (pr < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
ax.plot([0, 0.8], [0, 0.8], color=GREY, ls=":", lw=0.8)
ax.plot(mid, obs, "o-", color=NB, lw=1.4, ms=3)
ax.set_xlabel("predicted probability of reoffending"); ax.set_ylabel("observed frequency")
save(fig, "fig_calibration")

# 21 settlement as Bayesian bargaining: settlement range against prior disagreement
fig, ax = plt.subplots(figsize=(W, H))
d = np.linspace(0, 0.5, 300)
J = 100; Cp = 8; Cd = 8
pp_ = 0.5 + d / 2; pd_ = 0.5 - d / 2
lo = pd_ * J + Cd; hi = pp_ * J - Cp
ax.fill_between(d, lo, hi, where=hi < lo, color=GOLD, alpha=0.35, label="settlement zone")
ax.plot(d, pp_ * J - Cp, color=RED, lw=1.5, label="plaintiff minimum demand")
ax.plot(d, pd_ * J + Cd, color=NB, lw=1.5, label="defendant maximum offer")
ax.axvline(0.16, color=GREY, ls=":", lw=0.8)
ax.set_xlabel("gap between the parties' win probabilities"); ax.set_ylabel("amount")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_settlement")

# 22 Good's scale of evidence (bar)
fig, ax = plt.subplots(figsize=(W, 1.6))
edges = [0, 5, 10, 13, 20, 30]
labels = ["barely\nworth\nmention", "substantial", "strong", "very\nstrong", "decisive"]
cols = ["#e8e3cf", "#d9cc93", GOLD, "#8a95b5", NB]
for a, b, l, cc in zip(edges[:-1], edges[1:], labels, cols):
    ax.barh(0, b - a, left=a, color=cc, edgecolor="white")
    ax.text((a + b) / 2, 0, l, ha="center", va="center", fontsize=7, color="black" if cc != NB else "white")
ax.set_yticks([]); ax.set_xlabel("weight of evidence in decibans (10 log10 BF)")
for s in ["left"]:
    ax.spines[s].set_visible(False)
save(fig, "fig_scale")

# 23 prior sensitivity: posterior for guilt as the prior pool N varies, two LRs (Adams style)
fig, ax = plt.subplots(figsize=(W, H))
Npool = np.logspace(3, 7, 300)
for lr_dna, lr_def, col, lab in [(2e8, 1.0, NB, "DNA alone"), (2e8, 1 / 36.0, GOLD, "DNA with alibi and non identification"), (2e6, 1 / 36.0, RED, "weaker DNA with defence evidence")]:
    o = lr_dna * lr_def / (Npool - 1)
    ax.plot(Npool, o / (1 + o), color=col, lw=1.6, label=lab)
ax.set_xscale("log"); ax.set_xlabel("pool of possible offenders"); ax.set_ylabel("posterior probability of guilt")
ax.legend(frameon=False, loc="lower left")
save(fig, "fig_adams")

# 24 tort causation in Fairchild: probability each employer caused mesothelioma by exposure share
fig, ax = plt.subplots(figsize=(W, H))
shares = np.array([0.45, 0.30, 0.15, 0.10])
ax.bar(range(4), shares, color=[NB, GOLD, GREY, RED])
ax.set_xticks(range(4)); ax.set_xticklabels(["employer A", "employer B", "employer C", "employer D"])
ax.set_ylabel("posterior probability of causation")
save(fig, "fig_fairchild")

print("figures done")

# 25 Merton model: probability of default against leverage for several volatilities
fig, ax = plt.subplots(figsize=(W, H))
lev = np.linspace(0.3, 1.2, 300)  # face value of debt over asset value
T = 1.0
for sig, col in [(0.15, NB), (0.30, GOLD), (0.50, RED)]:
    d2 = (np.log(1 / lev) + (0.03 - 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    ax.plot(lev, stats.norm.cdf(-d2), color=col, lw=1.6, label=f"asset volatility {sig}")
ax.set_xlabel("debt relative to asset value"); ax.set_ylabel("one year default probability")
ax.legend(frameon=False)
save(fig, "fig_merton")

# 26 hindsight: expert estimates of default probability conditional on later failure (simulated)
fig, ax = plt.subplots(figsize=(W, H))
true_pd = rng.beta(2, 18, 20000)
failed = rng.uniform(size=true_pd.size) < true_pd
bias = 0.12
est_unbiased = np.clip(true_pd + rng.normal(0, 0.03, true_pd.size), 0, 1)
est_hind = np.clip(est_unbiased + bias * failed, 0, 1)
ax.hist(true_pd[failed], bins=40, range=(0, 0.6), density=True, histtype="step", color=NB, lw=1.5, label="ex ante probability, failed firms")
ax.hist(est_hind[failed], bins=40, range=(0, 0.6), density=True, histtype="step", color=RED, lw=1.5, label="expert estimate with hindsight")
ax.set_xlabel("default probability at the transfer date"); ax.set_ylabel("density")
ax.legend(frameon=False)
save(fig, "fig_hindsight")

# 27 regression discontinuity at a colonial border (simulated)
fig, ax = plt.subplots(figsize=(W, H))
x = rng.uniform(-100, 100, 600)
side = (x > 0).astype(float)
yv = 0.4 + 0.002 * x + 0.08 * side + rng.normal(0, 0.08, x.size)
ax.scatter(x, yv, s=3, color=GREY, alpha=0.5)
for s_, col in [(0, NB), (1, RED)]:
    m = side == s_
    b = np.polyfit(x[m], yv[m], 1)
    xx = np.linspace(-100, 0, 50) if s_ == 0 else np.linspace(0, 100, 50)
    ax.plot(xx, np.polyval(b, xx), color=col, lw=2)
ax.axvline(0, color=GOLD, ls="--", lw=1)
ax.set_xlabel("distance from former colonial border (km)"); ax.set_ylabel("household asset index")
ax.text(-95, 0.75, "French side", color=NB); ax.text(45, 0.75, "British side", color=RED)
save(fig, "fig_rd")

# 28 event study: cumulative abnormal returns around a corrective disclosure (simulated)
fig, ax = plt.subplots(figsize=(W, H))
days = np.arange(-20, 21)
for k in range(30):
    ar = rng.normal(0, 0.015, days.size)
    ax.plot(days, np.cumsum(ar), color=GREY, lw=0.5, alpha=0.5)
ar = rng.normal(0, 0.015, days.size); ar[days == 0] -= 0.12
ax.plot(days, np.cumsum(ar), color=RED, lw=1.8, label="stock with corrective disclosure on day 0")
ax.axvline(0, color=GOLD, ls="--", lw=1)
ax.set_xlabel("trading days relative to disclosure"); ax.set_ylabel("cumulative abnormal return")
ax.legend(frameon=False, loc="lower left")
save(fig, "fig_event")

# 29 heavy tailed versus normal prior: posterior mean of an effect against the observed estimate
fig, ax = plt.subplots(figsize=(W, H))
yv = np.linspace(-8, 8, 400)
ax.plot(yv, yv / 2, color=GREY, lw=1.6, label="normal prior: constant shrinkage")
# Cauchy prior posterior mean by numerical integration
th = np.linspace(-30, 30, 6001)
pm = []
for y0 in yv:
    w = stats.norm.pdf(y0 - th) * stats.cauchy.pdf(th)
    pm.append(np.sum(th * w) / np.sum(w))
ax.plot(yv, pm, color=NB, lw=1.8, label="heavy tailed prior: large signals kept")
ax.plot(yv, yv, color=GOLD, ls=":", lw=1)
ax.set_xlabel("observed effect (standard errors)"); ax.set_ylabel("posterior mean")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_heavytail")

# 30 ROC curve for a fact finder with signal strength d
fig, ax = plt.subplots(figsize=(W, H))
fpr = np.linspace(0, 1, 400)
for dp, col in [(0.5, GREY), (1.5, GOLD), (3.0, NB)]:
    thr = stats.norm.ppf(1 - fpr)
    ax.plot(fpr, 1 - stats.norm.cdf(thr - dp), color=col, lw=1.6, label=f"evidence strength d = {dp}")
ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=0.8)
ax.set_xlabel("Type I error rate (convicting the innocent)"); ax.set_ylabel("power (convicting the guilty)")
ax.legend(frameon=False, loc="lower right")
save(fig, "fig_roc")

# 31 optimal investment in accuracy: cost of process plus expected error cost
fig, ax = plt.subplots(figsize=(W, H))
e = np.linspace(0.1, 10, 400)
proc = 3 * e
err = 60 * np.exp(-0.5 * e)
ax.plot(e, proc, color=NB, lw=1.6, label="cost of process")
ax.plot(e, err, color=RED, lw=1.6, label="expected cost of error")
ax.plot(e, proc + err, color=GOLD, lw=2, label="total cost")
ax.axvline(e[np.argmin(proc + err)], color=GREY, ls="--", lw=0.8)
ax.set_xlabel("resources spent on a case"); ax.set_ylabel("cost"); ax.legend(frameon=False)
save(fig, "fig_accuracy")

# 32 predictive distribution of future loss of earnings
fig, ax = plt.subplots(figsize=(W, H))
sims = 200000
years = np.clip(rng.normal(22, 5, sims), 0, None)
wage = rng.lognormal(np.log(40000), 0.25, sims)
r = 0.005
pv = wage * (1 - (1 + r) ** (-years)) / r
ax.hist(pv / 1e6, bins=80, density=True, color=NB, alpha=0.8)
ax.axvline(np.mean(pv) / 1e6, color=GOLD, lw=1.5, label="mean")
ax.axvline(np.median(pv) / 1e6, color=RED, ls="--", lw=1.2, label="median")
ax.set_xlabel("present value of lost earnings (millions)"); ax.set_ylabel("density")
ax.legend(frameon=False)
save(fig, "fig_damages")

# 33 litigation funding portfolio: distribution of returns, single claim versus twenty
fig, ax = plt.subplots(figsize=(W, H))
def claim_returns(n, sims=100000):
    win = rng.uniform(size=(sims, n)) < 0.6
    return (np.where(win, 3.0, -1.0)).mean(axis=1)
for n, col in [(1, RED), (5, GOLD), (20, NB)]:
    rr = claim_returns(n)
    ax.hist(rr, bins=60, range=(-1.05, 3.05), density=True, histtype="step", color=col, lw=1.5, label=f"{n} claims")
ax.set_xlabel("return per unit invested"); ax.set_ylabel("density"); ax.legend(frameon=False)
save(fig, "fig_funding")
