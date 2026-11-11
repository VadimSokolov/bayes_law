"""Calculations from Bayes and Law (Polson and Sokolov).
Run with: python3 bayes_law.py"""
import numpy as np
from scipy import stats, special


def posterior_prob(prior_odds, lr):
    o = prior_odds * lr
    return o / (1 + o)


def island(N, p):
    """Posterior that a matching suspect is the source."""
    return 1 / (1 + N * p)


def island_general(prior, match):
    """prior[0], match[0] refer to the suspect."""
    prior, match = np.asarray(prior), np.asarray(match)
    return prior[0] / (prior[0] + np.sum(prior[1:] * match[1:]))


def simpson(p_partner=1 / 2500, p_other=1 / 20000):
    return p_partner / (p_partner + p_other)


def threshold(loss_false_conviction, loss_false_acquittal):
    lfc, lfa = loss_false_conviction, loss_false_acquittal
    return lfc / (lfc + lfa)


def sbb_bound(p):
    """Sellke, Bayarri and Berger bound on the Bayes factor."""
    return -1 / (np.e * p * np.log(p))


def lindley_bf01(z, n, tau=1.0):
    r = n * tau ** 2
    return np.sqrt(1 + r) * np.exp(-0.5 * z ** 2 * r / (1 + r))


def castaneda_log10_bf(k=339, n=870, p=0.791):
    lnum = (np.log(stats.beta.cdf(p, k + 1, n - k + 1))
            + special.betaln(k + 1, n - k + 1) - np.log(p))
    lden = k * np.log(p) + (n - k) * np.log(1 - p)
    return (lnum - lden) / np.log(10)


def merton_pd(V, D, mu, sigma, T=1.0):
    num = np.log(V / D) + (mu - 0.5 * sigma ** 2) * T
    d2 = num / (sigma * np.sqrt(T))
    return stats.norm.cdf(-d2)


def combine(m1, s1, m2, s2):
    w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
    return (w1 * m1 + w2 * m2) / (w1 + w2), np.sqrt(1 / (w1 + w2))


def proportionate_payment(claim, premium_charged, premium_fair):
    return claim * premium_charged / premium_fair


if __name__ == "__main__":
    print("Simpson P(partner | battered, murdered) =",
          round(simpson(), 3))
    print("Island N=1e5, p=1e-6:", round(island(1e5, 1e-6), 3))
    pri = [1] + [1] * 10000 + [1]
    mat = [1] + [1e-6] * 10000 + [0.01]
    print("Island with brother:", round(island_general(pri, mat), 3))
    print("Blackstone threshold:", round(threshold(10, 1), 3))
    print("SBB bounds p=.05, .01:",
          round(sbb_bound(0.05), 2), round(sbb_bound(0.01), 2))
    print("Lindley BF01 z=2 n=1e4:", round(lindley_bf01(2, 1e4), 1))
    print("Castaneda log10 BF:", round(castaneda_log10_bf(), 1))
    sd = np.sqrt(870 * 0.791 * 0.209)
    print("Castaneda sd from mean:", round((870 * 0.791 - 339) / sd, 1))
    print("Merton PD sigma=.2/.4:",
          round(merton_pd(100, 85, 0.05, 0.2), 3),
          round(merton_pd(100, 85, 0.05, 0.4), 3))
    comb = combine(0.04, 0.02, 0.20 - 0.08, 0.05)
    print("Solvency combination:", [round(float(x), 3) for x in comb])
    print("Insurance Act payment:", proportionate_payment(10, 1.0, 2.5))
    cab = 0.8 * 0.15 / (0.8 * 0.15 + 0.2 * 0.85)
    print("Cab problem:", round(cab, 3))
