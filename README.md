# polymarket-faa-attribution-02 — rev3

**Method:** Gaussian random walk with a level-decaying centre and a
saturating dispersion profile. Per market, from the case file alone:
centre(h) = c * exp(-0.001 * c * h) where c is the coherence-adjusted
last price; width(h) = sigma * sqrt(h) * (1 + h/120)^(-0.1), where
sigma = 0.6 * clipped-std of 1-minute changes over the last 120 min
(changes clipped to ±0.02), floored at 0.0003 and at 0.010 * last
price. Five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) from the Gaussian
z-scores. Coherence: FAA markets use the Iran → foreign-actor
ordering; any other family shrinks centres toward a sum of 1 (75%
of an excess removed, 50% of a shortfall added).

**Verified scores (local scorer, this kit, cases p00–p04):**
mean pinball loss **0.291986** vs live rev2 **0.296299** and house
baseline **0.398458**. Beats the baseline on every case
(p00 0.3617/0.5361, p01 0.2982/0.3804, p02 0.3366/0.4084,
p03 0.2586/0.3686, p04 0.2049/0.2988). Per case vs rev2: p00
0.3617/0.3890, p01 0.2982/0.2955, p02 0.3366/0.3366, p03
0.2586/0.2576, p04 0.2049/0.2028. Short-history regime (every market
truncated to its last 960 min): 0.291840 vs rev2 0.295842.

The mean gain is carried mostly by p00 (the case whose young
second-wave markets most resemble the final's young FAA markets);
the other four cases sit within ±0.003 of rev2. Constants sit in a
flat region (decay K 0.0007–0.0013 and beta 0.05–0.15 all score
0.2921–0.2928), so the score is not a cherry-picked spike.

**What was tried and rejected** (13 experiment rounds in exp*.py):
empirical h-step quantiles (0.42), fitted OLS centre (leave-one-case-out
0.371), drift/momentum centre (0.30–0.50), OU mean reversion
(0.47–1.05), asymmetric volatility, two-sigma narrow-core/wide-tails,
skew, IQR volatility, centre smoothing, growth exponents 0.40/0.45/0.55,
raw empirical h-step dispersion as width (0.50–0.76).

Pure stdlib Python, deterministic, no network, well under the 8 s
per-case cap. Entry for make.fun competition
polymarket-faa-attribution-02.
