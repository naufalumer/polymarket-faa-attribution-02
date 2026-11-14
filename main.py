#!/usr/bin/env python3
"""polymarket-faa-attribution-02 rev3: level-decaying centre with a
saturating dispersion profile.

For every market of a Polymarket event family, forecast the Yes price at
every minute of the 240 minutes after a checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per minute. Scored by mean pinball loss.

The model (per market, from the case file alone) is a Gaussian random
walk whose centre and width both come from the market's own recent
history, but with two measured departures from a plain random walk:

  1. Level-decaying centre. The centre does not sit still at the last
     price: it decays toward zero at a rate proportional to its own
     level, centre(h) = c * exp(-K * c * h). A regression over the 5
     practice cases (100 market-cases) found the realised 240-minute
     change is negatively correlated with the market's price level
     (corr -0.34): high-priced markets fall further than low-priced
     ones, as the field's probability mass redistributes toward the
     tail. The decay is therefore steepest for the leaders (c ~ 0.3
     decays ~7% over the horizon) and negligible for the dead tail
     (c ~ 0.003 decays ~0.07%). A uniform decay and a sum-aware decay
     were both tried and both scored worse; the level-weighted form is
     the one that survives. The one-time coherence adjustment (below)
     is applied to the level before the decay, so the Iran -> foreign
     actor ordering survives the decay unchanged.

  2. Saturating dispersion. The width does not grow as sqrt(h). The
     measured dispersion of h-step price changes (recent 1440-minute
     window, pooled over the practice markets) grows much slower than
     sqrt(h) -- the 240-minute standard deviation is only ~4x the
     1-minute one, not ~15x -- because price moves are steps: a market
     jumps to a new level and then sits, it does not keep diffusing.
     The width is therefore sigma * sqrt(h) * (1 + h/H0)^(-BETA) with
     H0 = 120, BETA = 0.1, i.e. the sqrt(h) growth damped by ~10% at
     the far horizon. A pure power law (h^0.45, h^0.4) and the raw
     empirical h-step dispersion were both tried and both scored much
     worse; the damped form is a mild correction, not a replacement.

  3. Short-window-only volatility. sigma is the population standard
     deviation of 1-minute price changes over the last 120 minutes,
     each change clipped to +/-0.02 so one jump cannot dominate,
     times MULT = 0.6, floored at 0.0003 and at REL_FLOOR * last price
     (0.010). The long 1440-minute window that rev2 blended in
     (weight 0.2) was dropped: volatility clusters (lag-1
     autocorrelation of |changes| is +0.1 to +0.5), so the recent
     regime is the informative one, and the long window only dilutes
     it. EWMA volatility (30/60/120-minute halflife) was tried and
     did not beat this.

Coherence (unchanged in kind from the earlier entries, because the
final needs it): the centres are adjusted for the family before the
decay. If both FAA attribution markets are present, the foreign-actor
centre is raised to the Iran centre when it sits below it (an official
attribution to Iran IS an attribution to a foreign actor). Otherwise
(mutually exclusive outcomes, as in the practice) the centres are
shrunk toward a sum of 1 (75% of an excess removed, 50% of a
shortfall added).

What was tried and REJECTED on the same 5 practice cases (all scored
with the kit's local scorer; rev2 = 0.296299, house baseline =
0.398458 on the same cases):
  * empirical h-step quantiles (own history, several windows): 0.42
  * fitted OLS centre (slope, level deviation, vol, sum excess;
    leave-one-case-out mean 0.371 -- the fitted centre does not
    generalise at all)
  * drift / momentum centre (positive or negative slope): 0.30-0.50
  * mean reversion to the long-run mean (OU): 0.47-1.05
  * asymmetric up/down volatility, two-sigma (narrow core + wide
    tails), skew, IQR volatility, centre smoothing, growth exponents
    0.40/0.45/0.55: all 0.295-0.35
  * raw empirical h-step dispersion as the width: 0.50-0.76
  The full search is in rev3/exp*.py (13 experiment rounds).

Verified scores (local scorer, this kit, cases p00-p04):
  mean pinball loss 0.291986 vs live rev2 0.296299 and house baseline
  0.398458 -- beats the baseline on EVERY case (p00 0.3617 vs
  0.5361, p01 0.2982 vs 0.3804, p02 0.3366 vs 0.4084, p03 0.2586 vs
  0.3686, p04 0.2049 vs 0.2988). Per case vs rev2: p00 0.3617 vs
  0.3890, p01 0.2982 vs 0.2955, p02 0.3366 vs 0.3366, p03 0.2586 vs
  0.2576, p04 0.2049 vs 0.2028. The mean gain (-0.0043, -1.5%) is
  carried mostly by p00, the case whose young second-wave markets
  most resemble the final's young FAA markets; the other four cases
  are within +/-0.003 of rev2. The same variant on a short-history
  regime (every market truncated to its last 960 minutes): 0.291840
  vs rev2 0.295842. The constants sit in a flat region (decay K
  0.0007-0.0013 and BETA 0.05-0.15 all score 0.2921-0.2928), so the
  score is not a cherry-picked spike.

Caveat (honest): the kit's practice is cut from Polymarket's "Next
CEO of NVIDIA?" event, not from the FAA event, so the practice score
is a relative comparison between variants only, NOT a final
expectation. The final's four FAA markets are young (created
2026-11-13) and jumpy.

Pure stdlib Python, deterministic (two runs on the same case write
identical bytes), no network, well under the 8 s per-case cap.
"""

import json
import math
import os
import statistics
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_ZDIST = statistics.NormalDist()
Z_VALUES = tuple(_ZDIST.inv_cdf(tau) for tau in QUANTILES)

# --- model constants (tuned on the practice cases p00-p04) ---
SHORT_MINUTES = 120  # minutes of history used for the volatility estimate
CLIP_STEP = 0.02     # 1-minute price changes are clipped to +/- this
VOL_MULT = 0.6       # overall multiplier on the clipped volatility
ABS_FLOOR = 0.0003   # smallest per-minute spread, in price units
REL_FLOOR = 0.010    # sigma never below this fraction of the last price
GROWTH_HALF = 120.0  # horizon (minutes) where the growth damping bends
GROWTH_BETA = 0.1    # damping exponent on the sqrt(h) growth
DECAY_K = 0.001      # centre decay rate, per unit of price per minute
DOWN_SHRINK = 0.75   # fraction of a sum > 1 excess removed from centres
UP_SHRINK = 0.5      # fraction of a sum < 1 shortfall added to centres

# Slugs of the FAA attribution family's markets, for the ordering rule.
IRAN_SLUG = "faa-outage-attributed-to-iran"
FOREIGN_SLUG = "faa-outage-attributed-to-a-foreign-actor"


def _round6(value):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, value)), 6) + 0.0


def _recent_vol(prices):
    """Population std of 1-minute changes over the recent window.

    Each change is clipped to +/-CLIP_STEP first, so a single jump
    does not dominate the spread. Uses whatever history exists when
    a market is younger than the window.
    """
    window = prices[-SHORT_MINUTES:] if len(prices) > SHORT_MINUTES else prices
    if len(window) < 2:
        return 0.0
    steps = []
    for earlier, later in zip(window, window[1:]):
        move = later - earlier
        if move > CLIP_STEP:
            move = CLIP_STEP
        elif move < -CLIP_STEP:
            move = -CLIP_STEP
        steps.append(move)
    count = len(steps)
    mean = sum(steps) / count
    var = sum((s - mean) ** 2 for s in steps) / count
    return math.sqrt(var)


def _coherent_levels(markets):
    """Last prices, with the cross-market coherence rule of the family.

    Two families, two rules, chosen from the case's own market slugs:
      * FAA attribution (both the Iran and the foreign-actor markets
        present): the four hypotheses overlap, so no sum rule applies.
        The one logical order that does hold is Iran => foreign actor:
        when the books disagree (Iran priced above foreign actor), the
        foreign-actor level is raised to the Iran level. A max, not an
        average: the higher price is the binding piece of information.
      * Anything else (the practice: mutually exclusive outcomes): the
        levels are shrunk toward a sum of 1.
    """
    levels = [m["prices"][-1] if m["prices"] else 0.5 for m in markets]
    by_slug = {}
    for idx, m in enumerate(markets):
        by_slug.setdefault(m.get("slug", ""), idx)
    if IRAN_SLUG in by_slug and FOREIGN_SLUG in by_slug:
        iran = by_slug[IRAN_SLUG]
        foreign = by_slug[FOREIGN_SLUG]
        if levels[foreign] < levels[iran]:
            levels[foreign] = levels[iran]
        return levels
    total = sum(levels)
    if total > 1.0:
        factor = 1.0 - DOWN_SHRINK * (total - 1.0) / total
    elif 0.0 < total < 1.0:
        factor = 1.0 + UP_SHRINK * (1.0 - total) / total
    else:
        factor = 1.0
    return [x * factor for x in levels]


def _market_rows(market, level, horizon):
    """240 rows of five quantiles for one market.

    The level decays toward zero at a rate proportional to itself
    (leaders fall, the dead tail barely moves), and the width grows
    more slowly than sqrt(h) (price moves are steps, not diffusion).
    """
    prices = market["prices"]
    if not prices:
        raise ValueError("market %s has no prices" % market["id"])
    last = prices[-1]
    sigma = _recent_vol(prices) * VOL_MULT
    if sigma < ABS_FLOOR:
        sigma = ABS_FLOOR
    rel = REL_FLOOR * last
    if sigma < rel:
        sigma = rel
    rows = []
    for minute in range(1, horizon + 1):
        width = sigma * math.sqrt(minute) * (1.0 + minute / GROWTH_HALF) ** (-GROWTH_BETA)
        centre = level * math.exp(-DECAY_K * level * minute)
        row = []
        prev = 0.0
        for z in Z_VALUES:
            value = _round6(centre + z * width)
            if value < prev:
                value = prev
            row.append(value)
            prev = value
        rows.append(row)
    return rows


def _forecast_case(case):
    """Forecast every market of one case."""
    horizon = int(case["horizon_minutes"])
    markets = case["markets"]
    levels = _coherent_levels(markets)
    return {"forecasts": [
        {"id": m["id"], "q": _market_rows(m, level, horizon)}
        for m, level in zip(markets, levels)
    ]}


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    if not inputs:
        raise FileNotFoundError("no *.in case file in %s" % in_dir)
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        if list(case["quantiles"]) != list(QUANTILES):
            raise ValueError(
                "%s asks for quantiles %r, not %r"
                % (path.name, case["quantiles"], QUANTILES)
            )
        horizon = case["horizon_minutes"]
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(
                "%s has an invalid horizon_minutes %r" % (path.name, horizon)
            )
        out = _forecast_case(case)
        with open(out_dir / (case_id + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
