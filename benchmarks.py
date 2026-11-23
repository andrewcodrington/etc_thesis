"""
benchmarks.py - Single source of truth for the discrete 1-cent-grid benchmarks (Addendum Table 11).
Computed from the market model in _market_outcome(), not hard-coded:
  Q = 100 * (2.00 - p_min); base shares (2.00 - p_i) / sum_j (2.00 - p_j); bias moves share to the lowest price.

Competitive benchmark (Appendix A): firms take turns in order A, B, C, ... Each turn, one firm picks the
1-cent price that maximises its own profit that round, given rivals' current prices (ties -> lowest price).
Every firm starts at START_PRICE. The benchmark averages prices, units and joint profit over the final
BR_KEEP turns of a BR_TURNS-turn run.
Monopoly benchmark: the common price on the interior grid ($1.01-$1.99) that maximises joint profit.
Highest joint profit: best two-price configuration (k firms at p_low, n - k at p_high, p_low <= p_high).

Usage: python3 benchmarks.py   -> prints Table 11 and writes benchmarks_table11.csv
"""
import csv
import sys
from functools import lru_cache

START_PRICE = 1.50   # every firm starts at the $1.50 monopoly price
BR_TURNS = 3000
BR_KEEP = 600

UNIT_COST = 1.00     # marginal cost (the bottom of the price grid)
CHOKE_PRICE = 2.00   # demand is zero at this price (the top of the price grid)

_SENS_ALIAS = {
    "high": "high", "sensitive": "high",
    "medium": "medium",
    "low": "low", "insensitive": "low",
}
_BETA = {"high": 1.0, "medium": 0.5, "low": 0.0}
GRID_1C = [round(UNIT_COST + 0.01 * k, 2) for k in range(round((CHOKE_PRICE - UNIT_COST) / 0.01) + 1)]
_INTERIOR = GRID_1C[1:-1]


def _market_outcome(prices, beta: float):
    n = len(prices)
    min_p = min(prices)
    q_total = max(0.0, 100.0 * (CHOKE_PRICE - min_p))
    weights = [max(0.0, CHOKE_PRICE - p) for p in prices]
    w_sum = sum(weights)
    probs = [w / w_sum if w_sum > 0 else 1.0 / n for w in weights]
    low = [i for i, p in enumerate(prices) if p == min_p]
    high = [i for i, p in enumerate(prices) if p > min_p]
    if beta > 0 and high:
        shift = sum(probs[i] for i in high) * beta
        for i in low:
            probs[i] += shift / len(low)
        for i in high:
            probs[i] *= (1.0 - beta)
    units = [round(q_total * s) for s in probs]
    profits = [(p - UNIT_COST) * u for p, u in zip(prices, units)]
    return units, profits


def best_response_path(beta: float, n: int, start: float = START_PRICE, turns: int = BR_TURNS):
    """Turn-based myopic best responses on the 1-cent grid; returns the price vector after every turn."""
    state = [float(start)] * n
    path = []
    for t in range(turns):
        i = t % n
        best_pi, best_p = -1.0, state[i]
        for cand in GRID_1C:
            trial = tuple(state[:i] + [cand] + state[i + 1:])
            pi = _market_outcome(trial, beta)[1][i]
            if pi > best_pi + 1e-12:
                best_pi, best_p = pi, cand
        state[i] = best_p
        path.append(tuple(state))
    return path


@lru_cache(maxsize=None)
def _compute(key: str, n: int) -> dict:
    beta = _BETA[key]

    mono_p, mono_jp = _INTERIOR[0], -1.0
    for p in _INTERIOR:
        jp = sum(_market_outcome((p,) * n, beta)[1])
        if jp > mono_jp:
            mono_jp, mono_p = jp, p

    best_jp, best_cfg = mono_jp, (n, mono_p, mono_p)   # (number of low-priced firms, p_low, p_high)
    if beta < 1.0:
        for k in range(1, n):
            for i_l, p_l in enumerate(_INTERIOR):
                for p_h in _INTERIOR[i_l:]:
                    jp = sum(_market_outcome((p_l,) * k + (p_h,) * (n - k), beta)[1])
                    if jp > best_jp:
                        best_jp, best_cfg = jp, (k, p_l, p_h)

    tail = best_response_path(beta, n)[-BR_KEEP:]
    tot_p, tot_u, tot_rev, tot_jp = 0.0, 0, 0.0, 0.0
    for prices in tail:
        units, profs = _market_outcome(prices, beta)
        tot_p += sum(prices)
        tot_u += sum(units)
        tot_rev += sum(p * u for p, u in zip(prices, units))
        tot_jp += sum(profs)
    comp_post = tot_p / (len(tail) * n)

    return {
        "comp_post": round(comp_post, 2),
        "comp_paid": round(tot_rev / tot_u, 2) if tot_u > 0 else round(comp_post, 2),
        "comp_joint_profit": round(tot_jp / len(tail), 2),
        "mono_price": round(mono_p, 2),
        "mono_joint_profit": round(mono_jp, 2),
        "max_asym_joint_profit": round(best_jp, 2),
        "best_n_low": best_cfg[0],
        "best_p_low": round(best_cfg[1], 2),
        "best_p_high": round(best_cfg[2], 2),
    }


def get_benchmark(price_sensitivity: str, num_agents: int) -> dict:
    key = _SENS_ALIAS[str(price_sensitivity).strip().lower()]
    return dict(_compute(key, int(num_agents)))


PRICE_INDEX_SPAN = 0.50   # $1.00 cost to the $1.50 3-firm monopoly price; same scale as lowest_price_index


def table11_rows() -> list:
    """Addendum Table 11 rows. Derived columns:
    comp_price_index        (comp_post - $1.00) / PRICE_INDEX_SPAN
    comp_share_of_mono_pct  competitive joint profit as % of monopoly joint profit
    best_share_of_mono_pct  highest (two-price) joint profit as % of monopoly joint profit
    excess_index_at_best    I_pi at the highest joint profit; blank where I_pi is undefined (insensitive)
    """
    rows = []
    for key, label in (("high", "Sensitive"), ("medium", "Medium"), ("low", "Insensitive")):
        for n in (3, 5, 7):
            b = get_benchmark(key, n)
            at_best = calvano_delta(b["max_asym_joint_profit"], key, n)
            rows.append({
                "Sensitivity": label, "Firms": n, **b,
                "comp_price_index": round((b["comp_post"] - UNIT_COST) / PRICE_INDEX_SPAN, 2),
                "comp_share_of_mono_pct": round(100.0 * b["comp_joint_profit"] / b["mono_joint_profit"], 1),
                "best_share_of_mono_pct": round(100.0 * b["max_asym_joint_profit"] / b["mono_joint_profit"], 1),
                "excess_index_at_best": round(at_best, 2) if at_best == at_best else None,
            })
    return rows


def calvano_delta(mean_joint_profit: float, price_sensitivity: str, num_agents: int) -> float:
    """Index of Excess Profit (I_pi). Undefined (NaN) where C >= M (beta = 0)."""
    key = _SENS_ALIAS[str(price_sensitivity).strip().lower()]
    if key == "low":
        return float("nan")
    b = get_benchmark(key, num_agents)
    denom = b["mono_joint_profit"] - b["comp_joint_profit"]
    if denom <= 0:
        return float("nan")
    return (mean_joint_profit - b["comp_joint_profit"]) / denom


def delta_best_achievable(mean_joint_profit: float, price_sensitivity: str, num_agents: int) -> float:
    b = get_benchmark(price_sensitivity, num_agents)
    denom = b["max_asym_joint_profit"] - b["comp_joint_profit"]
    if denom <= 0:
        return float("nan")
    return (mean_joint_profit - b["comp_joint_profit"]) / denom


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "benchmarks_table11.csv"
    rows = table11_rows()
    print(f"Table 11 benchmarks (start ${START_PRICE:.2f}, {BR_TURNS:,} turns, final {BR_KEEP} averaged)")
    cols = list(rows[0].keys())
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join("" if r[c] is None else str(r[c]) for c in cols))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out}")
