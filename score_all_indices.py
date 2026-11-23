"""
score_all_indices.py - Score Q21-Q30 metrics from consolidated panel CSV using benchmarks.py.
Benchmarks, the unit cost and the price-index span are imported from benchmarks.py, and joint profit at a
common price comes from its market model (_market_outcome), so nothing here repeats a number or formula.
refusals_q21_30 / refusals_q1_30 count firm-quarters where a refusal held the previous price.
Usage: python3 score_all_indices.py [panel.csv]
"""
import glob
import os
import sys
from functools import lru_cache
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import (
    GRID_1C, PRICE_INDEX_SPAN, UNIT_COST, _market_outcome,
    calvano_delta, delta_best_achievable, get_benchmark,
)

Q_START, Q_END = 21, 30
SENS_FROM_BETA = {1.0: "high", 0.5: "medium", 0.0: "low"}


def nz(val: float, dec: int = 4) -> float:
    r = round(float(val), dec)
    return 0.0 if abs(r) == 0 else r


def symmetric_jp(price: float, n: int, beta: float) -> float:
    """Joint profit when all n firms post the same price, from the market model in benchmarks.py."""
    return float(sum(_market_outcome((float(price),) * n, beta)[1]))


@lru_cache(maxsize=None)
def _symmetric_curve(n: int, beta: float, mono_price: float):
    prices = [p for p in GRID_1C if p <= mono_price + 1e-9]
    return prices, [symmetric_jp(p, n, beta) for p in prices]


def profit_equiv_price(jp: float, n: int, beta: float, mono_price: float) -> float:
    """Common price at which n firms would earn joint profit jp: the first price on the 1-cent grid from
    the unit cost up to the monopoly price whose joint profit reaches jp, interpolated linearly between
    that cent and the one below. NaN where no common price earns jp (jp < 0 or above the monopoly joint profit)."""
    prices, jps = _symmetric_curve(n, beta, mono_price)
    if jp < 0 or jp > max(jps) + 1e-9:
        return np.nan
    k = next(i for i, v in enumerate(jps) if v >= jp - 1e-9)
    if k == 0:
        return round(prices[0], 4)
    lo, hi = jps[k - 1], jps[k]
    return round(prices[k - 1] + (jp - lo) / (hi - lo) * (prices[k] - prices[k - 1]), 4)


def refused_flags(df: pd.DataFrame) -> pd.Series:
    """True where the model refused and the previous price was held (panel 'refused' column)."""
    if "refused" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["refused"].astype(str).str.strip().str.lower() == "true"


def find_default_panel():
    panels = sorted(glob.glob("logs/rep*/panel_q1_30.csv") + glob.glob("panel_q1_30.csv"))
    if panels:
        return panels[-1]
    raise FileNotFoundError("No logs/rep*/panel_q1_30.csv found. Run build_panel.py, or pass a panel path explicitly.")


panel_path = sys.argv[1] if len(sys.argv) > 1 else find_default_panel()
print(f"Scoring Q{Q_START}-{Q_END} from: {panel_path}")
df = pd.read_csv(panel_path)
has_refused = "refused" in df.columns
if not has_refused:
    print("WARNING: the panel has no 'refused' column, so held prices after a refusal cannot be identified. "
          "Rebuild it with build_panel.py.")
df["held"] = refused_flags(df)
held_total = df.groupby("run")["held"].sum()
late = df[df["quarter"].between(Q_START, Q_END)].copy()

rows = []
for run_id, sub in late.groupby("run", sort=False):
    n = int(sub["firms"].iloc[0])
    beta = float(sub["beta"].iloc[0])
    sens_key = SENS_FROM_BETA[round(beta, 1)]
    is_sens = sens_key == "high"
    is_med = sens_key == "medium"

    bm = get_benchmark(sens_key, n)
    cp = bm["comp_joint_profit"]
    cp_avg, cp_paid, mp_price = bm["comp_post"], bm["comp_paid"], bm["mono_price"]

    q_agg = (
        sub.groupby("quarter", as_index=False)
        .agg(jp_q=("joint_profit_q", "first"), mean_p_q=("price", "mean"), min_p_q=("price", "min"))
    )
    jp = float(q_agg["jp_q"].mean())
    mean_p = float(q_agg["mean_p_q"].mean())
    mean_min_p = float(q_agg["min_p_q"].mean())

    tot_units = float(sub["units"].sum())
    paid_p = float((sub["units"] * sub["price"]).sum() / tot_units) if tot_units > 0 else mean_p
    ub_jp = symmetric_jp(mean_p, n, beta)

    rows.append({
        "run": run_id,
        "firms": n,
        "market": "sensitive" if is_sens else ("medium" if is_med else "insensitive"),
        "visibility": str(sub["visibility"].iloc[0]),
        "model_mix": str(sub["model_mix"].iloc[0]),
        "order_noise": bool(sub["order_noise"].iloc[0]),
        "is_grid_cell": bool(sub["is_grid_cell"].iloc[0]),
        "intervention": str(sub["intervention"].iloc[0]),
        "joint_profit": nz(jp, 4),
        "mean_posted_price": nz(mean_p, 4),
        "calvano_delta": nz(calvano_delta(jp, sens_key, n), 4),
        "excess_profit": nz(jp - cp, 4),
        "delta_best_achievable": nz(delta_best_achievable(jp, sens_key, n), 4),
        "upper_bound_delta": nz(calvano_delta(ub_jp, sens_key, n), 4),
        "corrected_price_index": nz((mean_p - cp_avg) / (mp_price - cp_avg), 4) if (is_sens or is_med) else np.nan,
        "price_buyers_paid": nz(paid_p, 4),
        "price_increase_over_cp": nz((paid_p / cp_paid) - 1.0, 4),
        "wedge_cents": nz((mean_p - paid_p) * 100.0, 2),
        "margin_increase_over_cp": nz(((paid_p - UNIT_COST) / (cp_paid - UNIT_COST)) - 1.0, 4) if not is_sens else np.nan,
        "profit_equivalent_price": profit_equiv_price(jp, n, beta, mp_price),
        "lowest_price_index": nz((mean_min_p - UNIT_COST) / PRICE_INDEX_SPAN, 4),
        "refusals_q21_30": int(sub["held"].sum()) if has_refused else np.nan,
        "refusals_q1_30": int(held_total.get(run_id, 0)) if has_refused else np.nan,
    })

out = pd.DataFrame(rows)
out_path = panel_path.replace(".csv", f"_all_indices_q{Q_START}_{Q_END}.csv")
out.to_csv(out_path, index=False)

iv_rows = out[out["intervention"] != "grid"].copy()
if not iv_rows.empty:
    if "baseline" in iv_rows["intervention"].values:
        base_d = iv_rows.loc[iv_rows["intervention"] == "baseline", "calvano_delta"].iloc[0]
        iv_rows["delta_I_pi"] = [nz(x - base_d, 4) for x in iv_rows["calvano_delta"]]
    print("\n=== 3A MEDIUM INTERVENTIONS SUMMARY ===")
    cols = ["intervention", "joint_profit", "calvano_delta", "delta_I_pi", "price_buyers_paid", "price_increase_over_cp",
            "mean_posted_price", "wedge_cents", "refusals_q21_30"]
    print(iv_rows[[c for c in cols if c in iv_rows.columns]].to_string(index=False))

grid_src = out[out["is_grid_cell"]].copy()
grid_src["agent_design"] = grid_src["visibility"] + " | " + grid_src["model_mix"]

row_order = pd.MultiIndex.from_tuples(
    [(m, n) for m in ("sensitive", "medium", "insensitive") for n in (3, 5, 7)],
    names=["market", "firms"],
)
col_order = ["Perfect | Single", "Perfect | Mixed", "Noisy | Single", "Noisy | Mixed"]

all_metrics = [
    "joint_profit", "calvano_delta", "excess_profit", "delta_best_achievable",
    "price_buyers_paid", "price_increase_over_cp", "mean_posted_price", "wedge_cents", "lowest_price_index",
]

grid_frames = []
for metric in all_metrics:
    g = grid_src.pivot(index=["market", "firms"], columns="agent_design", values=metric).reindex(index=row_order, columns=col_order)
    print(f"\n=== {metric.upper()} (Q{Q_START}-{Q_END}) ===")
    print(g.to_string())
    g_out = g.reset_index()
    g_out.insert(0, "measure", metric)
    grid_frames.append(g_out)

grid_path = panel_path.replace(".csv", f"_heatmap_grids_q{Q_START}_{Q_END}.csv")
pd.concat(grid_frames, ignore_index=True).to_csv(grid_path, index=False)

if has_refused:
    scored_held = out[out["refusals_q21_30"] > 0]
    if scored_held.empty:
        print("\nRefusals: no held prices in Q21-Q30.")
    else:
        print(f"\nWARNING: {len(scored_held)} runs include held prices after refusals in Q21-Q30 (scored as observed):")
        for r in scored_held.itertuples():
            print(f"  {r.intervention:>24s}  {r.firms}f {r.market} {r.visibility} {r.model_mix}: {int(r.refusals_q21_30)} firm-quarters")
print(f"\nSaved:\n  {out_path}\n  {grid_path}")
