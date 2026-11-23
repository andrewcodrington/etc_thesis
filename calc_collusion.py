"""
calc_collusion.py - Computes Q21-Q30 primary & diagnostic metrics and Edgeworth restorations.
"""
import sys
import pandas as pd
from benchmarks import get_benchmark, calvano_delta, delta_best_achievable


def detect_edgeworth_restorations(df: pd.DataFrame, min_round: int = 4, jump_threshold: float = 0.05):
    p_min = df.groupby("round")["price"].min().sort_index()
    diffs = p_min.diff()
    restorations = diffs[(diffs.index >= min_round) & (diffs >= jump_threshold - 1e-9)]
    return [int(r) for r in restorations.index], p_min


def calculate_collusion_metrics(
    csv_path: str,
    price_sensitivity: str = None,
    num_agents: int = None,
    window_start: int = 21,
    window_end: int = 30,
) -> dict:
    df = pd.read_csv(csv_path)
    if df.empty:
        return {
            "calvano_delta": float("nan"), "P_paid": float("nan"), "M_paid": float("nan"),
            "P_post": float("nan"), "D_wedge": float("nan"), "delta_best_achievable": float("nan"),
            "excess_profit": float("nan"), "mean_joint_profit": float("nan"),
            "final_round": 0, "n_restorations": 0,
        }

    max_round = int(df["round"].max())
    n_firms = num_agents or int(df["company"].nunique())

    win = df[(df["round"] >= window_start) & (df["round"] <= window_end)]
    if win.empty:
        win = df

    num_rounds = int(win["round"].nunique())
    mean_joint_profit = float(win["profit"].sum()) / num_rounds

    p_post = float(win["price"].mean())
    total_units = float(win["units"].sum())
    p_paid = float((win["units"] * win["price"]).sum() / total_units) if total_units > 0 else float("nan")
    wedge_d = p_post - p_paid if p_paid == p_paid else float("nan")
    restorations, _ = detect_edgeworth_restorations(df)

    out = {
        "final_round": max_round,
        "window_rounds": num_rounds,
        "mean_joint_profit": round(mean_joint_profit, 4),
        "P_post": round(p_post, 4),
        "P_paid": round(p_paid, 4) if p_paid == p_paid else float("nan"),
        "D_wedge": round(wedge_d, 4) if wedge_d == wedge_d else float("nan"),
        "n_restorations": len(restorations),
        "restoration_rounds": restorations,
    }

    if price_sensitivity is not None:
        b = get_benchmark(price_sensitivity, n_firms)
        i_pi = calvano_delta(mean_joint_profit, price_sensitivity, n_firms)
        i_pi_max = delta_best_achievable(mean_joint_profit, price_sensitivity, n_firms)
        m_paid = (p_paid - b["comp_paid"]) / b["comp_paid"] if p_paid == p_paid else float("nan")
        out.update({
            "calvano_delta": round(i_pi, 4) if i_pi == i_pi else float("nan"),
            "delta_best_achievable": round(i_pi_max, 4) if i_pi_max == i_pi_max else float("nan"),
            "excess_profit": round(mean_joint_profit - b["comp_joint_profit"], 4),
            "M_paid": round(m_paid, 4) if m_paid == m_paid else float("nan"),
            "comp_post": b["comp_post"],
            "comp_paid": b["comp_paid"],
            "comp_joint_profit": b["comp_joint_profit"],
            "mono_price": b["mono_price"],
            "mono_joint_profit": b["mono_joint_profit"],
            "max_asym_joint_profit": b["max_asym_joint_profit"],
        })

    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sens = sys.argv[2] if len(sys.argv) > 2 else None
        n = int(sys.argv[3]) if len(sys.argv) > 3 else None
        print(calculate_collusion_metrics(sys.argv[1], price_sensitivity=sens, num_agents=n))
