"""
build_panel.py - Consolidates the run logs listed in logs/rep<N>/factorial_results.csv and
intervention_results.csv into logs/rep<N>/panel_q1_30.csv (the schema used by score_all_indices.py
and export_report_assets.py).

Only logs referenced by those two results files are included. Any other run_*.csv in the folder
(abandoned reruns, stray copies) is listed and ignored. The build stops with an error if a grid cell
or an intervention would appear more than once, or if a referenced log is missing.

The panel's `refused` column is True where the model refused and the firm's previous price was held
(llm_runner.get_agent_decision), so those firm-quarters can be told apart from model decisions.
Usage: python3 build_panel.py [--replicate 1]
"""
import argparse
import os
import pandas as pd

MAX_ROUNDS = 30


def refused_flags(rdf: pd.DataFrame) -> pd.Series:
    """True where the run log marks a held price after a refusal (False for logs without the column)."""
    if "refused" not in rdf.columns:
        return pd.Series(False, index=rdf.index)
    return rdf["refused"].astype(str).str.strip().str.lower() == "true"
SENS_MAP = {"high": ("Sensitive", 1.0), "medium": ("Medium", 0.5), "low": ("Insensitive", 0.0)}
IV_LABELS = {
    "Refine goal": "refine_goal",
    "Weaken long-term language": "weaken_longterm_language",
    "Reduce precision": "reduce_precision",
    "Reduce frequency": "reduce_frequency",
    "Inject marketshare noise": "inject_marketshare_noise",
    "Reduce model parameters": "reduce_model_params",
    "Inject pricing noise": "inject_pricing_noise",
    "Mixed model": "mixed_model",
}


def _base(path) -> str:
    s = "" if pd.isna(path) else str(path).strip()
    return os.path.basename(s) if s else ""


def _collect(run_dir: str) -> dict:
    """Return {log file name: metadata} for every log the two results files reference."""
    fr_path = os.path.join(run_dir, "factorial_results.csv")
    iv_path = os.path.join(run_dir, "intervention_results.csv")
    if not os.path.exists(fr_path) and not os.path.exists(iv_path):
        raise FileNotFoundError(f"Neither factorial_results.csv nor intervention_results.csv found in {run_dir}")

    refs, problems = {}, []
    if os.path.exists(fr_path):
        fr = pd.read_csv(fr_path)
        cells = fr[["num_agents", "price_sensitivity", "visibility", "model_variance"]].astype(str).agg(" | ".join, axis=1)
        dup = sorted(set(cells[cells.duplicated(keep=False)]))
        if dup:
            problems.append("factorial_results.csv lists these cells more than once: " + "; ".join(dup))
        for _, r in fr.iterrows():
            n, sens, vis, var = int(r["num_agents"]), r["price_sensitivity"], r["visibility"], r["model_variance"]
            b = _base(r.get("log_file"))
            if not b:
                print(f"  skipped (run failed, no log): {n}f {sens} {vis} {var}")
                continue
            is_base = (n, sens, vis, var) == (3, "medium", "perfect", "single")
            refs[b] = {"firms": n, "sens": sens, "vis": vis, "mix": var,
                       "intervention": "baseline" if is_base else "grid", "is_grid_cell": True}

    if os.path.exists(iv_path):
        iv = pd.read_csv(iv_path)
        dup = sorted(set(iv["intervention"][iv["intervention"].duplicated(keep=False)]))
        if dup:
            problems.append("intervention_results.csv lists these interventions more than once: " + "; ".join(dup))
        for _, r in iv.iterrows():
            name, b = str(r["intervention"]), _base(r.get("log_file"))
            if not b or b in refs:
                continue  # failed run, or a reused grid cell that is already included once
            if name == "Baseline":
                refs[b] = {"firms": 3, "sens": "medium", "vis": "perfect", "mix": "single",
                           "intervention": "baseline", "is_grid_cell": True}
            elif name in IV_LABELS:
                refs[b] = {"firms": 3, "sens": "medium",
                           "vis": "noisy" if name == "Inject pricing noise" else "perfect",
                           "mix": "mixed" if name == "Mixed model" else "single",
                           "intervention": IV_LABELS[name], "is_grid_cell": False}
            else:
                problems.append(f"unknown intervention '{name}' in intervention_results.csv")

    groups = {}
    for b, m in refs.items():
        key = (m["firms"], m["sens"], m["vis"], m["mix"]) if m["is_grid_cell"] else ("intervention", m["intervention"])
        groups.setdefault(key, []).append(b)
    for key, files in groups.items():
        if len(files) > 1:
            problems.append(f"{key} would appear {len(files)} times: " + ", ".join(sorted(files)))

    if problems:
        raise ValueError("Cannot build panel:\n  " + "\n  ".join(problems))
    return refs


def build_panel(rep: str = "1") -> str:
    run_dir = f"logs/rep{rep}"
    refs = _collect(run_dir)

    on_disk = {f for f in os.listdir(run_dir) if f.startswith("run_") and f.endswith(".csv")}
    for f in sorted(on_disk - set(refs)):
        print(f"  ignored (not listed in the results files): {f}")
    missing = sorted(set(refs) - on_disk)
    if missing:
        raise FileNotFoundError("Logs listed in the results files are missing from " + run_dir + ": " + ", ".join(missing))

    frames = []
    for b, m in sorted(refs.items()):
        rdf = pd.read_csv(os.path.join(run_dir, b))
        if rdf.empty:
            raise ValueError(f"{b} is empty")
        n_rounds = rdf["round"].nunique()
        if n_rounds < MAX_ROUNDS:
            print(f"  warning: {b} has {n_rounds} quarters (expected {MAX_ROUNDS})")

        q_agg = rdf.groupby("round", as_index=False).agg(
            joint_profit_q=("profit", "sum"), mean_price_q=("price", "mean"), tot_u=("units", "sum"),
        )
        rev_q = (rdf["units"] * rdf["price"]).groupby(rdf["round"]).sum().rename("tot_rev").reset_index()
        q_agg = q_agg.merge(rev_q, on="round")
        q_agg["paid_price_q"] = (q_agg["tot_rev"] / q_agg["tot_u"]).where(q_agg["tot_u"] > 0, q_agg["mean_price_q"])
        rdf = rdf.merge(q_agg[["round", "joint_profit_q", "mean_price_q", "paid_price_q"]], on="round", how="left")

        sens_lbl, beta_val = SENS_MAP[m["sens"]]
        vis_lbl = "Noisy" if m["vis"] == "noisy" else "Perfect"
        mix_lbl = "Mixed" if m["mix"] == "mixed" else "Single"
        frames.append(pd.DataFrame({
            "run": b[:-4],
            "firms": m["firms"],
            "beta": beta_val,
            "order_noise": m["vis"] == "noisy",
            "sensitivity": sens_lbl,
            "visibility": vis_lbl,
            "model_mix": mix_lbl,
            "intervention": m["intervention"],
            "is_grid_cell": m["is_grid_cell"],
            "cell_id": f"{m['firms']}f | {sens_lbl} | {vis_lbl} | {mix_lbl}",
            "quarter": rdf["round"].astype(int),
            "firm": rdf["company"],
            "model": rdf["model"],
            "price": rdf["price"].astype(float),
            "units": rdf["units"].astype(int),
            "firm_profit": rdf["profit"].astype(float).round(4),
            "joint_profit_q": rdf["joint_profit_q"].astype(float).round(4),
            "mean_price_q": rdf["mean_price_q"].astype(float).round(4),
            "paid_price_q": rdf["paid_price_q"].astype(float).round(4),
            "refused": refused_flags(rdf),
        }))

    panel = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(run_dir, "panel_q1_30.csv")
    panel.to_csv(out_path, index=False)
    print(f"Built consolidated panel ({panel['run'].nunique()} runs, {len(panel)} rows): {out_path}")
    held = panel[panel["refused"]]
    if not held.empty:
        print(f"  note: {len(held)} firm-quarters in {held['run'].nunique()} runs hold the previous price after a "
              f"refusal ({int(held['quarter'].between(21, 30).sum())} of them in Q21-Q30); see the 'refused' column.")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", default="1")
    build_panel(ap.parse_args().replicate)
