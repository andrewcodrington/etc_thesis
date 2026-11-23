"""
run_experiments.py - Orchestrator for the 36-cell grid (Q1-Q30), multi-seed Q-learning & PPO baselines,
8 policy interventions on the 3-firm medium baseline, automatic panel consolidation, and the report
figures and table cuts (export_report_assets.py) once the panel is complete.
"""
import argparse
import asyncio
import os
import sys
import numpy as np
import pandas as pd
from config import (
    MarketConfig, PROVIDER_MODELS, PROVIDERS, set_providers, ACTIVE_PROVIDERS,
    GRID_FIRMS, GRID_SENSITIVITIES, GRID_VISIBILITIES, SINGLE_PROVIDER_MAP,
    load_provider_map, write_provider_map, model_for,
)
from llm_runner import LLMPricingSimulation, QuotaExhaustedError, RefusalError
from calc_collusion import calculate_collusion_metrics
from benchmarks import get_benchmark
from build_panel import build_panel
from export_report_assets import build_report_assets, report_ready, appendix_b_event_aligned
import llm_runner

RESUME_HINT = "Re-run the same command with --resume once the limit clears; completed cells are kept."


def summarise(log_file, n, sens):
    m = calculate_collusion_metrics(log_file, price_sensitivity=sens, num_agents=n)
    return {
        "calvano_delta": m["calvano_delta"],
        "P_paid": m["P_paid"],
        "M_paid": m["M_paid"],
        "P_post": m["P_post"],
        "D_wedge": m["D_wedge"],
        "delta_best_achievable": m["delta_best_achievable"],
        "excess_profit": m["excess_profit"],
        "mean_joint_profit": m["mean_joint_profit"],
        "comp_post": m["comp_post"],
        "comp_paid": m["comp_paid"],
        "comp_joint_profit": m["comp_joint_profit"],
        "mono_joint_profit": m["mono_joint_profit"],
        "n_restorations": m["n_restorations"],
        "final_round": m["final_round"],
        **refusal_stats(log_file),
        "log_file": log_file,
    }


NO_REFUSAL_STATS = {"refusals": float("nan"), "refusals_q21_30": float("nan"),
                    "first_refusal_round": float("nan"), "refusing_models": ""}


def refusal_stats(log_file):
    """Firm-quarters where the model refused and the previous price was held (run log 'refused' column)."""
    try:
        d = pd.read_csv(log_file)
    except Exception:
        return dict(NO_REFUSAL_STATS)
    if "refused" not in d.columns:
        return {"refusals": 0, "refusals_q21_30": 0, "first_refusal_round": float("nan"), "refusing_models": ""}
    ref = d[d["refused"].astype(str).str.lower() == "true"]
    return {
        "refusals": len(ref),
        "refusals_q21_30": int(ref["round"].between(21, 30).sum()),
        "first_refusal_round": ref["round"].min() if len(ref) else float("nan"),
        "refusing_models": "; ".join(sorted(ref["model"].unique())),
    }


def with_refusals(row):
    """Add (or refresh) refusal counts on a results row from its log file, e.g. rows kept by --resume."""
    log = row.get("log_file")
    if isinstance(log, str) and log and os.path.exists(log):
        row.update(refusal_stats(log))
    return row


def empty_summary(n, sens):
    b = get_benchmark(sens, n)
    nan = float("nan")
    return {
        "calvano_delta": nan, "P_paid": nan, "M_paid": nan, "P_post": nan, "D_wedge": nan,
        "delta_best_achievable": nan, "excess_profit": nan, "mean_joint_profit": nan,
        "comp_post": b["comp_post"], "comp_paid": b["comp_paid"],
        "comp_joint_profit": b["comp_joint_profit"], "mono_joint_profit": b["mono_joint_profit"],
        "n_restorations": 0, "final_round": nan, **NO_REFUSAL_STATS, "log_file": "",
    }


def models_used(cfg):
    return "; ".join(sorted({cfg.get_model_for_firm(f) for f in cfg.company_names}))


def resolve_provider_map(args):
    if args.single_provider:
        return {k: args.single_provider for k in SINGLE_PROVIDER_MAP}
    if args.provider_map:
        return load_provider_map(args.provider_map)
    return dict(SINGLE_PROVIDER_MAP)


SMOKE_DIR = "logs/smoke"


async def run_smoke_test(rep, max_rounds):
    print(f"\n--- Smoke test 1/2: one call per 2025 model (logs in {SMOKE_DIR}/) ---")
    cfg = MarketConfig(num_agents=3, replicate=rep, log_root=SMOKE_DIR)
    sim = LLMPricingSimulation(cfg)
    prompt = sim._build_prompt("A", 1)
    all_models = sorted({m for sizes in PROVIDER_MODELS.values() for m in sizes.values()})
    ok = True
    for model in all_models:
        try:
            d = await sim.call_model(model, prompt)
            tok = d.get("tokens", (0, 0))
            print(f"  OK   {model:28s} price {d['price']:.2f}  tokens in/out: {tok[0]}/{tok[1]}")
        except (RuntimeError, QuotaExhaustedError, RefusalError) as e:
            ok = False
            print(f"  FAIL {model:28s} {e}")
    os.remove(sim.log_file)

    print("\n--- Smoke test 2/2: 3 firms, medium sensitivity, perfect visibility, mixed models ---")
    cfg = MarketConfig(
        num_agents=3, price_sensitivity="medium", visibility="perfect",
        model_variance="mixed", max_rounds=max_rounds, replicate=rep, log_root=SMOKE_DIR,
    )
    try:
        log = await LLMPricingSimulation(cfg).run()
        print(f"Smoke run OK ({log}):", summarise(log, 3, "medium"))
    except (RuntimeError, QuotaExhaustedError, RefusalError) as e:
        ok = False
        print(f"Smoke run failed: {e}")
    print("\nSmoke test passed." if ok else "\nSmoke test found problems.")


def _expected_label(var, n, sens, vis, pmap):
    return "mixed" if var == "mixed" else pmap[(n, sens, vis)]


QL_COLS = ["price_sensitivity", "num_agents", "periods", "seed", "ql_index"]
PPO_COLS = ["price_sensitivity", "num_agents", "visibility", "env_steps", "seed", "rl_index"]
RL_CELLS = [(n, s) for s in ("high", "medium") for n in GRID_FIRMS]


def _load_records(path, cols, keep):
    if not (keep and os.path.exists(path)):
        return []
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks column(s) {missing}; move it aside so seeds trained for different lengths don't mix.")
    return df[cols].to_dict("records")


def run_ql_seeds(records, path, n, sens, periods, runs):
    from rl_runner import run_qlearning_benchmark
    runs = max(1, runs)
    match = lambda r: r["price_sensitivity"] == sens and int(r["num_agents"]) == n and int(r["periods"]) == periods
    have = {int(r["seed"]) for r in records if match(r)}
    bias_val = MarketConfig(price_sensitivity=sens).bias
    for seed_i in range(runs):
        if seed_i in have:
            continue
        val = run_qlearning_benchmark(num_agents=n, bias=bias_val, periods=periods, seed=seed_i)
        records.append({"price_sensitivity": sens, "num_agents": n, "periods": periods, "seed": seed_i, "ql_index": round(float(val), 4)})
        pd.DataFrame(records, columns=QL_COLS).to_csv(path, index=False)
    return float(np.mean([float(r["ql_index"]) for r in records if match(r) and int(r["seed"]) < runs]))


def run_ppo_seeds(records, path, n, sens, vis, steps, runs):
    from rl_runner import run_rl_benchmark
    runs = max(1, runs)
    match = lambda r: (r["price_sensitivity"] == sens and int(r["num_agents"]) == n
                       and r["visibility"] == vis and int(r["env_steps"]) == steps)
    have = {int(r["seed"]) for r in records if match(r)}
    bias_val = MarketConfig(price_sensitivity=sens).bias
    for seed_i in range(runs):
        if seed_i in have:
            continue
        val = run_rl_benchmark(num_agents=n, bias=bias_val, visibility=vis, env_steps=steps, seed=seed_i)
        records.append({"price_sensitivity": sens, "num_agents": n, "visibility": vis, "env_steps": steps,
                        "seed": seed_i, "rl_index": round(float(val), 4)})
        pd.DataFrame(records, columns=PPO_COLS).to_csv(path, index=False)
    vals = [float(r["rl_index"]) for r in records if match(r) and int(r["seed"]) < runs]
    vals = [v for v in vals if v == v]
    if len(vals) < runs:
        print(f"  warning: PPO {n}f {sens} {vis}: {runs - len(vals)} of {runs} seeds produced no index")
    return float(np.mean(vals)) if vals else float("nan")


def run_ql_only(rep, ql_periods, ql_runs):
    os.makedirs(f"logs/rep{rep}", exist_ok=True)
    ql_csv = f"logs/rep{rep}/ql_runs.csv"
    records = _load_records(ql_csv, QL_COLS, keep=True)
    print(f"\n--- Q-learning only: {ql_runs} seeds x {ql_periods:,} periods per cell -> {ql_csv} ---")
    for n, sens in RL_CELLS:
        mean_idx = run_ql_seeds(records, ql_csv, n, sens, ql_periods, ql_runs)
        print(f"  {sens:6s} {n} firms: mean index {mean_idx:.3f}")


async def run_factorial_grid(
    rep, pmap, max_rounds, skip_rl=False, ql_periods=2_500_000, ql_runs=30, ppo_steps=2_050_000, ppo_runs=5,
    resume=False, parallel=1, cell_filter=None,
):
    print("\n--- Running 36-Cell Factorial Grid (Q1-Q30, scored Q21-Q30) ---")
    os.makedirs(f"logs/rep{rep}", exist_ok=True)
    out_csv = f"logs/rep{rep}/factorial_results.csv"
    ql_csv = f"logs/rep{rep}/ql_runs.csv"
    ppo_csv = f"logs/rep{rep}/ppo_runs.csv"
    write_provider_map(pmap, f"logs/rep{rep}/provider_map.csv")

    results, done, rl_done, ql_done = [], set(), {}, {}
    ql_records = _load_records(ql_csv, QL_COLS, keep=resume)
    ppo_records = _load_records(ppo_csv, PPO_COLS, keep=resume)
    if resume and os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        for _, row in old.iterrows():
            key = (int(row["num_agents"]), row["price_sensitivity"], row["visibility"], row["model_variance"])
            if pd.notna(row.get("P_post")) and row.get("provider") == _expected_label(key[3], *key[:3], pmap):
                results.append(with_refusals(row.to_dict()))
                done.add(key)
        print(f"Resuming: {len(done)} of 36 cells already complete.")

    def get_ppo(n, sens, vis):
        if sens == "low" or vis != "perfect":
            return float("nan")
        k = (n, sens, vis)
        if k not in rl_done:
            rl_done[k] = run_ppo_seeds(ppo_records, ppo_csv, n, sens, vis, ppo_steps, ppo_runs)
        return rl_done[k]

    def get_ql(n, sens, vis):
        if sens == "low" or vis != "perfect":
            return float("nan")
        k = (n, sens)
        if k not in ql_done:
            ql_done[k] = run_ql_seeds(ql_records, ql_csv, n, sens, ql_periods, ql_runs)
        return ql_done[k]

    def cfg_for(n, sens, vis, var):
        return MarketConfig(
            num_agents=n, price_sensitivity=sens, visibility=vis, model_variance=var,
            single_provider=pmap[(n, sens, vis)] if var == "single" else None,
            max_rounds=max_rounds, replicate=rep,
        )

    def save():
        order = {
            (n, s, v, m): i
            for i, (n, s, v, m) in enumerate(
                (n, s, v, m) for n in GRID_FIRMS for s in GRID_SENSITIVITIES for v in GRID_VISIBILITIES for m in ["single", "mixed"]
            )
        }
        results.sort(key=lambda r: order[(int(r["num_agents"]), r["price_sensitivity"], r["visibility"], r["model_variance"])])
        pd.DataFrame(results).to_csv(out_csv, index=False)

    jobs = [
        (n, s, v, m)
        for n in GRID_FIRMS for s in GRID_SENSITIVITIES for v in GRID_VISIBILITIES for m in ["single", "mixed"]
        if (n, s, v, m) not in done and (cell_filter is None or (n, s, v, m) in cell_filter)
    ]
    jobs.sort(key=lambda j: ("deepseek" not in models_used(cfg_for(*j)), -j[0]))
    sem = asyncio.Semaphore(max(1, parallel))

    async def run_job(job):
        n, sens, vis, var = job
        cfg = cfg_for(*job)
        async with sem:
            print(f"\nRunning: {n}f, {sens}, {vis}, {var} ({cfg.provider_label}) [rep {rep}]")
            try:
                log_file = await LLMPricingSimulation(cfg).run()
                summary = summarise(log_file, n, sens)
            except RuntimeError as e:
                print(f"Run failed: {e}. Recording NaN.")
                summary = empty_summary(n, sens)
        results.append({
            "replicate": rep, "num_agents": n, "price_sensitivity": sens,
            "visibility": vis, "model_variance": var, "provider": cfg.provider_label,
            "models": models_used(cfg), "providers": "+".join(ACTIVE_PROVIDERS),
            "max_rounds": max_rounds,
            "ql_index": ql_done.get((n, sens), float("nan")),
            "rl_index": rl_done.get((n, sens, vis), float("nan")),
            **summary,
        })
        save()

    tasks = [asyncio.create_task(run_job(j)) for j in jobs]
    try:
        await asyncio.gather(*tasks)
    except QuotaExhaustedError as e:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        save()
        print(f"\nStopped: {e}\n{RESUME_HINT}")
        raise

    if not skip_rl:
        for row in results:
            n = int(row["num_agents"])
            sens, vis = row["price_sensitivity"], row["visibility"]
            if pd.isna(row.get("ql_index", float("nan"))):
                row["ql_index"] = get_ql(n, sens, vis)
                save()
            if pd.isna(row.get("rl_index", float("nan"))):
                row["rl_index"] = get_ppo(n, sens, vis)
                save()
    save()
    print(f"\nGrid results saved to {out_csv}")


def load_grid_result(rep, visibility, model_variance, provider):
    path = f"logs/rep{rep}/factorial_results.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[
        (df["num_agents"] == 3) & (df["price_sensitivity"] == "medium") &
        (df["visibility"] == visibility) & (df["model_variance"] == model_variance) & (df["provider"] == provider)
    ]
    if row.empty or pd.isna(row.iloc[0]["calvano_delta"]):
        return None
    return float(row.iloc[0]["calvano_delta"]), row.iloc[0]["log_file"]


async def run_or_reuse(rep, label, cfg, visibility, model_variance):
    hit = load_grid_result(rep, visibility, model_variance, cfg.provider_label)
    if hit is not None:
        print(f"{label}: reusing grid cell (3 firms, medium, {visibility}, {cfg.provider_label}), I_pi {hit[0]:.4f}")
        return hit[0], hit[1], "grid"
    log = await LLMPricingSimulation(cfg).run()
    m = calculate_collusion_metrics(log, price_sensitivity="medium", num_agents=3)
    return m["calvano_delta"], log, "run"


INTERVENTIONS = [
    "Refine goal", "Weaken long-term language", "Reduce precision",
    "Reduce frequency", "Inject pricing noise", "Inject marketshare noise",
    "Mixed model", "Reduce model parameters",
]


async def run_report_fig6_interventions(rep, pmap, max_rounds, resume=False, parallel=1):
    """Report Figure 6: 8 interventions on the 3-firm, medium-sensitivity, perfect-visibility baseline."""
    print("\n--- Running Report Figure 6 Interventions (3-Firm Medium Baseline) ---")
    os.makedirs(f"logs/rep{rep}", exist_ok=True)
    out_csv = f"logs/rep{rep}/intervention_results.csv"
    base_provider = pmap[(3, "medium", "perfect")]

    def cfg_for(**kw):
        return MarketConfig(
            num_agents=3, price_sensitivity="medium", visibility="perfect",
            single_provider=base_provider, max_rounds=max_rounds, replicate=rep, **kw,
        )

    prev = {}
    if resume and os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        prev = {r["intervention"]: r for r in old.to_dict("records") if pd.notna(r["calvano_delta"])}

    rows = []
    if "Baseline" in prev:
        base = prev["Baseline"]
        base_index = float(base["calvano_delta"])
        rows.append(with_refusals(base))
    else:
        try:
            base_index, base_log, base_src = await run_or_reuse(rep, "Baseline", cfg_for(), "perfect", "single")
        except RuntimeError as e:
            print(f"Baseline failed: {e}")
            return
        rows.append({
            "replicate": rep, "intervention": "Baseline", "provider": base_provider,
            "calvano_delta": base_index, "delta": 0.0, "source": base_src,
            **refusal_stats(base_log), "log_file": base_log,
        })
    print(f"Baseline I_pi: {base_index:.4f}")

    order = ["Baseline"] + INTERVENTIONS

    def save():
        rows.sort(key=lambda r: order.index(r["intervention"]))
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    for inv in INTERVENTIONS:
        if inv in prev:
            rows.append(with_refusals(prev[inv]))

    sem = asyncio.Semaphore(max(1, parallel))

    async def run_one(inv):
        cfg = cfg_for()
        if inv == "Inject pricing noise":
            cfg.visibility = "noisy"
        elif inv == "Mixed model":
            cfg.model_variance = "mixed"
        elif inv == "Reduce model parameters":
            cfg.model_size = "mid"
        else:
            cfg.active_intervention = inv
        async with sem:
            try:
                if inv == "Inject pricing noise":
                    idx, log, src = await run_or_reuse(rep, inv, cfg, "noisy", "single")
                elif inv == "Mixed model":
                    idx, log, src = await run_or_reuse(rep, inv, cfg, "perfect", "mixed")
                else:
                    cfg.get_model_for_firm("A")
                    log = await LLMPricingSimulation(cfg).run()
                    idx = calculate_collusion_metrics(log, price_sensitivity="medium", num_agents=3)["calvano_delta"]
                    src = "run"
                rows.append({
                    "replicate": rep, "intervention": inv, "provider": cfg.provider_label,
                    "calvano_delta": idx, "delta": round(idx - base_index, 4), "source": src,
                    **refusal_stats(log), "log_file": log,
                })
            except (RuntimeError, ValueError) as e:
                print(f"Intervention '{inv}' failed: {e}")
                rows.append({
                    "replicate": rep, "intervention": inv, "provider": cfg.provider_label,
                    "calvano_delta": float("nan"), "delta": float("nan"), "source": "failed",
                    **NO_REFUSAL_STATS, "log_file": "",
                })
        save()

    tasks = [asyncio.create_task(run_one(inv)) for inv in INTERVENTIONS if inv not in prev]
    try:
        await asyncio.gather(*tasks)
    except QuotaExhaustedError as e:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        save()
        print(f"\nStopped: {e}\n{RESUME_HINT}")
        raise
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2025 Agentic Collusion Experiments")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--interventions", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--replicate", default="1")
    parser.add_argument("--exclude-provider", action="append", default=[], choices=PROVIDERS, metavar="PROVIDER")
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--provider-map", metavar="CSV")
    parser.add_argument("--single-provider", choices=PROVIDERS)
    parser.add_argument("--show-provider-map", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", choices=["single", "mixed"])
    parser.add_argument("--cells", metavar="CSV")
    parser.add_argument("--parallel", type=int, default=1, metavar="N")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--ql-periods", type=int, default=2_500_000, help="Tabular Q-learning periods per run (default 2.5M).")
    parser.add_argument("--ql-runs", type=int, default=30, help="Independent Q-learning seeds per cell (default 30).")
    parser.add_argument("--ppo-steps", type=int, default=2_050_000, help="PPO environment steps per cell (default 2.05M).")
    parser.add_argument("--ppo-runs", type=int, default=5, help="Independent PPO seeds per cell (default 5).")
    parser.add_argument("--ql-only", action="store_true",
                        help="Only run Q-learning seeds at --ql-periods/--ql-runs, appending to ql_runs.csv (e.g. the 6M check).")
    parser.add_argument("--edgeworth-align", action="store_true",
                        help="Addendum Appendix B: event-aligned restorations vs a 200-shuffle permutation benchmark (uses panel_q1_30.csv).")
    parser.add_argument("--align-cells", choices=["clustered", "all"], default="clustered",
                        help="Runs for the event windows: 3-firm sensitive + medium (default) or all grid runs.")
    args = parser.parse_args()

    if args.max_rounds < 30:
        parser.error("--max-rounds must be at least 30 for Q21-Q30 scoring.")

    llm_runner.RESUME_PARTIAL = args.resume

    cell_filter = None
    if args.cells:
        cf = pd.read_csv(args.cells)
        cell_filter = {(int(x.num_agents), x.price_sensitivity, x.visibility, x.model_variance) for x in cf.itertuples()}
    if args.only:
        allc = {(n, s, v, m) for n in GRID_FIRMS for s in GRID_SENSITIVITIES for v in GRID_VISIBILITIES for m in [args.only]}
        cell_filter = allc if cell_filter is None else (cell_filter & allc)

    set_providers(args.exclude_provider)
    pmap = resolve_provider_map(args)

    if args.show_provider_map:
        print(f"{'firms':>5}  {'sensitivity':11}  {'visibility':10}  provider   model")
        for (n, s, v), p in sorted(
            pmap.items(),
            key=lambda kv: (kv[0][0], GRID_SENSITIVITIES.index(kv[0][1]), GRID_VISIBILITIES.index(kv[0][2])),
        ):
            print(f"{n:>5}  {s:11}  {v:10}  {p:9}  {model_for(p)}")
        sys.exit(0)

    if not any([args.smoke, args.grid, args.interventions, args.all, args.ql_only, args.edgeworth_align]):
        print("Specify --smoke, --grid, --interventions, --all, --ql-only, or --edgeworth-align")
        sys.exit(0)

    try:
        if args.smoke:
            asyncio.run(run_smoke_test(args.replicate, args.max_rounds))
        if args.ql_only:
            run_ql_only(args.replicate, args.ql_periods, args.ql_runs)
        if args.grid or args.all:
            asyncio.run(run_factorial_grid(
                args.replicate, pmap, args.max_rounds, skip_rl=args.skip_rl,
                ql_periods=args.ql_periods, ql_runs=args.ql_runs,
                ppo_steps=args.ppo_steps, ppo_runs=args.ppo_runs,
                resume=args.resume, parallel=args.parallel, cell_filter=cell_filter,
            ))
        if args.interventions or args.all:
            asyncio.run(run_report_fig6_interventions(
                args.replicate, pmap, args.max_rounds, resume=args.resume, parallel=args.parallel,
            ))
        if args.grid or args.interventions or args.all:
            panel_path = build_panel(args.replicate)
            ready, why = report_ready(panel_path)
            if ready:
                build_report_assets(
                    panel_path,
                    ql_path=f"logs/rep{args.replicate}/ql_runs.csv",
                    ppo_path=f"logs/rep{args.replicate}/ppo_runs.csv",
                    align_cells=args.align_cells,
                )
            else:
                print(f"Report assets not built yet ({why}). Run export_report_assets.py once the panel is complete.")
        if args.edgeworth_align:
            panel_path = f"logs/rep{args.replicate}/panel_q1_30.csv"
            if not os.path.exists(panel_path):
                panel_path = build_panel(args.replicate)
            appendix_b_event_aligned(panel_path, cells=args.align_cells)
    except QuotaExhaustedError:
        sys.exit(2)
