"""
export_report_assets.py - The single plotting script. Builds every Report and Addendum figure, the
Appendix B event study and the Word-ready CSV table cuts from the consolidated panel
(logs/rep<N>/panel_q1_30.csv, written by build_panel.py). All benchmarks come from benchmarks.py.

Every figure comes in two versions: *_labelled (titles, axis and tick labels, legends, values) and
*_notext (PNG + SVG with no text, tick marks kept, plot filling the frame, sized for the Word page)
to label by hand. One exception: the Figure 2 heatmap keeps its cell values (and N/A cells) in the
_notext version. Every other value label (Figure 6 bars, Appendix B counts) is left out; the numbers
are in the table cuts.

Outputs (in ./report_assets/ unless --out-dir is given):
  Tables       cut_table7a_profit_and_I_pi.csv, cut_table7b_P_paid_and_M_paid.csv,
               cut_table7c_P_post_and_D_wedge.csv, cut_table9_interventions.csv,
               cut_table11_benchmarks.csv, cut_appendix_b_restorations.csv
  Refusals     cut_refusals.csv: every run with a firm-quarter where the model refused and the previous
               price was held (panel 'refused' column), with counts for Q1-Q30 and the scored Q21-Q30.
               A warning names any grid cell or intervention whose scored quarters include held prices.
  Figure 2     report_fig2_heatmap_labelled.png, report_fig2_heatmap_notext.png/.svg
  Figure 3     report_fig3_dumbbells_labelled.png (3 panels),
               report_fig3_<sensitive|medium|insensitive>_notext.png/.svg (one per market)
  Figure 4     report_fig4_edgeworth_labelled.png (2x2),
               report_fig4_row1_AB_notext.png/.svg, report_fig4_row2_CD_notext.png/.svg (one per row)
  Figure 6     report_fig6_interventions_labelled.png, report_fig6_interventions_notext.png/.svg
  Addendum 2   addendum_fig2_grouped_bars_labelled.png, addendum_fig2_grouped_bars_notext.png/.svg
               (Q-learning / PPO bars when ql_runs.csv / ppo_runs.csv exist)
  Appendix B   appendix_b_edgeworth_aligned_labelled.png/.svg, appendix_b_edgeworth_aligned_notext.png/.svg,
               appendix_b_event_windows.csv, appendix_b_permutation_medians.csv

Table 9 and Figure 6 report Delta I_pi as the difference of the two-decimal indices printed in Table 9,
so every row adds up as shown. Their rows and bars follow IV_ORDER, a fixed order that does not re-sort.

A restoration is a quarter in which the market's lowest price rises by at least JUMP (5 cents) on the
previous quarter. Figure 4, the Appendix B cut and Appendix B Panel B count them from Q4.
Appendix B Panel A centres a window (t - PRE to t + POST) on every restoration at t = EVENT_START..30 in
the selected grid runs and plots the lowest price relative to t - 1.
  Gain = p_min(t + 1) - p_min(t - 1): how much of the jump is still standing one quarter later.
  Permutation benchmark: within each run, the quarter-to-quarter changes in p_min over Q(EVENT_START)-Q30
  are shuffled N_SHUFFLES times (earlier quarters kept as observed); restorations, gains and windows are
  recomputed on every shuffled path.

Usage:
  python3 export_report_assets.py [panel.csv] [ql_runs.csv] [--ppo ppo_runs.csv] [--out-dir DIR]
                                  [--align-cells clustered|all] [--appendix-b-only]
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import get_benchmark, calvano_delta, table11_rows, START_PRICE, BR_TURNS, BR_KEEP

NAVY, SLATE, CORAL, TEAL = "#1B365D", "#4A6FA5", "#D95F5F", "#007EA7"
AMBER, BENCH_COL, STEM_COL, AXIS_COL = "#E88D67", "#2E6B3E", "#DCE2EA", "#555555"
GREY, RL_EDGE, PPO_FILL = "#C9D1DB", "#4A5859", "#9DB4C0"

ROW_ORD = [(m, n) for m in ("Sensitive", "Medium", "Insensitive") for n in (3, 5, 7)]
COL_ORD = ["Perfect | Single", "Perfect | Mixed", "Noisy | Single", "Noisy | Mixed"]
CELL_ORD = [(n, d) for n in (3, 5, 7) for d in COL_ORD]

# Table 9 rows and Figure 6 bars, top to bottom (fixed on purpose).
IV_ORDER = [
    ("baseline", "Baseline"), ("refine_goal", "Refine goal"), ("inject_pricing_noise", "Inject pricing noise"),
    ("reduce_model_params", "Reduce model parameters"), ("weaken_longterm_language", "Weaken long-term language"),
    ("inject_marketshare_noise", "Inject market share noise"), ("mixed_model", "Mixed model"),
    ("reduce_precision", "Reduce precision"), ("reduce_frequency", "Reduce frequency"),
]
# Interventions that reuse a 3-firm medium grid cell rather than a run of their own.
GRID_REUSE = {"inject_pricing_noise": "Noisy | Single", "mixed_model": "Perfect | Mixed"}

JUMP = 0.05
COUNT_FROM = 4
EVENT_START = 8
PRE, POST = 4, 6
OFFSETS = list(range(-PRE, POST + 1))
N_SHUFFLES = 200
SEED = 0

NUDE_W = 12.62 / 2.54   # width (inches) of the _notext figures: the Word text width the Figure 3 panels use

QL_FIG_PERIODS = int(os.environ.get("QL_FIG_PERIODS", "2500000"))
PPO_FIG_STEPS = int(os.environ.get("PPO_FIG_STEPS", "2050000"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def nz(val: float, dec: int = 2) -> float:
    r = round(float(val), dec)
    return 0.0 if abs(r) == 0 else r


def fmt_signed(val: float, dec: int = 2, suffix: str = "") -> str:
    v = nz(val, dec)
    if v == 0.0:
        return f"{0.0:.{dec}f}{suffix}"
    return f"{v:+.{dec}f}{suffix}"


def _latest(patterns):
    cands = sorted(f for p in patterns for f in glob.glob(p))
    return cands[-1] if cands else None


def default_panel() -> str:
    path = _latest(["logs/rep*/panel_q1_30.csv", "panel_q1_30.csv"])
    if path is None:
        raise FileNotFoundError("No logs/rep*/panel_q1_30.csv found. Run build_panel.py, or pass a panel path explicitly.")
    return path


def _save(fig, out_dir, stem, exts=("png",), **kw):
    for ext in exts:
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=300 if ext == "png" else None, **kw)


def _nude_axes(ax):
    """_notext styling (as the Figure 3 panels): tick marks stay, every label goes, bottom/left spines only."""
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, labeltop=False, labelright=False,
                   length=3.5, width=0.8, color=AXIS_COL, direction="out")
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True); ax.spines[side].set_color(AXIS_COL); ax.spines[side].set_linewidth(0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def min_price_paths(panel: pd.DataFrame) -> dict:
    """Lowest posted price per quarter, one Series per run."""
    return {run: g.groupby("quarter")["price"].min().sort_index() for run, g in panel.groupby("run")}


def restoration_quarters(p_min: pd.Series, start: int = COUNT_FROM) -> list:
    d = p_min.diff()
    return [int(q) for q in d.index if q >= start and d.loc[q] >= JUMP - 1e-9]


def refused_flags(df: pd.DataFrame) -> pd.Series:
    """True where the model refused and the previous price was held (panel 'refused' column)."""
    if "refused" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["refused"].astype(str).str.strip().str.lower() == "true"


def check_refused_column(df: pd.DataFrame) -> bool:
    if "refused" in df.columns:
        return True
    print("WARNING: the panel has no 'refused' column, so held prices after a refusal cannot be identified. "
          "Rebuild it with build_panel.py.")
    return False


def run_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Q21-Q30 metrics, one row per run."""
    late = df[df["quarter"].between(21, 30)].copy()
    late["rev"] = late["units"] * late["price"]
    late["held"] = refused_flags(late).astype(int)
    keys = ["run", "firms", "beta", "sensitivity", "visibility", "model_mix", "intervention", "is_grid_cell"]
    pq = late.groupby(keys + ["quarter"], as_index=False).agg(
        jp=("joint_profit_q", "first"), p_post=("price", "mean"), units=("units", "sum"), rev=("rev", "sum"),
        held=("held", "sum"),
    )
    s = pq.groupby(keys, as_index=False).agg(
        mean_jp=("jp", "mean"), p_post=("p_post", "mean"), tot_u=("units", "sum"), tot_rev=("rev", "sum"),
        refusals_q21_30=("held", "sum"),
    )
    s["is_grid_cell"] = s["is_grid_cell"].astype(bool)
    s["p_paid"] = np.where(s["tot_u"] > 0, s["tot_rev"] / s["tot_u"].where(s["tot_u"] > 0, 1), s["p_post"])
    s["wedge_c"] = (s["p_post"] - s["p_paid"]) * 100.0
    s["I_pi"] = [calvano_delta(float(r.mean_jp), r.sensitivity, int(r.firms)) for r in s.itertuples()]
    s["M_paid_pct"] = [
        ((r.p_paid / get_benchmark(r.sensitivity, int(r.firms))["comp_paid"]) - 1.0) * 100.0
        for r in s.itertuples()
    ]
    s["design"] = s["visibility"] + " | " + s["model_mix"]
    return s


def report_ready(panel_path: str):
    """(True, "") when the panel holds all 36 grid cells and every Table 9 intervention."""
    df = pd.read_csv(panel_path, usecols=["is_grid_cell", "intervention", "firms", "sensitivity", "visibility", "model_mix"])
    grid = df[df["is_grid_cell"].astype(bool)]
    n_cells = grid.groupby(["firms", "sensitivity", "visibility", "model_mix"]).ngroups
    if n_cells < 36:
        return False, f"{n_cells} of 36 grid cells in the panel"
    have = set(df["intervention"])
    missing = [label for key, label in IV_ORDER if key not in have and key not in GRID_REUSE]
    if missing:
        return False, "missing interventions: " + ", ".join(missing)
    return True, ""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_grid_cuts(grid: pd.DataFrame, out_dir: str):
    def save_cut(fmt_fn, filename):
        g = grid.copy()
        g["_val"] = [fmt_fn(r) for r in g.itertuples()]
        piv = g.pivot(index=["sensitivity", "firms"], columns="design", values="_val").reindex(index=ROW_ORD, columns=COL_ORD)
        piv.to_csv(os.path.join(out_dir, filename))

    save_cut(lambda r: f"{r.mean_jp:.2f} ({'N/A' if pd.isna(r.I_pi) else f'{nz(r.I_pi, 2):.2f}'})", "cut_table7a_profit_and_I_pi.csv")
    save_cut(lambda r: f"{r.p_paid:.2f} ({fmt_signed(r.M_paid_pct, 1, '%')})", "cut_table7b_P_paid_and_M_paid.csv")
    save_cut(lambda r: f"{r.p_post:.2f} ({fmt_signed(r.wedge_c, 1, 'c')})", "cut_table7c_P_post_and_D_wedge.csv")


def table9(s: pd.DataFrame, grid: pd.DataFrame):
    """Table 9 rows in IV_ORDER. Delta_I_pi is the difference of the printed two-decimal indices."""
    base_ipi = nz(s.loc[s["intervention"] == "baseline", "I_pi"].iloc[0], 2)
    rows = []
    for key, label in IV_ORDER:
        sub = s[s["intervention"] == key]
        if not sub.empty:
            r = sub.iloc[0]
        elif key in GRID_REUSE:
            r = grid[(grid["firms"] == 3) & (grid["sensitivity"] == "Medium") & (grid["design"] == GRID_REUSE[key])].iloc[0]
        else:
            raise KeyError(f"Intervention '{key}' not found in the panel")
        ipi = nz(r["I_pi"], 2)
        rows.append({
            "Intervention": label,
            "Mean_Profit": nz(r["mean_jp"], 2),
            "I_pi": ipi,
            "Delta_I_pi": nz(ipi - base_ipi, 2),
            "P_paid_M_paid": f"{r['p_paid']:.2f} ({fmt_signed(r['M_paid_pct'], 1, '%')})",
            "P_post_D": f"{r['p_post']:.2f} ({fmt_signed(r['wedge_c'], 1, 'c')})",
        })
    return pd.DataFrame(rows), base_ipi


def write_restoration_cut(panel_grid: pd.DataFrame, out_dir: str):
    meta = panel_grid.groupby("run")[["firms", "sensitivity", "visibility", "model_mix"]].first()
    meta["restorations"] = pd.Series({run: len(restoration_quarters(p)) for run, p in min_price_paths(panel_grid).items()})
    cut = meta.groupby(["firms", "sensitivity", "visibility", "model_mix"], as_index=False)["restorations"].sum()
    cut.to_csv(os.path.join(out_dir, "cut_appendix_b_restorations.csv"), index=False)


def write_refusal_cut(df: pd.DataFrame, out_dir: str):
    """cut_refusals.csv: one row per run with any firm-quarter where a refusal held the previous price.
    The file is written (header only) even when there are none, so its presence shows the check ran."""
    cols = ["run", "intervention", "cell_id", "refusals_q1_30", "refusals_q21_30", "first_quarter", "firms_held", "models"]
    path = os.path.join(out_dir, "cut_refusals.csv")
    if not check_refused_column(df):
        return
    held = df[refused_flags(df)]
    rows = []
    for run, g in held.groupby("run", sort=True):
        rows.append({
            "run": run,
            "intervention": g["intervention"].iloc[0],
            "cell_id": g["cell_id"].iloc[0],
            "refusals_q1_30": len(g),
            "refusals_q21_30": int(g["quarter"].between(21, 30).sum()),
            "first_quarter": int(g["quarter"].min()),
            "firms_held": "; ".join(sorted(g["firm"].astype(str).unique())),
            "models": "; ".join(sorted(g["model"].astype(str).unique())),
        })
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    if rows:
        scored = [r for r in rows if r["refusals_q21_30"] > 0]
        print(f"WARNING: {len(rows)} runs contain held prices after refusals; {len(scored)} of them inside Q21-Q30 "
              f"(their tables and figures include those quarters). Details: {path}")
        for r in scored:
            print(f"  {r['intervention']:>24s}  {r['cell_id']}: {r['refusals_q21_30']} held firm-quarters in Q21-Q30")
    else:
        print("Refusals: none in the panel (no held prices).")


# ---------------------------------------------------------------------------
# Report Figure 2: heatmap of I_pi
# ---------------------------------------------------------------------------
def fig2_heatmap(grid: pd.DataFrame, out_dir: str):
    vals = grid.pivot(index=["sensitivity", "firms"], columns="design", values="I_pi").reindex(index=ROW_ORD, columns=COL_ORD).values.astype(float)
    pos_cmap = LinearSegmentedColormap.from_list("pos_red", ["#FFF5F5", CORAL])
    neg_cmap = LinearSegmentedColormap.from_list("neg_blue", ["#F5F8FB", "#C6DAEE"])

    def draw_cells(ax):
        """Cells, cell values and divider lines: shared by both versions."""
        for r in range(9):
            for c in range(4):
                v = vals[r, c]
                if r >= 6 or pd.isna(v):
                    bg, txt, tc = "#F1F3F6", "N/A", "#777777"
                else:
                    v0 = nz(v, 2)
                    bg = pos_cmap(min(1.0, v0 / 0.60)) if v0 > 0 else neg_cmap(min(1.0, abs(v0) / 1.35))
                    txt, tc = ("0.00" if v0 == 0 else f"{v0:+.2f}"), "#111111"
                ax.add_patch(plt.Rectangle((c, 8 - r), 1, 1, facecolor=bg, edgecolor="white", linewidth=2.2))
                ax.text(c + 0.5, 8 - r + 0.5, txt, ha="center", va="center", fontsize=10, weight="bold", color=tc)
        ax.axhline(6, color=NAVY, lw=1.8); ax.axhline(3, color=NAVY, lw=1.8); ax.axvline(2, color=NAVY, lw=1.8)
        ax.axis("off")

    fig, ax = plt.subplots(figsize=(NUDE_W, 12.2 / 2.54), dpi=300)
    draw_cells(ax)
    ax.set_xlim(-0.03, 4.03); ax.set_ylim(-0.03, 9.03)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save(fig, out_dir, "report_fig2_heatmap_notext", exts=("png", "svg"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=300)
    draw_cells(ax)
    ax.set_xlim(-1.45, 4.0); ax.set_ylim(-1.15, 9.0)
    for r, (_, fn) in enumerate(ROW_ORD):
        ax.text(-0.22, 8 - r + 0.5, f"{fn}", ha="center", va="center", fontsize=10)
    for im, ml in enumerate(["Sensitive", "Medium", "Insensitive"]):
        ax.text(-0.92, 7.5 - im * 3.0, ml, ha="center", va="center", fontsize=10, weight="bold", color=NAVY)
    for c, sl in enumerate(["Single", "Mixed", "Single", "Mixed"]):
        ax.text(c + 0.5, -0.30, sl, ha="center", va="center", fontsize=9.5)
    ax.text(1.0, -0.72, "Perfect", ha="center", va="center", fontsize=10, weight="bold", color=NAVY)
    ax.text(3.0, -0.72, "Noisy", ha="center", va="center", fontsize=10, weight="bold", color=NAVY)
    ax.set_title("Index of Excess Profit (I_pi, Q21-30)", fontsize=10.5, color=NAVY, weight="bold", pad=12)
    plt.tight_layout()
    _save(fig, out_dir, "report_fig2_heatmap_labelled")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report Figure 3: advertised vs buyer-paid dumbbells
# (12.62cm x 6.91cm no-text SVG & PNG per market, plus one labelled 3-panel PNG)
# ---------------------------------------------------------------------------
def _draw_dumbbells(ax, sub_m, sens_lbl):
    for iy, (n, d) in enumerate(CELL_ORD):
        rv = sub_m.loc[(n, d)]
        ax.plot([rv["p_paid"], rv["p_post"]], [iy, iy], color=STEM_COL, lw=2.6, solid_capstyle="round", zorder=1)
        ax.scatter(rv["p_paid"], iy, color=TEAL, marker="o", s=60, zorder=3)
        ax.scatter(rv["p_post"], iy, color=AMBER, marker="D", s=28, zorder=4)
    for blk_i, n in enumerate((3, 5, 7)):
        cp_v = get_benchmark(sens_lbl, n)["comp_paid"]
        ax.plot([cp_v, cp_v], [blk_i * 4 - 0.4, blk_i * 4 + 3.4], color=BENCH_COL, ls="--", lw=1.2, zorder=2)
    ax.axhline(3.5, color="#D8DDE6", lw=0.9, zorder=0); ax.axhline(7.5, color="#D8DDE6", lw=0.9, zorder=0)
    ax.set_xlim(0.98, 1.55); ax.set_ylim(-0.6, 11.6); ax.invert_yaxis()
    ax.grid(axis="x", color="#F4F6F8", lw=0.7, zorder=0)
    ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]); ax.set_yticks(range(12))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def fig3_dumbbells(grid: pd.DataFrame, out_dir: str):
    markets = [("sensitive", "A. Price-sensitive (beta = 1.0)"), ("medium", "B. Medium sensitivity (beta = 0.5)"),
               ("insensitive", "C. Price-insensitive (beta = 0.0)")]
    for sens_lbl, _ in markets:
        sub_m = grid[grid["sensitivity"].str.lower() == sens_lbl].set_index(["firms", "design"])
        fig, ax = plt.subplots(figsize=(12.62 / 2.54, 6.91 / 2.54), dpi=300)
        _draw_dumbbells(ax, sub_m, sens_lbl)
        ax.tick_params(axis="both", which="major", bottom=True, left=True, labelbottom=False, labelleft=False, length=3.5, width=0.8, color=AXIS_COL, direction="out")
        for side in ("bottom", "left"):
            ax.spines[side].set_visible(True); ax.spines[side].set_color(AXIS_COL); ax.spines[side].set_linewidth(0.8)
        plt.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.045)
        _save(fig, out_dir, f"report_fig3_{sens_lbl}_notext", exts=("png", "svg"))
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 9.0), sharex=True, dpi=300)
    for ax, (sens_lbl, title) in zip(axes, markets):
        _draw_dumbbells(ax, grid[grid["sensitivity"].str.lower() == sens_lbl].set_index(["firms", "design"]), sens_lbl)
        ax.set_yticklabels([f"{n}f  {d}" for n, d in CELL_ORD], fontsize=8.5)
        ax.set_xticklabels(["$1.00", "$1.10", "$1.20", "$1.30", "$1.40", "$1.50"], fontsize=9)
        ax.set_title(title, fontsize=10, color=NAVY, weight="bold", loc="left")
    axes[2].set_xlabel("Price ($)", fontsize=10)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAL, markersize=8, label="Buyer-paid price (P_paid)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=AMBER, markersize=6, label="Mean advertised price (P_post)"),
        Line2D([0], [0], color=BENCH_COL, ls="--", lw=1.2, label="Competitive paid benchmark (C_paid)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.97), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Advertised vs. buyer-paid prices (Q21-30)", fontsize=11, color=NAVY, weight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_dir, "report_fig3_dumbbells_labelled")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report Figure 4: Edgeworth cycles in the 3-firm price-sensitive cells
# (no-text rows for Word, plus a labelled 2x2)
# ---------------------------------------------------------------------------
FIG4_PANELS = [
    ("A", "Perfect", "Single", "A. Perfect visibility, single model"),
    ("B", "Perfect", "Mixed",  "B. Perfect visibility, mixed models"),
    ("C", "Noisy",   "Single", "C. Noisy visibility, single model"),
    ("D", "Noisy",   "Mixed",  "D. Noisy visibility, mixed models"),
]


def _draw_edgeworth(ax, sub, show_y_labels=False, show_x_labels=False, title=None):
    p_min = sub.groupby("quarter")["price"].min().sort_index()
    resets = p_min.loc[restoration_quarters(p_min)]
    for _, fsub in sub.groupby("firm"):
        fsub = fsub.sort_values("quarter")
        ax.plot(fsub["quarter"], fsub["price"], color=STEM_COL, lw=1.2, zorder=1)
    b = get_benchmark("high", 3)
    ax.axhline(b["comp_post"], color=BENCH_COL, ls=(0, (6, 4)), lw=1.3, zorder=2)
    ax.axhline(b["mono_price"], color=BENCH_COL, ls=(0, (6, 4)), lw=1.3, zorder=2)
    ax.plot(p_min.index, p_min.values, color=TEAL, lw=2.0, zorder=3)
    ax.scatter(resets.index, resets.values, color=AMBER, marker="D", s=38, zorder=4)
    ax.set_xlim(0.2, 30.8); ax.set_ylim(0.97, 2.03)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_yticks([1.00, 1.25, 1.50, 1.75, 2.00])
    # The panels share one y-axis formatter (sharey=True): give every panel the same labels and
    # hide them per panel with tick_params, or the last panel's labels replace everyone's.
    ax.set_yticklabels(["$1.00", "", "$1.50", "", "$2.00"], fontsize=9.5)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, color="#333333",
                   labelbottom=show_x_labels, labelleft=show_y_labels)
    ax.grid(axis="x", color="#E6E9EF", lw=0.7, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8); ax.spines["bottom"].set_linewidth(0.8)
    if title:
        ax.set_title(f"{title}: {len(resets)} cycles", fontsize=9.5, color=NAVY, weight="bold", loc="left")
    return len(resets)


def fig4_edgeworth(panel_grid: pd.DataFrame, out_dir: str) -> dict:
    s3 = panel_grid[(panel_grid["firms"] == 3) & (panel_grid["sensitivity"] == "Sensitive")]

    def cell(vis, mix):
        return s3[(s3["visibility"] == vis) & (s3["model_mix"] == mix)]

    for row_tag, pair in [("row1_AB", FIG4_PANELS[:2]), ("row2_CD", FIG4_PANELS[2:])]:
        fig, axes = plt.subplots(1, 2, figsize=(15.5 / 2.54, 4.2 / 2.54), dpi=300, sharey=True)
        for ax, (_, vis, mix, _) in zip(axes, pair):
            _draw_edgeworth(ax, cell(vis, mix))
        plt.tight_layout(w_pad=1.0)
        _save(fig, out_dir, f"report_fig4_{row_tag}_notext", exts=("png", "svg"))
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.2), dpi=300, sharex=True, sharey=True)
    counts = {}
    for i, (ax, (tag, vis, mix, title)) in enumerate(zip(axes.flat, FIG4_PANELS)):
        counts[tag] = _draw_edgeworth(ax, cell(vis, mix), show_y_labels=(i % 2 == 0), show_x_labels=(i >= 2), title=title)
    fig.supxlabel("Quarter (Q1-Q30)", fontsize=10); fig.supylabel("Price ($)", fontsize=10)
    plt.tight_layout()
    _save(fig, out_dir, "report_fig4_edgeworth_labelled")
    plt.close(fig)
    return counts


# ---------------------------------------------------------------------------
# Report Figure 6: interventions (bars in IV_ORDER, values from Table 9)
# ---------------------------------------------------------------------------
def fig6_interventions(t9: pd.DataFrame, base_ipi: float, out_dir: str):
    iv_plot = t9[t9["Intervention"] != "Baseline"]
    vals = [nz(v, 2) for v in iv_plot["Delta_I_pi"].values]
    pad = max(0.08, 0.15 * (max(vals) - min(vals)))   # room for the value labels; 0.08 at the report's range
    xlim = (min(-0.08, min(vals) - pad), max(0.08, max(vals) + pad))
    for labels in (True, False):
        fig, ax = plt.subplots(figsize=(8.0, 4.6) if labels else (NUDE_W, 8.6 / 2.54), dpi=300)
        bars = ax.barh(iv_plot["Intervention"], vals, color=[SLATE if v < 0 else CORAL for v in vals], height=0.6, zorder=2)
        ax.axvline(0, color="#333333", lw=0.9, zorder=3)
        ax.set_xlim(*xlim)
        if labels:
            for bar, v in zip(bars, vals):
                off = -0.012 if v < 0 else 0.012
                lbl = "0.00" if v == 0 else f"{v:+.2f}"
                ax.text(v + off, bar.get_y() + bar.get_height() / 2, lbl, va="center", ha="right" if v < 0 else "left", fontsize=9, weight="bold")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#EEEEEE", lw=0.8, zorder=0)
        if labels:
            ax.set_xlabel(f"Change in Index of Excess Profit (I_pi) vs. Baseline ({base_ipi:.2f})", fontsize=9.5)
            plt.tight_layout()
            _save(fig, out_dir, "report_fig6_interventions_labelled")
        else:
            _nude_axes(ax)
            plt.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.05)
            _save(fig, out_dir, "report_fig6_interventions_notext", exts=("png", "svg"))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Addendum Figure 2: RL baselines vs agentic AI (perfect visibility)
# ---------------------------------------------------------------------------
def load_rl_means(path, kind):
    """Mean index per (market, firms) from ql_runs.csv ("ql") or ppo_runs.csv ("ppo") at the figure's training length."""
    if kind == "ql":
        col, length_col, length, name, unit, fname = "ql_index", "periods", QL_FIG_PERIODS, "Q-learning", "periods", "ql_runs.csv"
    else:
        col, length_col, length, name, unit, fname = "rl_index", "env_steps", PPO_FIG_STEPS, "PPO", "steps", "ppo_runs.csv"
    if not path or not os.path.exists(path):
        print(f"WARNING: no {fname} found - Addendum Figure 2 is drawn WITHOUT {name} bars.")
        return None, None
    q = pd.read_csv(path)
    if length_col not in q.columns:
        print(f"WARNING: {path} has no '{length_col}' column - {name} bars omitted.")
        return None, None
    q = q[q[length_col] == length]
    if "visibility" in q.columns:
        q = q[q["visibility"] == "perfect"]
    q = q[q[col].notna()]
    if q.empty:
        print(f"WARNING: {path} has no runs at {length:,} {unit} - {name} bars omitted.")
        return None, None
    q = q.assign(mkt=q["price_sensitivity"].str.lower().map({"high": "Sensitive", "sensitive": "Sensitive", "medium": "Medium"}))
    means = {}
    for mkt in ("Sensitive", "Medium"):
        for f in (3, 5, 7):
            sub = q[(q["mkt"] == mkt) & (q["num_agents"] == f)]
            means[(mkt, f)] = float(sub[col].mean()) if len(sub) else float("nan")
    gaps = [k for k, v in means.items() if v != v]
    if gaps:
        print(f"WARNING: no {name} runs for {gaps}; those bars are omitted.")
    seeds = int(q.groupby(["price_sensitivity", "num_agents"])["seed"].nunique().min())
    print(f"Loaded {name} runs from {path} ({length:,} {unit}, at least {seeds} seeds per cell)")
    return means, f"{name} mean ({length / 1e6:g}M {unit}, {seeds} seeds)"


def addendum_fig2(grid: pd.DataFrame, ql_path, ppo_path, out_dir: str):
    perf = grid[(grid["visibility"] == "Perfect") & (grid["sensitivity"].isin(["Sensitive", "Medium"]))].set_index(["sensitivity", "firms", "model_mix"])

    def llm(mix):
        return {(m, f): float(perf.loc[(m, f, mix), "I_pi"]) if (m, f, mix) in perf.index else float("nan")
                for m in ("Sensitive", "Medium") for f in (3, 5, 7)}

    series = []
    for kind, path, fill in (("ql", ql_path, STEM_COL), ("ppo", ppo_path, PPO_FILL)):
        means, label = load_rl_means(path, kind)
        if means:
            series.append((label, means, {"color": fill, "edgecolor": RL_EDGE, "lw": 0.8}))
    series += [("Agentic AI: Single (30 quarters)", llm("Single"), {"color": TEAL}),
               ("Agentic AI: Mixed (30 quarters)", llm("Mixed"), {"color": AMBER})]

    k = len(series)
    bar_h = 0.24 if k <= 3 else 0.2
    offsets = [((k - 1) / 2 - j) * bar_h for j in range(k)]
    all_vals = [v for _, vals, _ in series for v in vals.values() if v == v]
    xmin, xmax = min(-0.45, min(all_vals) - 0.08), max(0.65, max(all_vals) + 0.08)

    y_pos = np.arange(3)[::-1]

    def draw(axes, labels):
        for ax, mkt, title in zip(axes, ("Sensitive", "Medium"), ("Price-sensitive buyers", "Medium-sensitivity buyers")):
            ax.axvline(0, color="#333333", lw=0.8, zorder=2)
            if labels:
                ax.set_title(title, fontsize=9.5, fontweight="bold", pad=10, loc="left")
            for y, f in zip(y_pos, (3, 5, 7)):
                for off, (_, vals, style) in zip(offsets, series):
                    v = vals[(mkt, f)]
                    if v == v:
                        ax.barh(y + off, v, height=bar_h * 0.88, zorder=3, **style)
            ax.set_yticks(y_pos)
            if labels:
                ax.set_yticklabels(["3 firms", "5 firms", "7 firms"], fontsize=9)
            ax.set_xlim(xmin, xmax)
            if labels:
                ax.set_xlabel("Index of excess profit", fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(axis="x", color="#EEEEEE", lw=0.6, zorder=0)
            if not labels:
                _nude_axes(ax)

    fig, axes = plt.subplots(1, 2, figsize=(12.62 / 2.54, 6.0 / 2.54), dpi=300, sharey=True)
    draw(axes, labels=True)
    legend = [Patch(facecolor=style["color"], edgecolor=style.get("edgecolor"), label=label) for label, _, style in series]
    ncol = k if k <= 3 else 2
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.08 if ncol == k else 1.16), ncol=ncol, frameon=False, fontsize=8.5)
    plt.tight_layout()
    _save(fig, out_dir, "addendum_fig2_grouped_bars_labelled", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(NUDE_W, 5.0 / 2.54), dpi=300, sharey=True)
    draw(axes, labels=False)
    plt.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.07, wspace=0.08)
    _save(fig, out_dir, "addendum_fig2_grouped_bars_notext", exts=("png", "svg"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Addendum Appendix B: event-aligned restorations with a permutation benchmark
# ---------------------------------------------------------------------------
def _windows(p, ts):
    out = [[p.get(t + k, np.nan) - p.get(t - 1, np.nan) for k in OFFSETS] for t in ts]
    return np.array(out, dtype=float).reshape(-1, len(OFFSETS))


def _gains(d):
    idx = np.flatnonzero(d[:-1] >= JUMP - 1e-9)
    return d[idx] + d[idx + 1]


def appendix_b_event_aligned(panel_path=None, cells="clustered", out_dir="report_assets"):
    panel_path = panel_path or default_panel()
    os.makedirs(out_dir, exist_ok=True)
    print(f"Appendix B event-aligned restorations from: {panel_path}")

    df = pd.read_csv(panel_path)
    grid = df[df["is_grid_cell"].astype(bool)].copy()
    grid["design"] = grid["visibility"] + " | " + grid["model_mix"]
    if cells == "clustered":
        sel = grid[(grid["firms"] == 3) & grid["sensitivity"].isin(["Sensitive", "Medium"])]
    elif cells == "all":
        sel = grid
    else:
        raise ValueError("cells must be 'clustered' or 'all'")

    meta = grid.groupby("run")[["firms", "sensitivity", "design"]].first()
    meta["restorations"] = pd.Series({run: len(restoration_quarters(p)) for run, p in min_price_paths(grid).items()})
    count_grid = (meta.pivot_table(index=["sensitivity", "firms"], columns="design", values="restorations", aggfunc="sum")
                  .reindex(index=pd.MultiIndex.from_tuples(ROW_ORD), columns=COL_ORD))

    paths = min_price_paths(sel)
    events, obs_w, obs_g = [], [], []
    for run, p in paths.items():
        ts = restoration_quarters(p, EVENT_START)
        if ts:
            w = _windows(p, ts)
            obs_w.append(w)
            for t, row in zip(ts, w):
                events.append({"run": run, "quarter": t, "gain": round(p.get(t + 1, np.nan) - p.get(t - 1, np.nan), 4),
                               **{f"w{k:+d}": round(v, 4) for k, v in zip(OFFSETS, row)}})
        obs_g += list(_gains(p.diff().loc[EVENT_START:].to_numpy()))
    pd.DataFrame(events).to_csv(os.path.join(out_dir, "appendix_b_event_windows.csv"), index=False)
    obs_w = np.vstack(obs_w) if obs_w else np.empty((0, len(OFFSETS)))

    rng = np.random.default_rng(SEED)
    perm_medians, perm_paths = [], []
    for _ in range(N_SHUFFLES):
        g_all, w_all = [], []
        for run, p in paths.items():
            d = p.diff().loc[EVENT_START:].to_numpy()
            d_sh = rng.permutation(d)
            p_sh = p.copy()
            p_sh.loc[EVENT_START:] = p.loc[EVENT_START - 1] + np.cumsum(d_sh)
            g_all += list(_gains(d_sh))
            ts = [EVENT_START + int(i) for i in np.flatnonzero(d_sh >= JUMP - 1e-9)]
            if ts:
                w_all.append(_windows(p_sh, ts))
        perm_medians.append(float(np.median(g_all)) if g_all else np.nan)
        perm_paths.append(np.nanmedian(np.vstack(w_all), axis=0) if w_all else np.full(len(OFFSETS), np.nan))
    perm_medians = np.array(perm_medians)
    perm_paths = np.vstack(perm_paths)
    pd.DataFrame({"shuffle": range(N_SHUFFLES), "median_gain": perm_medians}).to_csv(
        os.path.join(out_dir, "appendix_b_permutation_medians.csv"), index=False)

    obs_med = float(np.median(obs_g)) if obs_g else np.nan
    summary = {
        "runs": len(paths),
        "restorations_in_windows": int(obs_w.shape[0]),
        "median_gain_cents": round(100 * obs_med, 1),
        "permutation_median_gain_cents": round(100 * float(np.nanmedian(perm_medians)), 1),
        "permutation_5_95_cents": tuple(float(v) for v in np.round(100 * np.nanpercentile(perm_medians, [5, 95]), 1)) if obs_g else (np.nan, np.nan),
        "share_of_shuffles_at_or_below_observed": round(float(np.mean(perm_medians <= obs_med)), 3) if obs_g else np.nan,
    }

    vals = count_grid.to_numpy(dtype=float)
    vmax = max(1.0, np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    if not obs_w.shape[0]:
        print("WARNING: no restorations in the selected runs; Panel A left empty.")

    def draw(ax, axb, labels):
        if obs_w.shape[0]:
            for row in obs_w:
                ax.plot(OFFSETS, 100 * row, color=GREY, lw=0.6, alpha=0.6, zorder=1)
            q25, q50, q75 = (100 * np.nanpercentile(obs_w, q, axis=0) for q in (25, 50, 75))
            ax.fill_between(OFFSETS, q25, q75, color=TEAL, alpha=0.18, lw=0, zorder=2)
            ax.plot(OFFSETS, q50, color=TEAL, lw=2.2, zorder=4, label=f"Observed median ({obs_w.shape[0]} restorations)")
            p5, p50, p95 = (100 * np.nanpercentile(perm_paths, q, axis=0) for q in (5, 50, 95))
            ax.fill_between(OFFSETS, p5, p95, color=AMBER, alpha=0.15, lw=0, zorder=2)
            ax.plot(OFFSETS, p50, color=AMBER, lw=1.8, ls="--", zorder=3, label=f"Permutation median ({N_SHUFFLES} shuffles)")
            if labels:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
        elif labels:
            ax.text(0.5, 0.5, "No restorations in the selected runs", ha="center", va="center", transform=ax.transAxes)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.axhline(0, color="#333333", lw=0.6)
        ax.set_xticks(OFFSETS)
        if labels:
            ax.set_xlabel("Quarters relative to restoration (t = 0)", fontsize=9)
            ax.set_ylabel("Lowest price vs quarter before restoration (cents)", fontsize=9)
            ax.set_title(f"A. Event-aligned restorations (Q{EVENT_START}-Q30, {cells} runs)", fontsize=9.5, color=NAVY, weight="bold", loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        axb.imshow(np.nan_to_num(vals, nan=0.0), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        if labels:
            for r in range(vals.shape[0]):
                for c in range(vals.shape[1]):
                    v = vals[r, c]
                    axb.text(c, r, "-" if np.isnan(v) else f"{int(v)}", ha="center", va="center", fontsize=8.5,
                             color="white" if np.isfinite(v) and v > 0.6 * vmax else "#111111")
        axb.set_xticks(range(len(COL_ORD)))
        if labels:
            axb.set_xticklabels([d.replace(" | ", "\n") for d in COL_ORD], fontsize=8)
        axb.set_yticks(range(len(ROW_ORD)))
        if labels:
            axb.set_yticklabels([f"{s} {n}f" for s, n in ROW_ORD], fontsize=8)
            axb.set_title(f"B. Restorations per run (Q{COUNT_FROM}-Q30, all grid cells)", fontsize=9.5, color=NAVY, weight="bold", loc="left")
        for y in (2.5, 5.5):
            axb.axhline(y, color="white", lw=2)
        if not labels:
            _nude_axes(ax)
            axb.tick_params(labelbottom=False, labelleft=False, length=3.5, width=0.8, color=AXIS_COL, direction="out")

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=300, gridspec_kw={"width_ratios": [1.35, 1]})
    draw(ax, axb, labels=True)
    plt.tight_layout()
    _save(fig, out_dir, "appendix_b_edgeworth_aligned_labelled", exts=("png", "svg"))
    plt.close(fig)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(NUDE_W, 5.2 / 2.54), dpi=300, gridspec_kw={"width_ratios": [1.35, 1]})
    draw(ax, axb, labels=False)
    plt.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.07, wspace=0.08)
    _save(fig, out_dir, "appendix_b_edgeworth_aligned_notext", exts=("png", "svg"))
    plt.close(fig)

    print("Appendix B summary:", summary)
    print(f"Saved appendix_b_edgeworth_aligned_labelled/_notext (.png/.svg) and appendix_b_*.csv in ./{out_dir}/")
    return summary


# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------
def build_report_assets(panel_path=None, ql_path=None, ppo_path=None, out_dir="report_assets", align_cells="clustered"):
    panel_path = panel_path or default_panel()
    ql_path = ql_path or _latest(["logs/rep*/ql_runs.csv", "ql_runs.csv"])
    ppo_path = ppo_path or _latest(["logs/rep*/ppo_runs.csv", "ppo_runs.csv"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"Building assets in ./{out_dir}/ from: {panel_path}")

    df = pd.read_csv(panel_path)
    panel_grid = df[df["is_grid_cell"].astype(bool)]
    s = run_summary(df)
    grid = s[s["is_grid_cell"]].copy()

    write_grid_cuts(grid, out_dir)
    t9, base_ipi = table9(s, grid)
    t9.to_csv(os.path.join(out_dir, "cut_table9_interventions.csv"), index=False)
    pd.DataFrame(table11_rows()).to_csv(os.path.join(out_dir, "cut_table11_benchmarks.csv"), index=False)
    print(f"Table 11 benchmarks: start ${START_PRICE:.2f}, {BR_TURNS:,} turns, final {BR_KEEP} averaged")
    write_restoration_cut(panel_grid, out_dir)
    write_refusal_cut(df, out_dir)

    fig2_heatmap(grid, out_dir)
    fig3_dumbbells(grid, out_dir)
    print("Edgeworth cycle counts:", fig4_edgeworth(panel_grid, out_dir))
    fig6_interventions(t9, base_ipi, out_dir)
    addendum_fig2(grid, ql_path, ppo_path, out_dir)
    appendix_b_event_aligned(panel_path, cells=align_cells, out_dir=out_dir)
    print(f"All figures & CSV table cuts written to ./{out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build every report figure and table cut from the consolidated panel.")
    ap.add_argument("panel", nargs="?", default=None, help="panel_q1_30.csv (default: latest logs/rep*/panel_q1_30.csv)")
    ap.add_argument("ql_runs", nargs="?", default=None, help="ql_runs.csv (default: latest logs/rep*/ql_runs.csv)")
    ap.add_argument("--ppo", default=None, help="ppo_runs.csv (default: latest logs/rep*/ppo_runs.csv)")
    ap.add_argument("--out-dir", default="report_assets")
    ap.add_argument("--align-cells", choices=["clustered", "all"], default="clustered",
                    help="Appendix B event windows: 3-firm sensitive + medium runs (default) or all grid runs.")
    ap.add_argument("--appendix-b-only", action="store_true", help="Only build the Appendix B figure and CSVs.")
    a = ap.parse_args()
    if a.appendix_b_only:
        appendix_b_event_aligned(a.panel, cells=a.align_cells, out_dir=a.out_dir)
    else:
        build_report_assets(a.panel, a.ql_runs, a.ppo, out_dir=a.out_dir, align_cells=a.align_cells)
