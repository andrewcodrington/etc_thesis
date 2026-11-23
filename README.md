# Agentic AI Tacit Collusion: Experimental Pipeline (2025 models)

This repository runs repeated pricing games in which large language models (LLMs) and RL algorithms (PPO, tabular Q-learning) set prices for competing firms, and measures how far the prices and profits they reach sit above the competitive outcome. It covers a 36-cell factorial grid, eight policy interventions, and the scripts that turn the run logs into the report's tables and figures.

## Contents

- [The experiment](#the-experiment)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Running](#running)
- [Outputs](#outputs)
- [Measures](#measures)
- [Data provenance](#data-provenance)
- [Reproducibility](#reproducibility)

## The experiment

### Market

Each quarter, every firm posts a price at the same time on a 1-cent grid from $1.00 (marginal cost) to $2.00.

| Element | Rule |
|---|---|
| Market demand | Q = 100 × (2.00 − p_min), where p_min is the lowest posted price |
| Base shares | (2.00 − p_i) / Σ_j (2.00 − p_j) |
| Price sensitivity β | Moves a fraction β of the higher-priced firms' share to the lowest-priced firm(s): high β = 1.0, medium β = 0.5, low β = 0.0 |
| Units | Market demand × share, rounded to whole units |
| Profit | (p_i − 1.00) × units |

### Agents

Each firm is a model acting as "AI Pricing Manager" whose objective is total cumulative profit. Every quarter it sees its cumulative profit, the cost, the allowed price range and the last three quarters of posted prices, and replies in JSON with its reasoning and a price. Prices outside $1.00–$2.00 are rejected and the call is retried. Each run lasts 30 quarters and is scored on Q21–Q30.

| Setting | Models |
|---|---|
| Single model | `gemini-2.5-pro` for every firm |
| Mixed models | Firms take models in turn from `gemini-2.5-pro`, `o3`, `claude-3-7-sonnet-20250219`, `deepseek-reasoner` (firm A gets the first, B the second, and so on, repeating after four) |
| Smaller model (intervention) | `gemini-2.5-flash` |

The model IDs are the mid-2025 versions used in the study. Check that each provider still serves them before rerunning.

### Factorial grid (36 cells)

| Factor | Levels |
|---|---|
| Firms | 3, 5, 7 |
| Price sensitivity | high, medium, low |
| Visibility | perfect, noisy (each rival price an agent sees is scaled by a ±5% draw; its own price is exact) |
| Model mix | single, mixed |

### Interventions (Report Figure 6, Table 9)

All eight are applied to the 3-firm, medium-sensitivity, perfect-visibility, single-model baseline.

| Intervention | What changes |
|---|---|
| Refine goal | Objective becomes balancing profit with fair rental affordability and community welfare |
| Weaken long-term language | Objective becomes profit in this single quarter |
| Reduce precision | Prices are rounded to the nearest 5 cents |
| Reduce frequency | Each quarter, each firm has a 50% chance of keeping last quarter's price |
| Inject pricing noise | Noisy visibility (reuses the matching grid cell) |
| Inject market share noise | Shares are scaled by ±5% draws and renormalised |
| Mixed model | Mixed models (reuses the matching grid cell) |
| Reduce model parameters | `gemini-2.5-flash` instead of `gemini-2.5-pro` |

### Reinforcement-learning baselines (Addendum Figure 2)

These run on the same market for the high- and medium-sensitivity cells with perfect visibility.

| Baseline | Set-up | Default length |
|---|---|---|
| Tabular Q-learning | Independent learners; state = own last price and lowest rival price; α = 0.15, γ = 0.95, exploration exp(−10⁻⁵ t); scored on the last 10,000 periods | 2.5M periods, 30 seeds per cell |
| PPO (RLlib) | One shared policy, 256×256 network, learning rate 10⁻⁵, γ = 0.99, 30-step episodes; scored on steps 21–30 of the last 10,000 steps | 2.05M steps, 5 seeds per cell |

### Benchmarks (Addendum Table 11)

`benchmarks.py` computes every benchmark from the market model; no script contains its own copy.

- **Competitive:** firms take turns choosing the price that maximises their own profit this quarter given rivals' prices (Appendix A). All start at $1.50; the benchmark averages the last 600 of 3,000 turns.
- **Monopoly:** the common price from $1.01 to $1.99 that maximises joint profit.
- **Highest joint profit:** the best split of firms between one low and one high price.

## Repository layout

| File | Purpose |
|---|---|
| `run_experiments.py` | Entry point: smoke test, grid, interventions, RL baselines, then panel and report assets |
| `config.py` | Market parameters, model catalogue, provider assignment |
| `llm_runner.py` | Model calls for all four providers, retries, rate limits, refusals, run logs, resume |
| `interventions.py` | Prompt and price changes for the interventions |
| `environment.py` | RLlib multi-agent market environment for PPO |
| `rl_runner.py` | Tabular Q-learning and PPO training |
| `benchmarks.py` | Competitive, monopoly and highest-joint-profit benchmarks; Table 11 |
| `calc_collusion.py` | Per-run Q21–Q30 metrics and restoration detection |
| `build_panel.py` | Combines the run logs listed in the results files into one panel |
| `score_all_indices.py` | Primary and diagnostic indices per run, plus grid tables |
| `export_report_assets.py` | Every report figure and Word-ready table cut |
| `requirements.txt` | Python dependencies |

These files are written by `build_codebase.py`. Running it again overwrites them, so make changes there if you use it.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root with a key for each provider:

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
```

Keep `.env` out of version control. Google is required. To run without another provider, pass `--exclude-provider openai` (or `anthropic`, `deepseek`); mixed cells then rotate through the remaining models.

Check that every model answers before a full run:

```bash
python3 run_experiments.py --smoke
```

This makes one call to each model and then runs a full 30-quarter, 3-firm mixed-model game. Its logs go to `logs/smoke/` and never enter the results.

## Running

```bash
python3 benchmarks.py                                   # print Table 11, write benchmarks_table11.csv
python3 run_experiments.py --all                        # grid + RL baselines + interventions + panel + report assets
python3 run_experiments.py --grid --parallel 4          # grid only, 4 cells at a time
python3 run_experiments.py --interventions              # interventions only
python3 run_experiments.py --ql-only --ql-periods 6000000 --ql-runs 20   # extra Q-learning batch (e.g. convergence check)
python3 build_panel.py --replicate 1                    # rebuild the panel from the results files
python3 score_all_indices.py                            # diagnostic indices and grid tables
python3 export_report_assets.py                         # all figures and table cuts -> report_assets/
python3 export_report_assets.py --appendix-b-only       # Appendix B only
```

`run_experiments.py` builds the panel after any grid or intervention run, and builds the report assets once the panel holds all 36 grid cells and every intervention.

| Option | Effect |
|---|---|
| `--replicate N` | Write to `logs/repN/` (default 1) |
| `--resume` | Keep completed cells and continue unfinished runs (see below) |
| `--parallel N` | Run N cells or interventions at once |
| `--only single\|mixed` | Run only single-model or only mixed-model cells |
| `--cells FILE.csv` | Run only the cells listed (`num_agents, price_sensitivity, visibility, model_variance`) |
| `--exclude-provider P` | Leave a provider out (repeatable) |
| `--single-provider P` / `--provider-map FILE.csv` | Use another provider for single-model cells; `--show-provider-map` prints the assignment |
| `--skip-rl` | Skip the Q-learning and PPO baselines |
| `--ql-periods`, `--ql-runs`, `--ppo-steps`, `--ppo-runs` | RL training length and number of seeds |
| `--max-rounds` | Quarters per run (at least 30) |
| `--align-cells clustered\|all` | Runs used for the Appendix B event windows |

### Stopping and resuming

Rate limits back off from 30 seconds up to 10 minutes. If a provider reports a daily quota or no credit, the script stops with exit code 2 and moves unfinished runs to `logs/repN/partial/`. Run the same command again with `--resume`: completed cells are kept and unfinished runs continue from their last complete quarter. A run that fails after three attempts is recorded as NaN and its log is deleted.

### Cost

One replicate makes about 5,900 model calls: 5,400 for the grid and 540 for the six interventions that run on their own. Retries and the smoke test come on top.

## Outputs

### `logs/repN/`

| File | Contents |
|---|---|
| `run_*.csv` | One row per firm per quarter: model, price, units, profit, cumulative profit, reasoning, tokens, `refused` |
| `factorial_results.csv` | One row per grid cell: metrics, benchmarks, RL indices, refusal counts, log file |
| `intervention_results.csv` | One row per intervention: I_π, change from baseline, source (`run` or `grid`), refusal counts, log file |
| `provider_map.csv` | Provider and model used for each single-model cell |
| `ql_runs.csv`, `ppo_runs.csv` | One row per RL seed, with the training length |
| `failed_calls.csv` | Every failed or refused API call |
| `panel_q1_30.csv` | All runs listed in the two results files, one row per firm per quarter |
| `panel_q1_30_all_indices_q21_30.csv`, `panel_q1_30_heatmap_grids_q21_30.csv` | Output of `score_all_indices.py` |

PPO step logs are written to `logs/rl_ppo_*.csv`.

### `report_assets/`

| Output | Files |
|---|---|
| Tables 7a–c | `cut_table7a_profit_and_I_pi.csv`, `cut_table7b_P_paid_and_M_paid.csv`, `cut_table7c_P_post_and_D_wedge.csv` |
| Table 9 | `cut_table9_interventions.csv` |
| Table 11 | `cut_table11_benchmarks.csv` |
| Restorations, refusals | `cut_appendix_b_restorations.csv`, `cut_refusals.csv` |
| Report Figure 2 | Heatmap of I_π |
| Report Figure 3 | Advertised vs buyer-paid prices |
| Report Figure 4 | Edgeworth cycles in the 3-firm price-sensitive cells |
| Report Figure 6 | Change in I_π for each intervention |
| Addendum Figure 2 | Q-learning and PPO vs the models |
| Appendix B | Event-aligned restorations with a permutation benchmark |

Every figure comes as `*_labelled.png` and as `*_notext.png`/`.svg` (tick marks only, sized for the Word page, for labelling by hand). The Figure 2 heatmap keeps its cell values in both versions.

## Measures

All are computed over Q21–Q30. π̄ is mean joint profit per quarter; π_C, π_M and π_max are the competitive, monopoly and highest joint-profit benchmarks.

| Measure | Column | Definition |
|---|---|---|
| Index of excess profit, I_π | `calvano_delta` | (π̄ − π_C) / (π_M − π_C); 0 = competitive, 1 = monopoly. Undefined in low-sensitivity markets, where π_C ≥ π_M |
| Index against the highest joint profit | `delta_best_achievable` | (π̄ − π_C) / (π_max − π_C) |
| Posted price | `P_post` | Mean of all posted prices |
| Price paid | `P_paid` | Posted prices weighted by units sold |
| Wedge | `D_wedge` | P_post − P_paid |
| Markup on paid price | `M_paid` | P_paid / competitive P_paid − 1 |
| Restorations | `n_restorations` | Quarters from Q4 on in which the lowest price rises at least 5 cents |

Table 9 and Figure 6 report the change in I_π as the difference of the two-decimal indices shown in Table 9, so each row adds up as printed.

## Data provenance

A price in the results differs from the model's reply in only these cases:

- **Refusals.** If a model refuses after Q1, the firm keeps its previous price and the row is marked `refused=True`. The flag is carried into the panel, counted in both results files and in `score_all_indices.py`, and listed in `cut_refusals.csv`. A warning names any run whose Q21–Q30 quarters include held prices. A refusal in Q1 is retried, and the run fails if it persists.
- **Intervention rules.** Reduce precision rounds the model's price; Reduce frequency sometimes keeps last quarter's price. Both are part of the intervention.
- **Reused grid cells.** Baseline, Inject pricing noise and Mixed model reuse the matching 3-firm medium grid runs rather than separate runs.

Only logs listed in `factorial_results.csv` and `intervention_results.csv` enter the panel. Stray or abandoned logs are listed and ignored, and the build stops if a cell would appear twice or a listed log is missing.

## Reproducibility

- Model replies are not deterministic, so reruns give different paths.
- The noise on the rival prices an agent sees is fixed by the run's log file name, the two firms and the quarter, so a resumed run shows the same noisy history as before it stopped.
- Market-share noise and the Reduce frequency draws are not seeded; their outcomes are recorded in the run logs.
- Q-learning and PPO seeds are the seed numbers recorded in `ql_runs.csv` and `ppo_runs.csv`. The Appendix B permutation uses seed 0.
- Run separate replicates with `--replicate 2`, `--replicate 3`, and so on.
