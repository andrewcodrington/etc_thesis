"""
rl_runner.py - Independent Tabular Q-learning & PPO baselines (1-cent grid, 2.00 choke price).
"""
import csv
import math
import os
import uuid
import numpy as np
import pandas as pd
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.registry import register_env
from benchmarks import _market_outcome, calvano_delta
from environment import UnifiedMarketEnv


def _steps_logged(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1


def run_qlearning_benchmark(
    num_agents=3,
    bias=0.5,
    periods=2_500_000,
    eval_window=10_000,
    alpha=0.15,
    gamma=0.95,
    beta_decay=1e-5,
    full_state=False,
    seed=None,
):
    rng = np.random.default_rng(seed)
    sens = {1.0: "high", 0.5: "medium", 0.0: "low"}[bias]
    grid = [round(1.00 + 0.01 * k, 2) for k in range(101)]
    n_actions = len(grid)
    q_tables = [{} for _ in range(num_agents)]

    def get_q(firm_idx, state):
        tbl = q_tables[firm_idx]
        if state not in tbl:
            tbl[state] = np.zeros(n_actions, dtype=np.float64)
        return tbl[state]

    def state_for(firm_idx, prev_prices):
        if full_state:
            return prev_prices
        rivals = [prev_prices[j] for j in range(num_agents) if j != firm_idx]
        return (prev_prices[firm_idx], min(rivals))

    prices = tuple([1.25] * num_agents)
    eval_jps = []

    print(f"Training Tabular Q-learning ({periods:,} periods, seed={seed}): {num_agents} firms, bias={bias}...")
    for t in range(1, periods + 1):
        eps = math.exp(-beta_decay * t)
        states = [state_for(i, prices) for i in range(num_agents)]
        actions = []
        for i in range(num_agents):
            if rng.random() < eps:
                a = int(rng.integers(n_actions))
            else:
                q_vals = get_q(i, states[i])
                best = np.flatnonzero(q_vals >= q_vals.max() - 1e-12)
                a = int(rng.choice(best))
            actions.append(a)

        new_prices = tuple(grid[a] for a in actions)
        _, profits = _market_outcome(new_prices, bias)

        for i in range(num_agents):
            next_state = state_for(i, new_prices)
            q_curr = get_q(i, states[i])
            q_next_max = get_q(i, next_state).max()
            q_curr[actions[i]] += alpha * (profits[i] + gamma * q_next_max - q_curr[actions[i]])

        prices = new_prices
        if t > periods - eval_window:
            eval_jps.append(sum(profits))

    return calvano_delta(float(np.mean(eval_jps)), sens, num_agents)


def run_rl_benchmark(num_agents=3, bias=0.5, visibility="perfect", env_steps=2_050_000, seed=None):
    ray.init(include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    register_env("market", lambda cfg: UnifiedMarketEnv(cfg))
    os.makedirs("logs", exist_ok=True)

    seed_tag = f"_s{seed}" if seed is not None else ""
    log_file = os.path.abspath(f"logs/rl_ppo_{num_agents}a_bias{bias}_{visibility}{seed_tag}_{uuid.uuid4().hex[:6]}.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "step", "calvano_delta", "P_post", "P_paid", "joint_profit"])

    config = (
        PPOConfig()
        .environment(
            "market",
            env_config={"num_agents": num_agents, "bias": bias, "visibility": visibility, "max_steps": 30, "log_path": log_file},
        )
        .framework("torch")
        .env_runners(num_env_runners=0)
        .multi_agent(policies={"shared"}, policy_mapping_fn=lambda agent_id, episode, **kw: "shared")
        .rl_module(model_config=DefaultModelConfig(fcnet_hiddens=[256, 256]))
        .training(lr=1e-5, gamma=0.99, train_batch_size_per_learner=2050)
    )
    if seed is not None:
        config = config.debugging(seed=seed)
    algo = config.build_algo()
    print(f"Training PPO ({env_steps:,} steps, seed={seed}): {num_agents} firms, bias={bias}, info={visibility}...")

    while _steps_logged(log_file) < env_steps:
        algo.train()
    algo.stop()
    ray.shutdown()

    df = pd.read_csv(log_file).head(env_steps).tail(10_000)
    late = df[df["step"].between(21, 30)] if df["step"].between(21, 30).any() else df
    return float(late["calvano_delta"].mean()) if late["calvano_delta"].notna().any() else float("nan")
