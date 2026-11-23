"""
environment.py - Multi-agent market environment for RLlib (1-cent discrete grid, Q1-Q30 episodes).
Demand: Q(p_min) = 100 * (2.00 - p_min).
Base shares: (2.00 - p_i) / sum_j(2.00 - p_j).
"""
import csv
import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from benchmarks import calvano_delta


class UnifiedMarketEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.n_firms = config.get("num_agents", 3)
        self.unit_cost = config.get("unit_cost", 1.00)
        self.max_price = config.get("max_price", 2.00)
        self.bias = config.get("bias", 0.5)
        self.price_sensitivity = config.get(
            "price_sensitivity",
            {1.0: "high", 0.5: "medium", 0.0: "low"}.get(self.bias, "medium"),
        )
        self.visibility = config.get("visibility", "perfect")
        self.noise_range = config.get("noise_range", 0.05)
        self.max_steps = config.get("max_steps", 30)
        self.log_path = config.get("log_path")

        self.action_gradients = 101
        self.price_grid = np.round(np.linspace(self.unit_cost, self.max_price, self.action_gradients), 2)

        self.agents = self.possible_agents = [f"firm_{i}" for i in range(self.n_firms)]
        self.observation_spaces = {
            a: Box(0.0, 3.0, shape=(self.n_firms + 1,), dtype=np.float32) for a in self.agents
        }
        self.action_spaces = {a: Discrete(self.action_gradients) for a in self.agents}
        self.episode = 0
        self.prices = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode += 1
        self.prices = {a: float(self.unit_cost * 1.25) for a in self.agents}
        return self._get_obs(), {}

    def _get_obs(self):
        obs = {}
        for me in self.agents:
            vec = [self.prices[a] for a in self.agents]
            if self.visibility == "noisy":
                vec = [
                    p if a == me else min(max(p * float(np.random.uniform(1.0 - self.noise_range, 1.0 + self.noise_range)), self.unit_cost), self.max_price)
                    for a, p in zip(self.agents, vec)
                ]
            obs[me] = np.array(vec + [self.unit_cost], dtype=np.float32)
        return obs

    def step(self, action_dict):
        self.current_step += 1
        for a, action in action_dict.items():
            self.prices[a] = round(float(self.price_grid[int(action)]), 2)

        min_p = min(self.prices.values())
        total_market_demand = max(0.0, 100.0 * (self.max_price - min_p))

        sum_inverse = sum(max(0.0, self.max_price - p) for p in self.prices.values())
        if sum_inverse > 0:
            probs = {a: max(0.0, self.max_price - p) / sum_inverse for a, p in self.prices.items()}
        else:
            probs = {a: 1.0 / self.n_firms for a in self.prices}

        lowest = [a for a, p in self.prices.items() if p == min_p]
        higher = [a for a, p in self.prices.items() if p > min_p]
        if self.bias > 0 and higher:
            shift = sum(probs[a] for a in higher) * self.bias
            for a in lowest:
                probs[a] += shift / len(lowest)
            for a in higher:
                probs[a] *= (1.0 - self.bias)

        orders = {a: round(total_market_demand * probs[a]) for a in self.agents}
        rewards = {a: float((self.prices[a] - self.unit_cost) * orders[a]) for a in self.agents}

        if self.log_path:
            joint_profit = sum(rewards.values())
            total_units = sum(orders.values())
            p_post = float(np.mean(list(self.prices.values())))
            p_paid = sum(orders[a] * self.prices[a] for a in self.agents) / total_units if total_units > 0 else p_post
            i_pi = calvano_delta(joint_profit, self.price_sensitivity, self.n_firms)
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.episode,
                    self.current_step,
                    round(i_pi, 4) if i_pi == i_pi else "",
                    round(p_post, 4),
                    round(p_paid, 4),
                    round(joint_profit, 4),
                ])

        done = self.current_step >= self.max_steps
        return self._get_obs(), rewards, {"__all__": False}, {"__all__": done}, {}
