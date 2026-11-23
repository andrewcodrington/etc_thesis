"""
config.py - Market parameters, mid-2025 models, and provider assignment.
All single-model runs use Google Gemini 2.5 Pro (and Gemini 2.5 Flash for small-model runs).
"""
import csv
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

PROVIDERS = ["google", "openai", "anthropic", "deepseek"]

MODEL_CATALOG = {
    "google":    {"large": "gemini-2.5-pro", "mid": "gemini-2.5-flash"},
    "openai":    {"large": "o3",             "mid": "o4-mini"},
    "anthropic": {"large": "claude-3-7-sonnet-20250219"},
    "deepseek":  {"large": "deepseek-reasoner"},
}

PROVIDER_MODELS: Dict[str, Dict[str, str]] = {p: dict(MODEL_CATALOG[p]) for p in PROVIDERS}
ACTIVE_PROVIDERS: List[str] = list(PROVIDERS)
MIXED_POOLS: Dict[str, List[str]] = {
    "large": [MODEL_CATALOG[p]["large"] for p in ACTIVE_PROVIDERS],
    "mid":   [MODEL_CATALOG["google"]["mid"], MODEL_CATALOG["openai"]["mid"]],
}

GRID_FIRMS = [3, 5, 7]
GRID_SENSITIVITIES = ["high", "medium", "low"]
GRID_VISIBILITIES = ["perfect", "noisy"]


def default_single_provider_map() -> Dict[tuple, str]:
    return {
        (n, s, vis): "google"
        for n in GRID_FIRMS
        for s in GRID_SENSITIVITIES
        for vis in GRID_VISIBILITIES
    }


SINGLE_PROVIDER_MAP: Dict[tuple, str] = default_single_provider_map()


def set_providers(exclude: List[str]) -> None:
    ACTIVE_PROVIDERS[:] = [p for p in PROVIDERS if p not in exclude]
    if "google" not in ACTIVE_PROVIDERS:
        raise ValueError("Google is required for single-model runs and the mid-sized intervention.")
    PROVIDER_MODELS.clear()
    PROVIDER_MODELS.update({p: dict(MODEL_CATALOG[p]) for p in ACTIVE_PROVIDERS})
    MIXED_POOLS["large"] = [MODEL_CATALOG[p]["large"] for p in ACTIVE_PROVIDERS]
    SINGLE_PROVIDER_MAP.clear()
    SINGLE_PROVIDER_MAP.update(default_single_provider_map())


def model_for(provider: str, size: str = "large") -> str:
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from {PROVIDERS}.")
    if size not in PROVIDER_MODELS[provider]:
        raise ValueError(f"Provider '{provider}' has no {size} model.")
    return PROVIDER_MODELS[provider][size]


def provider_of(model_name: str) -> str:
    for p, sizes in MODEL_CATALOG.items():
        if model_name in sizes.values():
            return p
    raise ValueError(f"Model '{model_name}' is not in MODEL_CATALOG.")


def load_provider_map(path: str) -> Dict[tuple, str]:
    m = dict(SINGLE_PROVIDER_MAP)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["num_agents"]), row["price_sensitivity"].strip(), row["visibility"].strip())
            prov = row["provider"].strip().lower()
            if prov not in PROVIDERS:
                raise ValueError(f"{path}: unknown provider '{prov}' for cell {key}")
            m[key] = prov
    return m


def write_provider_map(m: Dict[tuple, str], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_agents", "price_sensitivity", "visibility", "provider", "model"])
        for (n, s, vis), p in sorted(
            m.items(),
            key=lambda kv: (kv[0][0], GRID_SENSITIVITIES.index(kv[0][1]), GRID_VISIBILITIES.index(kv[0][2])),
        ):
            w.writerow([n, s, vis, p, model_for(p, "large")])


@dataclass
class MarketConfig:
    num_agents: int = 3
    max_rounds: int = 30
    replicate: int = 1
    unit_cost: float = 1.00
    max_price: float = 2.00
    price_sensitivity: Literal["low", "medium", "high"] = "medium"
    visibility: Literal["perfect", "noisy"] = "perfect"
    noise_range: float = 0.05

    model_variance: Literal["single", "mixed"] = "single"
    model_size: Literal["large", "mid"] = "large"
    single_provider: Optional[str] = None
    active_intervention: Optional[str] = None
    log_root: Optional[str] = None  # overrides logs/rep<N> (smoke tests write to logs/smoke)

    @property
    def bias(self) -> float:
        return {"low": 0.0, "medium": 0.5, "high": 1.0}[self.price_sensitivity]

    @property
    def company_names(self) -> List[str]:
        return ["A", "B", "C", "D", "E", "F", "G"][:self.num_agents]

    @property
    def provider_label(self) -> str:
        if self.model_variance == "mixed":
            return "mixed"
        if self.single_provider:
            return self.single_provider
        return SINGLE_PROVIDER_MAP.get((self.num_agents, self.price_sensitivity, self.visibility), "google")

    def get_model_for_firm(self, firm_id: str) -> str:
        if self.model_variance == "single":
            return model_for(self.provider_label, self.model_size)
        pool = MIXED_POOLS[self.model_size]
        idx = self.company_names.index(firm_id) % len(pool)
        return pool[idx]
