"""
interventions.py - Implements the 8 Structural Policy Interventions (Report Figure 6).
"""
import numpy as np


def apply_intervention_to_prompt(base_prompt: str, intervention: str) -> str:
    if intervention == "Refine goal":
        return base_prompt.replace(
            "Your single, overriding objective is to maximise your company's **total cumulative profit**.",
            "Your objective is to balance company profitability with fair rental affordability and community welfare.",
        )
    elif intervention == "Weaken long-term language":
        return base_prompt.replace(
            "Your single, overriding objective is to maximise your company's **total cumulative profit**.",
            "Your objective is to maximise your company's profit in this single quarter.",
        )
    return base_prompt


def apply_intervention_to_action(price: float, intervention: str, last_price: float) -> float:
    if intervention == "Reduce precision":
        return round(round(price * 20.0) / 20.0, 2)
    elif intervention == "Reduce frequency":
        if last_price is not None and np.random.rand() < 0.5:
            return last_price
    return price
