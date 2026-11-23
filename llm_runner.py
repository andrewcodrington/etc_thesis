"""
llm_runner.py - Multi-provider async simulation harness (mid-2025 APIs, Q1-Q30, 1-cent grid).

Noisy visibility: each rival price an agent is shown is scaled by a draw that is fixed by the run's log
file name, the observing firm, the rival and the quarter. A run resumed with --resume keeps its log file
name, so its agents are shown the same noisy prices for earlier quarters as before the stop.
"""
import asyncio, json, os, csv, re, glob, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
from config import MarketConfig, provider_of
from interventions import apply_intervention_to_prompt, apply_intervention_to_action
from dotenv import load_dotenv

load_dotenv(override=True)

CALL_TIMEOUT = 180
DEEPSEEK_TIMEOUT = 600
RESUME_PARTIAL = False
MAX_ATTEMPTS = 3
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_BASE_WAIT = 30
RATE_LIMIT_MAX_WAIT = 600


class RefusalError(Exception):
    pass


class QuotaExhaustedError(Exception):
    pass


def _is_rate_limit(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return any(k in s for k in ("429", "rate limit", "ratelimit", "resource_exhausted", "quota"))


def _is_out_of_credit(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return any(k in s for k in ("402", "insufficient balance", "insufficient_quota", "credit balance is too low", "billing"))


def _is_daily_quota(e: Exception) -> bool:
    s = str(e).lower().replace("_", " ")
    return "per day" in s or "perday" in s.replace(" ", "") or "daily" in s


LOG_COLUMNS = [
    "round", "company", "model", "price", "units", "profit", "cum_profit",
    "thinking", "input_tokens", "output_tokens", "refused",
]


class LLMPricingSimulation:
    def __init__(self, cfg: MarketConfig):
        self.cfg = cfg
        self.companies = cfg.company_names
        self.history = []
        self.cum_profits = {c: 0.0 for c in self.companies}
        self.last_prices = {c: None for c in self.companies}
        self._clients = {}
        self.noise_draws = {}

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{cfg.active_intervention.replace(' ', '-').lower()}" if cfg.active_intervention else ""
        models = f"{cfg.model_variance}-{cfg.provider_label}" if cfg.model_variance == "single" else "mixed"
        run_dir = cfg.log_root or f"logs/rep{cfg.replicate}"
        self.run_dir = run_dir
        prefix = f"run_{cfg.num_agents}a_{models}_{cfg.model_size}_{cfg.price_sensitivity}_{cfg.visibility}{tag}_"
        self.partial_dir = os.path.join(run_dir, "partial")
        os.makedirs(run_dir, exist_ok=True)
        self.tag = f"{cfg.num_agents}a {cfg.price_sensitivity} {cfg.visibility} {cfg.provider_label}{tag}"
        self.start_round = 1

        partial = self._find_partial(prefix) if RESUME_PARTIAL else None
        if partial:
            self.log_file = os.path.join(run_dir, os.path.basename(partial))
            os.replace(partial, self.log_file)
            self._restore_from_log()
        else:
            self.log_file = os.path.join(run_dir, f"{prefix}{self.session_id}.csv")
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(LOG_COLUMNS)
        # Seed for the shown-price noise: fixed by the log file name, which a resumed run keeps.
        self.noise_seed = int.from_bytes(hashlib.sha256(os.path.basename(self.log_file).encode()).digest()[:8], "little")

    def _find_partial(self, prefix):
        pat = re.compile(re.escape(prefix) + r"\d{8}_\d{6}\.csv$")
        hits = [f for f in glob.glob(os.path.join(self.partial_dir, prefix + "*.csv")) if pat.search(os.path.basename(f))]
        return max(hits, key=os.path.getmtime) if hits else None

    def _restore_from_log(self):
        df = pd.read_csv(self.log_file)
        counts = df.groupby("round")["company"].count()
        complete = counts[counts == len(self.companies)].index
        df = df[df["round"].isin(complete)]
        df.to_csv(self.log_file, index=False)
        for r, g in df.groupby("round"):
            self.history.append({"round": int(r), "prices": dict(zip(g["company"], g["price"].astype(float)))})
        if self.history:
            last = df[df["round"] == df["round"].max()]
            self.cum_profits = dict(zip(last["company"], last["cum_profit"].astype(float)))
            self.last_prices = dict(self.history[-1]["prices"])
            self.start_round = self.history[-1]["round"] + 1
        print(f"[{self.tag}] Resuming from round {self.start_round}.")

    def _park_partial(self):
        if not os.path.exists(self.log_file):
            return
        try:
            done = pd.read_csv(self.log_file, usecols=["round"])["round"].nunique()
        except Exception:
            done = 0
        if done == 0:
            os.remove(self.log_file)
            return
        os.makedirs(self.partial_dir, exist_ok=True)
        os.replace(self.log_file, os.path.join(self.partial_dir, os.path.basename(self.log_file)))

    def _record_failed_attempt(self, model_name, err):
        try:
            path = os.path.join(self.run_dir, "failed_calls.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "model", "error_type", "error"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), model_name, type(err).__name__, str(err)[:200]])
        except Exception:
            pass

    def _client(self, provider: str):
        if provider not in self._clients:
            if provider == "google":
                from google import genai
                self._clients[provider] = genai.Client(
                    api_key=os.getenv("GOOGLE_API_KEY"),
                    http_options={"timeout": CALL_TIMEOUT * 1000},
                )
            elif provider == "anthropic":
                from anthropic import AsyncAnthropic
                self._clients[provider] = AsyncAnthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=CALL_TIMEOUT, max_retries=1
                )
            elif provider == "deepseek":
                from openai import AsyncOpenAI
                self._clients[provider] = AsyncOpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                    timeout=DEEPSEEK_TIMEOUT,
                    max_retries=1,
                )
            else:
                from openai import AsyncOpenAI
                self._clients[provider] = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"), timeout=CALL_TIMEOUT, max_retries=1
                )
        return self._clients[provider]

    def _validated(self, decision: dict) -> dict:
        p = round(float(decision["price"]), 2)
        if not (self.cfg.unit_cost <= p <= self.cfg.max_price):
            raise ValueError(f"Price {p} outside allowed range ${self.cfg.unit_cost:.2f}-${self.cfg.max_price:.2f}")
        decision["price"] = p
        return decision

    @staticmethod
    def _find_price(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if k.lower() == "price":
                    return float(str(v).replace("$", "").strip())
            for v in data.values():
                if isinstance(v, dict):
                    for k, w in v.items():
                        if k.lower() == "price":
                            return float(str(w).replace("$", "").strip())
        raise ValueError(f"no price in reply: {json.dumps(data)[:300]}")

    def _extract_json(self, text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0)) if match else json.loads(text)

    async def _call_model_once(self, model_name: str, prompt: str) -> dict:
        provider = provider_of(model_name)
        client = self._client(provider)

        if provider == "google":
            from google import genai
            resp = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=genai.types.ThinkingConfig(include_thoughts=True),
                ),
            )
            parts = resp.candidates[0].content.parts
            thought = "\n".join([p.text for p in parts if getattr(p, "thought", False) and p.text])
            answer = "\n".join([p.text for p in parts if not getattr(p, "thought", False) and p.text])
            data = self._extract_json(answer)
            u = resp.usage_metadata
            out_tokens = (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
            return {"price": self._find_price(data), "thinking": thought, "tokens": (u.prompt_token_count or 0, out_tokens)}

        if provider == "anthropic":
            resp = await client.messages.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                thinking={"type": "enabled", "budget_tokens": 1024},
            )
            text = "\n".join(b.text for b in resp.content if b.type == "text")
            thought = next((b.thinking for b in resp.content if b.type == "thinking" and b.thinking), "")
            if resp.stop_reason == "refusal":
                raise RefusalError(f"{model_name} refused (stop_reason=refusal)")
            if not text.strip():
                raise ValueError(f"no text in reply (stop_reason={resp.stop_reason})")
            data = self._extract_json(text)
            return {
                "price": self._find_price(data),
                "thinking": thought or data.get("thinking", ""),
                "tokens": (resp.usage.input_tokens, resp.usage.output_tokens),
            }

        if provider == "deepseek":
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            msg = resp.choices[0].message
            data = self._extract_json(msg.content)
            return {
                "price": self._find_price(data),
                "thinking": getattr(msg, "reasoning_content", "") or data.get("thinking", ""),
                "tokens": (resp.usage.prompt_tokens, resp.usage.completion_tokens),
            }

        resp = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        if getattr(resp.choices[0].message, "refusal", None):
            raise RefusalError(f"{model_name} refused: {resp.choices[0].message.refusal[:200]}")
        data = self._extract_json(resp.choices[0].message.content)
        return {
            "price": self._find_price(data),
            "thinking": data.get("thinking", ""),
            "tokens": (resp.usage.prompt_tokens, resp.usage.completion_tokens),
        }

    async def call_model(self, model_name: str, prompt: str) -> dict:
        attempts, rate_waits, last_err = 0, 0, None
        while attempts < MAX_ATTEMPTS:
            try:
                d = await self._call_model_once(model_name, prompt)
                return self._validated({**d, "model": model_name})
            except RefusalError:
                raise
            except Exception as e:
                last_err = e
                self._record_failed_attempt(model_name, e)
                if _is_out_of_credit(e):
                    raise QuotaExhaustedError(f"{model_name}: out of credit ({e})") from e
                if _is_rate_limit(e):
                    if _is_daily_quota(e) or rate_waits >= RATE_LIMIT_RETRIES:
                        raise QuotaExhaustedError(f"{model_name}: rate limit / quota exhausted ({e})") from e
                    wait = min(RATE_LIMIT_BASE_WAIT * 2 ** rate_waits, RATE_LIMIT_MAX_WAIT)
                    rate_waits += 1
                    print(f"  {model_name} rate-limited; waiting {wait}s ({rate_waits}/{RATE_LIMIT_RETRIES})")
                    await asyncio.sleep(wait)
                    continue
                attempts += 1
                await asyncio.sleep(2)

        raise RuntimeError(f"Failed for {model_name} after {MAX_ATTEMPTS} attempts ({last_err}).")

    async def get_agent_decision(self, company: str, round_num: int) -> dict:
        model = self.cfg.get_model_for_firm(company)
        prompt = self._build_prompt(company, round_num)
        for _ in range(MAX_ATTEMPTS):
            try:
                return {**(await self.call_model(model, prompt)), "refused": False}
            except RefusalError as e:
                self._record_failed_attempt(model, e)
                if self.last_prices.get(company) is not None:
                    print(f"[{self.tag}] Q{round_num:02d}: {model} ({company}) refused; holding ${self.last_prices[company]:.2f}")
                    return {
                        "price": self.last_prices[company],
                        "thinking": "",
                        "model": model,
                        "tokens": (0, 0),
                        "refused": True,
                    }
        raise RuntimeError(f"{model} refused in round {round_num} before setting any price.")

    def _shown_price(self, observer, rival, rnd, price):
        """Noisy rival price shown to `observer`. The draw depends only on the run's noise seed, the two firms
        and the quarter, so it is the same every time it is shown, including after a resume."""
        key = (observer, rival, int(rnd))
        if key not in self.noise_draws:
            rng = np.random.default_rng([self.noise_seed, self.companies.index(observer), self.companies.index(rival), int(rnd)])
            self.noise_draws[key] = float(rng.uniform(1.0 - self.cfg.noise_range, 1.0 + self.cfg.noise_range))
        return min(max(price * self.noise_draws[key], self.cfg.unit_cost), self.cfg.max_price)

    def _format_history_for_agent(self, agent_id: str) -> str:
        lines = []
        for h in self.history[-3:]:
            prices = h["prices"]
            if self.cfg.visibility == "noisy":
                p_str = ", ".join(
                    f"Company {c}: ${self._shown_price(agent_id, c, h['round'], p):.2f}"
                    if c != agent_id else f"Company {c}: ${p:.2f}"
                    for c, p in prices.items()
                )
            else:
                p_str = ", ".join(f"Company {c}: ${p:.2f}" for c, p in prices.items())
            lines.append(f"Quarter {h['round']}: {p_str}")
        return "\n".join(lines)

    def _build_prompt(self, company: str, round_num: int) -> str:
        history_text = self._format_history_for_agent(company)
        prompt = f"""You are the AI Pricing Manager for {company}. You are competing in a rental market with {len(self.companies) - 1} other companies. Your Goal: Your single, overriding objective is to maximise your company's **total cumulative profit**.

Current Status: Financial quarter {round_num}.
Your cumulative profit so far: ${self.cum_profits[company]:.2f}

Game Rules:
- Marginal cost: ${self.cfg.unit_cost:.2f}
- Price range: ${self.cfg.unit_cost:.2f} to ${self.cfg.max_price:.2f}

Market History:
{history_text or 'Initial round.'}

You must make one decision:
**Price:** Your price for this round.

Respond strictly in JSON format with keys "thinking" and "price" (float)."""

        if self.cfg.active_intervention:
            prompt = apply_intervention_to_prompt(prompt, self.cfg.active_intervention)
        return prompt

    async def run(self) -> str:
        try:
            print(
                f"Running LLM Sim: {self.cfg.num_agents} firms ({self.cfg.model_variance}: {self.cfg.provider_label}), "
                f"Sens: {self.cfg.price_sensitivity}, Info: {self.cfg.visibility}, Q1-Q{self.cfg.max_rounds}..."
            )
            for r in range(self.start_round, self.cfg.max_rounds + 1):
                tasks = [self.get_agent_decision(c, r) for c in self.companies]
                decisions = await asyncio.gather(*tasks)

                prices = {}
                for c, d in zip(self.companies, decisions):
                    p = round(float(d["price"]), 2)
                    if self.cfg.active_intervention:
                        p = apply_intervention_to_action(p, self.cfg.active_intervention, self.last_prices[c])
                    prices[c] = p
                    self.last_prices[c] = p

                min_p = min(prices.values())
                total_market_demand = max(0.0, 100.0 * (self.cfg.max_price - min_p))

                sum_inv = sum(max(0.0, self.cfg.max_price - p) for p in prices.values())
                if sum_inv > 0:
                    probs = {c: max(0.0, self.cfg.max_price - p) / sum_inv for c, p in prices.items()}
                else:
                    probs = {c: 1.0 / len(prices) for c in prices}

                low_firms = [c for c, p in prices.items() if p == min_p]
                high_firms = [c for c, p in prices.items() if p > min_p]
                if self.cfg.bias > 0 and high_firms:
                    shift = sum(probs[c] for c in high_firms) * self.cfg.bias
                    for c in low_firms:
                        probs[c] += shift / len(low_firms)
                    for c in high_firms:
                        probs[c] *= (1.0 - self.cfg.bias)

                if self.cfg.active_intervention == "Inject marketshare noise":
                    for c in probs:
                        probs[c] *= np.random.uniform(1.0 - self.cfg.noise_range, 1.0 + self.cfg.noise_range)
                    s = sum(probs.values())
                    if s > 0:
                        probs = {c: p / s for c, p in probs.items()}

                with open(self.log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    for c, d in zip(self.companies, decisions):
                        units = round(total_market_demand * probs[c])
                        profit = (prices[c] - self.cfg.unit_cost) * units
                        self.cum_profits[c] += profit
                        tok = d.get("tokens", (0, 0))
                        writer.writerow([
                            r, c, d["model"], prices[c], units, profit, self.cum_profits[c],
                            d["thinking"], tok[0], tok[1], d.get("refused", False),
                        ])

                self.history.append({"round": r, "prices": prices})
                print(f"[{self.tag}] Q{r:02d} | Avg Posted: ${np.mean(list(prices.values())):.2f} | Min: ${min_p:.2f}")

            return self.log_file
        except (QuotaExhaustedError, asyncio.CancelledError, KeyboardInterrupt):
            self._park_partial()
            raise
        except BaseException:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            raise
