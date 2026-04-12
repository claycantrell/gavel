"""
LLM-as-judge fitness evaluator for Ludii games.

Uses Claude to evaluate game quality on multiple dimensions, replacing
the hand-crafted threshold-based fitness function. Can be used standalone
or as a pre-filter before expensive simulation-based evaluation.
"""

import json
import typing

import anthropic

from config import UNCOMPILABLE_FITNESS, UNPLAYABLE_FITNESS, UNINTERESTING_FITNESS

LLM_JUDGE_SYSTEM = """You are an expert board game designer evaluating games written in the Ludii game description language.

Given a Ludii game description, evaluate it on these dimensions (each scored 0.0 to 1.0):

1. **coherence**: Are the rules internally consistent? Do pieces have valid movement? Do end conditions make sense? (0 = broken/contradictory, 1 = perfectly coherent)
2. **interestingness**: Does the game offer meaningful player decisions? Are there interesting tradeoffs? (0 = trivial/no decisions, 1 = rich strategic depth)
3. **balance**: Does the game appear fair between players? (0 = one side always wins, 1 = perfectly balanced)
4. **novelty**: How different is this from well-known games like Chess, Go, Hex, Checkers? (0 = exact clone, 1 = completely novel mechanics)
5. **completeness**: Will games reliably end? Are there clear victory conditions? (0 = games never end, 1 = always terminates cleanly)

Respond with ONLY a JSON object, no other text:
{"coherence": 0.X, "interestingness": 0.X, "balance": 0.X, "novelty": 0.X, "completeness": 0.X, "brief_rationale": "one sentence"}"""


class LLMFitnessEvaluator:
    """Evaluate game quality using an LLM judge."""

    def __init__(self, model_name: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model_name = model_name

    def evaluate(self, game_str: str) -> typing.Dict[str, float]:
        """
        Evaluate a single game string. Returns a dict with scores and a
        combined fitness value compatible with the evolution pipeline.
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=256,
                temperature=0.0,
                system=LLM_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": f"Evaluate this Ludii game:\n\n{game_str}"}],
            )

            text = response.content[0].text.strip()
            # Handle markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            scores = json.loads(text)

        except (json.JSONDecodeError, Exception) as e:
            return {
                "compilable": False, "playable": False,
                "llm_coherence": 0, "llm_interestingness": 0,
                "llm_balance": 0, "llm_novelty": 0, "llm_completeness": 0,
                "llm_rationale": f"LLM evaluation failed: {e}",
                "llm_fitness": UNCOMPILABLE_FITNESS,
                "game_str": game_str, "wins": [],
            }

        evaluation = {
            "compilable": scores.get("coherence", 0) > 0.3,
            "playable": scores.get("coherence", 0) > 0.3 and scores.get("completeness", 0) > 0.2,
            "llm_coherence": scores.get("coherence", 0),
            "llm_interestingness": scores.get("interestingness", 0),
            "llm_balance": scores.get("balance", 0),
            "llm_novelty": scores.get("novelty", 0),
            "llm_completeness": scores.get("completeness", 0),
            "llm_rationale": scores.get("brief_rationale", ""),
            "game_str": game_str,
            "wins": [],
            # Populate standard fitness keys with LLM equivalents so downstream code works
            "balance": scores.get("balance", 0),
            "completion": scores.get("completeness", 0),
            "drawishness": 0.5,  # LLM can't assess this without simulation
            "decision_moves": scores.get("interestingness", 0),
            "board_coverage_default": 0.5,
            "mean_turns": 10,  # neutral default
            "trace_score": scores.get("interestingness", 0) * scores.get("balance", 0),
        }

        # Combined fitness: weighted geometric mean of the LLM scores
        dims = [evaluation["llm_coherence"], evaluation["llm_interestingness"],
                evaluation["llm_balance"], evaluation["llm_completeness"]]

        if any(d <= 0 for d in dims):
            evaluation["llm_fitness"] = UNINTERESTING_FITNESS
        else:
            import numpy as np
            # Novelty gets half weight — it's less reliable
            weighted = dims + [evaluation["llm_novelty"] ** 0.5] if evaluation["llm_novelty"] > 0 else dims
            evaluation["llm_fitness"] = float(np.prod(weighted) ** (1.0 / len(weighted)))

        return evaluation

    def evaluate_batch(self, game_strs: typing.List[str]) -> typing.List[typing.Dict[str, float]]:
        """Evaluate multiple games. Uses sequential calls (async version in mutators)."""
        return [self.evaluate(gs) for gs in game_strs]
