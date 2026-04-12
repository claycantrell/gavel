from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
import os
import random
import typing

import anthropic
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from config import ArchiveGame, MutationSelectionStrategy, MutationStrategy
from ludii_parser import extract_parentheticals

LUDII_SYSTEM_PROMPT = """You are an expert in the Ludii game description language (L-GDL). Ludii represents board games as nested S-expressions using "ludemes" — high-level game-design primitives.

Here are two example Ludii games for reference:

Example 1 — Breakthrough (simple war game):
(game "Breakthrough" (players 2) (equipment {(board (square 8)) (piece "Pawn" Each (or (move Step Forward (to if:(is Empty (to)))) (move Step (directions {FR FL}) (to if:(is Enemy (who at:(to))) (apply (remove (to)))))))}) (rules (start {(place "Pawn1" (expand (sites Bottom) steps:1)) (place "Pawn2" (expand (sites Top) steps:1))}) (play (forEach Piece)) (end {(if (is In (last To) (sites Mover "Home")) (result Mover Win))})))

Example 2 — Yavalath (connection game with a twist):
(game "Yavalath" (players 2) (equipment {(board (hex 5)) (piece "Marker" Each)}) (rules (play (move Add (to (sites Empty)) (then (if (is Line 4 exact:True) (result Mover Loss))))) (end (if (is Line 5) (result Mover Win)))))

Key Ludii conventions:
- Games have three top-level sections: (players ...), (equipment {...}), (rules ...)
- Rules contain (start ...), (play ...), and (end ...) blocks
- Pieces define their own movement via (move ...) ludemes
- Common ludemes: Step, Slide, Hop, Add, Remove, Set, is Line, is In, forEach, if/then
- Board types: square, hex, rectangle, tri, graph, etc.
- Parentheses must balance. Every ( must have a matching ).

Your task: given a partial Ludii game with a section removed (marked <BLANK>), generate a replacement expression that fits syntactically and creates an interesting game. Output ONLY the replacement expression — no explanation, no markdown, no extra text."""


class BaseMutator(ABC):
    """Shared mutation logic: location selection, UCB tracking, and the mutate interface."""

    def __init__(self, num_return_sequences: int):
        self.num_return_sequences = num_return_sequences

        # Match the interface expected by MAPElitesSearch (.config.num_return_sequences)
        self.config = type('Config', (), {'num_return_sequences': num_return_sequences})()

        # UCB tracking
        self.successful_mutations_per_depth = defaultdict(int)
        self.successful_mutations_per_ludeme = defaultdict(int)
        self.samples_per_depth = defaultdict(int)
        self.samples_per_ludeme = defaultdict(int)
        self.num_samples = 0

    @property
    def _ucb_stats(self):
        return {
            "samples_per_depth": self.samples_per_depth,
            "successful_mutations_per_depth": self.successful_mutations_per_depth,
            "samples_per_ludeme": self.samples_per_ludeme,
            "successful_mutations_per_ludeme": self.successful_mutations_per_ludeme,
            "num_samples": self.num_samples
        }

    def _ucb_value(self, key, strategy: MutationSelectionStrategy):
        if strategy == MutationSelectionStrategy.UCB_DEPTH:
            samples_per_key = self.samples_per_depth
            successes_per_key = self.successful_mutations_per_depth
        elif strategy == MutationSelectionStrategy.UCB_LUDEME:
            samples_per_key = self.samples_per_ludeme
            successes_per_key = self.successful_mutations_per_ludeme
        else:
            raise NotImplementedError

        if samples_per_key[key] == 0:
            return float("inf")

        exploitation = successes_per_key[key] / samples_per_key[key]
        exploration = (2 * np.log(self.num_samples) / samples_per_key[key]) ** 0.5
        return exploitation + exploration

    def _select_mutation_location(self, game: ArchiveGame, strategy: MutationSelectionStrategy):
        """Select a balanced parenthetical in the game to replace."""
        parentheticals = [p for p in extract_parentheticals(game.game_str) if p[0] != ""]

        if strategy == MutationSelectionStrategy.RANDOM:
            prefix, middle, suffix, depth = random.choice(parentheticals)

        elif strategy == MutationSelectionStrategy.UCB_DEPTH:
            parentheticals_by_depth = defaultdict(list)
            for p in parentheticals:
                parentheticals_by_depth[p[3]].append(p)
            best_depth = max(parentheticals_by_depth.keys(), key=lambda d: self._ucb_value(d, strategy))
            prefix, middle, suffix, depth = random.choice(parentheticals_by_depth[best_depth])

        elif strategy == MutationSelectionStrategy.UCB_LUDEME:
            parentheticals_by_ludeme = defaultdict(list)
            for p in parentheticals:
                ludeme = p[1].split(" ")[0][1:]
                parentheticals_by_ludeme[ludeme].append(p)
            best_ludeme = max(parentheticals_by_ludeme.keys(), key=lambda l: self._ucb_value(l, strategy))
            prefix, middle, suffix, depth = random.choice(parentheticals_by_ludeme[best_ludeme])

        return prefix, middle, suffix, depth

    @abstractmethod
    def _generate_mutations(self, prefix: str, suffix: str) -> typing.List[str]:
        """Generate replacement game strings given the surrounding prefix and suffix."""
        ...

    def mutate(self, game: ArchiveGame,
               mutation_selection_strategy: MutationSelectionStrategy,
               mutation_strategy: MutationStrategy):
        prefix, middle, suffix, depth = self._select_mutation_location(game, mutation_selection_strategy)
        new_games = self._generate_mutations(prefix, suffix)
        return new_games, (prefix, middle, suffix, depth)

    def update_ucb_stats(self, mutation: typing.Tuple[str, str, str, int], success: bool):
        _, middle, _, depth = mutation
        ludeme = middle.split(" ")[0][1:]
        self.samples_per_depth[depth] += 1
        self.samples_per_ludeme[ludeme] += 1
        self.successful_mutations_per_depth[depth] += int(success)
        self.successful_mutations_per_ludeme[ludeme] += int(success)
        self.num_samples += 1


class AnthropicMutator(BaseMutator):
    """Mutator that calls the Anthropic Messages API with concurrent requests."""

    def __init__(self, model_name: str = "claude-opus-4-6", num_return_sequences: int = 3, temperature: float = 1.0):
        super().__init__(num_return_sequences)
        self.async_client = anthropic.AsyncAnthropic()
        self.model_name = model_name
        self.temperature = temperature

    async def _call_api_async(self, prefix: str, suffix: str) -> str:
        user_prompt = f"Here is a Ludii game with a section removed. Generate a replacement for <BLANK>.\n\n{prefix}<BLANK>{suffix}"

        response = await self.async_client.messages.create(
            model=self.model_name,
            max_tokens=512,
            temperature=self.temperature,
            system=LUDII_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        output = response.content[0].text.strip()
        if output.startswith("```"):
            output = output.split("\n", 1)[1] if "\n" in output else output[3:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()
        output = output.replace("\n", " ")
        return output

    async def _generate_mutations_async(self, prefix: str, suffix: str) -> typing.List[str]:
        tasks = [self._call_api_async(prefix, suffix) for _ in range(self.num_return_sequences)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_games = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Anthropic API error: {result}")
                continue
            new_games.append(f"{prefix}{result}{suffix}".strip())
        return new_games

    def _generate_mutations(self, prefix: str, suffix: str) -> typing.List[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop (e.g. Jupyter) — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self._generate_mutations_async(prefix, suffix)).result()
        else:
            return asyncio.run(self._generate_mutations_async(prefix, suffix))


class LLMMutator(BaseMutator):
    """Original mutator using a local HuggingFace causal LM with fill-in-the-middle."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: GenerationConfig):
        super().__init__(config.num_return_sequences)
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        # Store the HF generation config alongside the base config shim
        self.hf_config = config
        self.hf_config.pad_token_id = self.tokenizer.eos_token_id

    def _format_prompt(self, prefix: str, suffix: str):
        return f"<s><PRE> {prefix} <SUF>{suffix} <MID>"

    def _format_output(self, output_token_ids: typing.List[int]):
        output = self.tokenizer.decode(output_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        if "<MID> " in output:
            output = output.split("<MID> ")[1]
        output = self.tokenizer.decode(self.tokenizer.encode(output), skip_special_tokens=True)
        output = output.replace("\n", "")
        return output

    def _generate_mutations(self, prefix: str, suffix: str) -> typing.List[str]:
        prompt = self._format_prompt(prefix, suffix)

        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            generation_outputs = self.model.generate(
                input_ids=inputs,
                generation_config=self.hf_config,
                max_new_tokens=512,
            )

        outputs = [self._format_output(output.cpu().tolist()) for output in generation_outputs]
        new_games = [f"{prefix}{output}{suffix}".strip() for output in outputs]

        del inputs
        del generation_outputs
        torch.cuda.empty_cache()

        return new_games
