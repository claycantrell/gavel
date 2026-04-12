from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
import os
import random
import typing

import anthropic
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

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

LUDAX_SYSTEM_PROMPT = """You are an expert board game designer working with the Ludax game description language. Ludax uses S-expressions to describe deterministic abstract strategy board games.

Here are example Ludax games:

Example 1 — Yavalath (placement + line game with a twist — 4-in-a-row loses, but 3-in-a-row also loses):
(game "Yavalath"
    (players 2)
    (equipment (board (hexagon 9)) (pieces ("token" both)))
    (rules
        (play (repeat (P1 P2) (place "token" (destination (empty)))))
        (end (if (line "token" 4) (mover win)) (if (line "token" 3) (mover lose)))
    )
)

Example 2 — HopThrough (movement game — hop to reach opponent's side):
(game "HopThrough"
    (players 2 (set_forward (P1 up) (P2 down)))
    (equipment (board (square 8)) (pieces ("token" both)))
    (rules
        (start (place "token" P1 ((row 6) (row 7))) (place "token" P2 ((row 0) (row 1))))
        (play (repeat (P1 P2) (move (hop "token" direction:any))))
        (end (if (exists (and (occupied mover) (edge forward))) (mover win)))
    )
)

Example 3 — Reversi (placement + flip capture + scoring):
(game "Reversi"
    (players 2)
    (equipment (board (square 8)) (pieces ("token" both)))
    (rules
        (start (place "token" P1 (28 35)) (place "token" P2 (27 36)))
        (play (repeat (P1 P2)
            (place "token" (destination (empty))
                (result (exists (custodial "token" any)))
                (effects (flip (custodial "token" any))
                    (set_score mover (count (occupied mover)))
                    (set_score opponent (count (occupied opponent)))))
            (force_pass)))
        (end (if (passed both) (by_score)))
    )
)

Example 4 — Hex (connection game with set_forward and connected predicate):
(game "Hex"
    (players 2 (set_forward (P1 up) (P2 right)))
    (equipment (board (hex_rectangle 11 11)) (pieces ("token" both)))
    (rules
        (play (repeat (P1 P2) (place "token" (destination (empty)))))
        (end (if (>= (connected "token" ((edge forward) (edge backward))) 2) (mover win)))
    )
)

Example 5 — English Draughts (movement + hop capture + promotion):
(game "Draughts"
    (players 2 (set_forward (P1 up) (P2 down)))
    (equipment (board (square 8)) (pieces ("pawn" both) ("king" both)))
    (rules
        (start (place "pawn" P1 (40 42 44 46 49 51 53 55 56 58 60 62)) (place "pawn" P2 (1 3 5 7 8 10 12 14 17 19 21 23)))
        (play (repeat (P1 P2) (move (or
            (hop "pawn" direction:(forward_left forward_right) hop_over:opponent capture:true priority:0)
            (step "pawn" direction:(forward_left forward_right) priority:1)
            (hop "king" direction:diagonal hop_over:opponent capture:true priority:0)
            (step "king" direction:diagonal priority:1))
            (effects (promote "pawn" "king" (edge forward))
                (if (and (action_was mover hop) (can_move_again hop)) (extra_turn mover same_piece:true))))))
        (end (if (no_legal_actions) (mover win)))
    )
)

Key Ludax syntax rules:
- Top-level: (game "Name" (players N ...) (equipment ...) (rules ...))
- Boards: (square N), (rectangle W H), (hexagon D), (hex_rectangle W H)
- Pieces: (pieces ("name" P1|P2|both) ...)
- Movement: slide, hop, step (in move blocks); place (for placement)
- Capture: (capture (custodial ...)), implicit via hop with capture:true
- Effects: capture, promote, flip, extra_turn, set_score, increment_score
- End conditions: (line "piece" N), (no_legal_actions), (full_board), (by_score), (captured_all "piece"), (exists MASK)
- Connected predicate: (>= (connected "piece" ((mask1) (mask2))) 2) — NOTE: multi-mask uses DOUBLE parens ((mask1) (mask2))
- Win/loss: (mover win), (mover lose), (opponent win), (draw)
- Masks: (empty), (occupied mover|opponent), (edge forward|backward|...), (row N), (column N), (corners), (center), (and ...), (or ...), (not ...)
- Play: (repeat (P1 P2) ...) or (once_through (P1 P2) ...)
- Parentheses must balance. Every ( needs a matching ).

Your task: given a partial Ludax game with a section removed (marked <BLANK>), generate a replacement expression that fits syntactically and creates an interesting game. Output ONLY the replacement expression — no explanation, no markdown, no extra text."""

LUDAX_AGENTIC_PROMPT = """You are an expert board game designer working with the Ludax game description language. You design novel, interesting games by modifying existing ones.

Ludax syntax reference:
- Boards: (square N), (rectangle W H), (hexagon D), (hex_rectangle W H)
- Pieces: ("name" P1|P2|both) — defined in equipment
- Movement: slide (any distance), hop (jump over), step (one cell), place (drop piece)
- Directions: orthogonal, diagonal, any, forward, backward, forward_left, forward_right, up, down, left, right
- Capture: (capture (custodial "piece" N orientation:...)), hop with capture:true
- Effects: capture, promote "from" "to" MASK, flip MASK, extra_turn, set_score, increment_score
- End conditions: (line "piece" N), (no_legal_actions), (full_board), (by_score), (captured_all "piece"), (exists MASK)
- Connected: (>= (connected "piece" ((mask1) (mask2))) 2) — NOTE double parens for multi-mask
- Win/loss: (mover win), (mover lose), (opponent win), (draw)
- Masks: (empty), (occupied mover|opponent), (edge forward|backward|...), (row N), (column N), (corners), (and ...), (or ...), (not ...)
- Play: (repeat (P1 P2) ...) or (once_through (P1 P2) ...)

Parentheses must balance. All games must be deterministic (no dice/randomness)."""



# Semantic weight categories for mutation targeting
# High-weight = gameplay-changing, low-weight = structural/trivial
_LUDEME_WEIGHTS = {
    # End conditions — highest impact (change win/loss conditions)
    "end": 10, "if": 8, "line": 8, "connected": 8, "exists": 8,
    "no_legal_actions": 6, "full_board": 6, "by_score": 6, "captured_all": 6,
    ">=": 7, "<=": 7, "=": 7,
    # Win/loss outcomes
    "mover": 5, "opponent": 5, "draw": 5,
    # Movement rules — high impact
    "move": 9, "hop": 9, "step": 9, "slide": 9, "place": 7,
    "or": 7, "and": 6, "not": 6,
    # Effects — change what happens after a move
    "effects": 9, "capture": 8, "promote": 8, "flip": 8,
    "extra_turn": 7, "set_score": 6, "increment_score": 6,
    # Constraints on placement/movement
    "destination": 7, "result": 7,
    # Play structure
    "play": 6, "repeat": 4, "once_through": 4, "force_pass": 5,
    # Board (interesting but risky — can break start positions)
    "board": 3, "square": 3, "hexagon": 3, "hex_rectangle": 3, "rectangle": 3,
    # Masks (useful subexpressions)
    "empty": 2, "occupied": 3, "edge": 3, "row": 2, "column": 2,
    "corners": 3, "center": 3, "adjacent": 4, "custodial": 5,
}
# Everything not listed gets weight 0 (skipped)
_SKIP_LUDEMES = {"players", "equipment", "pieces", "rules", "start", "rendering",
                 "color", "set_forward", "game"}


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

    @staticmethod
    def _get_ludeme_name(parenthetical_text: str) -> str:
        """Extract the ludeme name from a parenthetical string like '(move ...)'."""
        inner = parenthetical_text[1:].lstrip()
        return inner.split(None, 1)[0] if inner else ""

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

        elif strategy == MutationSelectionStrategy.SEMANTIC:
            # Weight mutation points by gameplay impact
            # Filter out trivial/structural expressions, then sample weighted by impact
            candidates = []
            weights = []
            for p in parentheticals:
                ludeme = self._get_ludeme_name(p[1])
                if ludeme in _SKIP_LUDEMES:
                    continue
                # Also skip pure string literals and index lists
                if ludeme.startswith('"') or ludeme[0:1].isdigit():
                    continue
                w = _LUDEME_WEIGHTS.get(ludeme, 1)
                candidates.append(p)
                weights.append(w)

            if not candidates:
                # Fallback to random if everything is filtered
                candidates = parentheticals
                weights = [1] * len(candidates)

            # Weighted random selection
            total = sum(weights)
            probs = [w / total for w in weights]
            idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
            prefix, middle, suffix, depth = candidates[idx]

        return prefix, middle, suffix, depth

    @abstractmethod
    def _generate_mutations(self, prefix: str, suffix: str) -> typing.List[str]:
        """Generate replacement game strings given the surrounding prefix and suffix."""
        ...

    @staticmethod
    def _normalized_edit_distance(a: str, b: str) -> float:
        """Fast normalized edit distance (0=identical, 1=completely different)."""
        # Use character-level comparison; strip whitespace for fair comparison
        a_clean = "".join(a.split())
        b_clean = "".join(b.split())
        if a_clean == b_clean:
            return 0.0
        # Approximate via difflib ratio (faster than full Levenshtein)
        from difflib import SequenceMatcher
        return 1.0 - SequenceMatcher(None, a_clean, b_clean).ratio()

    def mutate(self, game: ArchiveGame,
               mutation_selection_strategy: MutationSelectionStrategy,
               mutation_strategy: MutationStrategy,
               min_novelty: float = 0.05):
        prefix, middle, suffix, depth = self._select_mutation_location(game, mutation_selection_strategy)
        new_games = self._generate_mutations(prefix, suffix)

        # Novelty filter: reject mutations too similar to the parent
        if min_novelty > 0:
            novel_games = []
            for g in new_games:
                dist = self._normalized_edit_distance(g, game.game_str)
                if dist >= min_novelty:
                    novel_games.append(g)
            new_games = novel_games

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

    def __init__(self, model_name: str = "claude-opus-4-6", num_return_sequences: int = 3,
                 temperature: float = 1.0, use_ludax: bool = False, max_repair_attempts: int = 2):
        super().__init__(num_return_sequences)
        self.async_client = anthropic.AsyncAnthropic()
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = LUDAX_SYSTEM_PROMPT if use_ludax else LUDII_SYSTEM_PROMPT
        self.dsl_name = "Ludax" if use_ludax else "Ludii"
        self.use_ludax = use_ludax
        self.max_repair_attempts = max_repair_attempts

    def _strip_markdown(self, text: str) -> str:
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return text.replace("\n", " ")

    async def _call_api_async(self, prefix: str, suffix: str, original: str = "") -> str:
        if original:
            user_prompt = (
                f"Here is a {self.dsl_name} game with a section removed. The original section was:\n{original}\n\n"
                f"Generate a DIFFERENT replacement for <BLANK> that changes the gameplay. "
                f"Do NOT reproduce the original. Be creative.\n\n{prefix}<BLANK>{suffix}"
            )
        else:
            user_prompt = f"Here is a {self.dsl_name} game with a section removed. Generate a replacement for <BLANK>.\n\n{prefix}<BLANK>{suffix}"

        messages = [{"role": "user", "content": user_prompt}]

        response = await self.async_client.messages.create(
            model=self.model_name, max_tokens=512, temperature=self.temperature,
            system=self.system_prompt, messages=messages,
        )
        output = self._strip_markdown(response.content[0].text.strip())

        # Self-repair loop: validate grammar, feed error back if invalid
        if self.use_ludax:
            from ludax_grammar import validate_game
            for attempt in range(self.max_repair_attempts):
                full_game = f"{prefix}{output}{suffix}".strip()
                is_valid, err = validate_game(full_game)
                if is_valid:
                    break

                # Feed the error back and ask for a fix
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content":
                    f"That replacement produces a grammar error:\n{err[:300]}\n\n"
                    f"Fix the syntax and output ONLY the corrected replacement expression."
                })
                response = await self.async_client.messages.create(
                    model=self.model_name, max_tokens=512, temperature=0.3,
                    system=self.system_prompt, messages=messages,
                )
                output = self._strip_markdown(response.content[0].text.strip())

        return output

    async def _generate_mutations_async(self, prefix: str, suffix: str,
                                        original: str = "") -> typing.List[str]:
        tasks = [self._call_api_async(prefix, suffix, original) for _ in range(self.num_return_sequences)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_games = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Anthropic API error: {result}")
                continue
            new_games.append(f"{prefix}{result}{suffix}".strip())
        return new_games

    def _generate_mutations(self, prefix: str, suffix: str) -> typing.List[str]:
        # Called by BaseMutator.mutate; uses stored original from mutate override
        return self._run_async(self._generate_mutations_async(prefix, suffix, self._current_middle))

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    def mutate(self, game: ArchiveGame,
               mutation_selection_strategy: MutationSelectionStrategy,
               mutation_strategy: MutationStrategy,
               min_novelty: float = 0.05):
        """Override to pass the original section to the API for anti-identity prompting."""
        prefix, middle, suffix, depth = self._select_mutation_location(game, mutation_selection_strategy)
        self._current_middle = middle
        new_games = self._generate_mutations(prefix, suffix)

        # Novelty filter
        if min_novelty > 0:
            novel = [g for g in new_games if self._normalized_edit_distance(g, game.game_str) >= min_novelty]
            new_games = novel

        return new_games, (prefix, middle, suffix, depth)


AGENTIC_SYSTEM_PROMPT = """You are an expert board game designer working with the Ludii game description language. You design novel, interesting games by modifying existing ones.

Key Ludii conventions:
- Games have (players ...), (equipment {...}), (rules ...) sections
- Rules contain (start ...), (play ...), and (end ...) blocks
- Common ludemes: Step, Slide, Hop, Add, Remove, forEach, if/then
- Board types: square, hex, rectangle, tri, etc.
- Parentheses must balance. Every ( must have a matching )."""


class AgenticMutator(BaseMutator):
    """
    Agentic mutator that uses a multi-turn propose-critique-refine loop.
    Instead of blindly filling in blanks, the model:
    1. Sees the full game and the section being mutated
    2. Proposes a replacement with design rationale
    3. Self-critiques for issues (syntax, balance, fun)
    4. Produces a refined final version
    """

    def __init__(self, model_name: str = "claude-opus-4-6", num_return_sequences: int = 3,
                 temperature: float = 1.0, use_ludax: bool = False, max_repair_attempts: int = 2):
        super().__init__(num_return_sequences)
        self.client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = LUDAX_AGENTIC_PROMPT if use_ludax else AGENTIC_SYSTEM_PROMPT
        self.dsl_name = "Ludax" if use_ludax else "Ludii"
        self.use_ludax = use_ludax
        self.max_repair_attempts = max_repair_attempts

    async def _propose_critique_refine(self, prefix: str, original_section: str, suffix: str) -> str:
        """Single agentic mutation: propose → critique → refine."""
        full_game = f"{prefix}{original_section}{suffix}"

        # Turn 1: Propose a mutation with rationale
        messages = [
            {"role": "user", "content": (
                f"Here is a {self.dsl_name} board game:\n\n{full_game}\n\n"
                f"I want to mutate this section: {original_section}\n\n"
                f"Propose a creative replacement that makes the game more interesting or novel. "
                f"Output the replacement expression, then explain your design rationale in 1-2 sentences."
            )}
        ]

        response = await self.async_client.messages.create(
            model=self.model_name, max_tokens=512, temperature=self.temperature,
            system=self.system_prompt, messages=messages,
        )
        proposal = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": proposal})

        # Turn 2: Self-critique
        messages.append({"role": "user", "content": (
            "Now critique your proposal. Check for:\n"
            "1. Syntax errors (unbalanced parentheses, invalid ludemes)\n"
            "2. Coherence (does it fit with the rest of the game?)\n"
            "3. Playability (will games actually terminate?)\n"
            "4. Interest (does it create meaningful decisions?)\n"
            "Be specific about any issues."
        )})

        response = await self.async_client.messages.create(
            model=self.model_name, max_tokens=256, temperature=0.3,
            system=self.system_prompt, messages=messages,
        )
        critique = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": critique})

        # Turn 3: Produce refined version
        messages.append({"role": "user", "content": (
            f"Based on your critique, produce the final refined replacement expression. "
            f"Output ONLY the {self.dsl_name} expression — no explanation, no markdown."
        )})

        response = await self.async_client.messages.create(
            model=self.model_name, max_tokens=512, temperature=0.5,
            system=self.system_prompt, messages=messages,
        )

        output = response.content[0].text.strip()
        if output.startswith("```"):
            output = output.split("\n", 1)[1] if "\n" in output else output[3:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()
        output = output.replace("\n", " ")

        # Turn 4+ (optional): Grammar self-repair loop
        if self.use_ludax:
            from ludax_grammar import validate_game
            for attempt in range(self.max_repair_attempts):
                full_game = f"{prefix}{output}{suffix}".strip()
                is_valid, err = validate_game(full_game)
                if is_valid:
                    break

                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content":
                    f"That produces a grammar error:\n{err[:300]}\n\n"
                    f"Fix the syntax. Output ONLY the corrected replacement expression."
                })
                response = await self.async_client.messages.create(
                    model=self.model_name, max_tokens=512, temperature=0.3,
                    system=self.system_prompt, messages=messages,
                )
                output = response.content[0].text.strip()
                if output.startswith("```"):
                    output = output.split("\n", 1)[1] if "\n" in output else output[3:]
                    if output.endswith("```"): output = output[:-3]
                    output = output.strip()
                output = output.replace("\n", " ")

        return output

    async def _generate_mutations_async(self, prefix: str, suffix: str,
                                         original_section: str = "") -> typing.List[str]:
        tasks = [self._propose_critique_refine(prefix, original_section, suffix)
                 for _ in range(self.num_return_sequences)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_games = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Agentic mutation error: {result}")
                continue
            new_games.append(f"{prefix}{result}{suffix}".strip())
        return new_games

    def _generate_mutations(self, prefix: str, suffix: str) -> typing.List[str]:
        # This is called by BaseMutator.mutate, but we need the original section
        # for the agentic loop. We'll get it from the stored context.
        return self._run_async(self._generate_mutations_async(prefix, suffix, self._current_middle))

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    def mutate(self, game: ArchiveGame,
               mutation_selection_strategy: MutationSelectionStrategy,
               mutation_strategy: MutationStrategy,
               min_novelty: float = 0.05):
        """Override mutate to pass the original section to the agentic loop."""
        prefix, middle, suffix, depth = self._select_mutation_location(game, mutation_selection_strategy)
        self._current_middle = middle
        new_games = self._generate_mutations(prefix, suffix)

        if min_novelty > 0:
            novel = [g for g in new_games if self._normalized_edit_distance(g, game.game_str) >= min_novelty]
            new_games = novel

        return new_games, (prefix, middle, suffix, depth)


if _HAS_TORCH:
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
else:
    LLMMutator = None
