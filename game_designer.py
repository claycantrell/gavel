#!/usr/bin/env python3
"""
Novel game designer: generates complete board games from thematic briefs.

Instead of mutating existing games, this module asks Claude to:
1. Invent a theme/backstory for a new game
2. Design mechanics that serve that theme
3. Output a complete Ludax .ldx game

The theme grounds the design decisions — "volcanic islands sinking into the sea"
leads to different mechanics than "two generals besieging a fortress."
"""

import json
import os
import random
import time
import typing

import anthropic

from ludax_grammar import validate_game
from ludax_fitness import evaluate_game

LOG_PATH = "exp_outputs/designer_run.log"

_log_file = None

def log(msg):
    print(msg, flush=True)
    global _log_file
    if _log_file is None:
        os.makedirs("exp_outputs", exist_ok=True)
        _log_file = open(LOG_PATH, "w")
    _log_file.write(msg + "\n")
    _log_file.flush()


THEME_PROMPT = """You are a creative board game designer. Invent a new board game concept.

Output a JSON object with these fields:
- "name": a evocative game name (2-3 words)
- "theme": a 2-3 sentence backstory/theme (what's the world? who are the players? what's at stake?)
- "board": one of: "square 5", "square 7", "square 8", "square 9", "hexagon 7", "hexagon 9", "hex_rectangle 7 7", "hex_rectangle 9 9", "rectangle 6 8"
- "mechanic": the core player action — one of: "placement" (drop pieces on empty cells), "movement" (slide/step/hop existing pieces), "placement_with_capture" (place + flip or custodial capture)
- "win_condition": what ends the game — must be achievable through board play alone (no dice, cards, or random events)
- "twist": one unique rule expressible through board geometry, capture/flip patterns, or scoring — no dice, cards, hidden info, or external components

Your theme MUST be inspired by the random seed words provided in the user message. Interpret them creatively — they are evocative sparks, not literal requirements. The theme should MOTIVATE the mechanics.

IMPORTANT: The game engine is DETERMINISTIC. No dice, no cards, no random events, no hidden information. All mechanics must work through piece placement, piece movement, capturing, flipping, and scoring on a visible board.

Output ONLY the JSON. No markdown fences."""


GAME_PROMPT = """You are an expert Ludax game designer. Given a game concept, output a complete, valid Ludax game.

=== LUDAX STRUCTURE ===
(game "Name"
    (players 2 (set_forward (P1 up) (P2 down))?)
    (equipment
        (board (BOARD_TYPE))
        (pieces ("name" P1|P2|both) ...)
    )
    (rules
        (start (place "piece" PLAYER (INDICES))...)?
        (play
            (repeat (P1 P2)
                ;; PLACEMENT: (place "piece" (destination MASK) (result PRED)? (effects ...)?)
                ;; MOVEMENT: (move (or (step/slide/hop ...) ...) (effects ...)?)
            )
        )
        (end
            (if PREDICATE (mover win|lose))
            ...
        )
    )
)

=== KEY SYNTAX ===
- Effects go INSIDE (place ...) or (move ...): (effects (capture ...) (promote ...) (flip ...) (set_score ...) (extra_turn ...))
- Custodial capture: (capture (custodial "piece" 1 orientation:orthogonal|diagonal|any))
- Flip: (flip (custodial "piece" any))
- Line win: (if (line "piece" N) (mover win))
- Connection: (if (>= (connected "piece" ((edge forward) (edge backward))) 2) (mover win))
- No moves: (if (no_legal_actions) (mover win|lose))
- Score: (if (full_board) (by_score)) with (set_score mover (count (occupied mover)))
- Elimination: (if (captured_all "piece") (mover win))
- Directions: orthogonal, diagonal, any, forward, backward, forward_left, forward_right
- Boards: (square N), (hexagon D), (hex_rectangle W H), (rectangle W H)
- set_forward required for forward/backward directions: (players 2 (set_forward (P1 up) (P2 down)))
- Start positions: (place "piece" P1 ((row 0) (row 1))) or (place "piece" P1 (INDEX INDEX ...))
- Piece names in rules MUST match equipment. Parentheses MUST balance.
- Movement ALWAYS needs the piece name: (step "pawn" direction:any), NOT (step "any")
- (hop "piece" direction:DIR hop_over:opponent capture:true) for jump captures
- (slide "piece" direction:DIR) for long-range movement
- Start row indices must exist on the board (e.g., hexagon 9 has rows 0-16)

=== MISTAKES THAT WILL BREAK YOUR GAME ===
- (connected ...) triggers on a SINGLE piece — do not use as a win condition unless
  you require connecting across the full board: (>= (connected "p" ((edge X) (edge Y))) 2)
- (line N) with N < 4 ends the game in 1-3 turns on most boards
- If end uses (by_score), you MUST have (set_score ...) in effects, else both scores are 0
- Each piece name must be unique — ("token" both), NOT ("token" P1) + ("token" P2)
- (row N) can fail on hexagon boards — use explicit indices for hex start positions
- (custodial N) with N > 1 almost never triggers — use N=1 for active captures
- Ludax has NO dice, NO cards, NO hidden information, NO random events

Output ONLY the (game ...) expression. No explanation."""


def _random_seed_words() -> str:
    """Generate evocative random word pairs as theme seeds."""
    from wonderwords import RandomWord
    r = RandomWord()
    adj = r.word(include_categories=["adjective"])
    noun1 = r.word(include_categories=["noun"])
    noun2 = r.word(include_categories=["noun"])
    return f"{adj} {noun1}, {noun2}"


def generate_theme(client: anthropic.Anthropic, model: str = "claude-sonnet-4-6",
                   seed_words: typing.Optional[str] = None) -> dict:
    """Generate a random game theme/concept from random seed words."""
    if seed_words is None:
        seed_words = _random_seed_words()
    resp = client.messages.create(
        model=model, max_tokens=512, temperature=1.0,
        system=THEME_PROMPT,
        messages=[{"role": "user", "content":
            f"Invent a new board game concept. Your creative seed words are: {seed_words}"}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"): raw = raw[:-3]
        raw = raw.strip()
    result = json.loads(raw)
    result["seed_words"] = seed_words
    return result


def generate_game(client: anthropic.Anthropic, concept: dict,
                  model: str = "claude-sonnet-4-6", max_repairs: int = 3) -> typing.Optional[str]:
    """Generate a complete Ludax game from a concept. Returns game string or None."""
    brief = (
        f"Game: \"{concept['name']}\"\n"
        f"Theme: {concept['theme']}\n"
        f"Board: {concept['board']}\n"
        f"Core mechanic: {concept['mechanic']}\n"
        f"Win condition: {concept['win_condition']}\n"
        f"Twist: {concept['twist']}\n\n"
        f"Design a complete Ludax game that brings this theme to life through its mechanics."
    )

    messages = [{"role": "user", "content": brief}]

    resp = client.messages.create(
        model=model, max_tokens=1024, temperature=0.7,
        system=GAME_PROMPT, messages=messages,
    )
    output = resp.content[0].text.strip()
    if output.startswith("```"):
        output = output.split("\n", 1)[1] if "\n" in output else output[3:]
        if output.endswith("```"): output = output[:-3]
        output = output.strip()
    game_str = output.replace("\n", " ")

    # Validate + repair loop
    for attempt in range(max_repairs):
        is_valid, err = validate_game(game_str)
        if is_valid:
            return game_str

        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content":
            f"Grammar error:\n{err}\n\nFix the syntax. Output ONLY the corrected game."
        })
        resp = client.messages.create(
            model=model, max_tokens=1024, temperature=0.3,
            system=GAME_PROMPT, messages=messages,
        )
        output = resp.content[0].text.strip()
        if output.startswith("```"):
            output = output.split("\n", 1)[1] if "\n" in output else output[3:]
            if output.endswith("```"): output = output[:-3]
            output = output.strip()
        game_str = output.replace("\n", " ")

    return None  # All repairs failed


def design_games(num_games: int = 10, model: str = "claude-sonnet-4-6"):
    """Generate and evaluate novel games from scratch."""
    client = anthropic.Anthropic()
    results = []

    log(f"=== NOVEL GAME DESIGNER ===")
    log(f"Generating {num_games} games with {model}\n")

    for i in range(num_games):
        t0 = time.time()

        # Step 1: Theme from random seed words
        try:
            concept = generate_theme(client, model)
        except Exception as e:
            log(f"Game {i+1}: THEME ERROR — {e}")
            continue

        log(f"Game {i+1}: \"{concept.get('name', '?')}\" (seeds: {concept.get('seed_words', '?')})")
        log(f"  Theme: {concept.get('theme', '?')[:80]}")
        log(f"  Board: {concept.get('board', '?')} | Mechanic: {concept.get('mechanic', '?')}")
        log(f"  Win: {concept.get('win_condition', '?')[:60]}")
        log(f"  Twist: {concept.get('twist', '?')[:60]}")

        # Step 2: Generate game
        try:
            game_str = generate_game(client, concept, model)
        except Exception as e:
            log(f"  GENERATION ERROR — {e}")
            continue

        if game_str is None:
            log(f"  GRAMMAR FAILED after repairs")
            continue

        # Step 3: Evaluate
        try:
            r = evaluate_game(game_str, num_random_games=30, skip_skill_trace=True)
        except Exception as e:
            log(f"  EVAL ERROR — {e}")
            continue

        if not r["compilable"] or not r["playable"]:
            log(f"  UNPLAYABLE — {r.get('error', '')[:60]}")
            continue

        b = max(r["balance"], 0.01)
        c = max(r["completion"], 0.01)
        d = max(r["decision_moves"], 0.01)
        fitness = round((b * c * d) ** (1/3), 3)
        # Penalize dead mechanics (effects defined but never fire)
        mf = r.get("mechanic_frequency", 1.0)
        if mf < 0.05 and r.get("score_volatility", 0) > 0:
            fitness = round(fitness * 0.3, 3)
        # Penalize games where decisions don't matter
        ov = r.get("outcome_variance", 10)
        if ov < 5:
            fitness = round(fitness * 0.3, 3)

        elapsed = time.time() - t0
        log(f"  PLAYABLE! f={fitness:.3f} bal={r['balance']:.2f} comp={r['completion']:.2f} turns={r['mean_turns']:.0f} ({elapsed:.0f}s)")

        results.append({
            "concept": concept,
            "game_str": game_str,
            "fitness": fitness,
            "result": r,
        })

    # Show best games
    log(f"\n{'='*60}")
    log(f"RESULTS: {len(results)}/{num_games} playable games")
    log(f"{'='*60}")

    results.sort(key=lambda x: x["fitness"], reverse=True)
    for i, g in enumerate(results[:5]):
        c = g["concept"]
        r = g["result"]
        log(f"\n{'─'*60}")
        log(f"#{i+1}: \"{c['name']}\" (f={g['fitness']:.3f})")
        log(f"Theme: {c['theme']}")
        log(f"Twist: {c['twist']}")
        log(f"Balance={r['balance']:.2f} Completion={r['completion']:.2f} Turns={r['mean_turns']:.0f} Wins={r['wins']}")

        game = g["game_str"]
        for kw in ["(players", "(equipment", "(rules", "(start", "(play", "(end", "(effects"]:
            game = game.replace(kw, "\n    " + kw)
        log(game)

    # Save all results
    save_path = "exp_outputs/novel_games.json"
    with open(save_path, "w") as f:
        json.dump([{"concept": g["concept"], "game_str": g["game_str"],
                     "fitness": g["fitness"]} for g in results], f, indent=2)
    log(f"\nSaved to {save_path}")

    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    model = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-4-6"
    design_games(num_games=n, model=model)
