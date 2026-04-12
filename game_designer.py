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
- "win_condition": what ends the game — must be achievable through board play alone (no dice, cards, or random events)
- "twist": one unique rule expressible through board geometry, piece interactions, or scoring — no dice, cards, hidden info, or external components

Your theme MUST be inspired by the random seed words provided in the user message. Interpret them creatively — they are evocative sparks, not literal requirements. The theme should MOTIVATE the mechanics.

IMPORTANT: The game engine is DETERMINISTIC. No dice, no cards, no random events, no hidden information. All mechanics must work through piece placement, piece movement, capturing, flipping, and scoring on a visible board.

Output ONLY the JSON. No markdown fences."""


GAME_ARCHETYPES = [
    ("line", "Form N pieces in a row to win",
     '(game "X" (players 2) (equipment (board (square 9)) (pieces ("token" both))) (rules (play (repeat (P1 P2) (place "token" (destination (empty))))) (end (if (line "token" 5) (mover win)) (if (full_board) (draw)))))'),

    ("territory", "Control the most pieces when the board fills",
     '(game "X" (players 2) (equipment (board (hex_rectangle 7 7)) (pieces ("stone" both))) (rules (play (repeat (P1 P2) (place "stone" (destination (empty)) (effects (capture (custodial "stone" 1 orientation:orthogonal)) (set_score mover (count (occupied mover))) (set_score opponent (count (occupied opponent))))))) (end (if (full_board) (by_score)))))'),

    ("connection", "Connect your pieces across opposite board edges",
     '(game "X" (players 2 (set_forward (P1 up) (P2 right))) (equipment (board (hex_rectangle 9 9)) (pieces ("token" both))) (rules (play (repeat (P1 P2) (place "token" (destination (empty))))) (end (if (>= (connected "token" ((edge forward) (edge backward))) 2) (mover win)))))'),

    ("race", "Move pieces to reach the opponent's side of the board",
     '(game "X" (players 2 (set_forward (P1 up) (P2 down))) (equipment (board (square 8)) (pieces ("runner" both))) (rules (start (place "runner" P1 ((row 0) (row 1))) (place "runner" P2 ((row 6) (row 7)))) (play (repeat (P1 P2) (move (step "runner" direction:forward)))) (end (if (exists (and (occupied mover) (edge forward))) (mover win)))))'),

    ("elimination", "Capture all opponent pieces to win",
     '(game "X" (players 2 (set_forward (P1 up) (P2 down))) (equipment (board (square 8)) (pieces ("warrior" both))) (rules (start (place "warrior" P1 ((row 0) (row 1))) (place "warrior" P2 ((row 6) (row 7)))) (play (repeat (P1 P2) (move (or (hop "warrior" direction:diagonal hop_over:opponent capture:true priority:0) (step "warrior" direction:diagonal priority:1))))) (end (if (no_legal_actions) (mover lose)))))'),

    ("flip_territory", "Place pieces that flip opponent pieces to your color",
     '(game "X" (players 2) (equipment (board (square 8)) (pieces ("disc" both))) (rules (start (place "disc" P1 (27 36)) (place "disc" P2 (28 35))) (play (repeat (P1 P2) (place "disc" (destination (empty)) (result (exists (custodial "disc" any))) (effects (flip (custodial "disc" any)) (set_score mover (count (occupied mover))) (set_score opponent (count (occupied opponent))))) (force_pass))) (end (if (passed both) (by_score)))))'),

    ("promotion_race", "Move pieces forward with promotion at the far edge",
     '(game "X" (players 2 (set_forward (P1 up) (P2 down))) (equipment (board (square 8)) (pieces ("pawn" both) ("king" both))) (rules (start (place "pawn" P1 ((row 0) (row 1))) (place "pawn" P2 ((row 6) (row 7)))) (play (repeat (P1 P2) (move (or (step "pawn" direction:(forward_left forward_right) priority:1) (step "king" direction:diagonal priority:1)) (effects (promote "pawn" "king" (edge forward)))))) (end (if (no_legal_actions) (mover lose)))))'),

    ("score_race", "Place pieces and score points — first to a target wins",
     '(game "X" (players 2) (equipment (board (hex_rectangle 7 7)) (pieces ("gem" both))) (rules (play (repeat (P1 P2) (place "gem" (destination (empty)) (effects (capture (custodial "gem" 1 orientation:any)) (increment_score mover 1))))) (end (if (>= (score mover) 10) (mover win)) (if (full_board) (by_score)))))'),
]

GAME_PROMPT = """You are an expert Ludax game designer. Given a game concept AND a game archetype with a reference example, output a complete, valid Ludax game.

IMPORTANT: Use the reference example as a SYNTAX GUIDE for how to structure your game. Your game should follow the same structural pattern but with DIFFERENT pieces, board, effects, and win conditions that fit your theme. Do NOT copy the example — transform it.

=== GAME SKELETON ===
(game "Name"
    (players 2 (set_forward (P1 up) (P2 down))?)
    (equipment (board BOARD) (pieces PIECE_DEFS))
    (rules (start ...)? (play PLAY_BLOCK) (end END_CONDITIONS))
)

=== BOARDS ===
(square N) | (hexagon D) | (hex_rectangle W H) | (rectangle W H)

=== PIECES ===
("name" both) — shared type | ("name" P1) — one player only
Names must be unique. Use ("token" both), NOT ("token" P1) + ("token" P2).

=== START (optional) ===
(place "piece" P1 ((row 0) (row 1)))     — row-based (square/rectangle only)
(place "piece" P1 (0 1 2 3 4))           — explicit indices (works on all boards)

=== PLAY ===
(repeat (P1 P2) ACTION) or (once_through (P1 P2) ACTION)

Actions:
  (place "p" (destination MASK) (result PRED)? (effects EFFECTS)?)
  (move (or MOVES...) (effects EFFECTS)?)

Moves:
  (step "p" direction:DIR)              — one cell
  (slide "p" direction:DIR)             — any distance
  (hop "p" direction:DIR hop_over:opponent capture:true) — jump over piece
  priority:N on moves means lower number = mandatory first

(force_pass) after action = pass when no legal moves

=== EFFECTS (inside place or move only) ===
(capture MASK)                          — remove pieces matching mask
(flip MASK)                             — change ownership of matched pieces
(promote "from" "to" MASK)              — upgrade piece type at mask
(extra_turn PLAYER same_piece:true?)    — take another turn
(set_score PLAYER FUNCTION)             — update score
(increment_score PLAYER FUNCTION)       — add to score
(if PREDICATE EFFECT)                   — conditional effect

=== END CONDITIONS ===
(if (line "p" N) (mover win|lose))      — N in a row
(if (>= (connected "p" ((edge A) (edge B))) 2) (mover win)) — span board edges
(if (no_legal_actions) (mover win|lose))
(if (full_board) (draw|by_score))
(if (captured_all "p") (mover win))
(if (exists MASK) (mover win))          — piece reaches a zone

=== MASKS ===
(empty) | (occupied mover|opponent) | (edge forward|backward|left|right)
(row N) | (column N) | (corners) | (center)
(and M M) | (or M M) | (not M) | (adjacent M direction:DIR)
(custodial "p" N orientation:orthogonal|diagonal|any)

=== DIRECTIONS ===
orthogonal | diagonal | any | forward | backward
forward_left | forward_right | up | down | left | right
(set_forward required for forward/backward: (players 2 (set_forward (P1 up) (P2 down))))

=== FUNCTIONS ===
(count MASK) | (score PLAYER) | (add F F) | (multiply F F) | (subtract F F)

=== WILL BREAK YOUR GAME ===
- (connected) triggers on ONE piece — only use with ((edge X) (edge Y)) to require full-board span
- (line N) with N < 4 ends in 1-3 turns
- (by_score) without (set_score) in effects = both scores are 0
- (row N) fails on hexagon boards — use explicit indices
- (custodial N) with N > 1 almost never triggers
- Ludax has NO dice, NO cards, NO hidden info, NO random events

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


def _pick_archetype(client: anthropic.Anthropic, concept: dict, model: str) -> typing.Tuple[str, str, str]:
    """Ask the LLM to pick the best archetype for this theme. Returns (name, description, example)."""
    archetype_list = "\n".join(f"- {name}: {desc}" for name, desc, _ in GAME_ARCHETYPES)
    resp = client.messages.create(
        model=model, max_tokens=50, temperature=0.3,
        system="Pick the game archetype that best fits this theme. Output ONLY the archetype name, nothing else.",
        messages=[{"role": "user", "content":
            f"Theme: {concept['theme']}\nTwist: {concept.get('twist','')}\n\nArchetypes:\n{archetype_list}"}],
    )
    chosen = resp.content[0].text.strip().lower().replace(" ", "_")
    for name, desc, example in GAME_ARCHETYPES:
        if name in chosen or chosen in name:
            return name, desc, example
    # Fallback: random
    return random.choice(GAME_ARCHETYPES)


def generate_game(client: anthropic.Anthropic, concept: dict,
                  model: str = "claude-sonnet-4-6", max_repairs: int = 3) -> typing.Optional[str]:
    """Generate a complete Ludax game from a concept. Returns game string or None."""
    # Pick archetype based on theme
    arch_name, arch_desc, arch_example = _pick_archetype(client, concept, model)
    concept["archetype"] = arch_name

    brief = (
        f"Theme: {concept['theme']}\n"
        f"Twist: {concept['twist']}\n\n"
        f"Archetype: {arch_name} — {arch_desc}\n"
        f"Reference example (use as SYNTAX GUIDE only, do NOT copy):\n{arch_example}\n\n"
        f"Game name: \"{concept['name']}\"\n"
        f"Board: {concept['board']}\n"
        f"Win condition: {concept['win_condition']}\n\n"
        f"Design a complete Ludax game that brings this theme to life. Transform the archetype — don't clone it."
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
        log(f"  Board: {concept.get('board', '?')}")
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

        log(f"  Archetype: {concept.get('archetype', '?')}")

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
