#!/usr/bin/env python3
"""
Game designer using Ludii syntax (which the LLM knows) + Ludax JAX backend.

The LLM writes Ludii .lud format → we transpile to Ludax → compile to JAX → evaluate.
"""

import concurrent.futures
import json
import os
import random
import time
import typing

import anthropic

os.makedirs("exp_outputs", exist_ok=True)
LOG_PATH = "exp_outputs/ludii_designer_run.log"
_log_file = None

def log(msg):
    print(msg, flush=True)
    global _log_file
    if _log_file is None:
        _log_file = open(LOG_PATH, "w")
    _log_file.write(msg + "\n")
    _log_file.flush()


LUDII_GAME_PROMPT = """You are an expert board game designer. Write a complete game in the Ludii game description language.

Ludii uses nested S-expressions. Here are 3 real games as syntax examples:

Tic-Tac-Toe:
(game "Tic-Tac-Toe" (players 2) (equipment {(board (square 3)) (piece "Disc" P1) (piece "Cross" P2)}) (rules (play (move Add (to (sites Empty)))) (end (if (is Line 3) (result Mover Win)))))

Breakthrough:
(game "Breakthrough" (players {(player N) (player S)}) (equipment {(board (square 8)) (piece "Pawn" Each (or {(move Step Forward (to if:(is Empty (to)))) (move Step (directions {FR FL}) (to if:(or (is Empty (to)) (is Enemy (who at:(to)))) (apply (remove (to)))))})) (regions P1 (sites Top)) (regions P2 (sites Bottom))}) (rules (start {(place "Pawn1" (expand (sites Bottom))) (place "Pawn2" (expand (sites Top)))}) (play (forEach Piece)) (end (if (is In (last To) (sites Mover)) (result Mover Win)))))

Yavalath:
(game "Yavalath" (players 2) (equipment {(board (rotate 90 (hex 5))) (piece "Marker" Each)}) (rules (play (move Add (to (sites Empty)))) (end {(if (is Line 4) (result Mover Win)) (if (is Line 3) (result Mover Loss))})))

Here are 5 MORE diverse examples showing different game types:

Hex (connection game):
(game "Hex" (players 2) (equipment {(board (hex Diamond 11)) (piece "Marker" Each) (regions P1 {(sites Side NE) (sites Side SW)}) (regions P2 {(sites Side NW) (sites Side SE)})}) (rules (play (move Add (to (sites Empty)))) (end (if (is Connected Mover) (result Mover Win)))))

Hasami Shogi (slide + custodial capture):
(game "Hasami Shogi" (players 2) (equipment {(board (square 9)) (piece "Fuhyo" P1 (move Slide Orthogonal (then (custodial (from (last To)) Orthogonal (between (max 1) if:(is Enemy (who at:(between))) (apply (remove (between)))) (to if:(is Friend (who at:(to)))))))) (piece "Tokin" P2 (move Slide Orthogonal (then (custodial (from (last To)) Orthogonal (between (max 1) if:(is Enemy (who at:(between))) (apply (remove (between)))) (to if:(is Friend (who at:(to))))))))}) (rules (start {(place "Fuhyo1" (sites Bottom)) (place "Tokin2" (sites Top))}) (play (forEach Piece)) (end (if (= (count Pieces Next) 1) (result Mover Win)))))

Konane (hop chain capture):
(game "Konane" (players 2) (equipment {(board (square 8)) (piece "Marker" Each)}) (rules (start {(place "Marker1" (sites Phase 1)) (place "Marker2" (sites Phase 0))}) (play (forEach Piece "Marker" (move Hop Orthogonal (between if:(is Enemy (who at:(between))) (apply (remove (between)))) (to if:(is Empty (to))) (then (if (can Move (move Hop (from (last To)) Orthogonal (between if:(is Enemy (who at:(between))) (apply (remove (between)))) (to if:(is Empty (to))))) (moveAgain)))))) (end (if (no Moves Next) (result Next Loss)))))

Key Ludii patterns:
- (piece "Name" Each (move Step/Slide/Hop ...)) — pieces define their own movement
- (play (forEach Piece)) — each piece moves according to its definition
- (move Add (to (sites Empty))) — placement on empty cells
- (move Step Forward/Orthogonal/Diagonal ...) — one-cell movement
- (move Slide Orthogonal/Diagonal ...) — slide any distance
- (move Hop ... (between if:...) (to if:...)) — jump capture
- (then ...) — effects after a move (custodial capture, moveAgain, etc.)
- (end (if (is Line N) (result Mover Win))) — N in a row wins
- (end (if (is Connected Mover) (result Mover Win))) — connect opposite edges
- (end (if (no Moves Next) (result Next Loss))) — no moves = loss
- (end (if (= (count Pieces Next) N) (result Mover Win))) — reduce to N pieces
- (start {(place "Piece1" (expand (sites Bottom)))}) — starting positions
- (start {(place "Piece1" (sites Phase 0))}) — checkerboard placement
- Boards: (square N), (hex N), (hex Diamond N), (rectangle H W)
- NO dice, NO cards, NO hidden information — deterministic games only

Design a NOVEL game that is NOT a copy of any example above. Combine mechanics in new ways.

Output ONLY the (game ...) expression. No explanation."""


THEME_PROMPT = """You are a creative board game designer. Invent a new board game concept.

Output a JSON object:
- "name": evocative game name (2-3 words)
- "theme": 2-3 sentence backstory
- "board": one of "square 5", "square 7", "square 8", "square 9", "hex 5", "hex 7", "rectangle 5 9", "rectangle 6 8"
- "win_condition": how the game ends (deterministic, no dice/cards)
- "twist": one unique rule

Your theme MUST be inspired by the seed words provided. The game engine is DETERMINISTIC — no dice, cards, hidden info, or random events.

Output ONLY the JSON. No markdown."""


def _random_seeds():
    from wonderwords import RandomWord
    r = RandomWord()
    return f"{r.word(include_categories=['adjective'])} {r.word(include_categories=['noun'])}, {r.word(include_categories=['noun'])}"


def _generate_one(i, model):
    """Generate one game: theme → Ludii code → transpile → JAX eval."""
    import sys
    sys.path.insert(0, '/Users/a12066/Documents/GitHub/ludax/src')
    from ludax.ludii_transpiler import transpile
    from ludax import LudaxEnvironment
    from ludax_fitness import evaluate_game
    import jax

    client = anthropic.Anthropic()
    t0 = time.time()

    # Step 1: Theme
    seeds = _random_seeds()
    try:
        resp = client.messages.create(
            model=model, max_tokens=512, temperature=1.0,
            system=THEME_PROMPT,
            messages=[{"role": "user", "content": f"Seed words: {seeds}"}])
        concept = json.loads(resp.content[0].text.strip().strip('`').strip())
        concept["seed_words"] = seeds
    except Exception as e:
        log(f"Game {i+1}: THEME ERROR — {e}")
        return None

    log(f"Game {i+1}: \"{concept.get('name','?')}\" (seeds: {seeds})")

    # Step 2: Generate Ludii code
    try:
        brief = (
            f"Theme: {concept['theme']}\n"
            f"Twist: {concept.get('twist','')}\n"
            f"Game name: \"{concept['name']}\"\n"
            f"Board: {concept.get('board','square 8')}\n"
            f"Win condition: {concept.get('win_condition','')}\n\n"
            f"Write this as a complete Ludii game."
        )
        resp = client.messages.create(
            model=model, max_tokens=1024, temperature=0.7,
            system=LUDII_GAME_PROMPT,
            messages=[{"role": "user", "content": brief}])
        lud = resp.content[0].text.strip()
        if lud.startswith("```"):
            lud = lud.split("\n", 1)[1] if "\n" in lud else lud[3:]
            if lud.endswith("```"): lud = lud[:-3]
            lud = lud.strip()
    except Exception as e:
        log(f"  Game {i+1}: LUD ERROR — {e}")
        return None

    # Step 3: Transpile to Ludax
    ldx = transpile(lud)
    if not ldx:
        log(f"  Game {i+1}: TRANSPILE FAIL")
        return None

    # Step 4: Compile and evaluate in JAX
    try:
        flat = ldx.replace("\n", " ")
        r = evaluate_game(flat, num_random_games=20, skip_skill_trace=True)
    except Exception as e:
        log(f"  Game {i+1}: JAX FAIL — {str(e)[:50]}")
        return None

    if not r["compilable"] or not r["playable"]:
        log(f"  Game {i+1}: UNPLAYABLE — {r.get('error','')[:50]}")
        return None

    b = max(r["balance"], 0.01)
    c = max(r["completion"], 0.01)
    d = max(r["decision_moves"], 0.01)
    fitness = round((b * c * d) ** (1/3), 3)
    ov = r.get("outcome_variance", 10)
    if ov < 5:
        fitness = round(fitness * 0.3, 3)

    elapsed = time.time() - t0
    log(f"  Game {i+1}: f={fitness:.3f} bal={r['balance']:.2f} comp={r['completion']:.2f} turns={r['mean_turns']:.0f} ({elapsed:.0f}s)")

    return {
        "concept": concept,
        "lud": lud,
        "ldx": ldx,
        "game_str": flat,
        "fitness": fitness,
        "result": r,
    }


def design_games(num_games=10, model="claude-sonnet-4-6", max_parallel=5):
    log(f"=== LUDII GAME DESIGNER ===")
    log(f"Generating {num_games} games with {model} ({max_parallel} parallel)\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(_generate_one, i, model): i for i in range(num_games)}
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            if result:
                results.append(result)

    log(f"\n{'='*60}")
    log(f"RESULTS: {len(results)}/{num_games} playable")
    log(f"{'='*60}")

    results.sort(key=lambda x: x["fitness"], reverse=True)
    for i, g in enumerate(results[:5]):
        c = g["concept"]
        r = g["result"]
        log(f"\n{'─'*60}")
        log(f"#{i+1}: \"{c['name']}\" (f={g['fitness']:.3f})")
        log(f"Theme: {c['theme']}")
        log(f"Balance={r['balance']:.2f} Completion={r['completion']:.2f} Turns={r['mean_turns']:.0f}")

    save_path = "exp_outputs/ludii_games.json"
    with open(save_path, "w") as f:
        json.dump([{"concept": g["concept"], "lud": g["lud"], "game_str": g["game_str"],
                     "fitness": g["fitness"]} for g in results], f, indent=2)
    log(f"\nSaved to {save_path}")
    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    model = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-4-6"
    design_games(num_games=n, model=model)
