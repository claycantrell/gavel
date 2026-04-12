#!/usr/bin/env python3
"""
Diagnose-and-fix iteration: play the game, identify what's broken,
ask the LLM to fix the specific problem.

Usage: python3 -u run_diagnose_iterate.py
"""
import json
import os
import random
import time
import typing

import anthropic

from ludax_grammar import validate_game
from ludax_fitness import evaluate_game

os.makedirs("exp_outputs", exist_ok=True)
LOG_PATH = "exp_outputs/diagnose_run.log"
_log_file = open(LOG_PATH, "w")

def log(msg):
    print(msg, flush=True)
    _log_file.write(msg + "\n")
    _log_file.flush()


GAME_PROMPT_SHORT = """You are a Ludax game designer fixing a broken game. Output ONLY the complete corrected (game ...) expression."""

MULTI_FIX_PROMPT = """You are a Ludax game designer fixing specific problems in a game.

Plan 2-4 COORDINATED find-and-replace edits that fix the diagnosed problems together.
Each edit should address part of the problem. Changes must be compatible with each other.

Respond with a JSON array of edits:
[{"find": "exact text to find", "replace": "replacement text"}]

The "find" strings must EXACTLY match text in the current game. Copy them precisely.
The "replace" strings must be valid Ludax syntax."""


def diagnose(r: dict) -> list:
    """Look at evaluation results and return a list of specific problems."""
    problems = []

    if r.get("mean_turns", 0) < 5:
        problems.append("INSTANT_END: Games end in fewer than 5 turns. The win condition triggers too early — increase the line length, change the win condition, or adjust start positions so pieces aren't already in winning configurations.")

    if r.get("balance", 0) < 0.3:
        wins = r.get("wins", [0, 0, 0])
        if len(wins) >= 3:
            if wins[1] > wins[2] * 3:
                problems.append(f"P1_DOMINATES: Player 1 wins {wins[1]} vs Player 2 wins {wins[2]}. First player has a huge advantage. Add a catch-up mechanic, swap rule, or asymmetric scoring to compensate.")
            elif wins[2] > wins[1] * 3:
                problems.append(f"P2_DOMINATES: Player 2 wins {wins[2]} vs Player 1 wins {wins[1]}. Second player has a huge advantage. Check if the board geometry or end condition favors the second mover.")
            else:
                problems.append(f"DRAWS_DOMINANT: {wins[0]} draws out of {sum(wins)} games. The game draws too often — make the end condition more decisive.")

    if r.get("completion", 1) < 0.7:
        problems.append("GAMES_DONT_END: Only {:.0f}% of games reach a conclusion. Add a fallback end condition like (if (full_board) (by_score)) or (if (no_legal_actions) ...).".format(r["completion"] * 100))

    if r.get("outcome_variance", 10) < 5:
        problems.append("DECISIONS_DONT_MATTER: The final score barely varies between games (std={:.1f}). Different moves should lead to different outcomes. Add capture/flip effects that change the board state dramatically.".format(r.get("outcome_variance", 0)))

    mf = r.get("mechanic_frequency", 1)
    if mf < 0.05 and r.get("score_volatility", 0) > 0:
        problems.append("DEAD_MECHANIC: The game defines capture/flip effects but they fire on less than 5% of turns. Lower the custodial distance to 1, or change the effect trigger to be more common.")

    if not problems:
        problems.append("MINOR_TUNING: No major issues found. Try adjusting the board size, win threshold, or adding a secondary win condition for more variety.")

    return problems


def fix_game(client, game_str: str, problems: list, concept: dict,
             model: str = "claude-sonnet-4-6", attempt_num: int = 0) -> typing.Optional[str]:
    """Fix specific problems via coordinated multi-edit. Escalates on repeated failure."""
    import re

    problem_text = "\n".join(f"- {p}" for p in problems)

    if attempt_num < 3:
        # First 3 attempts: coordinated multi-edit
        prompt = (
            f"This Ludax game has problems:\n\n"
            f"PROBLEMS:\n{problem_text}\n\n"
            f"CURRENT GAME:\n{game_str}\n\n"
            f"Plan 2-4 coordinated find-and-replace edits to fix these problems. "
            f"Each edit should target a specific part of the game that contributes to the problem."
        )
        messages = [{"role": "user", "content": prompt}]
        resp = client.messages.create(
            model=model, max_tokens=1024, temperature=0.5,
            system=MULTI_FIX_PROMPT, messages=messages,
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"): raw = raw[:-3]
            raw = raw.strip()

        try:
            edits = json.loads(raw)
        except json.JSONDecodeError:
            log(f"    Multi-edit returned invalid JSON, falling back to full rewrite")
            edits = []

        if edits:
            result = game_str
            applied = 0
            for edit in edits:
                find = edit.get("find", "")
                replace = edit.get("replace", "")
                if find and find in result:
                    candidate = result.replace(find, replace, 1)
                    ok, _ = validate_game(candidate.replace("\n", " "))
                    if ok:
                        result = candidate
                        applied += 1
            result = result.replace("\n", " ").strip()
            if applied > 0:
                ok, _ = validate_game(result)
                if ok:
                    log(f"    Applied {applied}/{len(edits)} edits")
                    return result
            log(f"    Multi-edit: {applied} edits applied but result invalid")

    # Fallback / escalation: full rewrite with radical change instruction
    escalation = ""
    if attempt_num >= 3:
        escalation = (
            "Previous incremental fixes have NOT worked. Make a RADICAL change: "
            "swap the win condition entirely, change the board size, add a completely "
            "different mechanic, or restructure the game from scratch while keeping the theme. "
        )

    prompt = (
        f"This Ludax game has problems:\n\n"
        f"PROBLEMS:\n{problem_text}\n\n"
        f"CURRENT GAME:\n{game_str}\n\n"
        f"{escalation}"
        f"Output the complete fixed game. Keep the theme but fix the gameplay."
    )

    messages = [{"role": "user", "content": prompt}]
    resp = client.messages.create(
        model=model, max_tokens=1024, temperature=0.7 if attempt_num >= 3 else 0.5,
        system=GAME_PROMPT_SHORT, messages=messages,
    )
    output = resp.content[0].text.strip()
    if output.startswith("```"):
        output = output.split("\n", 1)[1] if "\n" in output else output[3:]
        if output.endswith("```"): output = output[:-3]
        output = output.strip()

    code_lines = [l for l in output.split("\n") if not l.strip().startswith(";;")]
    result = " ".join(code_lines).strip()

    # Auto-fix duplicate piece names
    piece_defs = re.findall(r'\("(\w+)"\s+(P1|P2|both)\)', result)
    seen = {}
    for name, owner in piece_defs:
        if name in seen and seen[name] != owner and seen[name] != "both":
            result = re.sub(rf'\("{name}"\s+P[12]\)', f'("{name}" both)', result)
        seen[name] = owner

    ok, err = validate_game(result)
    if ok:
        return result

    # One repair attempt
    messages.append({"role": "assistant", "content": result})
    messages.append({"role": "user", "content": f"Grammar error:\n{err}\n\nFix. Output ONLY the corrected game."})
    resp = client.messages.create(model=model, max_tokens=1024, temperature=0.3,
        system=GAME_PROMPT_SHORT, messages=messages)
    output = resp.content[0].text.strip()
    if output.startswith("```"):
        output = output.split("\n", 1)[1] if "\n" in output else output[3:]
        if output.endswith("```"): output = output[:-3]
        output = output.strip()
    code_lines = [l for l in output.split("\n") if not l.strip().startswith(";;")]
    result = " ".join(code_lines).strip()

    ok, _ = validate_game(result)
    return result if ok else None


def fitness(r):
    if not r["compilable"] or not r["playable"]:
        return -1
    b = max(r["balance"], 0.01)
    c = max(r["completion"], 0.01)
    d = max(r["decision_moves"], 0.01)
    base = round((b * c * d) ** (1/3), 3)
    mf = r.get("mechanic_frequency", 1.0)
    if mf < 0.05 and r.get("score_volatility", 0) > 0:
        base = round(base * 0.3, 3)
    ov = r.get("outcome_variance", 10)
    if ov < 5:
        base = round(base * 0.3, 3)
    return base


# --- Main ---
with open("exp_outputs/novel_games.json") as f:
    data = json.load(f)
    all_games = data["games"] if isinstance(data, dict) else data

best = max(all_games, key=lambda g: g["fitness"])
seed_name = best["concept"]["name"]
seed_game = best["game_str"]
seed_concept = best["concept"]

log(f"=== DIAGNOSE & FIX: \"{seed_name}\" ===")
log(f"Theme: {seed_concept['theme'][:80]}")

client = anthropic.Anthropic()
current_game = seed_game
current_concept = seed_concept
t_start = time.time()

for iteration in range(6):
    log(f"\n--- Iteration {iteration + 1} ---")

    # Evaluate
    r = evaluate_game(current_game, num_random_games=30, skip_skill_trace=True)
    f = fitness(r)
    log(f"Fitness: {f:.3f} | Balance: {r.get('balance',0):.2f} | Completion: {r.get('completion',0):.2f} | Turns: {r.get('mean_turns',0):.0f}")
    log(f"Outcome variance: {r.get('outcome_variance',0):.1f} | Mechanic freq: {r.get('mechanic_frequency',0):.2f} | Wins: {r.get('wins', [])}")

    # Diagnose
    problems = diagnose(r)
    log(f"Diagnosis:")
    for p in problems:
        log(f"  {p[:100]}")

    if f >= 0.9 and "MINOR_TUNING" in problems[0]:
        log(f"Game is good enough (f={f:.3f}). Stopping.")
        break

    # Fix
    log(f"Asking LLM to fix (attempt {iteration + 1})...")
    fixed = fix_game(client, current_game, problems, current_concept, attempt_num=iteration)
    if fixed is None:
        log(f"Fix failed (grammar). Keeping current version.")
        continue

    # Evaluate fix
    r_new = evaluate_game(fixed, num_random_games=30, skip_skill_trace=True)
    f_new = fitness(r_new)

    if f_new > f:
        log(f"IMPROVED: {f:.3f} → {f_new:.3f}")
        current_game = fixed
    else:
        log(f"NO IMPROVEMENT: {f:.3f} → {f_new:.3f} (keeping old)")

elapsed = time.time() - t_start
log(f"\n=== FINAL RESULT ({elapsed:.0f}s) ===")
r_final = evaluate_game(current_game, num_random_games=30, skip_skill_trace=True)
f_final = fitness(r_final)
log(f"Fitness: {f_final:.3f} | Balance: {r_final.get('balance',0):.2f} | Turns: {r_final.get('mean_turns',0):.0f}")

game = current_game
for kw in ["(players", "(equipment", "(rules", "(start", "(play", "(end", "(effects"]:
    game = game.replace(kw, "\n    " + kw)
log(game)

with open("exp_outputs/diagnosed_game.json", "w") as f:
    json.dump({"game_str": current_game, "fitness": f_final, "concept": current_concept}, f, indent=2)
log(f"Saved to exp_outputs/diagnosed_game.json")
