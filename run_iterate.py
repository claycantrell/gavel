#!/usr/bin/env python3
"""
Iterate on the best novel game: mix single-point mutations, multi-edit,
and agentic refinement to improve fitness.

Usage: python3 -u run_iterate.py
Logs to: exp_outputs/iterate_run.log
"""
import json
import os
import random
import time

import anthropic

from mutators import (AnthropicMutator, MultiEditMutator,
                      _LUDEME_WEIGHTS, _SKIP_LUDEMES, LUDAX_SYSTEM_PROMPT)
from ludii_parser import extract_parentheticals
from ludax_grammar import validate_game
from ludax_fitness import evaluate_game
from config import ArchiveGame, MutationSelectionStrategy, MutationStrategy

# --- Logging ---
os.makedirs("exp_outputs", exist_ok=True)
LOG_PATH = "exp_outputs/iterate_run.log"
_log_file = open(LOG_PATH, "w")

def log(msg):
    print(msg, flush=True)
    _log_file.write(msg + "\n")
    _log_file.flush()


def fitness(r):
    if not r["compilable"] or not r["playable"]:
        return -1
    b = max(r["balance"], 0.01)
    c = max(r["completion"], 0.01)
    d = max(r["decision_moves"], 0.01)
    # Engagement: penalize dead mechanics, don't reward more engagement
    # If the game has effects that never fire (<5% of turns), apply penalty
    # Otherwise fitness is purely balance * completion * decisions
    base = round((b * c * d) ** (1/3), 3)
    mf = r.get("mechanic_frequency", 1.0)
    if mf < 0.05 and r.get("score_volatility", 0) > 0:
        # Game has scoring/effects but they're inactive — penalize
        return round(base * 0.3, 3)
    return base


def pick_target(game_str):
    ps = [p for p in extract_parentheticals(game_str) if p[0] != ""]
    cands, ws = [], []
    for p in ps:
        l = p[1][1:].lstrip().split(None, 1)[0] if p[1][1:].strip() else ""
        if l in _SKIP_LUDEMES or l.startswith('"') or l[0:1].isdigit():
            continue
        cands.append(p)
        ws.append(_LUDEME_WEIGHTS.get(l, 1))
    if not cands:
        return random.choice(ps)
    return cands[random.choices(range(len(cands)), weights=ws, k=1)[0]]


# --- Load best game ---
with open("exp_outputs/novel_games.json") as f:
    all_games = json.load(f)

best = max(all_games, key=lambda g: g["fitness"])
seed_name = best["concept"]["name"]
seed_game = best["game_str"]
seed_fitness = best["fitness"]

log(f"=== ITERATING ON: \"{seed_name}\" (f={seed_fitness}) ===")
log(f"Theme: {best['concept']['theme'][:80]}")

# Run MCTS skill trace on the parent ONCE to confirm strategic depth
from ludax_fitness import compute_skill_trace
log(f"Computing MCTS skill trace on parent...")
try:
    parent_skill = compute_skill_trace(seed_game, num_games=6, mcts_sims=30)
    log(f"Parent skill trace: {parent_skill:.2f} (MCTS vs random win rate)")
    if parent_skill < 0.55:
        log(f"WARNING: Parent has low strategic depth — MCTS barely beats random")
except Exception as e:
    parent_skill = -1
    log(f"Skill trace failed: {e}")
log(f"")

# --- Setup ---
random.seed(42)
client = anthropic.Anthropic()

single_mut = AnthropicMutator(
    model_name="claude-sonnet-4-6", num_return_sequences=1,
    temperature=1.0, use_ludax=True)
multi_mut = MultiEditMutator(
    model_name="claude-sonnet-4-6", num_return_sequences=2, temperature=1.0)

archive = {seed_name: {"game": seed_game, "fitness": seed_fitness, "parent": None, "method": "seed"}}
evolved = []
t_start = time.time()

NUM_GENS = 8
MUTATIONS_PER_GEN = 4  # mix of strategies

for gen in range(NUM_GENS):
    log(f"=== GEN {gen+1} ===")

    # Pick best game(s) from archive as parents
    ranked = sorted(archive.keys(), key=lambda n: archive[n]["fitness"], reverse=True)
    parents = ranked[:2]  # top 2

    for pname in parents:
        pg = archive[pname]["game"]
        pf = archive[pname]["fitness"]

        for mut_i in range(MUTATIONS_PER_GEN):
            # Rotate strategies: 50% single-point, 25% multi-edit, 25% single-point with different target
            roll = random.random()

            if roll < 0.5:
                # Single-point mutation with anti-identity
                method = "single"
                prefix, middle, suffix, _ = pick_target(pg)
                ludeme = middle[1:].split(None, 1)[0] if middle[1:].strip() else "?"
                try:
                    resp = client.messages.create(
                        model="claude-sonnet-4-6", max_tokens=256, temperature=1.0,
                        system=LUDAX_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content":
                            f"Original section:\n{middle}\n\nGenerate a DIFFERENT replacement. Be creative.\n\n{prefix}<BLANK>{suffix}"}])
                    repl = resp.content[0].text.strip().replace("\n", " ")
                    if repl.startswith("```"):
                        repl = repl.split("\n", 1)[1] if "\n" in repl else repl[3:]
                        if repl.endswith("```"): repl = repl[:-3]
                        repl = repl.strip()
                    candidates = [f"{prefix}{repl}{suffix}".strip()]
                except:
                    continue

            else:
                # Multi-edit mutation
                method = "multi"
                parent_ag = ArchiveGame(
                    game_str=pg, fitness_score=pf, evaluation={},
                    lineage=[], generation=gen, original_game_name=pname, epoch=gen)
                try:
                    candidates, _ = multi_mut.mutate(
                        parent_ag, MutationSelectionStrategy.SEMANTIC, MutationStrategy.STANDARD)
                except:
                    candidates = []

            for ci, mutated in enumerate(candidates):
                # Identity check
                if "".join(mutated.split()) == "".join(pg.split()):
                    continue

                ok, err = validate_game(mutated)

                # Repair for single-point
                if not ok and method == "single":
                    try:
                        resp2 = client.messages.create(
                            model="claude-sonnet-4-6", max_tokens=256, temperature=0.3,
                            system=LUDAX_SYSTEM_PROMPT,
                            messages=[
                                {"role": "user", "content": f"Fix <BLANK>:\n\n{prefix}<BLANK>{suffix}"},
                                {"role": "assistant", "content": repl},
                                {"role": "user", "content": f"Grammar error:\n{err[:200]}\n\nFix. Output ONLY the expression."}])
                        repl = resp2.content[0].text.strip().replace("\n", " ")
                        if repl.startswith("```"):
                            repl = repl.split("\n", 1)[1] if "\n" in repl else repl[3:]
                            if repl.endswith("```"): repl = repl[:-3]
                            repl = repl.strip()
                        mutated = f"{prefix}{repl}{suffix}".strip()
                        ok, _ = validate_game(mutated)
                    except:
                        pass

                if not ok:
                    log(f"  {pname}>g{gen}m{mut_i}: GRAMMAR FAIL [{method}]")
                    continue

                try:
                    r = evaluate_game(mutated, num_random_games=30, skip_skill_trace=True)
                    f = fitness(r)
                except:
                    log(f"  {pname}>g{gen}m{mut_i}: EVAL FAIL [{method}]")
                    continue

                if f <= 0:
                    log(f"  {pname}>g{gen}m{mut_i}: UNPLAYABLE [{method}]")
                    continue

                cname = f"g{gen}_{pname[:15]}_{method}{mut_i}"
                d = "+" if f > pf else "-" if f < pf else "="
                archive[cname] = {"game": mutated, "fitness": f, "result": r, "parent": pname, "method": method}
                evolved.append(cname)
                log(f"  {cname}: f={f:.3f}{d} bal={r['balance']:.2f} comp={r['completion']:.2f} turns={r['mean_turns']:.0f} [{method}]")

elapsed = time.time() - t_start
log(f"\nEvolution: {elapsed:.0f}s, {len(evolved)} novel variants\n")

# Show top results
log("=" * 60)
log("TOP VARIANTS")
log("=" * 60)
ranked = sorted(evolved, key=lambda n: archive[n]["fitness"], reverse=True)
from difflib import SequenceMatcher
for i, name in enumerate(ranked[:5]):
    g = archive[name]
    r = g.get("result", {})
    sim = SequenceMatcher(None, "".join(g["game"].split()), "".join(seed_game.split())).ratio()
    log(f"\n{'─'*60}")
    log(f"#{i+1}: {name} (f={g['fitness']:.3f}, method={g['method']}, parent={g['parent']})")
    log(f"Balance={r.get('balance',0):.2f} Completion={r.get('completion',0):.2f} Turns={r.get('mean_turns',0):.0f}")
    log(f"Similarity to seed: {sim:.2f}")
    game = g["game"]
    for kw in ["(players", "(equipment", "(rules", "(start", "(play", "(end", "(effects"]:
        game = game.replace(kw, "\n    " + kw)
    log(game[:600])

# Save
with open("exp_outputs/iterated_games.json", "w") as f:
    json.dump([{"name": n, "game_str": archive[n]["game"], "fitness": archive[n]["fitness"],
                "parent": archive[n]["parent"], "method": archive[n]["method"]}
               for n in evolved], f, indent=2)
log(f"\nSaved to exp_outputs/iterated_games.json")
