#!/usr/bin/env python3
"""
Generate novel board games via evolution.
Run: python3 -u run_evolution.py
Logs to: exp_outputs/evolution_run.log
"""
import os, time, random, sys

# Log everything to file AND stdout
os.makedirs("exp_outputs", exist_ok=True)
LOG_PATH = "exp_outputs/evolution_run.log"
_log_file = open(LOG_PATH, "w")

def log(msg):
    print(msg)
    _log_file.write(msg + "\n")
    _log_file.flush()

import anthropic
from mutators import LUDAX_SYSTEM_PROMPT, _LUDEME_WEIGHTS, _SKIP_LUDEMES
from ludii_parser import extract_parentheticals
from ludax_grammar import validate_game
from ludax_fitness import evaluate_game, compute_skill_trace
from difflib import SequenceMatcher

random.seed(99)
client = anthropic.Anthropic()

import ludax
games_dir = os.path.join(os.path.dirname(ludax.__file__), "games")
seeds = {}
for f in sorted(os.listdir(games_dir)):
    if f.endswith(".ldx") and f not in ("test.ldx", "gridworld.ldx"):
        with open(os.path.join(games_dir, f)) as fh:
            seeds[f.replace(".ldx", "")] = fh.read().replace("\n", " ").strip()


def fitness(r):
    if not r["compilable"] or not r["playable"]:
        return -1
    b = max(r["balance"], 0.01)
    c = max(r["completion"], 0.01)
    d = max(r["decision_moves"], 0.01)
    v = min(max(r.get("score_volatility", 0), 0.01), 1.0)
    return round((b * c * d * v) ** 0.25, 3)


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


# --- Seed ---
print("=== SEEDING ===")
use = ["yavalath", "hex", "gomoku", "hop_through", "reversi"]
archive = {}
for name in use:
    r = evaluate_game(seeds[name], num_random_games=20, skip_skill_trace=True)
    f = fitness(r)
    if f > 0:
        archive[name] = {"game": seeds[name], "fitness": f, "result": r, "parent": None}
        log(f"  {name}: f={f:.3f}")
log(f"Seeded {len(archive)} games\n")

# --- Evolve ---
evolved = []
t0 = time.time()

for gen in range(6):
    log(f"=== GEN {gen+1} ===")
    parents = random.sample(list(archive.keys()), min(3, len(archive)))

    for pname in parents:
        pg = archive[pname]["game"]
        prefix, middle, suffix, _ = pick_target(pg)
        ludeme = middle[1:].split(None, 1)[0] if middle[1:].strip() else "?"

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=256, temperature=1.0,
                system=LUDAX_SYSTEM_PROMPT,
                messages=[{"role": "user", "content":
                    f"Original section:\n{middle}\n\n"
                    f"Generate a DIFFERENT replacement. Be creative.\n\n{prefix}<BLANK>{suffix}"}])
            repl = resp.content[0].text.strip().replace("\n", " ")
            if repl.startswith("```"):
                repl = repl.split("\n", 1)[1] if "\n" in repl else repl[3:]
                if repl.endswith("```"):
                    repl = repl[:-3]
                repl = repl.strip()
        except Exception as e:
            log(f"  {pname}>{ludeme}: API ERROR")
            continue

        mutated = f"{prefix}{repl}{suffix}".strip()

        if "".join(mutated.split()) == "".join(pg.split()):
            log(f"  {pname}>{ludeme}: IDENTITY")
            continue

        ok, err = validate_game(mutated)
        if not ok:
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
                    if repl.endswith("```"):
                        repl = repl[:-3]
                    repl = repl.strip()
                mutated = f"{prefix}{repl}{suffix}".strip()
                ok, _ = validate_game(mutated)
            except:
                pass
            if not ok:
                log(f"  {pname}>{ludeme}: GRAMMAR FAIL")
                continue

        try:
            r = evaluate_game(mutated, num_random_games=20, skip_skill_trace=True)
            f = fitness(r)
        except:
            log(f"  {pname}>{ludeme}: EVAL FAIL")
            continue

        if f <= 0:
            log(f"  {pname}>{ludeme}: UNPLAYABLE")
            continue

        cname = f"g{gen}_{pname}_{ludeme}"
        d = "+" if f > archive[pname]["fitness"] else "-" if f < archive[pname]["fitness"] else "="
        archive[cname] = {"game": mutated, "fitness": f, "result": r, "parent": pname}
        evolved.append(cname)
        log(f"  {cname}: f={f:.3f}{d} bal={r['balance']:.2f} comp={r['completion']:.2f} turns={r['mean_turns']:.0f} [{ludeme}]")

log(f"\nEvolution: {time.time()-t0:.0f}s, {len(evolved)} novel games\n")

# --- Top 3 with skill trace ---
print("=" * 60)
print("TOP NOVEL GAMES")
print("=" * 60)

ranked = sorted(evolved, key=lambda n: archive[n]["fitness"], reverse=True)
shown = 0
for name in ranked:
    g = archive[name]
    gc = "".join(g["game"].split())
    max_sim = max(SequenceMatcher(None, gc, "".join(s.split())).ratio() for s in seeds.values())
    if max_sim > 0.92:
        continue

    shown += 1
    if shown > 3:
        break

    log(f"\nComputing skill trace for {name}...")
    try:
        skill = compute_skill_trace(g["game"], num_games=6, mcts_sims=30)
    except:
        skill = -1

    r = g["result"]
    log(f"\n{'─'*60}")
    log(f"#{shown}: {name}")
    log(f"Fitness={g['fitness']:.3f} Parent={g['parent']} Novelty={1-max_sim:.2f}")
    log(f"Balance={r['balance']:.2f} Completion={r['completion']:.2f} Skill={skill:.2f} Turns={r['mean_turns']:.0f} Wins={r['wins']}")

    game = g["game"]
    for kw in ["(players", "(equipment", "(rules", "(start", "(play", "(end", "(effects"]:
        game = game.replace(kw, "\n  " + kw)
    print(game)

if shown == 0:
    log("No sufficiently novel games.")

log(f"\nTotal: {time.time()-t0:.0f}s")
