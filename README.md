# GAVEL: AI Board Game Designer

Generates novel, playable board games from scratch using LLMs and evolutionary search. Themes are invented from random word seeds, mechanics are grounded in a formal game DSL, and games are evaluated via JAX-accelerated simulation.

Based on the [GAVEL paper](https://arxiv.org/abs/2407.09388) (NeurIPS 2024), rebuilt from the ground up.

## How it works

```
Random words (wonderwords)
    → LLM invents theme + backstory
    → Archetype selected (line, territory, race, elimination, etc.)
    → LLM writes rules as comments + Ludax code in one call
    → Grammar validation (Lark) + auto-fix (duplicate pieces, etc.)
    → JAX-accelerated evaluation (balance, completion, outcome variance, mechanic frequency)
    → Diagnose problems → LLM fixes specific issues → re-evaluate
    → Iterate until game is good
```

## Quick start

```bash
# Install
pip install -e ~/path/to/ludax    # JAX game engine
pip install anthropic lark numpy scikit-learn wonderwords
export ANTHROPIC_API_KEY=sk-ant-...

# Generate 10 games in parallel (~45 seconds)
python3 -u game_designer.py 10 claude-sonnet-4-6 42

# Diagnose and fix the best game
python3 -u run_diagnose_iterate.py

# Play in the browser
cd ~/path/to/ludax/examples/02-ludax_gui && python3 interactive.py --port 8080
```

## Game generation pipeline

### 1. Theme generation (`game_designer.py`)

Random word pairs (e.g., "skinny sunflower, dessert") seed a theme prompt. The LLM invents a name, backstory, board, win condition, and twist. No mechanic is prescribed — the archetype handles that.

### 2. Archetype rotation

10 archetypes rotate so every game uses a different mechanical foundation:

| Archetype | Description |
|---|---|
| `line` | Form N in a row to win |
| `territory` | Custodial capture, score by piece count |
| `connection` | Connect opposite board edges |
| `race_with_capture` | Move forward, hop-capture opponents |
| `elimination` | Hop-capture all opponent pieces, chain jumps |
| `flip_territory` | Place pieces that flip opponents (Reversi-style) |
| `promotion_battle` | Pawns promote to kings at the far edge |
| `asymmetric` | Different piece types per player |
| `score_race` | Capture scoring, first to target wins |
| `line_with_penalty` | Long line wins, short line LOSES (Yavalath-style) |

Each archetype includes a complete working Ludax example as a syntax guide.

### 3. Inline rule comments

The LLM writes rules in plain English as comments before the code:

```
;; SETUP: Empty hex board. No starting pieces.
;; TURN: Place one grove stone on any empty cell (except center).
;; EFFECTS: Custodial capture orthogonally (range 2). Diagonal capture (range 1).
;;          Flip pieces adjacent to center. Score = your piece count.
;; WIN: When board is full, highest score wins.
(game "Elm Accord" ...)
```

This forces coherent design thinking in a single API call.

### 4. Evaluation (`ludax_fitness.py`)

JAX-accelerated via [Ludax](https://github.com/gdrtodd/ludax):

- **Balance**: P1 vs P2 win rate from 30 random playouts
- **Completion**: fraction of games that reach a conclusion
- **Decision moves**: game length as a proxy for decision depth
- **Outcome variance**: std of final scores across games (do decisions matter?)
- **Mechanic frequency**: fraction of turns with significant board changes
- **MCTS skill trace**: does thinking beat random play? (on final candidates only)

Fitness penalties:
- Dead mechanics (effects that fire <5% of turns): 0.3x
- Low outcome variance (<5 std): 0.3x

### 5. Diagnose and fix (`run_diagnose_iterate.py`)

Instead of random mutation, the iteration loop:

1. **Plays the game** and collects metrics
2. **Diagnoses specific problems**: P1_DOMINATES, INSTANT_END, DECISIONS_DONT_MATTER, DEAD_MECHANIC, GAMES_DONT_END
3. **Tells the LLM exactly what's wrong** and asks for a targeted fix
4. **Escalates** after 3 failed attempts — tells the LLM to make radical changes

Result: 0.271 → 0.874 in a single targeted fix (vs 8 generations of random mutation for similar improvement).

## Results

Best games generated:

| Game | Fitness | Balance | Turns | How it was made |
|---|---|---|---|---|
| Plumage Wars | 0.974 | 0.92 | 77 | Single-shot generation |
| Ember Covenant | 0.928 | 0.80 | 61 | Single-shot (race archetype) |
| Elm Accord v2 | 0.874 | 0.91 | 95 | Diagnose-and-fix from 0.271 |
| Blue Note | 0.873 | 0.80 | 42 | Single-shot (jazz theme) |

Generation stats (best run): 10/10 playable, 10 different archetypes, 34 seconds parallel.

## Key files

| File | Purpose |
|---|---|
| `game_designer.py` | Theme generation, archetype rotation, game generation, parallel batch runs |
| `run_diagnose_iterate.py` | Diagnose-and-fix iteration loop |
| `run_iterate.py` | Random mutation iteration loop (legacy) |
| `run_evolution.py` | MAP-Elites evolution loop (legacy) |
| `ludax_fitness.py` | JAX evaluation: random playouts, engagement metrics, MCTS skill trace |
| `ludax_grammar.py` | Lark grammar validation with actionable error messages |
| `llm_fitness.py` | LLM-as-judge fitness evaluation |
| `mutators.py` | Mutation strategies: single-point, agentic, multi-edit |
| `ludii_parser.py` | S-expression parser, structural feature extraction |
| `archives.py` | MAP-Elites archive types |
| `evolution.py` | MAP-Elites search loop |

## Playing games

The Ludax GUI runs at `http://localhost:8080` with:
- All generated games in the dropdown
- Auto-generated rules panel explaining mechanics
- AI opponents: random, one-ply lookahead, MCTS (50 simulations)

## Architecture decisions

**Why Ludax over Ludii?** Ludii is more expressive (~850 games vs ~250) but requires Java subprocesses and is 100x slower. Ludax compiles to JAX — evaluation takes 1.5s instead of minutes. The expressiveness gap is smaller than it looks: Ludax supports hop capture, promotion, chain jumps, asymmetric pieces, flip, scoring — enough for rich games.

**Why not fine-tune a model?** The original GAVEL fine-tuned CodeLlama-13b for 40 hours. With Claude Sonnet, zero-shot generation with good prompts produces better results. The LLM understands game design; it just needs the right syntax reference and diagnostic feedback.

**Why diagnose-and-fix over random mutation?** Random mutation found improvements by luck (0.815 → 0.934 in 8 generations). Targeted diagnosis found the same improvement in 1 fix (0.271 → 0.874). Telling the LLM "P1 wins 90% of the time" is more useful than "change this random parenthetical."

## Original paper

```
@inproceedings{todd2024gavel,
  title={GAVEL: Generating Games Via Evolution and Language Models},
  author={Todd, Graham and Padula, Alexander and Stephenson, Matthew and Piette, Eric and Soemers, Dennis and Togelius, Julian},
  booktitle={NeurIPS 2024},
  year={2024}
}
```
