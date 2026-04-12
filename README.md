# GAVEL: Games via Evolution and Language Models

A system for automatically generating novel board games by combining evolutionary search (MAP-Elites) with large language models. Originally based on the [GAVEL paper](https://arxiv.org/abs/2407.09388) (NeurIPS 2024), this fork modernizes the entire pipeline.

## What changed

The original system used a fine-tuned CodeLlama-13b model with Java-based Ludii evaluation. This fork replaces every major component:

| Component | Original | This fork |
|---|---|---|
| **LLM** | Fine-tuned CodeLlama-13b (40hr training) | Claude Sonnet/Opus via Anthropic API (zero training) |
| **Game DSL** | Ludii `.lud` (Java subprocess) | [Ludax](https://github.com/gdrtodd/ludax) `.ldx` (JAX-accelerated) |
| **Grammar** | `raise NotImplementedError` | Formal Lark grammar validation + self-repair |
| **Evaluation** | 10 MCTS games via Java (~seconds each) | 100 vmapped random playouts + MCTS skill trace (~8s total) |
| **Mutation** | Fill-in-the-blank FITM | 4 strategies: single-point, agentic, multi-edit, semantic targeting |
| **Fitness** | Hand-crafted thresholds + harmonic mean | Balance + completion + decision + skill trace (+ optional LLM-as-judge) |
| **Archive** | PCA on 1000+ Java-extracted concepts | PCA on 57 parser-extracted structural features |
| **Dependencies** | torch, transformers, Java, CUDA | anthropic, jax, lark |

## Setup

```bash
# Clone and install
git clone https://github.com/gdrtodd/ludax  # JAX game engine
pip install -e ./ludax
pip install anthropic lark numpy scikit-learn

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick start

Generate games with the Ludax pipeline (no Java, no GPU required):

```bash
python evolution.py \
    --use_ludax \
    --use_anthropic --anthropic_model claude-sonnet-4-6 \
    --fitness_evaluation_strategy ludax \
    --archive_type structural \
    --mutation_selection_strategy semantic \
    --num_selections 3 --num_mutations 2 \
    --num_epochs 100 \
    --save_dir ./exp_outputs/ludax_run --overwrite
```

### Mutation strategies

**Single-point** (default with `--use_anthropic`): Replace one parenthetical expression. 97% grammar success rate. Fast, reliable, conservative changes.

**Agentic** (`--use_agentic`): Multi-turn propose-critique-refine loop. The model proposes a mutation, self-critiques for syntax/balance/fun issues, then refines. Higher quality mutations, catches its own errors.

**Multi-edit** (`--use_multi_edit`): Coordinated find-and-replace edits guided by design directions (e.g., "add capture mechanics", "change the board shape", "make the game asymmetric"). Each edit is validated individually. 50% success rate but produces coordinated multi-part changes that single-point mutations can't.

### Fitness evaluation strategies

- `ludax` — JAX-accelerated random playouts + MCTS skill trace. No Java needed.
- `llm_judge` — Claude evaluates game quality (coherence, interestingness, balance, novelty, completeness). No simulation needed.
- `adaptive` — LLM pre-filter, then full simulation only for promising games. Saves compute.
- `random` / `uct` / `one_ply` / `combined` — Original Java-based evaluation (requires Ludii fork).

### Archive types

- `structural` — PCA on 57 features extracted from the game AST (board type, piece count, movement/capture/end-condition ludemes). No Java dependency.
- `selected_concept` / `pca` / `pca_and_length` — Original Java-based concept extraction.

## Architecture

```
Game string (.ldx)
    -> Lark grammar validation (ludax_grammar.py)
    -> LudaxEnvironment compilation
    -> Random playouts via jax.vmap (ludax_fitness.py)
    -> MCTS vs random skill trace
    -> MAP-Elites archive (structural features from AST)
    -> Mutation (AnthropicMutator / AgenticMutator / MultiEditMutator)
    -> Grammar self-repair loop if invalid
    -> Novelty filter (reject near-identity mutations)
    -> Back to evaluation
```

### Key files

| File | Purpose |
|---|---|
| `evolution.py` | MAP-Elites search loop, CLI entry point |
| `mutators.py` | All mutation strategies (BaseMutator, AnthropicMutator, AgenticMutator, MultiEditMutator) |
| `ludax_fitness.py` | JAX-accelerated game evaluation + MCTS skill trace |
| `ludax_grammar.py` | Lark grammar validation with actionable error messages |
| `llm_fitness.py` | LLM-as-judge fitness evaluation |
| `ludii_parser.py` | S-expression parser, structural feature extraction |
| `archives.py` | MAP-Elites archive types (StructuralPCAArchive, etc.) |
| `config.py` | Enums, fitness thresholds, shared constants |

### Legacy files (original Ludii pipeline)

| File | Purpose |
|---|---|
| `java_api.py` | Java subprocess communication with Ludii |
| `fitness_helpers.py` | Original Java-based fitness evaluation |
| `java_helpers.py` | Ludii concept names, JAR paths |
| `train.py` | CodeLlama fine-tuning script |
| `ludii_datasets.py` | FITM training data generation |
| `ludii_fork/` | Embedded Ludii game engine (Java) |

## Original paper

The fine-tuned model checkpoint and dataset from the original paper are available on HuggingFace:
- [Model](https://huggingface.co/LudiiLMs/code-llama-13b-fitm-mask-heldout-1-epoch)
- [Dataset](https://huggingface.co/datasets/LudiiLMs/code-llama-13b-fitm-mask-heldout-1-epoch-base-data)

To run the original experiment (requires torch, transformers, Java):
```bash
python evolution.py --model LudiiLMs/code-llama-13b-fitm-mask-heldout-1-epoch \
    --fitness_evaluation_strategy uct --games_per_eval 10 \
    --num_fitness_evals 1 --thinking_time 0.25 --max_turns 50 \
    --archive_type pca --num_selections 3 --num_mutations 3 \
    --fitness_eval_timeout 300 --num_threads 6 \
    --save_dir ./exp_outputs/main_experiment --overwrite --add_current_date \
    --seed 1
```

## Citation

```
@inproceedings{todd2024gavel,
  title={GAVEL: Generating Games Via Evolution and Language Models},
  author={Todd, Graham and Padula, Alexander and Stephenson, Matthew and Piette, Eric and Soemers, Dennis and Togelius, Julian},
  booktitle={NeurIPS 2024, the Thirty-Eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
