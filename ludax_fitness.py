"""
Ludax-based fitness evaluation for evolved games.

Replaces the Java-based evaluation pipeline (java_api.py + fitness_helpers.py)
with GPU-accelerated evaluation via Ludax's JAX environment.
"""

import typing

import jax
import jax.numpy as jnp
import numpy as np

from config import UNCOMPILABLE_FITNESS, UNPLAYABLE_FITNESS, UNINTERESTING_FITNESS


def _random_playout(env, rng) -> typing.Tuple:
    """Play a single game with random moves. Returns (rewards, num_turns, terminated)."""
    state = env.init(rng)

    def body(carry):
        state, rng, turns = carry
        rng, key = jax.random.split(rng)
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        action = jax.random.categorical(key, logits)
        state = env.step(state, action)
        return (state, rng, turns + 1)

    def cond(carry):
        state, _, _ = carry
        return ~state.terminated & ~state.truncated

    (final_state, _, num_turns) = jax.lax.while_loop(cond, body, (state, rng, 0))
    return final_state.rewards, num_turns, final_state.terminated


def compile_and_check(game_str: str) -> typing.Tuple[typing.Any, str]:
    """
    Try to compile a game string into a LudaxEnvironment.
    Returns (env, error_msg). env is None if compilation fails.
    """
    try:
        from ludax import LudaxEnvironment
        env = LudaxEnvironment(game_str=game_str)
        return env, ""
    except Exception as e:
        return None, str(e)


def evaluate_game(game_str: str,
                  num_random_games: int = 50,
                  num_batches: int = 1,
                  seed: int = 42,
                  skip_skill_trace: bool = False) -> typing.Dict[str, float]:
    """
    Evaluate a single game string using Ludax.

    Pipeline:
    1. Compile the game (fail fast if invalid syntax)
    2. Run random playouts to assess balance, completion, and basic metrics
    3. Return a fitness-compatible evaluation dict

    All playouts are JIT-compiled and vmapped for speed.
    """
    evaluation = {
        "compilable": False, "playable": False,
        "balance": -1, "completion": -1, "drawishness": -1,
        "mean_turns": -1, "decision_moves": -1, "board_coverage_default": -1,
        "trace_score": -1, "wins": [], "game_str": game_str,
    }

    # Stage 1: Compile
    env, error = compile_and_check(game_str)
    if env is None:
        evaluation["error"] = f"Compilation failed: {error}"
        return evaluation

    evaluation["compilable"] = True

    # Stage 2: Check playability — can we take at least one step?
    try:
        rng = jax.random.PRNGKey(seed)
        state = env.init(rng)
        if state.legal_action_mask.sum() == 0:
            evaluation["error"] = "No legal actions from starting position"
            return evaluation
        evaluation["playable"] = True
    except Exception as e:
        evaluation["error"] = f"Playability check failed: {e}"
        return evaluation

    # Stage 3: Random playouts
    try:
        playout_fn = jax.jit(jax.vmap(lambda rng: _random_playout(env, rng)))

        all_rewards = []
        all_turns = []
        all_terminated = []

        for batch_idx in range(num_batches):
            batch_rng = jax.random.PRNGKey(seed + batch_idx)
            keys = jax.random.split(batch_rng, num_random_games)

            rewards, turns, terminated = playout_fn(keys)

            all_rewards.append(np.array(rewards))
            all_turns.append(np.array(turns))
            all_terminated.append(np.array(terminated))

        rewards = np.concatenate(all_rewards, axis=0)     # (N, 2)
        turns = np.concatenate(all_turns, axis=0)           # (N,)
        terminated = np.concatenate(all_terminated, axis=0) # (N,)

    except Exception as e:
        evaluation["error"] = f"Playout failed: {e}"
        return evaluation

    total_games = len(rewards)
    completed_games = terminated.sum()

    # --- Metrics ---

    # Completion: fraction of games that terminated (vs truncated at 2000 steps)
    evaluation["completion"] = float(completed_games / total_games)

    # Mean turns (completed games only)
    if completed_games > 0:
        evaluation["mean_turns"] = float(turns[terminated].mean())
    else:
        evaluation["mean_turns"] = 0

    # Win counts: [draws, p1_wins, p2_wins]
    p1_wins = int(((rewards[:, 0] > 0) & terminated).sum())
    p2_wins = int(((rewards[:, 1] > 0) & terminated).sum())
    draws = int(completed_games) - p1_wins - p2_wins
    evaluation["wins"] = [draws, p1_wins, p2_wins]

    # Balance: 1 - |p1_wr - p2_wr| over decisive games
    decisive = p1_wins + p2_wins
    if decisive > 0:
        evaluation["balance"] = 1.0 - abs(p1_wins - p2_wins) / decisive
    else:
        evaluation["balance"] = -1

    # Drawishness: 1 - draw_rate
    if completed_games > 0:
        evaluation["drawishness"] = 1.0 - draws / int(completed_games)
    else:
        evaluation["drawishness"] = -1

    # Decision moves: proxy via mean legal actions per turn
    # (Ludax tracks this implicitly; we use mean_turns > threshold as proxy)
    if evaluation["mean_turns"] > 3:
        evaluation["decision_moves"] = min(1.0, evaluation["mean_turns"] / 50.0)
    else:
        evaluation["decision_moves"] = 0

    # Board coverage: fraction of board used (rough proxy from turn count)
    if env.board_size > 0 and evaluation["mean_turns"] > 0:
        evaluation["board_coverage_default"] = min(1.0, evaluation["mean_turns"] / env.board_size)
    else:
        evaluation["board_coverage_default"] = 0

    # Skill trace: measure MCTS vs random win rate for promising games
    # Only compute for games that pass basic quality checks (saves ~8s per bad game)
    if skip_skill_trace:
        evaluation["trace_score"] = 0
    elif (evaluation["completion"] > 0.5 and evaluation["balance"] > 0.2
            and evaluation["mean_turns"] > 5):
        try:
            evaluation["trace_score"] = compute_skill_trace(game_str, num_games=6, mcts_sims=30, seed=seed)
        except Exception:
            evaluation["trace_score"] = 0
    else:
        evaluation["trace_score"] = 0

    return evaluation


def compute_skill_trace(game_str: str, num_games: int = 10,
                        mcts_sims: int = 50, seed: int = 42) -> float:
    """
    Measure strategic depth: how often does MCTS beat random play?

    Plays num_games where MCTS controls one player and random controls the other,
    alternating sides. Returns the MCTS win rate (0.5 = no skill, 1.0 = pure skill).

    This is the Ludax equivalent of the Java "FastTrace" evaluation, but ~100x faster.
    """
    env, error = compile_and_check(game_str)
    if env is None:
        return 0.0

    from ludax.policies.mcts import uct_mcts_policy
    from ludax.policies.simple import random_policy

    mcts_fn = uct_mcts_policy(env, num_simulations=mcts_sims, max_depth=15)
    random_fn = random_policy()
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    mcts_wins = 0
    total_decisive = 0

    for game_i in range(num_games):
        rng, init_key = jax.random.split(rng)
        # Batch size 1 for policies (they expect batched input)
        state = env.init(init_key)
        state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)

        # Alternate which player MCTS controls
        mcts_player = game_i % 2

        for turn in range(2000):
            if state.terminated or state.truncated:
                break

            rng, policy_key = jax.random.split(rng)
            current_player = int(state.current_player)

            if current_player == mcts_player:
                action = mcts_fn(state_b, policy_key)[0]
            else:
                action = random_fn(state_b, policy_key)[0]

            state = env.step(state, action)
            state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)

        if state.terminated:
            rewards = np.array(state.rewards)
            if rewards[mcts_player] > 0:
                mcts_wins += 1
                total_decisive += 1
            elif rewards[1 - mcts_player] > 0:
                total_decisive += 1
            # draws don't count

    if total_decisive == 0:
        return 0.5  # no decisive games, can't measure skill

    return mcts_wins / total_decisive


def close_evaluation(game_str: str):
    """No-op. Ludax doesn't need cleanup (no Java subprocesses)."""
    pass
