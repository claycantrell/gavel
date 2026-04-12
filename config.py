from dataclasses import dataclass
from enum import Enum
import typing

@dataclass
class ArchiveGame:
    game_str: str
    fitness_score: float
    evaluation: typing.Dict[str, float]
    lineage: list
    generation: int
    original_game_name: str
    epoch: int

class EliteSelectionStrategy(str, Enum):
    RANDOM = "random"
    UCB = "ucb"

class MutationSelectionStrategy(str, Enum):
    RANDOM = "random"
    UCB_DEPTH = "ucb_depth"
    UCB_LUDEME = "ucb_ludeme"

class MutationStrategy(str, Enum):
    STANDARD = "standard"
    GRAMMAR_ENFORCED = "grammar_enforced"

class FitnessEvaluationStrategy(str, Enum):
    RANDOM = "random"
    UCT = "uct"
    ONE_PLY = "one_ply"
    COMBINED = "combined"
    LLM_JUDGE = "llm_judge"
    ADAPTIVE = "adaptive"

class FitnessAggregationFn(str, Enum):
    MEAN = "mean"
    HARMONIC_MEAN = "harmonic_mean"
    MIN = "min"


# Fitness evaluation thresholds — shared between fitness_helpers and java_api
COMPLETION_THRESHOLD = 0.2
MEAN_TURNS_THRESHOLD = 3
DECISION_MOVES_THRESHOLD = 0.1
BOARD_COVERAGE_THRESHOLD = 0.1
MIN_BALANCE_THRESHOLD = 0.5
MIN_DECISION_MOVES_THRESHOLD = 0.5
MIN_SCORE = 0.01

UNCOMPILABLE_FITNESS = -3
UNPLAYABLE_FITNESS = -2
UNINTERESTING_FITNESS = -1

VALIDATION_GAMES = [
    "ArdRi",
    "Ataxx",
    "Breakthrough",
    "Gomoku",
    "Havannah",
    "Hex",
    "Knightthrough",
    "Konane",
    "Pretwa",
    "Reversi",
    "Shobu",
    "Tablut",
    "Tron",
    "Yavalath"
]