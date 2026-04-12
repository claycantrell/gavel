"""
Grammar validation for Ludax .ldx game descriptions.

Uses Lark with Ludax's grammar.lark to validate game strings before
expensive evaluation. Rejects syntactically invalid games immediately.
"""

import os
import typing

from lark import Lark
from lark.exceptions import UnexpectedInput

# Find the grammar file from the ludax package
try:
    import ludax
    _LUDAX_GRAMMAR_PATH = os.path.join(os.path.dirname(ludax.__file__), "grammar.lark")
except ImportError:
    _LUDAX_GRAMMAR_PATH = None

_parser: typing.Optional[Lark] = None


def _get_parser() -> Lark:
    """Lazy-load the Lark parser (expensive to construct, do it once)."""
    global _parser
    if _parser is None:
        if _LUDAX_GRAMMAR_PATH is None:
            raise RuntimeError("ludax package not installed — grammar.lark not found")
        _parser = Lark.open(_LUDAX_GRAMMAR_PATH, start="game", parser="earley")
    return _parser


def validate_game(game_str: str) -> typing.Tuple[bool, str]:
    """
    Validate a Ludax game string against the formal grammar.

    Returns (is_valid, error_message).
    error_message is empty if valid.
    """
    try:
        parser = _get_parser()
        parser.parse(game_str)
        return True, ""
    except UnexpectedInput as e:
        return False, str(e)
    except Exception as e:
        return False, f"Parser error: {e}"


def validate_and_filter(game_strs: typing.List[str]) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """
    Validate a batch of game strings. Returns (valid_games, errors).
    """
    valid = []
    errors = []
    for gs in game_strs:
        is_valid, err = validate_game(gs)
        if is_valid:
            valid.append(gs)
        else:
            errors.append(f"{err[:100]}...")
    return valid, errors
