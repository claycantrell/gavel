"""
Grammar validation for Ludax .ldx game descriptions.

Uses Lark with Ludax's grammar.lark to validate game strings before
expensive evaluation. Rejects syntactically invalid games immediately.
Provides actionable error messages for LLM self-repair.
"""

import os
import typing

from lark import Lark
from lark.exceptions import UnexpectedInput, UnexpectedCharacters, UnexpectedToken

# Find the grammar file from the ludax package
try:
    import ludax
    _LUDAX_GRAMMAR_PATH = os.path.join(os.path.dirname(ludax.__file__), "grammar.lark")
except ImportError:
    _LUDAX_GRAMMAR_PATH = None

_parser: typing.Optional[Lark] = None

# Map Lark terminal names to human-readable descriptions
_TERMINAL_NAMES = {
    "SQUARE": "square", "RECTANGLE": "rectangle", "HEXAGON": "hexagon",
    "HEX_RECTANGLE": "hex_rectangle", "FORCE_PASS": "force_pass",
    "LPAR": "(", "RPAR": ")", "LBRACE": "{", "RBRACE": "}",
    "PLACE": "place", "MOVE": "move", "SLIDE": "slide", "HOP": "hop", "STEP": "step",
    "EFFECTS": "effects", "CAPTURE": "capture", "PROMOTE": "promote", "FLIP": "flip",
    "EXTRA_TURN": "extra_turn", "SET_SCORE": "set_score", "INCREMENT_SCORE": "increment_score",
    "END": "end", "IF": "if", "LINE": "line", "CONNECTED": "connected",
    "MOVER": "mover", "OPPONENT": "opponent", "DRAW": "draw", "BY_SCORE": "by_score",
    "REPEAT": "repeat", "ONCE_THROUGH": "once_through",
    "DESTINATION": "destination", "RESULT": "result",
    "EMPTY": "empty", "OCCUPIED": "occupied", "EDGE": "edge",
    "WIN": "win", "LOSE": "lose",
    "PLAYERS": "players", "EQUIPMENT": "equipment", "RULES": "rules",
    "BOARD": "board", "PIECES": "pieces", "START": "start", "PLAY": "play",
    "OR": "or", "AND": "and", "NOT": "not",
}


def _get_parser() -> Lark:
    """Lazy-load the Lark parser (expensive to construct, do it once)."""
    global _parser
    if _parser is None:
        if _LUDAX_GRAMMAR_PATH is None:
            raise RuntimeError("ludax package not installed — grammar.lark not found")
        _parser = Lark.open(_LUDAX_GRAMMAR_PATH, start="game", parser="earley")
    return _parser


def _format_error(e: UnexpectedInput, game_str: str) -> str:
    """
    Convert a Lark parse error into an actionable message for LLM repair.

    Instead of raw parser output, explains:
    - What the LLM wrote at the error position
    - What the grammar expected at that position
    - The surrounding context
    """
    col = getattr(e, 'column', None)
    if col is None:
        return str(e)

    # Extract context around the error
    pos = col - 1  # Lark columns are 1-based
    start = max(0, pos - 30)
    end = min(len(game_str), pos + 30)
    context = game_str[start:end]
    pointer_offset = pos - start

    # What was written at the error position
    bad_char = game_str[pos] if pos < len(game_str) else "<end of input>"
    # Find the word/token at the error position
    word_end = pos
    while word_end < len(game_str) and game_str[word_end] not in " (){}":
        word_end += 1
    bad_word = game_str[pos:word_end] if word_end > pos else bad_char

    # What the grammar expected
    allowed = set()
    if hasattr(e, 'allowed') and e.allowed:
        for terminal in e.allowed:
            readable = _TERMINAL_NAMES.get(terminal, terminal.lower().strip('_'))
            if not readable.startswith('_') and not readable.startswith('anon'):
                allowed.add(readable)
    elif hasattr(e, 'expected') and e.expected:
        for terminal in e.expected:
            readable = _TERMINAL_NAMES.get(terminal, terminal.lower().strip('_'))
            if not readable.startswith('_') and not readable.startswith('anon'):
                allowed.add(readable)

    msg_parts = [f'Syntax error at column {col}: found "{bad_word}"']

    if allowed:
        allowed_sorted = sorted(allowed)
        if len(allowed_sorted) <= 8:
            msg_parts.append(f'Expected one of: {", ".join(allowed_sorted)}')
        else:
            msg_parts.append(f'Expected one of: {", ".join(allowed_sorted[:8])}...')

    msg_parts.append(f'Context: ...{context}...')
    msg_parts.append(f'         {" " * pointer_offset}^')

    # Add nesting hint if we can determine the parent context
    # Walk backwards from error to find the enclosing ludeme
    depth = 0
    for i in range(pos - 1, -1, -1):
        if game_str[i] == ')': depth += 1
        elif game_str[i] == '(':
            if depth == 0:
                # Found the enclosing open paren
                word_start = i + 1
                word_end_ctx = word_start
                while word_end_ctx < len(game_str) and game_str[word_end_ctx] not in " (){}":
                    word_end_ctx += 1
                enclosing = game_str[word_start:word_end_ctx]
                if enclosing:
                    msg_parts.append(f'This error is inside a ({enclosing} ...) block.')
                break
            depth -= 1

    return "\n".join(msg_parts)


def validate_game(game_str: str) -> typing.Tuple[bool, str]:
    """
    Validate a Ludax game string against the formal grammar.

    Returns (is_valid, error_message).
    error_message is empty if valid. When invalid, the error is formatted
    for LLM consumption with actionable context.
    """
    try:
        parser = _get_parser()
        parser.parse(game_str)
        return True, ""
    except UnexpectedInput as e:
        return False, _format_error(e, game_str)
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
            errors.append(err)
    return valid, errors
