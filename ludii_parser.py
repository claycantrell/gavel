"""
Tree-based parser for Ludii game descriptions.

Ludii games are S-expressions with parentheses for ludemes and curly braces
for lists. This module parses them into a tree of LudiiNode objects, replacing
the ad-hoc string manipulation in utils.py.
"""

from dataclasses import dataclass, field
import typing


@dataclass
class LudiiNode:
    """A node in the Ludii parse tree, representing a single parenthetical expression."""
    start: int          # index of opening '(' in source string
    end: int            # index of closing ')' in source string
    depth: int          # nesting depth (0 = root game node)
    children: list = field(default_factory=list)

    # Populated after parsing
    _source: str = ""

    @property
    def text(self) -> str:
        """The full text of this node including parentheses."""
        return self._source[self.start:self.end + 1]

    @property
    def ludeme_name(self) -> str:
        """The first word after the opening paren, e.g. 'game', 'piece', 'move'."""
        inner = self._source[self.start + 1:self.end]
        token = inner.lstrip().split(None, 1)
        return token[0] if token else ""

    def prefix(self) -> str:
        """Everything in the source before this node."""
        return self._source[:self.start]

    def suffix(self) -> str:
        """Everything in the source after this node."""
        return self._source[self.end + 1:]

    def all_nodes(self) -> typing.List["LudiiNode"]:
        """Depth-first traversal of this node and all descendants."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes


def parse(game_str: str) -> LudiiNode:
    """
    Parse a Ludii game string into a tree of LudiiNode objects.

    Single-pass O(n) parser that tracks parenthesis nesting. Curly braces
    are treated as grouping tokens but don't create nodes (matching Ludii's
    semantics where {...} is a list literal inside a ludeme).
    """
    stack: list[LudiiNode] = []
    root: typing.Optional[LudiiNode] = None
    depth = 0

    for idx, char in enumerate(game_str):
        if char == "(":
            node = LudiiNode(start=idx, end=-1, depth=depth, _source=game_str)
            if stack:
                stack[-1].children.append(node)
            else:
                root = node
            stack.append(node)
            depth += 1

        elif char == ")":
            depth -= 1
            if stack:
                stack[-1].end = idx
                stack.pop()

    if root is None:
        raise ValueError("No parenthetical expressions found in game string")

    return root


def extract_parentheticals(game_str: str) -> typing.List[typing.Tuple[str, str, str, int]]:
    """
    Drop-in replacement for utils._extract_parentheticals.

    Returns list of (prefix, parenthetical, suffix, depth) tuples for every
    parenthetical expression except the root (depth 0), sorted by depth then
    position to match the original implementation's ordering.
    """
    root = parse(game_str)
    result = []
    for node in root.all_nodes():
        if node.depth == 0:
            continue
        result.append((node.prefix(), node.text, node.suffix(), node.depth))
    # Original groups by depth (via defaultdict keyed by depth), then by position
    result.sort(key=lambda t: (t[3], len(t[0])))
    return result
