"""
Tree-based parser for Ludii game descriptions.

Ludii games are S-expressions with parentheses for ludemes and curly braces
for lists. This module parses them into a tree of LudiiNode objects, replacing
the ad-hoc string manipulation in utils.py.
"""

from collections import Counter
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

    def find(self, ludeme_name: str) -> typing.List["LudiiNode"]:
        """Find all descendant nodes with the given ludeme name."""
        return [n for n in self.all_nodes() if n.ludeme_name == ludeme_name]

    def child(self, ludeme_name: str) -> typing.Optional["LudiiNode"]:
        """Find the first direct child with the given ludeme name."""
        for c in self.children:
            if c.ludeme_name == ludeme_name:
                return c
        return None


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


# --- Structural feature extraction ---

# Ludeme categories for feature extraction
BOARD_TYPES = ["square", "hex", "rectangle", "tri", "graph", "circle", "spiral", "star",
               "alquerque", "morris", "cross", "kints", "pachisi", "mancala"]
MOVEMENT_LUDEMES = ["Step", "Slide", "Hop", "Leap", "FromTo", "Add", "Shoot", "Sow"]
CAPTURE_LUDEMES = ["remove", "custodial", "surround", "intervene", "enclose", "hop"]
END_LUDEMES = ["Line", "Connected", "Group", "Loop", "NoMoves", "NoPiece", "Fill",
               "Reach", "Checkmate", "Scoring", "Pattern", "Territory"]
EFFECT_LUDEMES = ["Push", "Flip", "Promote", "Set", "Swap", "Roll"]


def extract_structural_features(game_str: str) -> typing.Dict[str, float]:
    """
    Extract a dict of structural features from a Ludii game string using the AST.
    No Java required. Features are suitable for diversity characterization.
    """
    root = parse(game_str)
    all_nodes = root.all_nodes()
    ludeme_names = [n.ludeme_name.lower() for n in all_nodes]
    ludeme_counts = Counter(ludeme_names)

    features: typing.Dict[str, float] = {}

    # --- Board features ---
    for bt in BOARD_TYPES:
        features[f"board_{bt}"] = 1.0 if bt in ludeme_counts else 0.0

    # Board size: look for integer args in board nodes
    board_nodes = root.find("board")
    if board_nodes:
        board_text = board_nodes[0].text
        import re
        nums = re.findall(r'\b(\d+)\b', board_text)
        features["board_size"] = max((int(n) for n in nums), default=0) / 20.0  # normalize
    else:
        features["board_size"] = 0.0

    # --- Players ---
    players_nodes = root.find("players")
    if players_nodes:
        import re
        nums = re.findall(r'\d+', players_nodes[0].text)
        features["num_players"] = int(nums[0]) / 4.0 if nums else 0.5
    else:
        features["num_players"] = 0.5

    # --- Piece features ---
    piece_nodes = root.find("piece")
    features["num_piece_types"] = min(len(piece_nodes) / 10.0, 1.0)
    features["has_hand"] = 1.0 if "hand" in ludeme_counts else 0.0
    features["has_dice"] = 1.0 if any(d in ludeme_counts for d in ["dice", "roll"]) else 0.0

    # --- Movement features ---
    for ml in MOVEMENT_LUDEMES:
        features[f"move_{ml.lower()}"] = 1.0 if ml.lower() in ludeme_counts else 0.0

    # --- Capture features ---
    for cl in CAPTURE_LUDEMES:
        features[f"capture_{cl}"] = 1.0 if cl in ludeme_counts else 0.0

    # --- End condition features ---
    for el in END_LUDEMES:
        features[f"end_{el.lower()}"] = 1.0 if el.lower() in ludeme_counts else 0.0

    # --- Effect features ---
    for ef in EFFECT_LUDEMES:
        features[f"effect_{ef.lower()}"] = 1.0 if ef.lower() in ludeme_counts else 0.0

    # --- Structural complexity ---
    features["num_ludemes"] = min(len(all_nodes) / 100.0, 1.0)
    features["max_depth"] = min(max(n.depth for n in all_nodes) / 15.0, 1.0)
    features["has_phases"] = 1.0 if "phase" in ludeme_counts else 0.0
    features["has_forEach"] = 1.0 if "foreach" in ludeme_counts else 0.0
    features["has_if"] = 1.0 if "if" in ludeme_counts else 0.0

    # --- Game string length (proxy for rule complexity) ---
    features["game_length"] = min(len(game_str) / 2000.0, 1.0)

    return features


# Stable ordered list of feature names — populated on first call, then frozen
STRUCTURAL_FEATURE_NAMES: typing.List[str] = sorted(
    [f"board_{bt}" for bt in BOARD_TYPES] +
    ["board_size", "num_players", "num_piece_types", "has_hand", "has_dice"] +
    [f"move_{ml.lower()}" for ml in MOVEMENT_LUDEMES] +
    [f"capture_{cl}" for cl in CAPTURE_LUDEMES] +
    [f"end_{el.lower()}" for el in END_LUDEMES] +
    [f"effect_{ef.lower()}" for ef in EFFECT_LUDEMES] +
    ["num_ludemes", "max_depth", "has_phases", "has_forEach", "has_if", "game_length"]
)

def get_structural_feature_vector(game_str: str) -> typing.List[float]:
    """Extract structural features as a fixed-length numeric vector."""
    features = extract_structural_features(game_str)
    return [features.get(name, 0.0) for name in STRUCTURAL_FEATURE_NAMES]


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
