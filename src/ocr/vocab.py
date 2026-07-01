"""Character vocabulary and CTC encode/decode helpers for LRLPR plate text."""

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 36 symbols
BLANK_IDX = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1  # 37 (blank + 36 symbols)
PLATE_LEN = 7


def encode_text(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text]


def ctc_greedy_decode(indices: list[int]) -> str:
    """Collapse repeated indices and drop blanks (standard CTC greedy decode)."""
    chars = []
    prev = None
    for idx in indices:
        if idx != prev and idx != BLANK_IDX:
            chars.append(IDX_TO_CHAR[idx])
        prev = idx
    return "".join(chars)
