"""Progress-log helpers for the construction pipeline."""

from uacpy.common.dict_utils import KeySequence


def format_key(key_sequence: KeySequence) -> str:
    """Render a key sequence for logging as e.g. `LIPV.inner.mitral`, not its tuple-of-enum repr.

    Args:
        key_sequence: Keys addressing an entry, outermost first.

    Returns:
        The keys joined with dots.
    """
    return ".".join(str(key) for key in key_sequence)


def print_with_underline(text: str) -> None:
    """Print a section header for the progress log, underlined to its own length.

    Args:
        text: The header text.
    """
    print(text)
    print("-" * len(text))
