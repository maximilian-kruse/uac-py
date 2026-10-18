"""Key sequences: how the construction pipeline addresses entries, and its keyed-dict shape.

An anatomy config is authored as nested dicts (`{LIPV: {INNER: {...}}}`), which reads well when
entries are grouped by structure. Everything the pipeline computes is instead a flat dict keyed by
each entry's full key sequence as a tuple, e.g. `("LIPV", "inner", "mitral")`: `flatten` converts
the authored shape once, at link time, and no other code walks nested dicts.
"""

from collections.abc import Sequence

# A key sequence such as `("LIPV", "inner", "mitral")` addresses one entry of a nested config dict,
# outermost key first. Config references may be lists or tuples.
type KeySequence = Sequence[str]

# A flat dict keyed by full key sequences, as tuples so they are hashable. `StrEnum` members hash
# equal to their plain string values, so `("LIPV",)` and `(AtrialStructure.LIPV,)` are one key.
type KeySequenceDict[T] = dict[tuple[str, ...], T]


# ==================================================================================================
def flatten(nested: dict, prefix: tuple[str, ...] = ()) -> KeySequenceDict:
    """Flatten a nested dict into one entry per leaf, keyed by the leaf's full key sequence.

    Args:
        nested: Nested dict to traverse. Any non-dict value counts as a leaf.
        prefix: Key sequence prepended to every key. Used by the recursion.

    Returns:
        Dict mapping each leaf's key sequence (a tuple starting with `prefix`) to the leaf, in
        depth-first order.
    """
    flat = {}
    for key, value in nested.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat
