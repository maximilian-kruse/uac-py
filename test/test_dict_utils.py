"""Unit tests for `dict_utils.flatten`, the one place a nested config dict becomes keyed entries."""

import pytest

from uacpy.common import dict_utils


@pytest.mark.unit
def test_flatten_keys_each_leaf_by_its_full_key_sequence():
    """Every non-dict value is a leaf, keyed by the tuple of keys leading to it."""
    nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}

    assert dict_utils.flatten(nested) == {("a", "b"): 1, ("a", "c", "d"): 2, ("e",): 3}


@pytest.mark.unit
def test_flatten_preserves_depth_first_order():
    """Entries come out in the order a depth-first walk of the nested dict visits them."""
    nested = {"a": {"b": 1}, "c": 2, "d": {"e": 3}}

    assert list(dict_utils.flatten(nested)) == [("a", "b"), ("c",), ("d", "e")]


@pytest.mark.unit
def test_flatten_treats_a_none_leaf_as_a_leaf():
    """A `None` value (a result not computed yet) is a leaf, not skipped."""
    assert dict_utils.flatten({"a": {"b": None}}) == {("a", "b"): None}
