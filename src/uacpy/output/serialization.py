"""NumPy `.npz` (de)serialization of `UACSubmesh` data, the pipeline's per-segment UAC result.

Each segment's `UACSubmesh` is stored as one array per field, named by the segment's key sequence
joined with `/` and the field name, e.g. `LIPV/segment_1/alpha`. Arrays are stored natively, so
no encoding hooks are needed, and loading never unpickles (`allow_pickle=False`).
"""

import dataclasses
from collections import defaultdict
from pathlib import Path

import numpy as np

from uacpy.common import dict_utils
from uacpy.construction import model

_KEY_SEPARATOR = "/"


def save_uac_submeshdata(
    path: str | Path, uac_submesh_data: dict_utils.KeySequenceDict[model.UACSubmesh]
) -> None:
    """Write `UACSubmesh` data to a compressed `.npz` file, overwriting any existing file.

    Args:
        path: Destination file path. Should end in `.npz`, or NumPy appends the suffix.
        uac_submesh_data: `UACSubmesh` per segment (`UACData.uac_submesh_data`). Segment keys must
            not contain `/`.
    """
    arrays = {
        _KEY_SEPARATOR.join((*map(str, key), field.name)): getattr(submesh, field.name)
        for key, submesh in uac_submesh_data.items()
        for field in dataclasses.fields(model.UACSubmesh)
    }
    np.savez_compressed(path, **arrays)


def load_uac_submeshdata(path: str | Path) -> dict_utils.KeySequenceDict[model.UACSubmesh]:
    """Read `UACSubmesh` data from a file written by `save_uac_submeshdata`.

    Args:
        path: Source file path.

    Returns:
        `UACSubmesh` per segment, keyed by the segment's key sequence as a tuple of strings.
    """
    fields_by_key = defaultdict(dict)
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            *key, field_name = name.split(_KEY_SEPARATOR)
            fields_by_key[tuple(key)][field_name] = archive[name]
    return {key: model.UACSubmesh(**fields) for key, fields in fields_by_key.items()}
