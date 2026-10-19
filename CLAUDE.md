# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# uac-py
Universal Atrial Coordinates for the left atrium: parameterizes an LA surface mesh into
anatomically-segmented (α, β) charts (PVs, roof, mitral valve, appendage) so scalar/vector fields
(e.g. fiber orientation) transfer between patients through a shared coordinate system. Python
≥3.14, NumPy/pyvista/igraph/numba. Alpha stage, single author.

**Procedure (non-trivial tasks):** inspect code/conventions → short plan → smallest appropriate
change → run checks → review diff.

## Environment (pixi)
- NEVER bare `python`/`pip`/`conda`; always `pixi run ...`. Ask before adding deps (`pixi add`,
  `--pypi` only if not on conda-forge — `libigl`/`meshlib` are pypi-only, rest is conda-forge).
  Never hand-edit `pixi.lock`; commit it with `pyproject.toml` (holds all pixi config, no
  `pixi.toml`).
- Envs: `default` (numpy/pyvista/igraph/numba/libigl/meshlib/shapely), `dev` (+ruff,
  pre-commit), `examples` (+jupyter, plotting, trame), `test` (+pytest, pytest-cov). `meshlib` is a
  base dependency (not `examples`-only) since `test` needs it too, to import `basis.py` (whose
  hole-filling helper uses it) for `test/test_basis.py`'s unit tests. Tools need `-e`:
  - `pixi run -e dev ruff check src test notebooks` / `ruff format src test notebooks`
  - `pixi run -e test pytest`
  - `pixi run -e examples jupyter lab`
- `python` pinned `>=3.14`, `pyvista` pinned `>=0.49`: `pyvista==0.46.4` breaks on 3.14's `typing`
  internals, so the two move together — don't drop either pin without re-testing the notebooks
  and `test/`. `pyvista`'s VTK internals changed how chained-extraction filters (`extract_feature_
  edges`, `connectivity`/`split_bodies`) handle the reserved `vtkOriginalPointIds` field across
  this version jump — see `geometry.py`'s `get_feature_boundary`/`extract_region_from_boundary`
  for the workarounds, and don't assume this array means the same thing in both versions.

## Architecture
- `construction/`: config-driven pipeline. `model.py` — authored pure-data cell-complex dataclasses
  (`Marker`=0-cell, `BoundaryPath`/`ConnectionPath`=1-cell construction, `PathParameterization`/
  `UACMapping`=1-cell arclength/UAC assignment, `Segment`=2-cell), addressed by key-sequence
  (`("LIPV","inner","mitral")`) so an anatomy config can be authored in whatever order reads best,
  independent of build-dependency order (`common.dict_utils.KeySequence`), plus the `EndpointKind`/`PositionKind`
  enums they use and, in the same module, the computed counterparts (`ParameterizedPath`,
  `UACPath`, `Submesh`, `UACSubmesh`). `names.py` — generic vocabulary as `StrEnum`/`IntEnum`
  groups (`OstiumLoop`, `AtrialStructure`,
  `AnatomicalDirection`, `Axis`, `OstiumSegment`) anatomy configs import rather than retyping as string literals — a typo is
  an `AttributeError` at import time instead of a silently-wrong new dict key.
  `graph.py` — `link()` resolves a config's key-sequence references into a `ConstructionGraph` of
  live `Node` objects (one `PathNode`/`MarkerNode`/`ParameterizationNode`/`UACMappingNode`/
  `SegmentNode` per entry, key-sequence fields replaced by direct references to the nodes they
  name, plus a mutable `result` slot for that node's computed value) — a dangling reference is a
  `KeyError` here, at link time, not wherever the pipeline first reaches it. Each `Node` type also
  owns its own `construct*` method: the logic for turning that node's config plus its already-
  resolved dependencies' results into its own result, plus (for the segmentation-phase types) a
  `dependencies` property. `geometry.py` — stateless mesh algorithms (boundary/path extraction
  via igraph, region flood-fill via numba, arclength/UAC parameterization along a path). `UACSubmesh.to_polydata` (in `model.py`) is the one place a
  `UACSubmesh`'s `(alpha, beta)` + `connectivity` becomes a flat z=0 pyvista mesh — `check.py`,
  `mapper.py`, and `visualization.py` all build one for area checks or plotting, so it isn't
  re-derived three separate times). `constructor.py` — `UACConstructor`, a thin
  orchestrator over a linked `ConstructionGraph`: it decides *what order* to do work in
  (`construct_segmentation` derives its own order with a `networkx` topological sort over each
  node's `dependencies`, since paths/markers/parameterizations feed each other back and forth, not
  in fixed phases; `construct_uacs` runs the remaining fixed phases), but never *how* an individual
  node is computed — that split keeps `UACConstructor` from being both the thing that sequences
  work and the thing that implements every kind of it. `.data` collects the graph's results into
  `UACData` on read: flat dicts keyed by each entry's full key sequence as a tuple, e.g.
  `uac_submesh_data[("LIPV", "segment_1")]` — the pipeline's external contract, consumed that way by
  `check.py`/`output/`/`mapper.py`/notebooks. Only the *authored* config is nested; `graph.link`
  flattens it once via `dict_utils.flatten`, and nothing else walks nested dicts. `check.py` — three independent validity checks, each catching
  a defect the others can't: `check_uac_mesh_validity` (per-submesh fold-overs, via cell area),
  `check_segment_boundary_multiplicity` (structural, config-only, no mesh needed — every boundary
  *portion* a `Segment` references should be shared by exactly 1 or 2 segments; grouped by
  `(path, portion)`, not by path alone, since one ostium loop is several distinct portions, each
  its own edge), and `check_uac_tiling` (geometric, via `shapely` — no two segments' computed UAC
  regions should overlap; deliberately doesn't check for gaps, since this anatomy's UAC domain has
  genuine holes at the PV/MV/LAA openings, so "no gaps" isn't a valid generic invariant here).
- `common/`: `mvc.py`/`jacobian.py` are the numeric core (mean-value coordinates, per-triangle
  Jacobians) — keep pyvista-free. `mesh_utils.py` — pyvista/trimesh preprocessing.
  `dict_utils.py` — `KeySequence` (an address into a nested config dict), `KeySequenceDict[T]` (a flat
  dict keyed by full key-sequence tuples) and `flatten` (nested config -> keyed entries).
  `common/` is the base layer: it imports nothing else from `uacpy`.
- `output/`: `serialization.py` (NumPy `.npz`, one array per `UACSubmesh` field, named `key/parts/field`) and `visualization.py` (pyvista plots) are pipeline
  consumers — they import `construction`/`mapper`, so they must not live in `common/`.
  `logging.py` (`format_key`, `print_with_underline`) is the progress-log helper the pipeline
  itself calls; it imports only `common`, so `construction` can depend on it without a cycle.
- `basis.py`/`mapper.py`: cross-patient transfer. `basis.py` builds a per-cell tangent basis from
  a smooth whole-mesh parameterization; `mapper.py`'s `UACMapper` moves scalar/vector fields
  between the original mesh, its UAC submeshes, and another patient's, via cached Jacobians. Transfer between patients is a nearest-neighbor copy (no averaging), so
  fiber fields, being axial, need no special treatment: use `interpolate_to_other_uac_mesh`.
- `anatomies/`: one subpackage per anatomy, each a full `construction`-pipeline config — installed
  library code, not application data, so it's importable without any `sys.path` hacks.
  `anatomies/left_atrium_4_pvs/` hardcodes one LA topology (4 PVs + LAA + MV) and is written out
  explicitly as plain nested dicts of `model` dataclasses, one module per config kind (`paths`,
  `parameterizations`, `markers`, `uacs`, `segments`), re-exported by the package. Deliberately
  not generated by builders: the repetitive PV blocks stay readable and individually editable. The
  package docstring has the chart layout.
  All anatomy-config identifiers are `StrEnum`/`IntEnum` members from the single
  `construction/names.py`, split by divider into a generic section and an LA-specific section
  (`RoofConnector`, `MitralConnector`, `MitralVariant`, `SegmentId`, `MITRAL`); none is a bare
  string literal typed inline in those modules. Anatomy-specific *vocabulary* belongs in
  that LA-specific section of `names.py` (a new anatomy adds its own section there); a new
  anatomy is otherwise a new subpackage here, built the same way — never an anatomy-specific
  *branch of logic* in `construction/`, `mapper.py`, or `common/`.
- `notebooks/`: notebooks only — thin orchestration that reads a mesh, builds
  `UACConstructorSettings` from an `anatomies.<name>` config, and runs the pipeline. No anatomy
  data belongs here. Notebooks run in order: `preprocess_mesh` → `la_construction` → `la_basis` →
  `la_mapping`.

## Design
- Priorities: numerical correctness (no fold-overs, consistent conventions) > reproducibility >
  clear APIs > performance > convenience.
- Stateless functions in `geometry.py`/`common/`; only `UACConstructor`/`UACMapper` hold state,
  and only to cache expensive per-submesh data (boundary loops, Jacobians).
- Explicit over clever; small functions; no magic numbers in library code (name + cite); per-
  anatomy constants belong in that anatomy's `uacpy.anatomies.<name>` module. Breaking the public
  API is fine (no external users yet).

## Code style
- Ruff's enforced rule set is curated, not `select = ["ALL"]`: `E`/`W` (pycodestyle), `F`
  (Pyflakes), `I` (isort), `UP` (pyupgrade), `B` (flake8-bugbear), `SIM` (flake8-simplify), and
  `TID252` with `ban-relative-imports = "all"` (always import `uacpy.x.y`, never `from . import
  y` or `from .x import y`). Type hints, docstrings, and copyright headers are encouraged (see
  Docs below) but not lint-enforced — don't add `# noqa` for rule categories that aren't selected,
  there's nothing to suppress. No `# noqa` without a documented reason for the ones that are.
- Vectorise over mesh cells/vertices (numpy/einsum); loops only over small structural lists
  (key-sequence tuples, segmentation-graph nodes), never per-cell data. Descriptive snake_case, no
  new abbreviations.

## Docs
- Existing modules are mostly undocumented — this isn't lint-enforced, so it won't surface as a
  check failure, but add a docstring when you touch a function; no speculative repo-wide
  docstring pass unless asked.
- Docstring concepts, not mechanics: array shapes/dtypes, units (mesh coords are patient-space
  until `mesh_utils.preprocess_mesh`'s `dimension_scaling_factor`), non-obvious invariants (UAC
  `(alpha, beta)` ranges are anatomy-specific, not `[0, 1]`).

## Testing
- `test/unit/` (synthetic data, fast; mirrors `src/uacpy`'s layout) and `test/integration/` (real
  `data/patient_*`, `slow`-marked regression tests). `test/conftest.py` marks tests `unit`/
  `integration` by directory (no per-test decorator; `slow` is still applied by hand) and holds
  the session-scoped real-patient fixtures (`patient_01_mesh`, `patient_01_uac_data`,
  `load_patient_mesh`) so the pipeline runs once. Test modules can't import from `conftest`
  (importlib mode) -- share via fixtures.
- `pixi run test-unit` / `pixi run test-integration` / `pixi run -e test pytest` (config in
  `pyproject.toml`, coverage in `.coveragerc`). Pipeline-level tests import
  `uacpy.anatomies.left_atrium_4_pvs` directly -- the only real anatomy, so there's no synthetic
  fixture to exercise instead.
- Integration tests are golden-snapshot regressions: `test_construction_regression.py` (per-segment
  counts and (alpha, beta) ranges plus all three `check.py` checks), `test_basis_regression.py`
  (`data/patient_01/basis_vecs.npy`), `test_mapping_regression.py` (fiber transfer onto patient_01,
  `snapshots/mapped_fibers_to_patient_01.npz`, regenerated by running the module), and
  `test_preprocess_regression.py` (pinned counts). A failure means either a real pipeline change
  (investigate) or a deliberate change to the mesh/config (regenerate the snapshot -- never loosen
  tolerances or edit the reference values just to make it pass). The structural, config-only
  `unit/construction/test_anatomy_config.py` can catch a bad config edit before you run the
  pipeline.
- Unit tests prefer mathematical invariants over implementation details, on small synthetic
  meshes: `test_mvc.py` (boundary reproduction; MVC are generalized barycentric coordinates),
  `test_jacobian.py` (exact for an affine 2D-to-3D embedding), `test_geometry.py` (arclength
  parameterization, boundary/region extraction on a tagged grid). `assert_allclose`; seed via
  `np.random.default_rng(seed)`.

## Numerical code
- `mvc.py` = Floater's mean value coordinates; `jacobian.py` = per-triangle affine Jacobian via
  Cramer's rule — cite the method when you touch either.
- Before changing an algorithm: understand the math, preserve conventions (pyvista's flattened
  face format, the `(3, 2)` Jacobian layout, `(alpha, beta)` order), check shapes/broadcasting,
  and don't "fix" unusual-looking math (e.g. hole-filling before parameterization in `basis.py`)
  without understanding why it's there. Correctness first — this runs offline, per patient, not
  in a hot loop.
