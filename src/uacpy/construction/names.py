"""All vocabulary for anatomy configs.

The module has two sections. The generic section holds structural roles of an ostium and standard
clinical anatomical terms. The left-atrium-specific section holds the names of the 4-PV LA
config's own construction (connectors, segment ids, split variants).

Each group is its own `StrEnum`.
"""

from enum import StrEnum

# ==================================================================================================
# GENERIC VOCABULARY
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
class OstiumLoop(StrEnum):
    """The two concentric marker loops of a PV ostium.

    Antral is the loop in the mesh interior, ostial the loop on the mesh boundary.
    """

    ANTRAL = "antral"
    OSTIAL = "ostial"


# --------------------------------------------------------------------------------------------------
class AnatomicalDirection(StrEnum):
    """Standard anatomical directions used to name markers, segments, and split-path variants."""

    ANTERIOR = "anterior"
    POSTERIOR = "posterior"
    LATERAL = "lateral"
    SEPTAL = "septal"


# --------------------------------------------------------------------------------------------------
class Axis(StrEnum):
    """An ostium's two orthogonal marker axes, each a pair of opposite `AnatomicalDirection`s."""

    ANTERIOR_POSTERIOR = "anterior_posterior"
    SEPTAL_LATERAL = "septal_lateral"


# ==================================================================================================
# LEFT-ATRIUM-SPECIFIC VOCABULARY (`anatomies.left_atrium_4_pvs`)
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
class AtrialStructure(StrEnum):
    """Left-atrial structures with their own boundary: the 4 PVs, LAA, MV, and roof."""

    LIPV = "LIPV"
    LSPV = "LSPV"
    RSPV = "RSPV"
    RIPV = "RIPV"
    LAA = "LAA"
    MV = "MV"
    ROOF = "roof"


# "mitral" names the 4-PV LA pipeline's group of connectors tying the PVs and LAA to the mitral
# valve side of the atrium (clinically, mitral-isthmus-type lines).
MITRAL = "mitral"


# --------------------------------------------------------------------------------------------------
class RoofConnector(StrEnum):
    """The four shortest paths directly connecting adjacent PVs' inner boundary loops."""

    LIPV_LSPV = "LIPV_LSPV"
    LSPV_RSPV = "LSPV_RSPV"
    RSPV_RIPV = "RSPV_RIPV"
    RIPV_LIPV = "RIPV_LIPV"


# --------------------------------------------------------------------------------------------------
class MitralConnector(StrEnum):
    """The connections tying each PV, the LAA, and the MV together."""

    LIPV_LAA = "LIPV_LAA"
    LAA_MV = "LAA_MV"
    LSPV_MV = "LSPV_MV"
    RSPV_MV = "RSPV_MV"
    RIPV_MV = "RIPV_MV"
    LAA_LATERAL = "LAA_lateral"
    LAA_POSTERIOR = "LAA_posterior"


# --------------------------------------------------------------------------------------------------
class MitralVariant(StrEnum):
    """The direction-split variants of a `MitralConnector`, split by a mid-path marker.

    Used as `parameterization_configs` or marker keys; each value is `"<connector>_<direction>"`.
    """

    LSPV_MV_ANTERIOR = "LSPV_MV_anterior"
    LSPV_MV_LATERAL = "LSPV_MV_lateral"
    RIPV_MV_SEPTAL = "RIPV_MV_septal"
    RIPV_MV_POSTERIOR = "RIPV_MV_posterior"


# --------------------------------------------------------------------------------------------------
class SegmentId(StrEnum):
    """Identifiers for the non-PV `Segment`s that don't already have a name from elsewhere.

    The other three (`AtrialStructure.ROOF`, `AnatomicalDirection.ANTERIOR`,
    `AnatomicalDirection.SEPTAL`) double as their own `Segment`'s key already -- this covers
    only the ones that don't.
    """

    POSTERIOR_ROOF = "posterior_roof"
    POSTERIOR_MV = "posterior_mv"
    LATERAL_ROOF = "lateral_roof"
    LATERAL_MV = "lateral_mv"
    LAA = "laa"


# --------------------------------------------------------------------------------------------------
class OstiumSegment(StrEnum):
    """The three segments around a PV ostium, split by its spokes."""

    SEGMENT_1 = "segment_1"
    SEGMENT_2 = "segment_2"
    SEGMENT_3 = "segment_3"
