"""RuVector-style research helpers for CSI signal, geometry, and histories."""

from ruview.ruvector.bvp import attention_weighted_bvp, bvp_attention_weights
from ruview.ruvector.geometry import (
    CramerRaoBound,
    FresnelGeometryResult,
    GeometricDiversityIndex,
    GeometryError,
    ViewpointPosition,
    angular_distance,
    compute_effective_viewpoints,
    estimate_crb,
    estimate_crb_gdop,
    estimate_regularized_crb,
    geometric_diversity_index,
    solve_fresnel_geometry,
)
from ruview.ruvector.history import (
    CompressedBreathingBuffer,
    CompressedBreathingHistory,
    CompressedHeartbeatHistory,
    CompressedHeartbeatSpectrogram,
    FrameQuantizationMetadata,
    HistoryDiagnostics,
    RuVectorHistoryError,
    TierPolicy,
)
from ruview.ruvector.spectrogram import gate_spectrogram
from ruview.ruvector.subcarrier import (
    mincut_subcarrier_partition,
    subcarrier_importance_weights,
    subcarrier_partition,
)
from ruview.ruvector.triangulation import (
    SPEED_OF_LIGHT_M_S,
    TdoaMeasurement,
    TdoaResult,
    TriangulationError,
    solve_triangulation,
)

__all__ = [
    "CompressedBreathingBuffer",
    "CompressedBreathingHistory",
    "CompressedHeartbeatHistory",
    "CompressedHeartbeatSpectrogram",
    "CramerRaoBound",
    "FrameQuantizationMetadata",
    "FresnelGeometryResult",
    "GeometricDiversityIndex",
    "GeometryError",
    "HistoryDiagnostics",
    "RuVectorHistoryError",
    "SPEED_OF_LIGHT_M_S",
    "TdoaMeasurement",
    "TdoaResult",
    "TierPolicy",
    "TriangulationError",
    "ViewpointPosition",
    "angular_distance",
    "attention_weighted_bvp",
    "bvp_attention_weights",
    "compute_effective_viewpoints",
    "estimate_crb",
    "estimate_crb_gdop",
    "estimate_regularized_crb",
    "gate_spectrogram",
    "geometric_diversity_index",
    "mincut_subcarrier_partition",
    "solve_fresnel_geometry",
    "solve_triangulation",
    "subcarrier_importance_weights",
    "subcarrier_partition",
]
