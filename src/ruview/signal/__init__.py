"""Signal-processing primitives for CSI arrays and frames."""

from ruview.signal.csi_processor import (
    CsiWindow,
    amplitude_phase_to_complex,
    complex_amplitude,
    complex_phase,
    complex_to_amplitude_phase,
    stack_csi_frames,
)
from ruview.signal.baseline import (
    BaselineStats,
    DetectionDebouncer,
    RollingBaseline,
    adaptive_threshold,
    measure_baseline,
    relative_increase,
)
from ruview.signal.features import (
    SignalFeatures,
    calculate_motion_energy,
    extract_frame_features,
    extract_window_features,
    min_max_normalize,
    zscore_normalize,
)
from ruview.signal.hampel import HampelResult, hampel_filter
from ruview.signal.motion import MotionScore, MotionWeights, calculate_motion_score
from ruview.signal.phase import unwrap_phase
from ruview.signal.presence import PresenceResult, classify_presence
from ruview.signal.subcarrier import subcarrier_variance

__all__ = [
    "BaselineStats",
    "CsiWindow",
    "DetectionDebouncer",
    "HampelResult",
    "MotionScore",
    "MotionWeights",
    "PresenceResult",
    "RollingBaseline",
    "SignalFeatures",
    "adaptive_threshold",
    "amplitude_phase_to_complex",
    "calculate_motion_energy",
    "calculate_motion_score",
    "classify_presence",
    "complex_amplitude",
    "complex_phase",
    "complex_to_amplitude_phase",
    "extract_frame_features",
    "extract_window_features",
    "hampel_filter",
    "measure_baseline",
    "min_max_normalize",
    "relative_increase",
    "stack_csi_frames",
    "subcarrier_variance",
    "unwrap_phase",
    "zscore_normalize",
]
