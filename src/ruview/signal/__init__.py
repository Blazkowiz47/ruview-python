"""Signal-processing primitives for CSI arrays and frames."""

from ruview.signal.csi_processor import (
    CsiWindow,
    amplitude_phase_to_complex,
    complex_amplitude,
    complex_phase,
    complex_to_amplitude_phase,
    stack_csi_frames,
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
from ruview.signal.phase import unwrap_phase
from ruview.signal.subcarrier import subcarrier_variance

__all__ = [
    "CsiWindow",
    "HampelResult",
    "SignalFeatures",
    "amplitude_phase_to_complex",
    "calculate_motion_energy",
    "complex_amplitude",
    "complex_phase",
    "complex_to_amplitude_phase",
    "extract_frame_features",
    "extract_window_features",
    "hampel_filter",
    "min_max_normalize",
    "stack_csi_frames",
    "subcarrier_variance",
    "unwrap_phase",
    "zscore_normalize",
]
