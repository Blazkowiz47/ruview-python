"""NumPy helpers for CSI amplitude/phase conversion and window stacking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.core import CsiFrame


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def complex_amplitude(data: ArrayLike) -> FloatArray:
    """Return the magnitude of complex CSI samples."""

    return np.abs(np.asarray(data, dtype=np.complex128)).astype(np.float64, copy=False)


def complex_phase(data: ArrayLike) -> FloatArray:
    """Return wrapped phase in radians for complex CSI samples."""

    return np.angle(np.asarray(data, dtype=np.complex128)).astype(np.float64, copy=False)


def complex_to_amplitude_phase(data: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Convert complex CSI samples to ``(amplitude, phase)`` arrays."""

    complex_data = np.asarray(data, dtype=np.complex128)
    return complex_amplitude(complex_data), complex_phase(complex_data)


def amplitude_phase_to_complex(amplitude: ArrayLike, phase: ArrayLike) -> ComplexArray:
    """Convert polar CSI arrays back to complex samples."""

    amplitude_array = np.asarray(amplitude, dtype=np.float64)
    phase_array = np.asarray(phase, dtype=np.float64)
    try:
        broadcast_amplitude, broadcast_phase = np.broadcast_arrays(amplitude_array, phase_array)
    except ValueError as exc:
        raise ValueError("amplitude and phase must be broadcast-compatible") from exc
    return (broadcast_amplitude * np.exp(1j * broadcast_phase)).astype(np.complex128, copy=False)


@dataclass(frozen=True)
class CsiWindow:
    """CSI frames stacked as ``[time, spatial_stream, subcarrier]`` tensors."""

    frames: tuple[CsiFrame, ...]
    data: ComplexArray
    amplitude: FloatArray
    phase: FloatArray

    @classmethod
    def from_frames(cls, frames: Iterable[CsiFrame]) -> "CsiWindow":
        frame_tuple = tuple(frames)
        if not frame_tuple:
            raise ValueError("CsiWindow requires at least one CsiFrame")

        first_shape = frame_tuple[0].data.shape
        for index, frame in enumerate(frame_tuple):
            if frame.data.shape != first_shape:
                raise ValueError(
                    "all CsiFrame objects must have matching data shapes; "
                    f"frame 0 has {first_shape}, frame {index} has {frame.data.shape}"
                )

        data = np.stack([np.asarray(frame.data, dtype=np.complex128) for frame in frame_tuple])
        amplitude = np.stack(
            [np.asarray(frame.amplitude, dtype=np.float64) for frame in frame_tuple]
        )
        phase = np.stack([np.asarray(frame.phase, dtype=np.float64) for frame in frame_tuple])
        return cls(frame_tuple, data, amplitude, phase)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self.data.shape)

    @property
    def timestamps_ns(self) -> NDArray[np.int64]:
        return np.asarray(
            [frame.metadata.timestamp.as_nanos() for frame in self.frames],
            dtype=np.int64,
        )

    @property
    def sequence_numbers(self) -> NDArray[np.uint32]:
        return np.asarray([frame.metadata.sequence_number for frame in self.frames], dtype=np.uint32)

    def __len__(self) -> int:
        return len(self.frames)


def stack_csi_frames(frames: Sequence[CsiFrame] | Iterable[CsiFrame]) -> CsiWindow:
    """Stack frames into a ``CsiWindow``."""

    return CsiWindow.from_frames(frames)
