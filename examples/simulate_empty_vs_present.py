"""Small synthetic CSI example for the first visual lab."""

from __future__ import annotations

import numpy as np

from ruview.core import CsiFrame, CsiMetadata, FrequencyBand


def synthetic_frame(scale: float, phase_offset: float) -> CsiFrame:
    metadata = CsiMetadata("sim-node-1", FrequencyBand.BAND_5_GHZ, channel=36)
    subcarriers = np.arange(56, dtype=np.float64)
    streams = []
    for stream in range(3):
        amplitude = scale + 0.08 * np.sin(subcarriers / 6.0 + stream)
        phase = phase_offset + 0.15 * np.cos(subcarriers / 8.0 + stream)
        streams.append(amplitude * np.exp(1j * phase))
    return CsiFrame(metadata, np.vstack(streams))


def main() -> None:
    empty = synthetic_frame(scale=1.0, phase_offset=0.0)
    present = synthetic_frame(scale=1.35, phase_offset=0.3)
    print(f"empty mean amplitude:   {empty.mean_amplitude():.3f}")
    print(f"present mean amplitude: {present.mean_amplitude():.3f}")
    print(f"empty witness:          {empty.witness_hash().hex()}")


if __name__ == "__main__":
    main()

