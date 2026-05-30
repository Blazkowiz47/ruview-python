"""Compare deterministic synthetic CSI scenarios from the hardware simulator."""

from __future__ import annotations

import argparse

import numpy as np

from ruview.core import CsiFrame
from ruview.hardware import SyntheticCsiConfig, generate_synthetic_sequence


def summarize(label: str, frames: list[CsiFrame]) -> tuple[str, float, float, float, str]:
    window = np.stack([frame.data for frame in frames], axis=0)
    amplitude = np.abs(window)
    mean_amplitude = float(np.mean(amplitude))
    amplitude_std = float(np.std(amplitude))
    temporal_motion = float(np.mean(np.var(amplitude, axis=0)))
    witness = frames[0].witness_hash().hex()
    return label, mean_amplitude, amplitude_std, temporal_motion, witness[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--streams", type=int, default=3)
    parser.add_argument("--subcarriers", type=int, default=56)
    args = parser.parse_args()

    config = SyntheticCsiConfig(
        seed=args.seed,
        frames=args.frames,
        streams=args.streams,
        subcarriers=args.subcarriers,
    )

    rows = [
        summarize("empty_room", generate_synthetic_sequence("empty_room", config)),
        summarize("person_present", generate_synthetic_sequence("person_present", config)),
        summarize("stillness", generate_synthetic_sequence("stillness", config)),
        summarize("walking", generate_synthetic_sequence("walking", config)),
    ]

    print(
        f"synthetic CSI: seed={config.seed} frames={config.frames} "
        f"shape=({config.streams}, {config.subcarriers}) "
        f"sample_rate={config.sample_rate_hz:.1f} Hz"
    )
    print("scenario          mean_amp  amp_std  temporal_var  first_witness")
    for label, mean_amplitude, amplitude_std, temporal_motion, witness in rows:
        print(
            f"{label:<16}  {mean_amplitude:8.3f}  {amplitude_std:7.3f}  "
            f"{temporal_motion:12.5f}  {witness}"
        )


if __name__ == "__main__":
    main()
