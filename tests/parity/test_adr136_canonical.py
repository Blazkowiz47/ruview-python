from __future__ import annotations

import numpy as np

from ruview.core import CsiFrame, CsiMetadata, FrameId, FrequencyBand, Timestamp


def test_fixed_adr136_witness_vector() -> None:
    metadata = CsiMetadata(
        "node-1",
        FrequencyBand.BAND_5_GHZ,
        36,
        timestamp=Timestamp(1_700_000_000, 123),
        sequence_number=99,
    )
    metadata.set_model(7, 0x0102)
    frame = CsiFrame(
        metadata,
        np.array([[0.5 + 0j, 1.0 + 0.25j], [1.5 - 0.5j, 2.0 + 0.75j]], dtype=np.complex128),
        id=FrameId.from_uuid("00000000-0000-0000-0000-000000000001"),
    )

    assert frame.witness_hash().hex() == "9ac4fdb9b6b9b2cca62b6ac6751a50400d49146d16435a7d3aed8350e01d983e"
