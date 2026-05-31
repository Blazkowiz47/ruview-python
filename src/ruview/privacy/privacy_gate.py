"""Monotonic BFLD privacy-class demotion."""

from __future__ import annotations

from dataclasses import replace

from ruview.privacy.bfld import (
    BfldFrame,
    BfldPayload,
    InvalidDemote,
    PrivacyClass,
    crc32_of_payload,
)


class PrivacyGate:
    """Apply irreversible section stripping as frames move to stricter classes."""

    @staticmethod
    def demote(frame: BfldFrame, target: PrivacyClass) -> BfldFrame:
        target = PrivacyClass(target)
        current = PrivacyClass.from_u8(frame.header.privacy_class)
        if target.as_u8() < current.as_u8():
            raise InvalidDemote(from_class=current.as_u8(), to_class=target.as_u8())

        out = BfldFrame(replace(frame.header), bytes(frame.payload))
        try:
            payload = out.parse_payload()
        except ValueError:
            out.header.privacy_class = target.as_u8()
            out.header.payload_crc32 = crc32_of_payload(out.payload)
            return out

        if target.as_u8() >= PrivacyClass.Anonymous.as_u8():
            payload = BfldPayload(
                compressed_angle_matrix=b"",
                amplitude_proxy=payload.amplitude_proxy,
                phase_proxy=payload.phase_proxy,
                snr_vector=payload.snr_vector,
                csi_delta=None,
                vendor_extension=payload.vendor_extension,
            )

        if target.as_u8() >= PrivacyClass.Restricted.as_u8():
            payload = BfldPayload(
                compressed_angle_matrix=payload.compressed_angle_matrix,
                amplitude_proxy=b"",
                phase_proxy=b"",
                snr_vector=payload.snr_vector,
                csi_delta=payload.csi_delta,
                vendor_extension=payload.vendor_extension,
            )

        out = BfldFrame.from_payload(out.header, payload)
        out.header.privacy_class = target.as_u8()
        out.resync_payload_metadata()
        return out
