# Android Low-Level CSI Driver Research Plan

## Purpose

Explore whether an Android device, starting with the Pixel 9 Pro if approved,
can be turned into a WiFi Channel State Information (CSI) receiver for
camera-free spatial sensing experiments.

This is a research feasibility plan, not an app plan. A normal Android app
cannot access CSI. The useful work is below Java/Kotlin: chipset
identification, kernel/driver inspection, firmware capability discovery, and a
safe data-export path if CSI can be exposed.

## Working Hypothesis

The WiFi chipset estimates channel state internally to decode packets, but
Android normally exposes only high-level metrics such as RSSI, link speed,
channel, MCS, and scan results. CSI collection is possible only if the chipset
firmware and driver can export per-packet channel estimates or can be patched to
do so.

## Safety Boundary

- Do not modify the Pixel 9 Pro until supervisor approval is explicit.
- Assume bootloader unlock/root may wipe the device and may affect warranty,
  device integrity, security posture, and normal use.
- Keep ESP32 CSI collection as the safe baseline path.
- Treat Android CSI as an optional high-risk research track.
- Prefer read-only reconnaissance before flashing, rooting, kernel changes, or
  firmware patching.
- Do not collect personal network traffic payloads. Store CSI and metadata only.

## Target Architecture

```text
WiFi transmitter
  -> packets over the room channel
Android WiFi chipset / firmware
  -> per-packet CSI if available
Android kernel driver
  -> debugfs, tracepoint, vendor command, ioctl, or netlink export
native collector daemon
  -> binary/JSONL/CSV stream
Python collector / ruview-python
  -> parsing, labeling, notebooks, model experiments
```

## Candidate Export Paths

1. Vendor driver already exposes CSI-like debug data.
2. Vendor driver can be patched to expose existing firmware messages.
3. Firmware can be patched Nexmon-style to emit CSI records.
4. Kernel monitor-mode radiotap or vendor event path exposes enough PHY
   metadata for a weaker fallback.
5. If none of the above work, abandon Pixel CSI and use ESP32/other NICs.

## Milestone 0: Approval And Device Baseline

Goal: prepare without modifying the phone.

- Get written approval to unlock/root/modify the Pixel 9 Pro.
- Record exact device model, Android version, build number, kernel version, and
  WiFi chipset/firmware identifiers.
- Confirm whether the device is spare/test-only or must remain daily-driver
  safe.
- Back up anything important before bootloader unlock.
- Define the rollback plan: factory image, locked/unlocked state, and who owns
  recovery if the device stops booting.

Deliverables:

- `docs/android-csi/device-baseline.md`
- `docs/android-csi/risk-and-rollback.md`

## Milestone 1: Chipset And Driver Recon

Goal: learn what WiFi stack is actually present.

- Identify WiFi chipset from `dmesg`, `/vendor/firmware`, `/sys`, kernel config,
  and Android properties.
- Identify kernel driver modules and vendor interfaces.
- Check whether sources are available through Android kernel source drops or
  vendor repositories.
- Search for CSI, channel estimate, sounding, beamforming, debug, radiotap,
  monitor, and vendor command hooks.
- Compare findings against Nexmon/Nexmon CSI supported chipsets and techniques.

Deliverables:

- `docs/android-csi/chipset-recon.md`
- Table of possible hook points and their risk level.

## Milestone 2: Non-Root And Rootless Limits

Goal: prove what cannot be done from normal Android APIs.

- Check Android `WifiManager`, scan results, RTT APIs, and diagnostics for
  exposed PHY data.
- Build a tiny Android app or adb script that logs all available WiFi metadata.
- Confirm no raw CSI is exposed through public APIs.
- Document why Java/Kotlin alone is not enough.

Deliverables:

- `docs/android-csi/android-api-limits.md`
- Optional `android/metadata_probe/` toy app or adb script.

## Milestone 3: Read-Only Root Recon

Goal: inspect low-level surfaces after root, without modifying firmware.

- Root only after approval.
- Inspect debugfs, tracefs, procfs, sysfs, vendor logs, and WiFi driver logs.
- Watch driver/vendor events during controlled packet traffic.
- Search for hidden diagnostic toggles or vendor commands.
- Confirm whether monitor mode, radiotap, or packet injection are available.

Deliverables:

- `docs/android-csi/root-recon.md`
- Captured log snippets with payload-free metadata only.

## Milestone 4: Kernel/User Export Prototype

Goal: expose any available CSI-like data without firmware patching.

- If driver symbols/events expose channel estimates, create a small native
  collector.
- Prefer vendor command, netlink, debugfs, tracepoint, or ioctl before kernel
  patching.
- Define a compact CSI record schema:
  - timestamp
  - source/destination MAC hashes
  - channel/bandwidth
  - RSSI/noise/MCS if available
  - subcarrier count
  - complex CSI vector or amplitude/phase vector
  - firmware/driver version
- Stream records to local file first, then UDP/USB to laptop.

Deliverables:

- `src/ruview_android_csi/schema.md`
- `android/native_collector/` prototype if feasible
- Small sample capture with synthetic/non-sensitive traffic.

## Milestone 5: Firmware Patch Feasibility

Goal: decide whether Nexmon-style work is realistic for this device.

- Identify firmware image format, signing/encryption constraints, load path,
  and crash/recovery path.
- Check whether the chipset family has public Nexmon support or similar work.
- Map where CSI would be generated in firmware if symbols/reversing permit.
- Decide whether firmware patching is worth the risk versus buying supported
  hardware.

Deliverables:

- `docs/android-csi/firmware-feasibility.md`
- Go/no-go recommendation.

## Milestone 6: Controlled Collection Experiment

Goal: collect a minimal labeled CSI dataset if export works.

- Use a controlled transmitter: ESP32, laptop, or router sending fixed-rate UDP.
- Keep Android receiver fixed on a stand.
- Collect sessions:
  - empty room
  - person standing between TX/RX
  - walking across link
  - seated still
  - breathing still
- Save labels, positions, device orientation, channel, bandwidth, and distance.
- Convert captures into `ruview-python` replay format.

Deliverables:

- `data/recordings/android-csi/README.md`
- first labeled capture set
- parser tests for the Android CSI record format
- notebook comparing Android CSI to ESP32 CSI baseline

## Milestone 7: Decision Gate

Goal: choose whether Android CSI remains a project track.

Continue only if:

- CSI-like vectors are available at usable rate.
- Captures are stable across reboots.
- Device can recover safely after crashes/flashes.
- Data shows human motion/presence signal above noise.
- Collection does not require fragile manual intervention every run.

Otherwise:

- Freeze Android work as reconnaissance.
- Continue data collection with ESP32 CSI receivers.

## Initial Questions To Answer

- What WiFi chipset and firmware does the Pixel 9 Pro use?
- Is the bootloader unlockable on this exact device?
- Is root acceptable for this device owner?
- Are kernel sources and matching build configs available?
- Does any driver/vendor log mention CSI, channel matrix, sounding, or beamform
  feedback?
- Can we collect monitor/radiotap metadata even if full CSI is unavailable?

## Recommended Parallel Baseline

While Android approval and recon are pending, continue with ESP32 data
collection:

- 5x ESP32-WROOM-32 / ESP-32S CP2102 USB-C boards
- ESP32-CSI-Tool
- 1 transmitter, 4 receivers
- fixed stands/clamps
- laptop serial collector

This keeps the sensing project moving even if Android CSI is not feasible.
