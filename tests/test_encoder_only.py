"""无相机真实快照的精度、完整性与身份边界；不连接硬件。"""
import hashlib
import json

import pytest

from glove_calibrate.encoder_only import (
    ScanContinuity, build_metadata, parse_board_serial, parse_factory_block, parse_snapshot,
)
from glove_calibrate.usb.dgst_protocol import ProtocolError


def snapshot_text(sequence=17, timestamp=2**53 + 1234567):
    counts = [0, 1, 32768, 65535] + list(range(17))
    header = (f"sequence={sequence} timestamp_ns={timestamp} scan_us=846.875 "
              "period_us=5000.000 installed=0x1fffff transfer_ok=0x1fffff "
              "suspect=0x000009 overruns=3 scan_errors=0")
    rows = [f"CS{cs:02d} raw={raw:5d} angle={raw * 360 / 65536:9.4f} deg "
            f"transfer=ok{' suspect-rail' if cs in (0, 3) else ''} errors=0"
            for cs, raw in enumerate(counts)]
    return header + "\n" + "\n".join(rows)


def test_counts_and_scan_midpoint_do_not_lose_precision():
    sample = parse_snapshot(snapshot_text())
    assert sample["raw_counts_by_cs"][:4] == [0, 1, 32768, 65535]
    assert sample["snapshot_timestamp_ns"] == 2**53 + 1234567
    assert sample["scan_duration_ns"] == 846875
    assert sample["timestamp_ns"] == 2**53 + 1234567 - 846875 // 2
    assert sample["suspect_mask"] == 9  # 极值有真实证据，不擅自变成缺失或角度零位。


def test_incomplete_or_duplicate_channels_cannot_be_a_valid_frame():
    rows = snapshot_text().splitlines()
    with pytest.raises(ProtocolError):
        parse_snapshot("\n".join(rows[:-1]))
    rows[-1] = rows[-2]
    with pytest.raises(ProtocolError):
        parse_snapshot("\n".join(rows))


def test_spi_failure_is_not_published_as_an_angle():
    with pytest.raises(ProtocolError):
        parse_snapshot(snapshot_text().replace("transfer_ok=0x1fffff", "transfer_ok=0x1ffffe"))


def test_scan_gaps_are_preserved_duplicates_ignored_and_reversal_rejected():
    continuity = ScanContinuity()
    first = parse_snapshot(snapshot_text(sequence=100, timestamp=2**53 + 1000000))
    later = parse_snapshot(snapshot_text(sequence=105, timestamp=2**53 + 26000000))
    assert continuity.accept(first)
    assert not continuity.accept(first)
    assert continuity.accept(later)
    assert later["scan_sequence"] == 105
    with pytest.raises(ProtocolError):
        continuity.accept(first)


def factory_record():
    return {"schema": "dataglove.factory", "schema_version": 1,
            "device_id": "80fa9df13e5a285d", "hand": "left",
            "cs_by_joint": list(reversed(range(21)))}


def factory_block(factory):
    payload = json.dumps(factory).encode()
    header = f"DGSTDATA1 factory 1 {len(payload)} {hashlib.sha256(payload).hexdigest()}\n".encode()
    return (header + payload).ljust(65536, b"\xff")


def test_factory_corruption_or_other_board_cannot_silently_select_a_profile():
    block = factory_block(factory_record())
    damaged = block.replace(b'"hand": "left"', b'"hand": "righ"', 1)
    with pytest.raises(ProtocolError):
        parse_factory_block(damaged, board_device_id="80fa9df13e5a285d")
    with pytest.raises(ProtocolError):
        parse_factory_block(block, board_device_id="1111111111111111")
    assert parse_factory_block(b"\xff" * 65536, board_device_id="80fa9df13e5a285d") is None


def test_generated_identity_survives_reboot_without_overriding_factory_wiring():
    factory = parse_factory_block(factory_block(factory_record()), board_device_id="80fa9df13e5a285d")
    first, _ = build_metadata(board_device_id="80fa9df13e5a285d", boot_id="1feb0418-0caa-469d-b187-af73bedd9d91", hand="left", factory=factory)
    rebooted, _ = build_metadata(board_device_id="80fa9df13e5a285d", boot_id="2feb0418-0caa-469d-b187-af73bedd9d91", hand="right", factory=factory)
    other, _ = build_metadata(board_device_id="1111111111111111", boot_id="1feb0418-0caa-469d-b187-af73bedd9d91", hand="left", factory=None)
    assert first.device["device_id"] == rebooted.device["device_id"]
    assert first.device["device_id"] != other.device["device_id"]
    assert first.device["device_id"] != first.device["board_device_id"]
    assert rebooted.device["hand"] == "left"
    assert rebooted.sensors["encoder"]["cs_by_joint"] == factory["cs_by_joint"]


def test_factory_identity_retains_significant_serial_leading_zeros():
    serial = "0080fa9df13e5a285d"
    parsed_serial = parse_board_serial(f"Serial\\t: {serial.upper()}\n".replace("\\t", "\t"))
    factory = {**factory_record(), "device_id": serial}
    assert parse_factory_block(factory_block(factory), board_device_id=parsed_serial)["device_id"] == serial
