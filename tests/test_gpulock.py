import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gpulock


class DecodeTests(unittest.TestCase):
    def test_consumer_profile(self):
        report = gpulock.build_report(gpulock.mock_device("3090ti"))
        self.assertEqual(report.level, "locked")
        self.assertIn("消费级", report.hardware_profile)
        self.assertEqual(len(report.locked), 2)
        self.assertEqual(report.max_divisor, 2)
        self.assertIn("约 1/2", report.impact[0])
        self.assertIn("满速", report.impact[2])

    def test_pro_card_has_no_lock(self):
        report = gpulock.build_report(gpulock.mock_device("a6000"))
        self.assertEqual(report.level, "ok")
        self.assertIn("无锁", report.hardware_profile)
        self.assertEqual(report.locked, [])
        self.assertEqual(report.max_divisor, 1)
        self.assertIn("满速", report.effective)

    def test_cmp_profile(self):
        report = gpulock.build_report(gpulock.mock_device("170hx"))
        self.assertEqual(report.level, "locked")
        self.assertIn("CMP", report.hardware_profile)
        self.assertEqual(len(report.locked), 9)
        self.assertEqual(report.max_divisor, 32)

    def test_overridden_profile(self):
        report = gpulock.build_report(gpulock.mock_device("unlocked"))
        self.assertEqual(report.level, "overridden")
        self.assertIn("已被覆盖", report.effective)
        self.assertEqual(report.plm, 0xFFFFFFFF)
        self.assertIn("满速（已覆盖）", report.impact[0])

    def test_cmp50hx_profile(self):
        report = gpulock.build_report(gpulock.mock_device("cmp50hx"))
        self.assertEqual(report.chip, "TU102")
        self.assertEqual(report.level, "locked")
        self.assertIn("CMP 50HX", report.hardware_profile)
        self.assertEqual({fuse.name for fuse in report.locked}, {"FFMA", "IMLA0"})
        self.assertEqual(report.max_divisor, 2)
        self.assertEqual(report.readout_addr, gpulock.TU102_FEAT_READOUT)
        self.assertIn("FECS", report.unit_source)

    def test_cmp50hx_unlocked_profile(self):
        report = gpulock.build_report(gpulock.mock_device("cmp50hx_unlocked"))
        self.assertEqual(report.level, "overridden")
        self.assertIn("已被覆盖", report.effective)
        self.assertEqual(report.ss0, 0x88888888)
        self.assertEqual(report.ss1, 0x8)
        self.assertIn("满速", report.impact[0])

    def test_cmp50hx_plain_output(self):
        text = gpulock.format_plain(gpulock.mock_device("cmp50hx"))
        self.assertIn("FECS_READOUT", text)
        self.assertIn("0x00409668", text)

    def test_rate_decoding(self):
        self.assertEqual(gpulock.rate_for(0), (1, "满速"))
        self.assertEqual(gpulock.rate_for(1), (2, "1/2"))
        self.assertEqual(gpulock.rate_for(5), (32, "1/32"))
        self.assertEqual(gpulock.rate_for(6), (64, "1/64"))
        self.assertEqual(gpulock.rate_for(1, is_dp=True), (2, "降低"))
        self.assertEqual(gpulock.rate_for(None)[0], None)

    def test_bad_values(self):
        self.assertTrue(gpulock.bad_value(None))
        self.assertTrue(gpulock.bad_value(0xFFFFFFFF))
        self.assertTrue(gpulock.bad_value(0xBADF5040))
        self.assertFalse(gpulock.bad_value(0))
        self.assertFalse(gpulock.bad_plm(0xFFFFFFFF))
        self.assertTrue(gpulock.bad_plm(0xBADF5040))

    def test_bar(self):
        self.assertEqual(gpulock.lock_bar(2, 9, 9), "[##-------] 2/9 项")
        self.assertEqual(gpulock.lock_bar(0, 9, 9), "[---------] 0/9 项")
        self.assertEqual(gpulock.lock_bar(0, 9, 0), "[?] 未知")

    def test_read_bar0_from_fake_sysfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "0000:01:00.0"
            base.mkdir()
            (base / "resource").write_text("0x0 0xffffff 0x40200\n")
            data = bytearray(0x1000000)
            struct.pack_into("<I", data, 0, 0xB72000A1)
            struct.pack_into("<I", data, gpulock.FEAT_READOUT_1, 0x00400080)
            (base / "resource0").write_bytes(data)

            old_sys_pci = gpulock.SYS_PCI
            gpulock.SYS_PCI = Path(tmp)
            try:
                with mock.patch("os.geteuid", return_value=0):
                    regs = gpulock.read_bar0(
                        "0000:01:00.0", (gpulock.PMC_BOOT_0, gpulock.FEAT_READOUT_1)
                    )
            finally:
                gpulock.SYS_PCI = old_sys_pci

        self.assertEqual(regs[gpulock.PMC_BOOT_0], 0xB72000A1)
        self.assertEqual(regs[gpulock.FEAT_READOUT_1], 0x00400080)

    def test_plain_output(self):
        text = gpulock.format_plain(gpulock.mock_device("3080ti"))
        self.assertIn("锁定量", text)
        self.assertIn("FMLA32", text)
        self.assertIn("有锁", text)


if __name__ == "__main__":
    unittest.main()
