#!/usr/bin/env python3
"""gpulock — inspect NVIDIA Ampere SM speed-selector locks.

Read-only: maps BAR0 and reads registers. It never writes GPU registers.
The only possible write is the standard PCI ``enable`` bit if memory decoding
is off, which is what a driver would do before touching BAR0.
"""

from __future__ import annotations

import argparse
import curses
import errno
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

VERSION = "0.1.0"
SYS_PCI = Path("/sys/bus/pci/devices")

# BAR0 offsets (GA102/GA104 FEATURE block and OPT fuses).
PMC_BOOT_0 = 0x00000000
FEAT_OVR_PLM = 0x00823804
FEAT_READOUT_1 = 0x00823818
FEAT_SS0 = 0x0082381C
FEAT_SS1 = 0x00823820
FUSE_SS_PLM = 0x008200FC

# TU102 / CMP 50HX FECS block (not the Ampere FEATURE block).
TU102_FEAT_PLM = 0x00409650
TU102_FEAT_SS0 = 0x00409664
TU102_FEAT_READOUT = 0x00409668
TU102_FEAT_SS1 = 0x0040966C
TU102_FIELDS = (
    ("IMLA0", 0, 3),
    ("FMLA16", 3, 3),
    ("DP", 6, 1),
    ("FMLA32", 7, 3),
    ("FFMA", 10, 3),
    ("IMLA1", 13, 3),
    ("IMLA2", 16, 3),
    ("IMLA3", 19, 3),
    ("IMLA4", 22, 3),
)
CMP50HX_ID = 0x1E09

# (name, BAR0 offset, is one-bit DP fuse)
FUSE_UNITS = (
    ("DP", 0x00820224, True),
    ("FFMA", 0x0082059C, False),
    ("FMLA16", 0x008207D4, False),
    ("FMLA32", 0x008207D8, False),
    ("IMLA0", 0x008207DC, False),
    ("IMLA1", 0x008207E0, False),
    ("IMLA2", 0x008207E4, False),
    ("IMLA3", 0x008207E8, False),
    ("IMLA4", 0x008207EC, False),
)

ALL_OFFSETS = (
    PMC_BOOT_0,
    FEAT_OVR_PLM,
    FEAT_READOUT_1,
    FEAT_SS0,
    FEAT_SS1,
    FUSE_SS_PLM,
    TU102_FEAT_PLM,
    TU102_FEAT_SS0,
    TU102_FEAT_READOUT,
    TU102_FEAT_SS1,
) + tuple(addr for _, addr, _ in FUSE_UNITS)

# GA102/GA104 GeForce + CMP 90HX IDs from Zagitta/gpu-unlock main.c.
SUPPORTED_IDS = {
    0x2200, 0x2203, 0x2204, 0x2205, 0x2206, 0x2207, 0x2208, 0x220A,
    0x220D, 0x2216, 0x222B, 0x222F,
    0x2480, 0x2482, 0x2483, 0x2484, 0x2486, 0x2487, 0x2488, 0x2489,
    0x248A, 0x248C, 0x248D, 0x248E, 0x249C, 0x249D, 0x249F, 0x24A0,
    0x24A4, 0x24AC, 0x24AD, 0x24AF, 0x24BF, 0x24C0, 0x24C7, 0x24C8,
    0x24C9, 0x24DC, 0x24DD, 0x24DF, 0x24E0,
}

CHIP_NAMES = {
    0x170000A1: "GA100",
    0xB72000A1: "GA102",
    0xB74000A1: "GA104",
    0x162000A1: "TU102",
}

READOUT_LABELS = {
    0x00000000: "全满速（无锁 / 已解锁）",
    0x00400080: "RTX 30 消费卡锁定档（2 项 × 1/2）",
    0x016DB6ED: "CMP 170HX 重锁档（最大 1/32）",
}


class BarError(Exception):
    """BAR0 cannot be read for a user-actionable reason."""


# ---------------------------------------------------------------------------
# small helpers


def read_hex_file(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip(), 16)
    except (OSError, ValueError):
        return None


def bad_value(value: Optional[int]) -> bool:
    """True for an unreadable / PRI-violation / all-ones register value."""
    return value is None or value == 0xFFFFFFFF or (value & 0xFFFF0000) == 0xBADF0000


def bad_plm(value: Optional[int]) -> bool:
    """PLM 0xffffffff is a valid 'fully open' value, unlike other registers."""
    return value is None or (value & 0xFFFF0000) == 0xBADF0000


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def truncate_display(text: str, width: int) -> str:
    out = []
    used = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + cw > width:
            break
        out.append(ch)
        used += cw
    return "".join(out)


def pad_display(text: str, width: int) -> str:
    text = truncate_display(text, width)
    return text + " " * max(0, width - display_width(text))


def wrap_display(text: str, width: int) -> list[str]:
    if width <= 1:
        return [text]
    lines: list[str] = []
    current = ""
    used = 0
    for ch in text:
        if ch == "\n":
            lines.append(current)
            current = ""
            used = 0
            continue
        cw = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + cw > width:
            lines.append(current)
            current = ""
            used = 0
        current += ch
        used += cw
    lines.append(current)
    return lines or [""]


def norm_bdf(text: str) -> str:
    text = text.strip().lower()
    match = re.match(r"^(?:[0-9a-f]{4}){2}:([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])$", text)
    if match:
        return "0000:" + match.group(1)
    return text


def is_pro_name(name: str) -> bool:
    upper = name.upper()
    if any(key in upper for key in ("RTX A", "QUADRO", "TESLA", "CMP 170HX")):
        return True
    return bool(re.search(r"\bA(4000|5000|6000|100|10)\b", upper))


def arch_from_name(name: str) -> Optional[str]:
    match = re.search(r"\b(GA100|GA10\d|AD10\d|TU10\d|GP10\d|GV100|GH100|GB10\d)", name.upper())
    return match.group(1) if match else None


def style_for_level(level: str) -> str:
    return {
        "ok": "ok",
        "locked": "bad",
        "overridden": "warn",
        "warn": "warn",
        "error": "bad",
    }.get(level, "dim")


# ---------------------------------------------------------------------------
# device discovery


def nvidia_smi_map() -> dict[str, tuple[str, Optional[int]]]:
    """BDF -> (marketing name, PCI device id)."""
    queries = (
        "index,name,pci.bus_id,pci.device_id,pci.subsystem_id",
        "index,name,pci.bus_id,pci.device_id",
        "index,name,pci.bus_id",
    )
    for query in queries:
        try:
            proc = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if proc.returncode != 0:
            continue
        result: dict[str, tuple[str, Optional[int]]] = {}
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            name, bdf = parts[1], norm_bdf(parts[2])
            dev_id = None
            if len(parts) >= 4:
                try:
                    dev_id = int(parts[3], 16) & 0xFFFF
                except ValueError:
                    pass
            result[bdf] = (name, dev_id)
        if result:
            return result
    return {}


def lspci_names() -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        proc = subprocess.run(
            ["lspci", "-nn", "-D"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return result
    if proc.returncode != 0:
        return result
    for line in proc.stdout.splitlines():
        match = re.match(
            r"^(\S+)\s+.*?:\s+(.*?)\s+\[10de:([0-9a-fA-F]{4})\]", line
        )
        if not match:
            continue
        name = match.group(2)
        if name.startswith("NVIDIA Corporation "):
            name = name[len("NVIDIA Corporation ") :]
        result[match.group(1).lower()] = name
    return result


def enumerate_devices() -> list["Device"]:
    smi = nvidia_smi_map()
    lspci = lspci_names()
    devices: list[Device] = []
    try:
        entries = sorted(SYS_PCI.iterdir(), key=lambda item: item.name)
    except OSError:
        return devices

    for path in entries:
        if read_hex_file(path / "vendor") != 0x10DE:
            continue
        pci_class = read_hex_file(path / "class")
        if pci_class is None or (pci_class >> 16) != 0x03:
            continue

        bdf = path.name.lower()
        dev_id = read_hex_file(path / "device") or 0
        sub_vendor = read_hex_file(path / "subsystem_vendor")
        sub_device = read_hex_file(path / "subsystem_device")
        subsys = (
            f"{sub_vendor:04x}:{sub_device:04x}"
            if sub_vendor is not None and sub_device is not None
            else None
        )
        boot_vga = False
        try:
            boot_vga = (path / "boot_vga").read_text().strip() == "1"
        except OSError:
            pass

        bar0_size = 0
        try:
            first = (path / "resource").read_text().splitlines()[0].split()
            bar0_size = int(first[1], 16) - int(first[0], 16) + 1
        except (OSError, ValueError, IndexError):
            pass

        smi_entry = smi.get(bdf)
        name = (smi_entry[0] if smi_entry else None) or lspci.get(bdf) or f"NVIDIA GPU {dev_id:04x}"
        devices.append(
            Device(
                bdf=bdf,
                name=name,
                device_id=dev_id,
                subsys=subsys,
                boot_vga=boot_vga,
                bar0_size=bar0_size,
                pci_name=lspci.get(bdf),
            )
        )
    return devices


# ---------------------------------------------------------------------------
# BAR0 access


def read_bar0(bdf: str, offsets: tuple[int, ...]) -> dict[int, Optional[int]]:
    if os.geteuid() != 0:
        raise BarError("需要 root（sudo）才能读取 BAR0")
    base = SYS_PCI / bdf
    resource0 = base / "resource0"
    if not resource0.exists():
        raise BarError("找不到 resource0（设备可能没有 BAR0）")

    try:
        start_text, end_text, _ = (base / "resource").read_text().splitlines()[0].split()
        size = int(end_text, 16) - int(start_text, 16) + 1
    except (OSError, ValueError, IndexError) as exc:
        raise BarError(f"无法读取 BAR 信息：{exc}") from exc
    if size <= 0:
        raise BarError("BAR0 大小为 0")

    # Usually already on when a driver is bound.  This is the standard PCI
    # memory-enable bit, not a GPU register.
    try:
        enable = base / "enable"
        if enable.exists() and enable.read_text().strip() == "0":
            enable.write_text("1\n")
    except OSError:
        pass

    fd: Optional[int] = None
    bar: Optional[mmap.mmap] = None
    try:
        fd = os.open(resource0, os.O_RDONLY)
        bar = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        values: dict[int, Optional[int]] = {}
        for offset in offsets:
            if offset < 0 or offset + 4 > size:
                values[offset] = None
            else:
                values[offset] = struct.unpack_from("<I", bar, offset)[0]
        if all(value is None for value in values.values()):
            raise BarError(f"目标寄存器偏移超出 BAR0 范围（BAR0 只有 {size} 字节）")
        if all(bad_value(value) for value in values.values()):
            raise BarError(
                "BAR0 全部读回 0xffffffff / 0xbadf…：设备可能处于 D3、未上电或 BAR 未启用"
            )
        return values
    except PermissionError as exc:
        raise BarError(f"权限不足：{exc}") from exc
    except OSError as exc:
        if exc.errno == errno.EBUSY:
            raise BarError(
                "BAR0 被 NVIDIA 驱动占用（EBUSY）。请在 nvidia 未绑定时读取，"
                "或先解绑非显示卡：echo <BDF> | sudo tee /sys/bus/pci/drivers/nvidia/unbind"
            ) from exc
        raise BarError(f"打开/映射 BAR0 失败：{exc}") from exc
    finally:
        if bar is not None:
            try:
                bar.close()
            except Exception:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass


def read_device(device: "Device") -> None:
    if device.mock:
        return
    device.registers = {}
    device.error = None
    try:
        device.registers = read_bar0(device.bdf, ALL_OFFSETS)
    except BarError as exc:
        device.error = str(exc)


# ---------------------------------------------------------------------------
# decoding


@dataclass
class Device:
    bdf: str
    name: str
    device_id: int
    subsys: Optional[str]
    boot_vga: bool
    bar0_size: int
    registers: dict[int, Optional[int]] = field(default_factory=dict)
    error: Optional[str] = None
    mock: bool = False
    pci_name: Optional[str] = None


@dataclass
class FuseInfo:
    name: str
    addr: int
    is_dp: bool
    raw: Optional[int]
    label: str
    divisor: Optional[int]
    locked: bool


@dataclass
class Report:
    chip: Optional[str]
    readout: Optional[int]
    readout_label: str
    plm: Optional[int]
    ss0: Optional[int]
    ss1: Optional[int]
    fuses: list[FuseInfo]
    locked: list[FuseInfo]
    max_divisor: int
    hardware_profile: str
    effective: str
    impact: list[str]
    notes: list[str]
    fallback: Optional[str]
    level: str
    unit_source: str = "熔丝（OTP，硬件档位）"
    readout_name: str = "FEATURE_READOUT_1"
    readout_addr: int = FEAT_READOUT_1
    plm_name: str = "FEAT_OVR_PLM"
    plm_addr: int = FEAT_OVR_PLM
    ss0_name: str = "SS0"
    ss0_addr: int = FEAT_SS0
    ss1_name: str = "SS1"
    ss1_addr: int = FEAT_SS1


def rate_for(raw: Optional[int], is_dp: bool = False) -> tuple[Optional[int], str]:
    if bad_value(raw):
        return None, "不可读"
    assert raw is not None
    if raw == 0:
        return 1, "满速"
    if is_dp:
        return (2, "降低") if raw == 1 else (None, f"code {raw}")
    if 1 <= raw <= 6:
        divisor = 1 << raw
        return divisor, f"1/{divisor}"
    return None, f"code {raw}"


def _build_tu102_report(device: Device, regs: dict[int, Optional[int]], chip_raw: Optional[int]) -> Report:
    """CMP 50HX is TU102: the lock lives in the FECS block, not 0x8238xx."""
    readout = regs.get(TU102_FEAT_READOUT)
    plm = regs.get(TU102_FEAT_PLM)
    ss0 = regs.get(TU102_FEAT_SS0)
    ss1 = regs.get(TU102_FEAT_SS1)

    fuses: list[FuseInfo] = []
    for name, shift, width in TU102_FIELDS:
        if bad_value(readout):
            raw: Optional[int] = None
        else:
            assert readout is not None
            raw = (readout >> shift) & ((1 << width) - 1)
        divisor, label = rate_for(raw, is_dp=(name == "DP"))
        fuses.append(
            FuseInfo(
                name=name,
                addr=TU102_FEAT_READOUT,
                is_dp=(name == "DP"),
                raw=raw,
                label=label,
                divisor=divisor,
                locked=raw is not None and raw != 0,
            )
        )
    locked = [fuse for fuse in fuses if fuse.locked]
    readable = [fuse for fuse in fuses if fuse.raw is not None]
    max_divisor = max((fuse.divisor or 1) for fuse in locked) if locked else 1

    if not readable:
        hardware_profile = "未知（TU102 FECS 读出不可读）"
    elif not locked:
        if not bad_value(readout) and readout == 0:
            hardware_profile = "读回 9/9 项满速（0x0；无法区分原厂无锁/已覆盖）"
        else:
            hardware_profile = "无锁（9/9 项满速）"
    elif (
        {fuse.name for fuse in locked} == {"FFMA", "IMLA0"}
        and all(fuse.raw == 1 for fuse in locked)
    ):
        hardware_profile = "有锁：CMP 50HX 典型档（FFMA + DP4A × 1/2）"
    else:
        hardware_profile = f"有锁：{len(locked)}/9 项，最高 ×1/{max_divisor}"
    if readable and len(readable) < 9:
        hardware_profile += f"（仅 {len(readable)}/9 项可读）"

    ss0_full = not bad_value(ss0) and ss0 == 0x88888888
    ss1_full = not bad_value(ss1) and ss1 == 0x8
    if ss0_full and ss1_full:
        effective = "满速（SS0/SS1 已被覆盖）"
        level = "overridden"
    elif not bad_value(readout) and readout == 0:
        effective = "满速（FECS 读回全 0）"
        level = "overridden" if locked else "ok"
    elif locked:
        effective = f"受限（按 FECS 读出，最高 ×1/{max_divisor}）"
        level = "locked"
    elif readable:
        effective = "满速（FECS 字段全 0）"
        level = "ok"
    else:
        effective = "未知（FECS 读出不可读）"
        level = "unknown"

    overridden = ss0_full and ss1_full
    impact: list[str] = []
    for label, unit in (
        ("FP32 FFMA", "FFMA"),
        ("INT8 DP4A", "IMLA0"),
        ("FP16/BF16→FP32 张量", "FMLA32"),
        ("FP16 half2", "FMLA16"),
    ):
        fuse = next((item for item in fuses if item.name == unit), None)
        if fuse is None or fuse.raw is None:
            impact.append(f"{label}：不可读")
        elif overridden:
            impact.append(f"{label}：满速（已覆盖）")
        elif fuse.divisor == 1:
            impact.append(f"{label}：满速")
        elif fuse.divisor is None:
            impact.append(f"{label}：code {fuse.raw}")
        else:
            impact.append(f"{label}：约 {fuse.label}")

    notes: list[str] = [
        "CMP 50HX 是 TU102：锁在 FECS 0x409650 / 0x409664 / 0x40966C，不是 Ampere 的 0x823xxx。",
        "gpu-unlock 不支持 TU102；解锁用 xrip/cmp50hx-unlock（UEFI 或 patched driver）。",
        "0x409668 的九项字段布局来自 cmp50hx 文档；原厂主要限制 FFMA 和 IMLA0（DP4A）。",
    ]
    if device.mock:
        notes.append("当前为 mock 数据；CMP 50HX 的 0x401 是按文档推断的 FFMA+IMLA0 半速示例。")
    if plm == 0xFFFFFFFF:
        notes.append("PLM=0xffffffff：本次上电执行过解锁（或 EFI 留下了打开状态）。")
    elif plm == 0xFFFFFF8F:
        notes.append("PLM=0xffffff8f：正常的 FECS 写保护门闩。")
    if overridden and locked:
        notes.append("FECS 读回仍显示锁定字段，但 SS0/SS1 已被覆盖为满速；断电后恢复。")

    fallback = None if device.registers else model_hint(device)
    return Report(
        chip="TU102" if bad_value(chip_raw) else CHIP_NAMES.get(chip_raw, "TU102"),
        readout=None if bad_value(readout) else readout,
        readout_label="TU102 FECS 有效读出" if not bad_value(readout) else "不可读",
        plm=None if bad_plm(plm) else plm,
        ss0=None if bad_value(ss0) else ss0,
        ss1=None if bad_value(ss1) else ss1,
        fuses=fuses,
        locked=locked,
        max_divisor=max_divisor,
        hardware_profile=hardware_profile,
        effective=effective,
        impact=impact,
        notes=notes,
        fallback=fallback,
        level=level,
        unit_source="FECS 读出字段（0x409668）",
        readout_name="FECS_READOUT",
        readout_addr=TU102_FEAT_READOUT,
        plm_name="FECS_PLM",
        plm_addr=TU102_FEAT_PLM,
        ss0_name="FECS_SS0",
        ss0_addr=TU102_FEAT_SS0,
        ss1_name="FECS_SS1",
        ss1_addr=TU102_FEAT_SS1,
    )


def build_report(device: Device) -> Report:
    regs = device.registers
    chip_raw = regs.get(PMC_BOOT_0)
    chip = CHIP_NAMES.get(chip_raw) if not bad_value(chip_raw) else None
    arch_name = arch_from_name(device.pci_name or device.name)
    if device.device_id == CMP50HX_ID:
        return _build_tu102_report(device, regs, chip_raw)
    ampere = chip in CHIP_NAMES.values() or bool(
        arch_name and arch_name.startswith(("GA10", "GA100"))
    )
    readout = regs.get(FEAT_READOUT_1)
    plm = regs.get(FEAT_OVR_PLM)
    ss0 = regs.get(FEAT_SS0)
    ss1 = regs.get(FEAT_SS1)

    if bad_value(readout):
        readout_label = "不可读"
    elif readout in READOUT_LABELS:
        readout_label = READOUT_LABELS[readout]  # type: ignore[index]
    else:
        readout_label = f"未知档位（{readout:#010x}）"  # type: ignore[union-attr]

    fuses: list[FuseInfo] = []
    for name, addr, is_dp in FUSE_UNITS:
        raw = regs.get(addr)
        divisor, label = rate_for(raw, is_dp)
        fuses.append(
            FuseInfo(
                name=name,
                addr=addr,
                is_dp=is_dp,
                raw=None if bad_value(raw) else raw,
                label=label,
                divisor=divisor,
                locked=not bad_value(raw) and raw != 0,
            )
        )
    locked = [fuse for fuse in fuses if fuse.locked]
    readable = [fuse for fuse in fuses if not bad_value(fuse.raw)]
    max_divisor = max((fuse.divisor or 1) for fuse in locked) if locked else 1

    if not readable:
        hardware_profile = "未知（熔丝不可读）"
    elif not locked:
        hardware_profile = "无锁（9/9 项满速）"
    elif (
        {fuse.name for fuse in locked} == {"FMLA32", "IMLA4"}
        and all(fuse.raw == 1 for fuse in locked)
    ):
        hardware_profile = "有锁：消费级半速档（2/9 项 × 1/2）"
    elif len(locked) == 9 and all(
        fuse.raw == (1 if fuse.is_dp else 5) for fuse in fuses
    ):
        hardware_profile = "有锁：CMP 重锁档（8 项 × 1/32 + DP 降低）"
    else:
        hardware_profile = f"有锁：{len(locked)}/9 项，最高 ×1/{max_divisor}"
    if readable and len(readable) < 9:
        hardware_profile += f"（仅 {len(readable)}/9 项可读）"

    ss0_full = not bad_value(ss0) and ss0 == 0x88888888
    ss1_full = not bad_value(ss1) and ss1 == 0x8

    if ss0_full and ss1_full:
        effective = "满速（SS0/SS1 已被覆盖）"
        level = "overridden"
    elif ss0_full or ss1_full:
        effective = "部分覆盖（只写了一部分选择器）"
        level = "warn"
    elif not bad_value(readout) and readout == 0:
        effective = "满速（FEATURE_READOUT_1 = 0）"
        level = "overridden" if locked else "ok"
    elif locked:
        effective = f"受限（按熔丝，最高 ×1/{max_divisor}）"
        level = "locked"
    elif readable:
        effective = "满速（熔丝全 0）"
        level = "ok"
    else:
        effective = "未知（寄存器不可读）"
        level = "unknown"

    overridden = ss0_full and ss1_full
    impact: list[str] = []
    for label, unit in (
        ("FP16/BF16→FP32 张量", "FMLA32"),
        ("整数 MLA", "IMLA4"),
        ("FP32 FFMA", "FFMA"),
    ):
        fuse = next((item for item in fuses if item.name == unit), None)
        if fuse is None or fuse.raw is None:
            impact.append(f"{label}：不可读")
        elif fuse.divisor is None:
            impact.append(f"{label}：code {fuse.raw}")
        elif overridden:
            impact.append(f"{label}：满速（已覆盖）")
        elif fuse.divisor == 1:
            impact.append(f"{label}：满速")
        else:
            impact.append(f"{label}：约 {fuse.label}")

    notes: list[str] = []
    if device.error:
        notes.append(f"BAR0 读取失败：{device.error}")
        notes.append("用 sudo 运行；若仍 EBUSY，解绑非显示卡，或从 live USB 冷启动读取。")
    if device.mock:
        notes.append("当前为 mock 数据，仅用于演示 UI 与解码逻辑。")
    if readout == 0x016DB6ED:
        notes.append("这是 CMP 170HX（GA100），不在 gpu-unlock 的目标范围；用 cmp170hx 项目。")
    elif device.device_id in SUPPORTED_IDS:
        notes.append("此卡在 gpu-unlock 解锁支持列表内（GA102/GA104）；检测本身不依赖该列表。")
    elif is_pro_name(device.name):
        notes.append("专业卡（RTX A / Quadro / Tesla）通常出厂满速，不需要解锁。")
    else:
        notes.append("不在 gpu-unlock 解锁支持列表；检测仍可读，但不能用该项目解锁（无对应固件）。")
        arch = arch_from_name(device.pci_name or device.name) or chip
        if arch and arch.startswith(("GA10", "GA100")):
            notes.append("Ampere 架构：检测偏移与 GA102/GA104 同属一个 FEATURE 块。")
    if plm == 0xFFFFFFFF:
        notes.append("PLM=0xffffffff：本次上电执行过解锁（断电后会恢复 0xffffff8f）。")
    elif plm == 0xFFFFFF8F:
        notes.append("PLM=0xffffff8f：正常门闩，所有 Ampere 卡都一样，不是锁的判据。")
    if overridden and locked:
        notes.append("熔丝仍是锁定档，但 SS0/SS1 已被覆盖为满速；这是运行时解锁，不是永久改卡。")

    fallback = None
    if not device.registers:
        fallback = model_hint(device)
    if device.registers and not ampere:
        raw = f"{chip_raw:#010x}" if not bad_value(chip_raw) else "不可读"
        hardware_profile = f"未知架构（PMC_BOOT_0={raw}）；未验证"
        effective = "未知（该架构的锁语义未验证）"
        level = "unknown"
        notes.append("此卡不在已验证的 Ampere 列表内；熔丝/读回值只作原始参考，不能按本文档下结论。")

    return Report(
        chip=chip,
        readout=None if bad_value(readout) else readout,
        readout_label=readout_label,
        plm=None if bad_plm(plm) else plm,
        ss0=None if bad_value(ss0) else ss0,
        ss1=None if bad_value(ss1) else ss1,
        fuses=fuses,
        locked=locked,
        max_divisor=max_divisor,
        hardware_profile=hardware_profile,
        effective=effective,
        impact=impact,
        notes=notes,
        fallback=fallback,
        level=level,
    )


def model_hint(device: Device) -> str:
    if device.device_id == CMP50HX_ID:
        return "CMP 50HX（TU102）：需读 FECS 0x409668；解锁用 xrip/cmp50hx-unlock"
    if device.device_id in SUPPORTED_IDS:
        return "按 PCI ID：在 gpu-unlock 解锁支持列表内（GA102/GA104 消费档，通常有锁）"
    if is_pro_name(device.name):
        return "按型号：专业卡（通常无锁 / 满速）"
    return "不在 gpu-unlock 解锁支持列表；无法仅凭型号判断"


def ss_summary(value: Optional[int], count: int) -> str:
    if bad_value(value):
        return "不可读"
    assert value is not None
    enabled = sum(1 for index in range(count) if (value >> (4 * index)) & 0x8)
    return f"{enabled}/{count} 项 override 启用"


def lock_bar(count: int, total: int, readable: int) -> str:
    if readable == 0:
        return "[?] 未知"
    filled = max(0, min(count, total))
    return "[" + "#" * filled + "-" * (total - filled) + f"] {count}/{total} 项"


# ---------------------------------------------------------------------------
# plain / JSON output


def format_plain(device: Device) -> str:
    report = build_report(device)
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"GPU   {device.name}")
    lines.append(f"BDF   {device.bdf}    PCI ID   10de:{device.device_id:04x}" + (f"  (subsys {device.subsys})" if device.subsys else ""))
    chip = report.chip or arch_from_name(device.pci_name or device.name) or "未知"
    bar0 = f"{device.bar0_size // (1024 * 1024)} MiB" if device.bar0_size else "未知"
    supported = "是" if device.device_id in SUPPORTED_IDS else "否"
    lines.append(f"芯片  {chip}    BAR0  {bar0}    gpu-unlock 解锁支持  {supported}")
    lines.append("-" * 78)
    lines.append(f"硬件  {report.hardware_profile}")
    lines.append(f"锁定量  {lock_bar(len(report.locked), len(report.fuses), sum(1 for f in report.fuses if f.raw is not None))}")
    lines.append(f"当前  {report.effective}")
    if report.fallback:
        lines.append(f"推断  {report.fallback}")
    lines.append("影响  " + "；".join(report.impact))
    if device.error:
        lines.append(f"错误  {device.error}")
    lines.append("")
    lines.append("寄存器")
    lines.append(f"  {report.readout_name:<19} {report.readout_addr:#010x}  {fmt_reg(report.readout):<11} {report.readout_label}")
    lines.append(f"  {report.plm_name:<19} {report.plm_addr:#010x}  {fmt_reg(report.plm):<11} {plm_label(report.plm)}")
    lines.append(f"  {report.ss0_name:<19} {report.ss0_addr:#010x}  {fmt_reg(report.ss0):<11} {ss_summary(report.ss0, 8)}")
    lines.append(f"  {report.ss1_name:<19} {report.ss1_addr:#010x}  {fmt_reg(report.ss1):<11} {ss_summary(report.ss1, 1)}")
    lines.append("")
    lines.append(report.unit_source)
    lines.append(
        f"  {pad_display('单元', 8)} {pad_display('地址', 12)} {'值':>4}  {pad_display('倍率', 6)} 状态"
    )
    for fuse in report.fuses:
        raw = "—" if fuse.raw is None else str(fuse.raw)
        status = "锁定" if fuse.locked else ("满速" if fuse.raw is not None else "不可读")
        lines.append(
            f"  {pad_display(fuse.name, 8)} {pad_display(f'{fuse.addr:#010x}', 12)} {raw:>4}  "
            f"{pad_display(fuse.label, 6)} {status}"
        )
    lines.append("")
    lines.append("说明")
    for note in report.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)


def fmt_reg(value: Optional[int]) -> str:
    return "—" if value is None else f"{value:#010x}"


def plm_label(value: Optional[int]) -> str:
    if value is None:
        return "不可读"
    if value == 0xFFFFFFFF:
        return "已打开（本次上电解锁过）"
    if value == 0xFFFFFF8F:
        return "写保护（正常门闩）"
    return "其他值"


def report_json(device: Device) -> dict:
    report = build_report(device)
    return {
        "bdf": device.bdf,
        "name": device.name,
        "device_id": f"0x{device.device_id:04x}",
        "subsystem": device.subsys,
        "boot_vga": device.boot_vga,
        "bar0_size": device.bar0_size,
        "supported": device.device_id in SUPPORTED_IDS,
        "error": device.error,
        "registers": {
            f"0x{offset:08x}": (None if value is None else f"0x{value:08x}")
            for offset, value in sorted(device.registers.items())
        },
        "report": {
            "chip": report.chip,
            "hardware_profile": report.hardware_profile,
            "effective": report.effective,
            "level": report.level,
            "readout": report.readout,
            "readout_label": report.readout_label,
            "readout_name": report.readout_name,
            "readout_address": f"0x{report.readout_addr:08x}",
            "plm": report.plm,
            "ss0": report.ss0,
            "ss1": report.ss1,
            "max_divisor": report.max_divisor,
            "fuses": [
                {
                    "name": fuse.name,
                    "address": f"0x{fuse.addr:08x}",
                    "raw": fuse.raw,
                    "rate": fuse.label,
                    "divisor": fuse.divisor,
                    "locked": fuse.locked,
                }
                for fuse in report.fuses
            ],
            "unit_source": report.unit_source,
            "impact": report.impact,
            "notes": report.notes,
        },
    }


# ---------------------------------------------------------------------------
# mock devices for demo / tests


MOCKS = {
    "3090ti": ("NVIDIA GeForce RTX 3090 Ti", 0x2203, 0xB72000A1, 0x00400080, 0x30403100, 0x00000000, 0xFFFFFF8F, {"FMLA32": 1, "IMLA4": 1}),
    "3090": ("NVIDIA GeForce RTX 3090", 0x2204, 0xB72000A1, 0x00400080, 0x06740057, 0x00000003, 0xFFFFFF8F, {"FMLA32": 1, "IMLA4": 1}),
    "3080ti": ("NVIDIA GeForce RTX 3080 Ti", 0x2205, 0xB72000A1, 0x00400080, 0x10551033, 0x00000000, 0xFFFFFF8F, {"FMLA32": 1, "IMLA4": 1}),
    "3080": ("NVIDIA GeForce RTX 3080", 0x2206, 0xB72000A1, 0x00400080, 0x03676064, 0x00000000, 0xFFFFFF8F, {"FMLA32": 1, "IMLA4": 1}),
    "a6000": ("NVIDIA RTX A6000", 0x2230, 0xB72000A1, 0x00000000, 0x14170072, 0x00000002, 0xFFFFFF8F, {}),
    "170hx": ("NVIDIA CMP 170HX", 0x20C2, 0x170000A1, 0x016DB6ED, 0x51261070, 0x00000003, 0xFFFFFF8F, {"DP": 1, "FFMA": 5, "FMLA16": 5, "FMLA32": 5, "IMLA0": 5, "IMLA1": 5, "IMLA2": 5, "IMLA3": 5, "IMLA4": 5}),
    "unlocked": ("NVIDIA GeForce RTX 3090 Ti (unlocked)", 0x2203, 0xB72000A1, 0x00000000, 0x88888888, 0x00000008, 0xFFFFFFFF, {"FMLA32": 1, "IMLA4": 1}),
}


TU102_MOCKS = {
    # Stock readout 0x401 is synthetic: FFMA=1 and IMLA0=1 (the documented
    # CMP 50HX clamp).  A live card may report additional fields.
    "cmp50hx": ("NVIDIA CMP 50HX", 0x1E09, 0x162000A1, 0x00000401, 0x00000000, 0x00000000, 0xFFFFFF8F),
    "cmp50hx_unlocked": ("NVIDIA CMP 50HX (unlocked)", 0x1E09, 0x162000A1, 0x00000000, 0x88888888, 0x00000008, 0xFFFFFFFF),
}


def mock_device(kind: str) -> Device:
    if kind in TU102_MOCKS:
        name, dev_id, chip, readout, ss0, ss1, plm = TU102_MOCKS[kind]
        regs: dict[int, Optional[int]] = {
            PMC_BOOT_0: chip,
            TU102_FEAT_PLM: plm,
            TU102_FEAT_SS0: ss0,
            TU102_FEAT_READOUT: readout,
            TU102_FEAT_SS1: ss1,
        }
        return Device(
            bdf="0000:21:00.0",
            name=name,
            device_id=dev_id,
            subsys="1462:371f",
            boot_vga=False,
            bar0_size=16 * 1024 * 1024,
            registers=regs,
            mock=True,
            pci_name="TU102 [CMP 50HX]",
        )
    name, dev_id, chip, readout, ss0, ss1, plm, fuse_values = MOCKS[kind]
    regs: dict[int, Optional[int]] = {
        PMC_BOOT_0: chip,
        FEAT_READOUT_1: readout,
        FEAT_SS0: ss0,
        FEAT_SS1: ss1,
        FEAT_OVR_PLM: plm,
        FUSE_SS_PLM: 0xFFFFFFFF,
    }
    for unit, addr, _ in FUSE_UNITS:
        regs[addr] = fuse_values.get(unit, 0)
    return Device(
        bdf="0000:21:00.0",
        name=name,
        device_id=dev_id,
        subsys="1462:5090",
        boot_vga=False,
        bar0_size=16 * 1024 * 1024,
        registers=regs,
        mock=True,
    )


# ---------------------------------------------------------------------------
# TUI


class App:
    def __init__(self, stdscr, devices: list[Device], args: argparse.Namespace):
        self.stdscr = stdscr
        self.devices = devices
        self.args = args
        self.selected = 0
        self.scroll = 0
        self.status = ""
        self.colors: dict[str, int] = {}
        self._ensure_read(self.devices[0])
        if os.geteuid() != 0:
            self.status = "未 root：仅型号推断；sudo 可读寄存器"

    # -- setup ---------------------------------------------------------

    def init_colors(self) -> None:
        if not curses.has_colors():
            return
        curses.start_color()
        try:
            curses.use_default_colors()
            background = -1
        except curses.error:
            background = curses.COLOR_BLACK
        definitions = {
            "ok": (curses.COLOR_GREEN, background),
            "bad": (curses.COLOR_RED, background),
            "warn": (curses.COLOR_YELLOW, background),
            "header": (curses.COLOR_CYAN, background),
            "title": (curses.COLOR_CYAN, background),
            "dim": (curses.COLOR_WHITE, background),
        }
        for index, (name, (foreground, bg)) in enumerate(definitions.items(), start=1):
            try:
                curses.init_pair(index, foreground, bg)
                self.colors[name] = curses.color_pair(index)
            except curses.error:
                self.colors[name] = 0

    def attr(self, style: str) -> int:
        value = self.colors.get(style, 0)
        if style in ("header", "title"):
            value |= curses.A_BOLD
        elif style == "dim":
            value |= curses.A_DIM
        elif style == "tab_sel":
            value |= curses.A_REVERSE | curses.A_BOLD
        return value

    # -- data ----------------------------------------------------------

    def _ensure_read(self, device: Device) -> None:
        if device.mock or device.registers or device.error:
            return
        read_device(device)

    def reselect(self, index: int) -> None:
        self.selected = index % len(self.devices)
        self.scroll = 0
        self.status = ""
        self._ensure_read(self.devices[self.selected])

    def refresh(self) -> None:
        device = self.devices[self.selected]
        if device.mock:
            self.status = "mock 数据无需刷新"
            return
        device.registers = {}
        device.error = None
        read_device(device)
        self.status = "已重新读取" if not device.error else device.error

    # -- drawing -------------------------------------------------------

    def safe_addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        try:
            self.stdscr.addstr(y, x, text, attr)
        except curses.error:
            pass

    def draw(self) -> None:
        stdscr = self.stdscr
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if width < 40 or height < 10:
            self.safe_addstr(0, 0, "Terminal too small", curses.A_BOLD)
            stdscr.refresh()
            return

        title = " GPU 锁检测器 — Ampere SM Speed Selector "
        self.safe_addstr(0, 0, pad_display(title, width), self.attr("title"))

        # GPU tabs.
        x = 0
        for index, device in enumerate(self.devices):
            short = re.sub(r"^NVIDIA (GeForce |RTX |CMP )?", "", device.name)
            label = f" {index}:{short} [{device.bdf}] "
            if x + display_width(label) > width:
                break
            style = "tab_sel" if index == self.selected else "dim"
            self.safe_addstr(1, x, label, self.attr(style))
            x += display_width(label)

        device = self.devices[self.selected]
        report = build_report(device)
        body_top = 2
        body_height = height - body_top - 1

        content: list[tuple[str, str]] = []
        content.append((f"硬件  {report.hardware_profile}", style_for_level(report.level)))
        content.append((f"锁定量  {lock_bar(len(report.locked), len(report.fuses), sum(1 for f in report.fuses if f.raw is not None))}", style_for_level(report.level)))
        content.append((f"当前  {report.effective}", style_for_level(report.level)))
        if report.fallback:
            content.append((f"推断  {report.fallback}", "warn"))
        content.append(("影响  " + "；".join(report.impact), "dim"))
        if device.error:
            content.append((f"错误  {device.error}", "bad"))
        content.append(("", "dim"))
        content.append(("设备", "header"))
        content.append((f"  BDF       {device.bdf}", ""))
        content.append((f"  型号      {device.name}", ""))
        content.append((f"  PCI ID    10de:{device.device_id:04x}" + (f"  subsys {device.subsys}" if device.subsys else ""), ""))
        content.append((f"  芯片      {report.chip or arch_from_name(device.pci_name or device.name) or '未知'}", ""))
        content.append((f"  BAR0      {device.bar0_size // (1024 * 1024)} MiB" if device.bar0_size else "  BAR0      未知", ""))
        content.append((f"  解锁支持  {'是' if device.device_id in SUPPORTED_IDS else '否'}", ""))
        content.append(("", "dim"))
        content.append(("寄存器", "header"))
        content.append((f"  {report.readout_name:<19} {report.readout_addr:#010x}  {fmt_reg(report.readout):<11} {report.readout_label}", ""))
        content.append((f"  {report.plm_name:<19} {report.plm_addr:#010x}  {fmt_reg(report.plm):<11} {plm_label(report.plm)}", ""))
        content.append((f"  {report.ss0_name:<19} {report.ss0_addr:#010x}  {fmt_reg(report.ss0):<11} {ss_summary(report.ss0, 8)}", ""))
        content.append((f"  {report.ss1_name:<19} {report.ss1_addr:#010x}  {fmt_reg(report.ss1):<11} {ss_summary(report.ss1, 1)}", ""))
        content.append(("", "dim"))
        content.append((report.unit_source, "header"))
        content.append((f"  {pad_display('单元', 8)} {pad_display('地址', 12)} {'值':>4}  {pad_display('倍率', 6)} 状态", "dim"))
        for fuse in report.fuses:
            raw = "—" if fuse.raw is None else str(fuse.raw)
            if fuse.locked:
                status, style = "锁定", "bad"
            elif fuse.raw is not None:
                status, style = "满速", "ok"
            else:
                status, style = "不可读", "dim"
            content.append(
                (
                    f"  {pad_display(fuse.name, 8)} {pad_display(f'{fuse.addr:#010x}', 12)} {raw:>4}  "
                    f"{pad_display(fuse.label, 6)} {status}",
                    style,
                )
            )
        content.append(("", "dim"))
        content.append(("说明", "header"))
        for note in report.notes:
            content.append((f"  - {note}", "dim"))

        wrapped: list[tuple[str, str]] = []
        for text, style in content:
            for part in wrap_display(text, width - 2):
                wrapped.append((part, style))

        max_scroll = max(0, len(wrapped) - body_height)
        self.scroll = max(0, min(self.scroll, max_scroll))
        for row in range(body_height):
            index = self.scroll + row
            if index >= len(wrapped):
                break
            text, style = wrapped[index]
            self.safe_addstr(body_top + row, 1, pad_display(text, width - 2), self.attr(style))

        footer = " q 退出   ↑↓ 滚动   Tab 切换   r 重读 "
        if self.status:
            footer += f"  {self.status}"
        self.safe_addstr(height - 1, 0, pad_display(footer, width), self.attr("dim"))
        stdscr.refresh()

    # -- loop ----------------------------------------------------------

    def run(self) -> None:
        curses.curs_set(0)
        self.stdscr.keypad(True)
        self.init_colors()
        while True:
            self.draw()
            key = self.stdscr.getch()
            if key in (ord("q"), 27):
                break
            if key in (curses.KEY_DOWN, ord("j")):
                self.scroll += 1
            elif key in (curses.KEY_UP, ord("k")):
                self.scroll = max(0, self.scroll - 1)
            elif key == curses.KEY_NPAGE:
                self.scroll += 10
            elif key == curses.KEY_PPAGE:
                self.scroll = max(0, self.scroll - 10)
            elif key in (curses.KEY_RIGHT, ord("\t")):
                self.reselect(self.selected + 1)
            elif key in (curses.KEY_LEFT, curses.KEY_BTAB):
                self.reselect(self.selected - 1)
            elif key in (ord("r"), curses.KEY_F5):
                self.refresh()


# ---------------------------------------------------------------------------
# main


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gpulock",
        description="Inspect NVIDIA Ampere SM speed-selector locks (read-only).",
    )
    parser.add_argument("--once", action="store_true", help="print a plain report and exit")
    parser.add_argument("--json", action="store_true", help="print JSON and exit")
    parser.add_argument(
        "--mock",
        choices=sorted(set(MOCKS) | set(TU102_MOCKS)),
        help="use synthetic register values (demo)",
    )
    parser.add_argument("--bdf", help="only inspect this PCI address")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.mock:
        devices = [mock_device(args.mock)]
    else:
        devices = enumerate_devices()
    if args.bdf:
        target = args.bdf.lower()
        devices = [device for device in devices if device.bdf.lower().endswith(target)]
    if not devices:
        print("没有找到 NVIDIA 显示设备。", file=sys.stderr)
        return 2

    if args.json:
        for device in devices:
            read_device(device)
        payload = {
            "version": VERSION,
            "gpus": [report_json(device) for device in devices],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.once or not sys.stdout.isatty():
        for device in devices:
            read_device(device)
            print(format_plain(device))
            print()
        return 0

    try:
        curses.wrapper(lambda screen: App(screen, devices, args).run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
