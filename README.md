# gpulock — GPU 锁检测 TUI

一个只读的小工具：在终端里快速判断 NVIDIA Ampere（GA102/GA104）以及
CMP 50HX（TU102）显卡是否带有 **SM 发射率软件锁**，以及锁住了多少项、
每项被降到多少速率。

它读取的是 BAR0 里的寄存器，不跑分、不改卡、不写 GPU 寄存器。适合在买卡、收卡、
装 [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) 之前做一次快速核对。

## 它判断的“锁”是什么

不是 LHR 挖矿锁。这里说的是消费卡上 **FP16/BF16 → FP32 累加** 张量吞吐被砍半的
产品档位限制，由两根基线熔丝决定：

| 熔丝 | BAR0 偏移 | 消费卡 | 专业卡 |
|---|---|---:|---:|
| `FUSE_SS_FMLA32` | `0x008207D8` | `1`（1/2 速率） | `0`（满速） |
| `FUSE_SS_IMLA4` | `0x008207EC` | `1`（1/2 速率） | `0`（满速） |

九项发射单元（DP、FFMA、FMLA16、FMLA32、IMLA0–3、IMLA4）的完整熔丝值都会列出来。
`gpulock` 用 `FEATURE_READOUT_1`（`0x00823818`）作为交叉验证：

- `0x00400080` — RTX 30 消费卡锁定档（两项半速）
- `0x00000000` — 九项全满速（无锁，或本次上电已解锁）
- `0x016DB6ED` — CMP 170HX 重锁档（八项 1/32，DP 为 1 位降低）

## 安装 / 运行

只依赖 Python 3 标准库（`curses`、`mmap` 等），不需要 pip 安装任何东西。

```bash
# TUI（读取寄存器需要 root）
sudo python3 gpulock.py

# 不开 TUI，直接打印报告
sudo python3 gpulock.py --once

# JSON，方便脚本处理
sudo python3 gpulock.py --json

# 没有卡也能看界面和解码逻辑
python3 gpulock.py --mock 3090ti
python3 gpulock.py --mock 170hx --once
```

TUI 按键：

| 键 | 作用 |
|---|---|
| `↑` / `↓` / `PgUp` / `PgDn` | 滚动 |
| `Tab` / `←` / `→` | 切换显卡 |
| `r` / `F5` | 重新读取 |
| `q` / `Esc` | 退出 |

多卡机器会按 BDF 排成标签页；每张卡单独读取。

## 输出示例

```text
==============================================================================
GPU   NVIDIA GeForce RTX 3090 Ti
BDF   0000:21:00.0    PCI ID   10de:2203  (subsys 1462:5090)
芯片  GA102    BAR0  16 MiB    gpu-unlock 解锁支持  是
------------------------------------------------------------------------------
硬件  有锁：消费级半速档（2/9 项 × 1/2）
锁定量  [##-------] 2/9 项
当前  受限（按熔丝，最高 ×1/2）
影响  FP16/BF16→FP32 张量：约 1/2；整数 MLA：约 1/2；FP32 FFMA：满速

寄存器
  FEATURE_READOUT_1   0x00823818  0x00400080 RTX 30 消费卡锁定档（2 项 × 1/2）
  FEAT_OVR_PLM        0x00823804  0xffffff8f L3-only 写保护（正常门闩）
  SS0                 0x0082381c  0x30403100 0/8 项 override 启用
  SS1                 0x00823820  0x00000000 0/1 项 override 启用

熔丝（OTP，硬件档位）
  单元       地址              值  倍率     状态
  DP       0x00820224     0  满速     满速
  FFMA     0x0082059c     0  满速     满速
  FMLA16   0x008207d4     0  满速     满速
  FMLA32   0x008207d8     1  1/2    锁定
  IMLA0    0x008207dc     0  满速     满速
  ...
  IMLA4    0x008207ec     1  1/2    锁定
```

如果卡已经被 `gpu-unlock` 在本次上电解锁，会看到：

```text
硬件  有锁：消费级半速档（2/9 项 × 1/2）
当前  满速（SS0/SS1 已被覆盖）
```

也就是说：**“含锁”是熔丝层面的硬件档位；“当前满速”是运行时被覆盖。**
断电后熔丝仍然是 1，下一次启动需要重新跑解锁模块。

## BAR0 读不出来怎么办

`gpulock` 是用户态程序，需要 root + 对 `/sys/bus/pci/devices/<BDF>/resource0`
做只读 mmap。常见问题：

1. **没有 sudo**：会提示 `需要 root（sudo）`。
2. **`EBUSY`**：NVIDIA 驱动占着 BAR0。对**非显示卡**可以临时解绑：

   ```bash
   echo 0000:21:00.0 | sudo tee /sys/bus/pci/drivers/nvidia/unbind
   sudo python3 gpulock.py --bdf 0000:21:00.0 --once
   echo 0000:21:00.0 | sudo tee /sys/bus/pci/drivers/nvidia/bind
   ```

   显示卡不要随便解绑，会黑屏；用 SSH 或 live USB。
3. **全读回 `0xffffffff` / `0xbadf…`**：卡可能处于 D3、未上电，或容器里没有
   `CAP_SYS_RAWIO`。试试冷启动后立刻读，或换 live USB。
4. **容器里 EPERM**：容器通常 drop 了 `CAP_SYS_RAWIO`，在宿主机上跑。

没有 root 时，工具仍会根据 PCI ID / 型号给出“推断”结论：

- 在 `gpu-unlock` 解锁支持列表（GA102/GA104 消费卡）→ 通常有锁
- RTX A / Quadro / Tesla → 通常无锁
- 其他芯片 → 无法仅凭型号判断

## 判读规则

- `锁定量 [##-------] 2/9 项`：九项发射单元里有两项被降速。
- 消费级 RTX 30：`FMLA32` 和 `IMLA4` = 1（各 1/2），所以 FP16/BF16→FP32
  张量路径大约是满速的一半；FP32 FFMA 不受影响。
- CMP 170HX：八项熔丝是 5（1/32），DP 是 1 位降低；这是另一套重锁，不在 `gpu-unlock` 的支持范围。
- 专业卡（A5000/A6000 等）：九项全 0，本来就满速，不需要解锁。
- `SS0`/`SS1` 是**逐 die 的运行时值**，原厂每张卡都可能不同，不能单独用来判断
  “有没有锁”；`FEAT_OVR_PLM = 0xffffff8f` 也是所有 Ampere 卡都一样的门闩，
  不是锁的判据。
- `FEATURE_READOUT_1` 是只读的有效值，最适合做“当前是否满速”的交叉验证。

## 支持范围

寄存器偏移和熔丝含义来自：

- [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock)（GA102/GA104 设备列表、SS0/SS1、PLM）
- [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx) 的寄存器文档
  （`FEATURE_READOUT_1`、九项 `FUSE_SS_*`、消费档/专业档/CMP 档的实测值）

工具主要面向 GA102/GA104。其他 Ampere 芯片的偏移可能不同，结果仅供参考；
40 系（AD10x）不在 `gpu-unlock` 的支持范围，也不应套用这里的结论。

本工具只做检测，不解锁。确认有锁且卡在解锁支持列表里后，按 `gpu-unlock` 的 README
安装模块；解锁发生在 initramfs 里、`nvidia` 绑定之前，启动后手动 `modprobe` 无效。

## 检测范围 vs 解锁范围

先区分两件事：

- **检测**：读 BAR0 里的熔丝/读回值，判断这张卡当前是什么档位。
- **解锁**：`gpu-unlock` 只支持 GA102/GA104 的特定设备 ID，因为它需要对应的 V67 固件和 SEC2 Booter。

所以工具里的 `gpu-unlock 解锁支持：否` **不代表不能检测**。只要寄存器偏移对得上，
专业卡、CMP、甚至其他 Ampere SKU 都可以读。

## “全系显卡”能支持到什么程度

### Ampere（GA100 / GA10x）：可以，基本是同一套偏移

公开的 15 卡 / 11 SKU 探测数据（A100 SXM4 40G、A100 PCIe 40G/80G、A10、A16、
A5000、A6000、RTX 3080、3080 Ti、3090、3090 Ti、CMP 170HX、Drive A100）显示：

- `FEATURE_READOUT_1` 在 `0x00823818`
- `FEAT_OVR_PLM` 在 `0x00823804`
- `FEAT_OVR_SM_SPD` / `_1` 在 `0x0082381C` / `0x00823820`
- 九项 `FUSE_SS_*` 在 `0x00820224` … `0x008207EC`

而且分档很干净：

| 卡档 | FMLA32 / IMLA4 | FEATURE_READOUT_1 |
|---|---:|---:|
| 消费级 GeForce | `1` / `1` | `0x00400080` |
| 专业/数据中心 | `0` / `0` | `0x00000000` |
| CMP 170HX | `5` / `5` | `0x016DB6ED` |

所以 **Ampere 全系做“读取 + 判断”是现实的**，工具现在的解码逻辑已经不依赖设备 ID。
例外是 **A16**：它的 BAR0 只有 256 KB，大部分熔丝寄存器超出 BAR0，读不到，只能报未知。
GA106/GA107 不在上述 15 卡样本里，偏移大概率相同，但需要实卡确认。

### Turing 及更老：不能直接套用

Turing 的 fuse 块地址就和 Ampere 不同（例如 `NV_FUSE_OPT_SECURE_GSP_DEBUG_DIS`
在 TU102 是 `0x0002174C`，Ampere 是 `0x0082074C`），`0x00823814` 也不是它的
`FEATURE_READOUT`。而且这些卡不一定存在同一套“SM 发射率选择”机制，限速方式可能是
FP64 比率、双 FP32、NVLink 等别的产品分层。**换架构就必须重新找寄存器、重新验证语义。**
例外是 CMP 50HX 的 TU102 FECS 块，已经单独做了 profile（见下文）。

### Ada / Hopper / Blackwell：寄存器块会搬家

- Ada（AD10x）：`gpu-unlock` issue #4 明确说 AD103 的 PLM/SS 可写掩码和速度编码还没解决；
  能读不代表能判断，更不能套 Ampere 的结论。
- Hopper（GH100）：公开头文件里没有 `0x00823814` 这个 `FEATURE_READOUT`。
- Blackwell（GB100）：`NV_FUSE_ZB_FEATURE_READOUT` 在 `0x00003814`，不是 `0x00823814`。

结论：**“全系”如果指 Ampere 全系，可以；如果指所有 NVIDIA 代，不能靠一个固定偏移。**
正确做法是“芯片 ID → 架构 profile → 对应偏移/语义”，未知架构只做 raw dump，不硬下结论。

如果要做成真正的全系工具，下一步是把 `gpulock.py` 重构成：

```text
profiles/
  ampere_ga10x.py   # 已验证：当前这套偏移
  ada_ad10x.py      # TODO：待实卡/文档验证
  turing_tu10x.py   # TODO：不同 fuse 块
  ...
```

再加一个 `--raw` 模式，把 `0x00823800`–`0x00823FFC` 和 fuse 块原样 dump 成 JSON。
这样新卡可以先收集数据，再补 profile，而不是猜一个偏移就宣布“支持”。

## CMP 50HX / TU102

`gpulock` 现在也带一个 **TU102 CMP 50HX** 的 profile。它和 Ampere 不是同一套寄存器：

| 寄存器 | BAR0 偏移 | 作用 |
|---|---|---:|
| `FECS_PLM` | `0x00409650` | FECS feature-override 写保护；`0xffffff8f` = 正常，`0xffffffff` = 已打开 |
| `FECS_SS0` | `0x00409664` | SM speed override word 0；满速 `0x88888888` |
| `FECS_READOUT` | `0x00409668` | 九项有效读出（只读） |
| `FECS_SS1` | `0x0040966C` | SM speed override word 1；满速 `0x8` |

`0x409668` 的字段布局来自 `xrip/cmp50hx-unlock` 的文档：

| 字段 | 位 |
|---|---:|
| `IMLA0` | `2:0` |
| `FMLA16` | `5:3` |
| `DP` | `6` |
| `FMLA32` | `9:7` |
| `FFMA` | `12:10` |
| `IMLA1` | `15:13` |
| `IMLA2` | `18:16` |
| `IMLA3` | `21:19` |
| `IMLA4` | `24:22` |

每个字段 0 = 满速，1 = 1/2，2 = 1/4，以此类推；CMP 50HX 原厂主要把
`FFMA`（FP32-FMA）和 `IMLA0`（DP4A）压到 1/2。EFI 解锁后 `SS0=0x88888888`、
`SS1=0x8`，`0x409668` 应回到 0。

```bash
# 用 mock 看 CMP 50HX 的报告
python3 gpulock.py --mock cmp50hx
python3 gpulock.py --mock cmp50hx_unlocked

# 实卡
sudo python3 gpulock.py --bdf 0000:01:00.0 --once
```

注意：

- `--mock cmp50hx` 的 `0x401` 是按文档推断的 FFMA+IMLA0 半速示例，不是实测值；
  实卡读回多少就显示多少。
- `gpu-unlock` 不支持 TU102；CMP 50HX 用 [`xrip/cmp50hx-unlock`](https://github.com/xrip/cmp50hx-unlock)
  （UEFI 预启动解锁，或 patched kernel module）。
- 这个 profile 只覆盖 CMP 50HX / TU102 的 FECS 块；其他 Turing（TU104/TU106/TU116/TU117）
  没有验证，不要默认套用。
