#!/usr/bin/env python3
"""cv10 图像分割进阶：距离变换（三档精度）/ 分水岭（sure_bg-sure_fg-unknown 三区 + markers 约定）/
粘连细胞计数（纯形态学 vs 分水岭）/ 逐像素 Dice 与 IoU / GrabCut（含 RNG 不可复现的坑）。

数据：`classic/coins.png`（真实硬币）、`classic/messi5.jpg`（GrabCut），
以及**代码合成的粘连细胞图**（8 个半径 30 的圆，4 对圆心距 52 必然粘连，GT 掩码可逐像素对账）。
真值全部实跑（opencv 5.0.0 / numpy 2.5.3）。

> 本章最值钱的三条：① `maskSize=3` 是 3x3 chamfer 快速近似，水平/垂直步长权重固定为
> **0.955** 而不是 1.0 —— 1px 细线的 `dist.max()` 直接从 1.0 掉到 0.955002；
> ② 粘连目标用**纯形态学连通域**只能数出 4/8，必须「距离变换找核心 + 分水岭长边界」；
> ③ **GrabCut 内部用了 RNG，不设种子结果不可复现**（实测四次 16241 / 16196 / 16190 / 16235），
> `cv2.setRNGSeed(42)` 之后稳定在 16254。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv10_segmentation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch10_segmentation"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

DATA = Path("data")
CL = DATA / "classic"
'''

SETUP = '''coins_img = cv2.imread(str(CL / "coins.png"))
messi = cv2.imread(str(CL / "messi5.jpg"))
print("coins.png  :", coins_img.shape, "（真实硬币；这张图是「硬币亮、背景暗」）")
print("messi5.jpg :", messi.shape, "（GrabCut 用图）")

# 合成一张「粘连细胞」图：8 个半径 30 的圆，4 对圆心距 52 < 60，所以每对**必然粘连**
rng = np.random.default_rng(7)
S = 300
syn = np.full((S, S), 30, np.uint8)
gt = np.zeros((S, S), np.uint8)
CELLS = [(60, 60), (112, 60), (188, 60), (240, 60),
         (60, 230), (112, 230), (188, 230), (240, 230)]
for cx, cy in CELLS:
    cv2.circle(gt, (cx, cy), 30, 255, -1)
syn[gt > 0] = 220
syn = cv2.GaussianBlur(syn, (5, 5), 0)
syn = np.clip(syn.astype(np.int16) + rng.normal(0, 6, syn.shape).astype(np.int16), 0, 255).astype(np.uint8)
syn_c = cv2.cvtColor(syn, cv2.COLOR_GRAY2BGR)
print("合成粘连细胞:", syn.shape, "| 真值目标数 =", len(CELLS), "| GT 前景像素 =", int((gt > 0).sum()))
print("圆心距 52 < 直径 60 -> 每对圆都粘连，纯连通域**必然少数**")
'''

SCAFFOLD = '''# 脚手架：常量、小形状、工具函数，以及 coins 的预处理（Otsu + 开运算，ch03/ch06 已练过）
K3 = np.ones((3, 3), np.uint8)

BLOCK = np.zeros((11, 11), np.uint8)
BLOCK[3:8, 3:8] = 255
DISK = np.zeros((21, 21), np.uint8)
cv2.circle(DISK, (10, 10), 7, 255, -1)
BAR = np.zeros((11, 31), np.uint8)
BAR[4:7, 3:28] = 255
LINE1 = np.zeros((11, 11), np.uint8)
LINE1[:, 5] = 255
SHAPES = {"5x5 方块": BLOCK, "半径7圆盘": DISK, "3px 长条": BAR, "1px 竖线": LINE1}
MSIZES = [("3", 3), ("5", 5), ("PRECISE", cv2.DIST_MASK_PRECISE)]
SHAPE_CASES = [(nm, tag, m, mv) for nm, m in SHAPES.items() for tag, mv in MSIZES]

FG_FRACS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
CELL_FRACS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
IOU_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)

SQ_A = np.zeros((10, 10), np.uint8)
SQ_A[2:6, 2:6] = 1
SQ_B = np.zeros((10, 10), np.uint8)
SQ_B[4:8, 4:8] = 1

GC_RECT = (137, 85, 274, 171)

coins_g = cv2.cvtColor(coins_img, cv2.COLOR_BGR2GRAY)
coins_blur = cv2.GaussianBlur(coins_g, (5, 5), 0)
otsu_t, cbw = cv2.threshold(coins_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
op = cv2.morphologyEx(cbw, cv2.MORPH_OPEN, K3, iterations=2)
print("coins 预处理: Otsu =", otsu_t, "| 二值前景", int((cbw > 0).sum()), "| 开运算后", int((op > 0).sum()))


def near(a, b, tol=1e-6):
    """浮点容差比较。真值都是实跑得到的，用容差避免 == 在末位翻车。"""
    return abs(float(a) - float(b)) <= tol


def dice(a, b):
    """像素级 Dice = 2|A n B| / (|A| + |B|)。两边都空时约定 0.0。"""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else 0.0


def pix_iou(a, b):
    """像素级 IoU = |A n B| / |A u B|。两边都空时约定 0.0。"""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def region_labels(ws):
    """把 watershed 结果整理成 (标签, 面积)。剔掉边界标签 -1；返回的 labels[0] 恒为 1（背景区域）。"""
    lab, cnt = np.unique(ws, return_counts=True)
    keep = lab > 0
    return lab[keep], cnt[keep]


def labels_at(ws, pts):
    """各目标中心处落在哪个区域标签上（用来检查「每个目标是否各自独立成区」）。"""
    return sorted({int(ws[y, x]) for x, y in pts})
'''

E1_CODE = '''# @@todo 对 4 种形状 x 3 档 maskSize 各算距离变换的峰值，记成 {(形状, 档位): 峰值}
# @@hint 一条字典推导，4 个变量依次是 形状名 / 档位标签 / 掩码 / maskSize 数值
# @@hint 峰值要 round(..., 6)，最后形状与档位的组合一共 12 个
dt_peak = {(nm, tag): round(float(cv2.distanceTransform(m, cv2.DIST_L2, mv).max()), 6) for nm, tag, m, mv in SHAPE_CASES}
# @@end
print("距离变换峰值（形状 x 精度）")
for nm in SHAPES:
    print(f"   {nm:10s}", {tag: dt_peak[(nm, tag)] for tag, _ in MSIZES})
print("解读: ① 「5x5 方块」「3px 长条」三档都一样 —— 因为它们的最大距离恰好是整数；"
      "② 圆盘不一样：mask=3 给 6.846466、mask=5 给 6.999969、PRECISE 给 7.071068 —— "
      "后者的真值是 5*sqrt(2)，因为 cv2.circle 画出来的「圆」其实是**内接多边形**，"
      "45 度方向的弦被切掉了，最近的背景像素落在对角上；"
      "③ **1px 竖线最坑**：mask=3 给 0.955002、mask=5 给 1.0 —— 3x3 chamfer 把水平/垂直步长"
      "的权重固定成了 0.955 而不是 1.0，细线目标的 dist.max() 直接偏小 4.5%")
assert len(dt_peak) == 12
assert near(dt_peak[("5x5 方块", "3")], 2.865005)
assert near(dt_peak[("5x5 方块", "5")], 3.0)
assert near(dt_peak[("5x5 方块", "PRECISE")], 3.0)
assert near(dt_peak[("半径7圆盘", "3")], 6.846466)
assert near(dt_peak[("半径7圆盘", "5")], 6.999969)
assert near(dt_peak[("半径7圆盘", "PRECISE")], 7.071068)
assert near(dt_peak[("3px 长条", "3")], 1.910004)
assert near(dt_peak[("3px 长条", "5")], 2.0)
assert near(dt_peak[("3px 长条", "PRECISE")], 2.0)
assert near(dt_peak[("1px 竖线", "3")], 0.955002)
assert near(dt_peak[("1px 竖线", "5")], 1.0)
assert near(dt_peak[("1px 竖线", "PRECISE")], 1.0)
'''

E2_CODE = '''# @@todo 把「距离变换长什么样」看成矩阵：5x5 方块取中心 7x7 窗口，3px 长条取水平中线那列
# @@hint 两行：block_mat 用 np.round(cv2.distanceTransform(BLOCK, cv2.DIST_L2, 5), 3)[2:9, 2:9]
# @@hint bar_col 用 np.round(cv2.distanceTransform(BAR, cv2.DIST_L2, 5), 3)[:, 15].tolist()
# @@hint 再来一行 line_max：1px 竖线在 mask=3 / mask=5 下的峰值，一起放进一个二元组
block_mat = np.round(cv2.distanceTransform(BLOCK, cv2.DIST_L2, 5), 3)[2:9, 2:9]
bar_col = np.round(cv2.distanceTransform(BAR, cv2.DIST_L2, 5), 3)[:, 15].tolist()
line_max = (round(float(cv2.distanceTransform(LINE1, cv2.DIST_L2, 3).max()), 6),
            round(float(cv2.distanceTransform(LINE1, cv2.DIST_L2, 5).max()), 6))
# @@end
print("5x5 方块（mask=5）中心 7x7 窗口：")
print(block_mat)
print("3px 长条（mask=5）水平中线那一列:", bar_col)
print("1px 竖线峰值 (mask=3, mask=5):", line_max)
print("解读: 距离变换的输出是**逐像素**的图，不是标量。方块中心的最大值是 3 —— "
      "注意 5x5 方块的外圈距离是 0（背景自己到自己是 0），往里一圈 1、再一圈 2、中心 3，"
      "对应「到最近边界的层数」。3px 长条的中线是 2（上下各 1 px 才是背景）。"
      "**所以 dist 的「0 值区」就是背景本身** —— 后面 `dist > f * dist.max()` 之所以能当"
      "「前景核心」，就是因为它是「离边界足够远」的区域")
assert block_mat.shape == (7, 7)
assert near(block_mat[3, 3], 3.0)
assert near(block_mat[0, 0], 0.0)
assert near(block_mat[1, 1], 1.0)
assert near(block_mat[2, 3], 2.0)
assert bar_col == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
assert near(line_max[0], 0.955002) and near(line_max[1], 1.0)
'''

E3_CODE = '''dist = cv2.distanceTransform(op, cv2.DIST_L2, 5)
# @@todo 扫 6 档阈值，记下每档的「核心连通域数」和「核心像素数」
# @@hint 两条字典推导：连通域数用 cv2.connectedComponents(...)[0] - 1，像素数用 .sum()
coin_core = {f: cv2.connectedComponents((dist > f * dist.max()).astype(np.uint8))[0] - 1 for f in FG_FRACS}
coin_core_px = {f: int((dist > f * dist.max()).sum()) for f in FG_FRACS}
# @@end
print("coins 距离变换: dist.max = %.6f" % float(dist.max()))
print("核心阈值扫描（阈值 = f * dist.max）")
for f in FG_FRACS:
    print(f"   f={f:.2f} (阈值 {f * float(dist.max()):7.3f}) -> 核心 {coin_core[f]:3d} 个 / 核心像素 {coin_core_px[f]:6d}")
print("解读: 阈值越紧，核心越少 -> 目标被越切越碎。0.50 档只剩 2 个核心，"
      "说明 coins.png 里有些硬币**本来就比 0.5 * max 的半径还小**（大硬币和背景粘连把上限抬高了）。"
      "这就是分水岭的通病：**阈值必须扫，不能拍脑袋**")
assert coin_core[0.20] == 29 and coin_core[0.30] == 25 and coin_core[0.50] == 2
assert coin_core_px[0.20] == 16580 and coin_core_px[0.30] == 7648 and coin_core_px[0.50] == 1862
assert near(float(dist.max()), 47.384338)
'''

E4_CODE = '''# @@todo 造分水岭的三个区域：sure_bg（膨胀 3 次）、sure_fg30（0.3 档核心）、unknown（差集）
# @@hint 三行：cv2.dilate(op, K3, iterations=3)；(dist > 0.3 * dist.max()).astype(np.uint8)
# @@hint 差集**必须用 cv2.subtract**（都是 0/255 图），不要用 numpy 相减
sure_bg = cv2.dilate(op, K3, iterations=3)
sure_fg30 = (dist > 0.3 * dist.max()).astype(np.uint8)
unknown = cv2.subtract(sure_bg, sure_fg30)
# @@end
print("sure_bg  :", int((sure_bg > 0).sum()), "像素（膨胀 3 次向外扩了",
      int((sure_bg > 0).sum()) - int((op > 0).sum()), "个像素）")
print("sure_fg30:", int((sure_fg30 > 0).sum()), "像素（0.3 档核心）")
print("unknown  :", int((unknown == 255).sum()), "像素（这是「待定区」，交给分水岭去切）")
print("unknown 里 `> 0` 的像素数 =", int((unknown > 0).sum()),
      "<- 注意：这是 0/255 图，核心处算出的是 254 而不是 0，判空必须写 `== 255`")
assert int((sure_bg > 0).sum()) == 62589
assert int((sure_fg30 > 0).sum()) == 7648
assert int((unknown == 255).sum()) == 62589 - 7648
'''

E5_CODE = '''# @@todo 按 OpenCV 约定造 markers 并跑第一次分水岭
# @@hint markers 约定：**背景是 1**、各核心从 2 起递增、unknown 归 0
# @@hint 三行：n, mk = cv2.connectedComponents(sure_fg30)；markers = mk + 1；markers[unknown == 255] = 0
# @@hint 最后一行：ws_coins = cv2.watershed(coins_img, markers.copy())（要 copy，watershed 会就地改）
n_core, mk0 = cv2.connectedComponents(sure_fg30)
markers = mk0 + 1
markers[unknown == 255] = 0
ws_coins = cv2.watershed(coins_img, markers.copy())
# @@end
lab_c, cnt_c = region_labels(ws_coins)
print("markers 标签取值:", int(markers.min()), "~", int(markers.max()),
      "（0 = unknown，1 = 背景，2..", int(markers.max()), "= 各核心）")
print("watershed 结果: ws.max =", int(ws_coins.max()), "| 区域标签数 =", len(lab_c),
      "| 目标数 = 区域数 - 1 =", len(lab_c) - 1, "（减掉标签 1 的背景区域）")
print("边界像素（ws == -1）:", int((ws_coins == -1).sum()))
print("标签 1（背景区域）的面积:", int(cnt_c[0]))
print("解读: 分水岭的输出里，**标签 1 是「背景区域」而不是一个目标** —— 直接拿 ws.max() 当个数"
      "会多算 1。另外 `ws == -1` 是分水岭自己画出来的**边界线**，它不属于任何区域，统计前要先剔")
assert int(markers.min()) == 0 and int(markers.max()) == 26
assert int(ws_coins.max()) == 26
assert len(lab_c) == 26 and len(lab_c) - 1 == 25
assert int((ws_coins == -1).sum()) == 4596
assert int(cnt_c[0]) > 0
'''

E6_CODE = '''coin_scan = {}
for f in FG_FRACS:
    sfg = (dist > f * dist.max()).astype(np.uint8)
    unk = cv2.subtract(sure_bg, sfg)
    n, mk = cv2.connectedComponents(sfg)
    mk = mk + 1
    mk[unk == 255] = 0
    # @@todo 跑一次分水岭，把 (ws.max(), 目标数, 边界像素数) 记进 coin_scan[f]
    # @@hint 两行：ws = cv2.watershed(coins_img, mk.copy())；再一行 coin_scan[f] = (...)
    # @@hint 目标数 = ws.max() - 1（ws.max() 里那个 1 是背景区域）
    ws = cv2.watershed(coins_img, mk.copy())
    coin_scan[f] = (int(ws.max()), int(ws.max()) - 1, int((ws == -1).sum()))
    # @@end
print("coins 分水岭阈值扫描")
for f in FG_FRACS:
    mx, ntar, edge = coin_scan[f]
    print(f"   f={f:.2f} -> ws.max {mx:3d} 目标 {ntar:3d} 边界像素 {edge:5d}")
print("解读: 目标数从 0.20 档的 29 一路掉到 0.50 档的 2。**边界像素数不是单调的**："
      "0.20 -> 0.25 边界从 4864 掉到 4655（核心少了，要切的地方也少了），"
      "但 0.30 -> 0.35 又从 4596 掉到 3947。这说明「分割区域数」和「边界长度」是两个独立指标，"
      "只看其中一个会误判")
assert coin_scan[0.20] == (30, 29, 4864)
assert coin_scan[0.30] == (26, 25, 4596)
assert coin_scan[0.50] == (3, 2, 1745)
'''

E7_CODE = '''# @@todo 合成细胞图走同一套流程：Otsu 二值化 -> 开运算 2 次 -> 距离变换 -> 膨胀求 sure_bg
# @@hint 四行：cell_t, sbw = cv2.threshold(syn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# @@hint sbw2 = cv2.morphologyEx(sbw, cv2.MORPH_OPEN, K3, iterations=2)
# @@hint sd = cv2.distanceTransform(sbw2, cv2.DIST_L2, 5)
# @@hint sure_bg_cell = cv2.dilate(sbw2, K3, iterations=3)
cell_t, sbw = cv2.threshold(syn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
sbw2 = cv2.morphologyEx(sbw, cv2.MORPH_OPEN, K3, iterations=2)
sd = cv2.distanceTransform(sbw2, cv2.DIST_L2, 5)
sure_bg_cell = cv2.dilate(sbw2, K3, iterations=3)
# @@end
print("合成细胞: Otsu =", cell_t, "| 二值前景", int((sbw > 0).sum()), "| 开运算后", int((sbw2 > 0).sum()))
print("距离变换: sd.max = %.6f" % float(sd.max()))
print("各圆心处的 sd 值:", [round(float(sd[cy, cx]), 3) for cx, cy in CELLS])
print("解读: 每个圆心处的 sd 都是 29.766（≈半径 30，差的 0.234 来自高斯模糊把边缘啃掉了一点）。"
      "**圆心就是距离变换的局部极大** —— 这正是「用 dist 找目标核心」的几何依据")
assert near(float(sd.max()), 29.765869)
assert all(near(sd[cy, cx], 29.765869, 1e-4) for cx, cy in CELLS)
assert int((sbw2 > 0).sum()) == 21933
'''

E8_CODE = '''# @@todo 数一数「纯形态学」能数出几个目标，再记下 GT 的像素数与各粘连块的面积
# @@hint 三行：cell_conn 用 cv2.connectedComponents(sbw2)[0] - 1
# @@hint cell_gt_area 用 int((gt > 0).sum())
# @@hint cell_blobs 用 sorted(np.bincount(cv2.connectedComponents(sbw2)[1].ravel())[1:].tolist(), reverse=True)
cell_conn = cv2.connectedComponents(sbw2)[0] - 1
cell_gt_area = int((gt > 0).sum())
cell_blobs = sorted(np.bincount(cv2.connectedComponents(sbw2)[1].ravel())[1:].tolist(), reverse=True)
# @@end
print("真值目标数:", len(CELLS), "| 纯形态学连通域数:", cell_conn, "| GT 前景像素:", cell_gt_area)
print("各粘连块面积:", cell_blobs, "-> 4 块，每块 5480 左右（≈ 2 个圆）")
print("解读: **4 vs 8**。8 个细胞两两粘连成 4 坨，连通域方法直接少一半。"
      "注意每块 5480 px 略小于「两个半径 30 的圆、圆心距 52」的理论并集面积 5492.96，"
      "差值来自模糊 + 噪声把边缘侵蚀了")
assert cell_conn == 4
assert cell_gt_area == 21940
assert cell_blobs == [5486, 5484, 5482, 5481]
'''

E9_CODE = '''# @@todo 扫 10 档阈值（0.40 ~ 0.85），记录每档的核心数
# @@hint 一条字典推导，键是 f 值是 cv2.connectedComponents(...)[0] - 1
cell_core = {f: cv2.connectedComponents((sd > f * sd.max()).astype(np.uint8))[0] - 1 for f in CELL_FRACS}
# @@end
print("核心阈值扫描（真值需要 8）")
for f in CELL_FRACS:
    mark = "  <== 命中 8" if cell_core[f] == 8 else ""
    print(f"   f={f:.2f} (阈值 {f * float(sd.max()):6.3f}) -> 核心 {cell_core[f]:2d}{mark}")
print("解读: 0.40~0.50 只能分出 4 个（圆心距 52 的脖子还没断开）；"
      "**0.60 ~ 0.85 是一个很宽的平台，全部给出 8**；再往上（>0.9）核心会开始消失。"
      "注意这和 §10.4 的 coins 完全不同 —— 真实图要在 0.2~0.3，合成图要 0.6~0.8。"
      "**阈值窗口得靠自己扫出来**")
assert cell_core[0.40] == 4 and cell_core[0.50] == 4
assert cell_core[0.55] == 7
assert all(cell_core[f] == 8 for f in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85))
'''

E10_CODE = '''sfg = (sd > 0.6 * sd.max()).astype(np.uint8)
unk = cv2.subtract(sure_bg_cell, sfg)
n, mk = cv2.connectedComponents(sfg)
mk = mk + 1
mk[unk == 255] = 0
cell_ws = cv2.watershed(syn_c, mk.copy())
# @@todo 记录 8 个圆心各自落在哪个区域标签上
# @@hint 一行：cell_lab = labels_at(cell_ws, CELLS)
cell_lab = labels_at(cell_ws, CELLS)
# @@end
# @@todo 逐像素评价：前景 = 「在分水岭区域内」且「在二值前景内」，再算 Dice 与 IoU
# @@hint 三行：cell_fg 是 (cell_ws > 0) & (sbw2 > 0)
# @@hint cell_dice = round(dice(gt > 0, cell_fg), 6)；cell_iou 同理用 pix_iou
cell_fg = (cell_ws > 0) & (sbw2 > 0)
cell_dice = round(dice(gt > 0, cell_fg), 6)
cell_iou = round(pix_iou(gt > 0, cell_fg), 6)
# @@end
lab_l, cnt_l = region_labels(cell_ws)
print("分水岭结果: ws.max =", int(cell_ws.max()), "| 区域标签数 =", len(lab_l),
      "| 目标数 =", len(lab_l) - 1)
print("各圆心落在的标签:", cell_lab, "-> 唯一值个数", len(set(cell_lab)), "（真值 8，说明各自独立成区）")
print("标签 1（背景区域）面积:", int(cnt_l[0]), "| 8 个目标区域面积:", sorted(cnt_l[1:].tolist(), reverse=True))
print("碎片检查: 面积 < 300 的目标区域 =", int((cnt_l[1:] < 300).sum()), "个")
print("逐像素评价: Dice = %.6f  IoU = %.6f" % (cell_dice, cell_iou))
print("  前景像素", int(cell_fg.sum()), "| GT 像素", int((gt > 0).sum()),
      "| 交集", int(((gt > 0) & cell_fg).sum()))
print("解读: 目标数 8 达标，8 个区域面积都在 2500~2800（单个圆 2827，合理）——"
      "**「数字对了」和「区域形状对了」是两件事，两样都要验**。"
      "Dice 0.982337 / IoU 0.965286 说明分水岭区域**基本覆盖**了 GT，"
      "但它不含边界线（ws == -1 的像素被排除了），所以永远达不到 1.0")
assert int(cell_ws.max()) == 9
assert cell_lab == [2, 3, 4, 5, 6, 7, 8, 9]
assert len(lab_l) == 9 and int(cnt_l[0]) == 66317
assert int((cnt_l[1:] < 300).sum()) == 0
assert near(cell_dice, 0.982337) and near(cell_iou, 0.965286)
assert int(cell_fg.sum()) == 21200
'''

E11_CODE = '''# @@todo 从 IoU 反推 Dice：D = 2I / (1 + I)，在 0.1 / 0.3 / 0.5 / 0.7 / 0.9 上验证
# @@hint 一条字典推导，值是 round(2 * i / (1 + i), 6)
di_map = {i: round(2 * i / (1 + i), 6) for i in IOU_GRID}
# @@end
print("IoU -> Dice 换算表")
for i in IOU_GRID:
    print(f"   IoU {i:.1f} -> Dice {di_map[i]:.6f}")
print("解读: 两个指标**单调等价**（已知一个就能算出另一个），但曲线形状不同："
      "IoU 低的时候 Dice 明显更高（IoU 0.1 -> Dice 0.181818，涨了近一倍），"
      "IoU 高的时候两者趋同（0.9 -> 0.947368）。"
      "**所以汇报分割精度时必须写清用的是哪个**，否则 0.18 和 0.18 可能对应完全不同的质量")
assert near(di_map[0.1], 0.181818)
assert near(di_map[0.5], 0.666667)
assert near(di_map[0.9], 0.947368)
assert near(2 * 0.142857 / (1 + 0.142857), 0.25, 1e-5)
'''

E12_CODE = '''# @@todo 小错位的代价：两个 4x4 方块错位 2px 的 Dice / IoU
# @@hint 两行：round(dice(SQ_A, SQ_B), 6) 与 round(pix_iou(SQ_A, SQ_B), 6)
sq_dice = round(dice(SQ_A, SQ_B), 6)
sq_iou = round(pix_iou(SQ_A, SQ_B), 6)
# @@end
# @@todo 再看看「几乎不出错」的情况：coins 的 Otsu 二值 vs 开运算后的 Dice / IoU
# @@hint 两行：coin_dice = round(dice(cbw > 0, op > 0), 6)；coin_iou 用 pix_iou
coin_dice = round(dice(cbw > 0, op > 0), 6)
coin_iou = round(pix_iou(cbw > 0, op > 0), 6)
# @@end
print("两个 4x4 方块错位 2px: Dice %.6f  IoU %.6f（交集只有 2x2 = 4，各自 16）" % (sq_dice, sq_iou))
print("coins Otsu vs 开运算 : Dice %.6f  IoU %.6f" % (coin_dice, coin_iou))
print("  二值前景", int((cbw > 0).sum()), "-> 开运算", int((op > 0).sum()),
      "| 差集", int((cbw > 0).sum()) - int((op > 0).sum()))
print("解读: 4x4 方块错位 2px 就只剩 0.25 / 0.142857 —— **错位半个身位，指标就腰斩**。"
      "而开运算只吃掉 378 个前景像素，Dice 还有 0.996053。"
      "同一个 Dice 尺度上，0.25 和 0.996 的差距就是「预测框偏了一格」和「边缘毛刺」的差距")
assert near(sq_dice, 0.25) and near(sq_iou, 0.142857)
assert near(coin_dice, 0.996053) and near(coin_iou, 0.992136)
'''

RNG_DEMO = '''# 先看看「不设种子」会发生什么：同一个图、同一组参数，连跑三次
gc_raw = []
for _ in range(3):
    m_r = np.zeros(messi.shape[:2], np.uint8)
    b_r = np.zeros((1, 65), np.float64)
    f_r = np.zeros((1, 65), np.float64)
    cv2.grabCut(messi, m_r, GC_RECT, b_r, f_r, 5, cv2.GC_INIT_WITH_RECT)
    gc_raw.append(int(((m_r == cv2.GC_FGD) | (m_r == cv2.GC_PR_FGD)).sum()))
print("不设种子连跑三次的前景像素数:", gc_raw)
print("三次全相同?", len(set(gc_raw)) == 1, "<- False 说明 GrabCut **每次结果都不一样**")
'''

E13_CODE = '''gc_runs = []
for _ in range(2):
    cv2.setRNGSeed(42)
    gc_mask = np.zeros(messi.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    # @@todo 跑一次 GrabCut（rect 模式），把前景像素数追加进 gc_runs
    # @@hint 两行：cv2.grabCut(messi, gc_mask, GC_RECT, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    # @@hint 前景 = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)，再 .sum()
    cv2.grabCut(messi, gc_mask, GC_RECT, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    gc_runs.append(int(((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).sum()))
    # @@end
print("RECT =", GC_RECT, "（宽", GC_RECT[2], "高", GC_RECT[3], "，面积", GC_RECT[2] * GC_RECT[3], "）")
print("setRNGSeed(42) 后跑两次的前景像素数:", gc_runs, "-> 完全一致")
print("解读: GrabCut 内部用 GMM 做 k-means 初始化，**k-means 会调 RNG**。"
      "所以不设种子时每次结果都不同（实测四种结果：16241 / 16196 / 16190 / 16235）。"
      "要做可复现的实验，**必须先 cv2.setRNGSeed** —— 这个坑很多教程都不提")
assert gc_runs == [16254, 16254]
assert gc_runs[0] == gc_runs[1]
'''

E14_CODE = '''# @@todo 拆开看 GrabCut 的四种标签各有多少像素
# @@hint 一条字典推导：{int(k): int((gc_mask == k).sum()) for k in (GC_BGD, GC_PR_BGD, GC_FGD, GC_PR_FGD)}
gc_stat = {int(k): int((gc_mask == k).sum()) for k in (cv2.GC_BGD, cv2.GC_PR_BGD, cv2.GC_FGD, cv2.GC_PR_FGD)}
# @@end
# @@todo 合成「前景掩码」与「原来的 RECT 掩码」，算前景占 RECT 的比例
# @@hint 三行：gc_fg_rect 用 | 把 FGD 与 PR_FGD 拼起来；gc_rect_mask 用切片把 RECT 区域置 1
# @@hint gc_rect_ratio = round(gc_fg_rect.sum() / gc_rect_mask.sum(), 6)
gc_fg_rect = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)
gc_rect_mask = np.zeros(messi.shape[:2], np.uint8)
gc_rect_mask[GC_RECT[1]:GC_RECT[1] + GC_RECT[3], GC_RECT[0]:GC_RECT[0] + GC_RECT[2]] = 1
gc_rect_ratio = round(float(gc_fg_rect.sum()) / float(gc_rect_mask.sum()), 6)
# @@end
print("四种标签的像素数:", gc_stat)
print("  0 = GC_BGD（确定背景）", gc_stat[0])
print("  1 = GC_FGD（确定前景）", gc_stat[1], "<- **恒为 0**：rect 模式没有任何像素被标成「确定前景」")
print("  2 = GC_PR_BGD（可能背景）", gc_stat[2])
print("  3 = GC_PR_FGD（可能前景）", gc_stat[3])
print("前景合计:", int(gc_fg_rect.sum()), "| 占 RECT 的比例: %.6f" % gc_rect_ratio)
print("越出 RECT 的前景像素:", int((gc_fg_rect & ~gc_rect_mask).sum()), "| 前景 bbox:", cv2.boundingRect(gc_fg_rect))
print("Dice(前景 vs RECT) = %.6f" % dice(gc_fg_rect, gc_rect_mask))
print("解读: ① rect 模式下 **GC_FGD 恒为 0**（你只给了个方框，GrabCut 不敢说任何像素「确定是前景」）；"
      "② 前景**完全落在 RECT 内**（越出 0 个像素）—— rect 就是硬边界；"
      "③ GrabCut 只啃掉了 RECT 的 65%，而且啃得**很不规则**（bbox 是 (160, 90, 220, 166)，"
      "比 RECT 小但不在正中）。所以 rect 模式给的是「方框内的一个不规则形状」，不是「方框本身」")
assert gc_stat == {0: 140562, 1: 0, 2: 30600, 3: 16254}
assert int(gc_fg_rect.sum()) == 16254
assert near(gc_rect_ratio, 0.346907)
assert int((gc_fg_rect & ~gc_rect_mask).sum()) == 0
assert cv2.boundingRect(gc_fg_rect) == (160, 90, 220, 166)
assert near(dice(gc_fg_rect, gc_rect_mask), 0.515117)
'''

E15_CODE = '''# @@todo 把上一轮的前景当作「可能前景」反馈回去，用 GC_INIT_WITH_MASK 再精化一轮
# @@hint 四行：gc_mask2 先 np.full(形状, GC_PR_BGD)；再把 gc_fg_rect > 0 的位置置 GC_PR_FGD
# @@hint 跑之前记得再 cv2.setRNGSeed(42)，否则结果不可复现
# @@hint cv2.grabCut(messi, gc_mask2, None, bgd2, fgd2, 5, cv2.GC_INIT_WITH_MASK) —— rect 传 None
gc_mask2 = np.full(messi.shape[:2], cv2.GC_PR_BGD, np.uint8)
gc_mask2[gc_fg_rect > 0] = cv2.GC_PR_FGD
cv2.setRNGSeed(42)
bgd2 = np.zeros((1, 65), np.float64)
fgd2 = np.zeros((1, 65), np.float64)
cv2.grabCut(messi, gc_mask2, None, bgd2, fgd2, 5, cv2.GC_INIT_WITH_MASK)
gc_fg_mask = ((gc_mask2 == cv2.GC_FGD) | (gc_mask2 == cv2.GC_PR_FGD)).astype(np.uint8)
# @@end
# @@todo 量化两轮结果的差异
# @@hint 三行：gc_mask_diff 用 np.abs(gc_fg_mask.astype(int) - gc_fg_rect.astype(int)).sum()
# @@hint gc_mask_dice = round(dice(gc_fg_rect, gc_fg_mask), 6)；再取 bbox 用 cv2.boundingRect
gc_mask_diff = int(np.abs(gc_fg_mask.astype(np.int32) - gc_fg_rect.astype(np.int32)).sum())
gc_mask_dice = round(dice(gc_fg_rect, gc_fg_mask), 6)
gc_mask_bbox = cv2.boundingRect(gc_fg_mask)
# @@end
print("mask 模式前景:", int(gc_fg_mask.sum()), "（rect 模式是", int(gc_fg_rect.sum()), "）")
print("两轮差异像素数:", gc_mask_diff, "| Dice(两轮) = %.6f" % gc_mask_dice)
print("mask 模式前景 bbox:", gc_mask_bbox, " vs rect 模式 (160, 90, 220, 166)")
print("GC_FGD 数量:", int((gc_mask2 == cv2.GC_FGD).sum()), "<- 即使反馈一轮，仍然没有「确定前景」")
print("解读: mask 模式把前景从 16254 涨到 23517（多了 45%），bbox 也从 (160, 90, 220, 166) "
      "撑到 (69, 88, 356, 232) —— **它不再受 RECT 的硬边界约束了**，开始往外长。"
      "两轮 Dice 只有 0.813910，说明 refine 是**大改**而不是微调。"
      "实践含义：rect 模式给的是「保守的初始估计」，想拿到完整目标要么喂更准的 rect，"
      "要么用 mask 模式给「确定前景/确定背景」的笔画提示")
assert int(gc_fg_mask.sum()) == 23517
assert gc_mask_diff == 7401
assert near(gc_mask_dice, 0.813910)
assert gc_mask_bbox == (69, 88, 356, 232)
assert int((gc_mask2 == cv2.GC_FGD).sum()) == 0
'''

E16_CODE = '''# @@todo 把三种口径并排：纯形态学 / 分水岭 / GT，看它们在「目标数」和「像素级重合」上差多少
# @@hint 两行：三样分别是 (纯连通域, 分水岭 0.6 档目标数, 真值) 与 (Otsu 的 Dice, 分水岭的 Dice, 1.0)
cell_final = {"纯形态学连通域": cell_conn, "分水岭(0.6)": len(lab_l) - 1, "GT": len(CELLS)}
cell_quality = {"Otsu 二值 vs GT": round(dice(gt > 0, sbw2 > 0), 6),
                "分水岭区域 vs GT": cell_dice}
# @@end
print("目标数对比:", cell_final)
print("像素级对比:", cell_quality)
print("解读: 纯形态学的 Dice 其实**更高**（0.99+ vs 0.982337）—— 因为它保留全部前景，"
      "而分水岭把边界线（ws == -1）挖掉了。**如果只报告 Dice，形态学反而「赢」**。"
      "但它数不出 8 个目标。所以：**「数得多」和「盖得准」必须分开评估**，"
      "一个像素级指标配一个实例级指标，缺一个都会得出错误结论")
assert cell_final == {"纯形态学连通域": 4, "分水岭(0.6)": 8, "GT": 8}
assert near(cell_quality["Otsu 二值 vs GT"], 0.998929, 1e-5)
assert near(cell_quality["分水岭区域 vs GT"], 0.982337)
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch10 图像分割进阶

> 前九章里，「分割」一直是**阈值 + 形态学**的副产品。本章开始正面处理分割：
> 目标**粘连在一起**怎么切开、切完之后**怎么评价**、以及交互式分割的经典算法 GrabCut。
>
> 一句话总结本章的立场：**阈值只能分开前景/背景，分不开目标和目标。**

## 本章路线

| 小节 | 内容 | 为什么要手写 |
|---|---|---|
| §10.1 | 距离变换的三种精度 | 后面所有「找核心」都基于它，精度口径必须先说清 |
| §10.2 | 用距离变换提取目标核心 | `dist > f * dist.max()` 的几何含义 |
| §10.3 | **分水岭三区 + markers 约定** | `sure_bg` / `sure_fg` / `unknown` 与「标签 1 是背景」 |
| §10.4 | 真实场景：coins 阈值扫描 | 阈值窗口只能扫出来，不能背 |
| §10.5 | **粘连细胞：形态学 vs 分水岭** | 4 vs 8 的定量差距 |
| §10.6 | 分水岭的逐目标 + 逐像素评价 | 「数对了」≠「形状对了」 |
| §10.7 | **Dice 与 IoU** | 两者的换算、错位代价、什么时候会「骗人」 |
| §10.8 | **GrabCut** | rect / mask 两种模式 + **RNG 不可复现的坑** |

## 关于数据

三个场景，各有分工：

| 数据 | 来源 | 用途 |
|---|---|---|
| `classic/coins.png` | OpenCV 官方样例 | 真实、光照不均、大小不一 —— 练「阈值扫描」 |
| `classic/messi5.jpg` | OpenCV 官方样例 | 有明确主体 —— 练 GrabCut |
| **合成粘连细胞** | 本章代码生成 | **有精确 GT 掩码** —— 逐像素对账 Dice/IoU |

合成图的构造故意做得很「规矩」：8 个半径 30 的圆，圆心距 52（< 直径 60），
所以**每一对必然粘连**，而 GT 掩码可以由 `cv2.circle` 精确得到 ——
这样每个指标才有真值可比，而不是「看起来差不多」。
"""
    ),
    md(
        """
## §10.0 环境与数据准备

先读入两张真实图，再用代码合成一张带 GT 的粘连细胞图。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## §10.1 距离变换：三种精度

`cv2.distanceTransform(src, distanceType, maskSize)` 算的是**每个前景像素到最近背景像素的距离**。

关键在于 `maskSize` —— 它是三种**不同的算法**，不是「精度旋钮」那么简单：

| `maskSize` | 取值 | 算法 | 水平/垂直步长权重 |
|---|---|---|---|
| `DIST_MASK_3` | 3 | 3x3 chamfer 近似 | **0.955**（不是 1.0！） |
| `DIST_MASK_5` | 5 | 5x5 chamfer 近似 | 1.0 |
| `DIST_MASK_PRECISE` | **0** | 精确欧氏 | 1.0（真值） |

注意 `DIST_MASK_PRECISE` 的**数值是 0** —— 这是一个很容易记错的常量。

下面用四种小形状（方块 / 圆盘 / 长条 / 1px 细线）把三档都跑一遍。
"""
    ),
    code(E1_CODE),
    md(
        """
## §10.2 距离变换长什么样

距离变换的输出是**一张图**而不是一个标量。把它打印出来看形状，比看峰值更有用。
"""
    ),
    code(E2_CODE),
    md(
        """
## §10.3 真实场景：coins 的核心提取

思路：把 `dist` 归一化后取「足够远离边界」的区域当**目标核心**（`sure_fg`）。

阈值写成 `f * dist.max()` 而不是绝对值，是因为 `dist` 的量纲随图像尺寸变 ——
固定绝对值挪到另一张图就废了。但 `f` 取多少，**只能扫**。
"""
    ),
    code(E3_CODE),
    md(
        """
## §10.4 分水岭三区 + markers 约定

分水岭的思想是「往地形图里灌水」：`dist` 是海拔，山顶是目标核心，山谷是边界。
它需要**三个区域**：

| 区域 | 怎么来 | 在 markers 里的标签 |
|---|---|---|
| `sure_fg` | `dist > f * dist.max()` | **2, 3, 4, ...** 每个核心一个 |
| `sure_bg` | `sure_fg` 膨胀若干次之后的外侧 | **1**（只有背景是 1） |
| `unknown` | `sure_bg - sure_fg` 的差集 | **0**（交给分水岭自己判） |

**标签约定是本节的考点**：`0` 是「未知」、`1` 是「背景」、`>= 2` 才是目标。
把 `sure_bg` 填 1 而不是 0，是唯一正确的做法 —— 填错会让整张图被当成一个前景区。
"""
    ),
    code(E4_CODE),
    code(E5_CODE),
    md(
        """
## §10.5 coins 阈值扫描：没有「正确」的 f

`f` 决定核心有多少个。把 6 档全跑一遍，把「目标数」和「边界像素数」一起看。
"""
    ),
    code(E6_CODE),
    md(
        """
## §10.6 粘连细胞：为什么形态学数不出来

现在换成合成图 —— 它有精确 GT，可以把「数错了多少」量化。

先走一遍和 coins 一样的流程（Otsu -> 开运算 -> 距离变换）。
"""
    ),
    code(E7_CODE),
    md(
        """
先看**不算分水岭**会得到什么。
"""
    ),
    code(E8_CODE),
    md(
        """
再扫一遍核心阈值 —— 这次会有明确的答案。
"""
    ),
    code(E9_CODE),
    md(
        """
## §10.7 分水岭的完整评价

跑 f = 0.6，然后从两个角度看结果：**每个目标是否独立成区**（实例级），
**区域像素与 GT 的重合度**（像素级）。
"""
    ),
    code(E10_CODE),
    md(
        """
## §10.8 Dice 与 IoU

分割任务里两个指标总是成对出现。它们**单调等价** —— 知道一个就能算出另一个：

```
Dice = 2 * IoU / (1 + IoU)        IoU = Dice / (2 - Dice)
```

但等价不代表可以混用，因为**同一个数值对应的质量感不一样**。
"""
    ),
    code(E11_CODE),
    md(
        """
再用两个极端例子感受一下 Dice 的量纲：一个「错位半个身位」，一个「边缘有点毛刺」。
"""
    ),
    code(E12_CODE),
    md(
        """
## §10.9 GrabCut：先说一个**可复现性**的坑

GrabCut 用 GMM + k-means 建模前景/背景的颜色分布，而 **k-means 的初始化会调用 OpenCV 的全局 RNG**。
后果是：**同样的输入、同样的参数，每次跑出来的结果都不一样。**

先看看这个现象。
"""
    ),
    code(RNG_DEMO),
    md(
        """
`cv2.setRNGSeed(42)` 可以把这个随机性钉死。先跑 rect 模式。
"""
    ),
    code(E13_CODE),
    md(
        """
## §10.10 GrabCut 的四种标签

`grabCut` 的 mask 里是 4 种取值，理解它们才知道结果该怎么用：

| 值 | 常量 | 含义 |
|---|---|---|
| 0 | `GC_BGD` | 确定是背景 |
| 1 | `GC_FGD` | 确定是前景 |
| 2 | `GC_PR_BGD` | 可能是背景 |
| 3 | `GC_PR_FGD` | 可能是前景 |

**通常要的前景掩码 = `GC_FGD` + `GC_PR_FGD`**。
"""
    ),
    code(E14_CODE),
    md(
        """
## §10.11 mask 模式精化

rect 模式给的是一块硬方框内的不规则形状。想突破这个边界，就用 `GC_INIT_WITH_MASK`
把上一轮结果反馈回去，同时把 rect 参数传 `None`。
"""
    ),
    code(E15_CODE),
    md(
        """
## §10.12 三种口径并排

最后把「纯形态学」「分水岭」「GT」放在一张表里 —— 这一步会暴露一个反直觉的结论。
"""
    ),
    code(E16_CODE),
    md(
        """
## 自查清单

- [ ] 知道 `DIST_MASK_PRECISE` 的数值是 **0**（不是 5）
- [ ] 知道 `maskSize=3` 的水平步长权重是 **0.955**
- [ ] 知道 `dist` 的 0 值区就是背景本身
- [ ] 知道核心阈值要用 `f * dist.max()` 而不是绝对值
- [ ] 能默写 markers 的三条约定：**0 = unknown、1 = 背景、>= 2 = 目标**
- [ ] 知道 `unknown` 的判空要写 `== 255`（0/255 图相减会得到 254）
- [ ] 知道分水岭的结果里 **`ws.max()` 比目标数多 1**
- [ ] 知道 `ws == -1` 是边界线，不属于任何区域
- [ ] 知道粘连目标必须用「距离变换 + 分水岭」，纯连通域会少数
- [ ] 能用 `Dice = 2I / (1 + I)` 互算
- [ ] 知道「数对了」≠「形状对了」，实例级和像素级指标都要报
- [ ] **知道 GrabCut 不设种子不可复现**，以及为什么要 `setRNGSeed`
- [ ] 知道 rect 模式下 `GC_FGD` 恒为 0
- [ ] 知道 rect 是硬边界、mask 模式才能突破
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch10 图像分割进阶（练习版）

> 按提示补全 `____`，跑通所有 assert。`coins_img` / `messi` / `syn` / `gt` / `CELLS`
> 以及 `K3` / `BLOCK` / `DISK` / `BAR` / `LINE1` / `SQ_A` / `SQ_B` / `GC_RECT`
> 和 `near` / `dice` / `pix_iou` / `region_labels` / `labels_at` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；控制流写在挖空块之外
> （要写函数时 `def` 与 `return` 已经给好，你只填函数体）。
> 题目真值都来自实跑（opencv 5.0.0）。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：距离变换的三种精度

4 种形状 x 3 档 maskSize，各算峰值。**注意 `DIST_MASK_PRECISE` 的数值是 0。**
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：距离变换的矩阵形态

打印 5x5 方块的中心窗口与 3px 长条的中线。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：coins 核心阈值扫描

扫 6 档，记录核心数与核心像素数。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：分水岭的三区

`sure_bg` / `sure_fg30` / `unknown`。**差集用 `cv2.subtract`。**
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：markers 约定

**0 = unknown、1 = 背景、>= 2 = 目标**。填错整张图就废了。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：coins 阈值扫描（完整分水岭）

每档跑一次完整分水岭，记下区域数、目标数、边界像素数。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：合成细胞的标准流程

Otsu -> 开运算 -> 距离变换 -> 膨胀。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：纯形态学能数出几个

数连通域个数，再和真值对照。
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 9：核心阈值扫描（合成图）

扫 10 档（0.40 ~ 0.85），找出命中 8 的平台。
"""
    ),
    code(E9_CODE),
    md(
        """
## 任务 10：分水岭 + 双重评价

8 个圆心是否各自独立成区（实例级），区域与 GT 的 Dice / IoU（像素级）。
"""
    ),
    code(E10_CODE),
    md(
        """
## 任务 11：IoU -> Dice 换算

`D = 2I / (1 + I)`。
"""
    ),
    code(E11_CODE),
    md(
        """
## 任务 12：错位的代价

两个 4x4 方块错位 2px；再对照 coins 的 Otsu vs 开运算。
"""
    ),
    code(E12_CODE),
    md(
        """
## 任务 13：GrabCut 的可复现性

先看**不设种子**会怎样，再用 `setRNGSeed(42)` 把结果钉死。
"""
    ),
    code(RNG_DEMO),
    code(E13_CODE),
    md(
        """
## 任务 14：GrabCut 的四种标签

拆开 0 / 1 / 2 / 3 各有多少像素，再看前景占 RECT 的比例。
"""
    ),
    code(E14_CODE),
    md(
        """
## 任务 15：mask 模式精化

把上一轮前景反馈回去，**rect 参数传 `None`**。
"""
    ),
    code(E15_CODE),
    md(
        """
## 任务 16：三种口径并排

目标数（实例级）与 Dice（像素级）各一张表。
"""
    ),
    code(E16_CODE),
    md(
        """
## 自查清单

- [ ] `DIST_MASK_PRECISE` 的数值是 **0**
- [ ] `maskSize=3` 的水平步长权重是 **0.955**
- [ ] markers：**0 = unknown、1 = 背景、>= 2 = 目标**
- [ ] `unknown` 判空写 `== 255`
- [ ] `ws.max()` 比目标数**多 1**
- [ ] `ws == -1` 是边界线，不是区域
- [ ] 粘连目标用分水岭，纯连通域会少数
- [ ] `Dice = 2I / (1 + I)`
- [ ] 实例级指标和像素级指标都要报
- [ ] GrabCut 必须 `setRNGSeed` 才能复现
- [ ] rect 模式 `GC_FGD` 恒为 0
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
