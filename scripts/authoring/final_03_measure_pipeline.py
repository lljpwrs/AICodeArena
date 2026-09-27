#!/usr/bin/env python3
"""cv final_03 全流程（分割测量）：阈值 -> 形态学 -> 距离变换/分水岭分离 ->
逐目标测量（面积/周长/等效直径/圆度）-> 与 GT 对账 -> 标定校正 -> 可视化报告。

数据：本章**自造**一个 640x480 的工业测量场景（9 个已知直径的圆，其中两对互相重叠）+
`classic/coins.png`（真实硬币，与 ch10 同口径）。
真值全部实跑（opencv 5.0.0 / numpy 2.5.3）。

> 本章的核心是**「分割得好」和「量得准」是两件事**：
> 像素级 Dice = **0.998681**（几乎完美），但实例级连通域只有 **7** 个（GT 9 个），
> 而孤立目标的直径**系统性偏小 1.019971 px**（相对 1.43%），
> 粘连目标最差要偏 **+7.5171%**。
>
> 第二个核心是**两条路线的取舍**：
> 纯阈值路线的直径偏差随阈值单调漂移 **+0.310897 px / 每 10 阈值**（阈值卡准才准）；
> 分水岭路线几乎完全免疫阈值（**-0.000667 px / 每 10 阈值**），
> 但有一个 **-1.02 px 的固定偏置** —— 因为 `cv2.watershed` 的最终边界由**图像梯度**决定，
> 阈值只影响标记（marker）。**稳但偏 vs 准但不稳**，这才是工程要权衡的东西。
>
> 第三个核心是**标定件校正**：用一个已知直径的标准件求出比例 1.017502，
> 直径误差从 1.019971 px 降到 0.236334 px（**降 76.83%**），毫米误差从 0.254993 mm 降到 0.059084 mm。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/final_03_measure_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv" / "final"
NAME = "final_03_measure_pipeline"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

CL = Path("../data/classic")
'''

SETUP = '''# 常量：场景尺寸、标定、真值圆、以及「哪些是孤立目标 / 哪些和邻居粘连」
K3 = np.ones((3, 3), np.uint8)
PX_PER_MM = 4.0
W, H = 640, 480
GT_CIRCLES = [(90, 90, 30), (200, 90, 42), (262, 90, 26), (450, 90, 38), (110, 250, 34),
              (300, 250, 48), (378, 250, 30), (150, 400, 36), (400, 400, 44)]
ISOLATED = [0, 3, 4, 7, 8]
CONTACTED = [1, 2, 5, 6]
CALIB_IDX = 0
CORE_RATIO = 0.45


def make_scene(seed=11):
    """造工业测量场景：白底 + 9 个深色圆（其中 #1/#2 与 #5/#6 互相重叠）。"""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 235, np.uint8)
    for cx, cy, r in GT_CIRCLES:
        cv2.circle(img, (cx, cy), r, (60, 60, 60), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)


def gt_mask():
    """真值掩码。"""
    m = np.zeros((H, W), np.uint8)
    for cx, cy, r in GT_CIRCLES:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def near(a, b, tol=1e-6):
    """浮点容差比较。真值都实跑得到，用容差避免 == 在末位翻车。"""
    return abs(float(a) - float(b)) <= tol


img = make_scene()
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
GTM = gt_mask()
print("场景", img.shape, "| 1 px = %.2f mm | 目标 %d 个（孤立 %d / 粘连 %d）"
      % (1 / PX_PER_MM, len(GT_CIRCLES), len(ISOLATED), len(CONTACTED)))
print("GT 直径(px):", [2 * c[2] for c in GT_CIRCLES])
print("GT 直径(mm):", [round(2 * c[2] / PX_PER_MM, 2) for c in GT_CIRCLES])
print("GT 前景像素 %d" % int((GTM > 0).sum()))
print("提示: 本章的 GT 是**精确已知**的（自己画的圆），所以能逐目标算误差 —— "
      "真实工业场景里这一步要靠标准件（量块/标定板）")
'''

SCAFFOLD = '''# 脚手架：分割 / 清理 / 计数 / 匹配 / 区域收集（缺口在后面的函数体里）
def seg_otsu(image):
    """Otsu 自动阈值 + 反相（目标是暗的，所以用 THRESH_BINARY_INV）。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(t), mask


def cleanup(mask, it_open=1, it_close=2):
    """开运算去毛刺 + 闭运算补小洞。"""
    op = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K3, iterations=it_open)
    cl = cv2.morphologyEx(op, cv2.MORPH_CLOSE, K3, iterations=it_close)
    return op, cl


def count_cc(mask):
    """数轮廓（面积 > 100 才算），用来做「纯连通域」计数基线。"""
    cs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return [c for c in cs if cv2.contourArea(c) > 100]


def nearest_gt(cx, cy):
    """把测得的目标中心匹配到最近的真值圆，返回它在 GT_CIRCLES 里的下标。"""
    return min(range(len(GT_CIRCLES)), key=lambda i: np.hypot(cx - GT_CIRCLES[i][0], cy - GT_CIRCLES[i][1]))


def region_labels(ws):
    """取分水岭结果里的目标区域（标签 >= 2；标签 1 是背景、-1 是边界线）。"""
    labs, cnts = np.unique(ws, return_counts=True)
    keep = labs >= 2
    return labs[keep], cnts[keep]


def collect(ws):
    """把分水岭结果整成「GT 下标 -> 测量字典」，方便逐目标对账。"""
    out = {}
    labs, _ = region_labels(ws)
    for lab in labs:
        d = measure((ws == lab).astype(np.uint8))
        out[nearest_gt(d["cx"], d["cy"])] = d
    return out


print("脚手架就绪：seg_otsu / cleanup / count_cc / nearest_gt / region_labels / collect")
'''

T1_CODE = '''def watershed_split(bgr, mask, core_ratio=CORE_RATIO, dil=3):
    """距离变换 -> 核心 / 背景 / 待定三区 -> markers -> 分水岭。返回 (ws, dist, 核心连通域数)。"""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # @@todo 造三个区域：核心（dist > core_ratio * max）、背景（膨胀 dil 次）、待定（背景 - 核心）
    # @@hint 两行：_, sure_fg = cv2.threshold(dist, core_ratio * dist.max(), 255, 0)
    # @@hint 一行：sure_fg = np.uint8(sure_fg)
    # @@hint 一行：sure_bg = cv2.dilate(mask, K3, iterations=dil)
    # @@hint 一行：unknown = cv2.subtract(sure_bg, sure_fg)（两张 0/255 图必须用 cv2.subtract）
    _, sure_fg = cv2.threshold(dist, core_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(mask, K3, iterations=dil)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # @@end
    # @@todo 按 OpenCV 约定造 markers（0 = 待定、1 = 背景、>= 2 = 各核心），再跑分水岭
    # @@hint 三行：n, markers = cv2.connectedComponents(sure_fg)
    # @@hint 一行：markers = markers + 1
    # @@hint 一行：markers[unknown == 255] = 0
    # @@hint 一行：ws = cv2.watershed(bgr, markers.copy())（watershed 会就地改 markers）
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    ws = cv2.watershed(bgr, markers.copy())
    # @@end
    return ws, dist, n


print("watershed_split 就绪")
'''

T2_CODE = '''def measure(m):
    """量一个二值区域：像素面积 / contourArea / 周长 / 等效直径 / 最小外接圆直径 / 圆度。

    等效直径 = 2*sqrt(A/pi)  —— 把「面积」换算成一个等面积圆的直径。
    圆度      = 4*pi*A/P^2   —— 理想圆为 1；数字图像里因阶梯效应上限约 0.87。
    """
    area = int((m > 0).sum())
    cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    c = max(cs, key=cv2.contourArea)
    # @@todo 算出四个测量量：周长、最小外接圆、等效直径、圆度
    # @@hint 两行：perim = float(cv2.arcLength(c, True))
    # @@hint 一行：(cx, cy), r_enc = cv2.minEnclosingCircle(c)
    # @@hint 一行：d_eq = 2.0 * np.sqrt(area / np.pi)
    # @@hint 一行：rnd = 4.0 * np.pi * area / perim ** 2
    perim = float(cv2.arcLength(c, True))
    (cx, cy), r_enc = cv2.minEnclosingCircle(c)
    d_eq = 2.0 * np.sqrt(area / np.pi)
    rnd = 4.0 * np.pi * area / perim ** 2
    # @@end
    return dict(area=area, area_ct=float(cv2.contourArea(c)), perim=round(perim, 4),
                d_eq=round(d_eq, 6), min_circle=round(2 * r_enc, 6), roundness=round(rnd, 6),
                cx=float(cx), cy=float(cy))


print("measure 就绪")
'''

E1_CODE = '''t_otsu, mask = seg_otsu(img)
op, cl = cleanup(mask)
# @@todo 像素级对账：交集、并集、Dice、IoU
# @@hint 两行：inter = int(((cl > 0) & (GTM > 0)).sum())；union = int(((cl > 0) | (GTM > 0)).sum())
# @@hint 两行：dice = 2 * inter / (int((cl > 0).sum()) + int((GTM > 0).sum()))；iou = inter / union
inter = int(((cl > 0) & (GTM > 0)).sum())
union = int(((cl > 0) | (GTM > 0)).sum())
dice = 2 * inter / (int((cl > 0).sum()) + int((GTM > 0).sum()))
iou = inter / union
# @@end

# @@todo 用 count_cc 数连通域，并列出各面积（升序）
# @@hint 两行：boxes = count_cc(cl)；n_cc = len(boxes)
# @@hint 一行：cc_areas = sorted(round(cv2.contourArea(c)) for c in boxes)
boxes = count_cc(cl)
n_cc = len(boxes)
cc_areas = sorted(round(cv2.contourArea(c)) for c in boxes)
# @@end

print("Otsu 阈值 %.1f | GT 前景 %d | Otsu 前景 %d | 开 %d | 闭 %d"
      % (t_otsu, int((GTM > 0).sum()), int((mask > 0).sum()), int((op > 0).sum()), int((cl > 0).sum())))
print("像素级对账: 交集 %d 并集 %d -> Dice %.6f | IoU %.6f" % (inter, union, dice, iou))
print("实例级计数: 连通域 %d 个 | GT 目标 %d 个 | 差 %d" % (n_cc, len(GT_CIRCLES), len(GT_CIRCLES) - n_cc))
print("连通域面积(升序):", cc_areas)
print("解读: **这就是本章第一个必须讲清楚的事 —— 像素级和实例级是两个不同的指标。**"
      "Dice 0.998681 看起来「分割几乎完美」，但它**只会告诉你像素对不对**；"
      "而连通域只有 7 个（GT 9 个）—— 因为两对圆粘在一起，被算成了一个目标。"
      "**只报 Dice 的报告，等于在隐藏 2 个漏检。**")
assert near(t_otsu, 146.0)
assert inter == 38629 and union == 38731
assert near(dice, 0.998681) and near(iou, 0.997366)
assert n_cc == 7 and len(GT_CIRCLES) == 9
assert cc_areas == [2732, 3524, 3948, 4402, 5948, 7389, 9881]
'''

E2_CODE = '''core_scan = {}
for ratio in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
    ws_r, dist_r, n_r = watershed_split(img, cl, core_ratio=ratio)
    labs_r, cnts_r = region_labels(ws_r)
    m_r = collect(ws_r)
    # @@todo 记下这个档位的 (区域数, 核心数, 最小核心面积, 平均|相对误差|, 粘连目标最差|相对误差|)
    # @@hint 一行：rel = [abs((m_r[gi]["d_eq"] - 2*GT_CIRCLES[gi][2]) / (2*GT_CIRCLES[gi][2])) for gi in m_r]
    # @@hint 两行：hits = [gi for gi in CONTACTED if gi in m_r]
    # @@hint 一行：worst = max([abs((m_r[gi]["d_eq"] - 2*GT_CIRCLES[gi][2]) / (2*GT_CIRCLES[gi][2])) for gi in hits] or [0.0])
    # @@hint 一行：core_scan[ratio] = (len(cnts_r), n_r - 1, int(cnts_r.min()), round(float(np.mean(rel)), 6), round(worst, 6))
    rel = [abs((m_r[gi]["d_eq"] - 2 * GT_CIRCLES[gi][2]) / (2 * GT_CIRCLES[gi][2])) for gi in m_r]
    hits = [gi for gi in CONTACTED if gi in m_r]
    worst = max([abs((m_r[gi]["d_eq"] - 2 * GT_CIRCLES[gi][2]) / (2 * GT_CIRCLES[gi][2])) for gi in hits] or [0.0])
    core_scan[ratio] = (len(cnts_r), n_r - 1, int(cnts_r.min()),
                        round(float(np.mean(rel)), 6), round(worst, 6))
    # @@end

print("核心阈值扫描（dist.max = %.6f，每档取 core_ratio * max 当绝对核心阈值）" % float(dist_r.max()))
print("  ratio | 区域数 | 核心数 | 最小核心 | 平均|相对误差| | 粘连最差|相对误差|")
for ratio in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
    r = core_scan[ratio]
    print("   %.2f |  %3d   |  %3d   |  %5d   |    %.6f    |    %.6f" % (ratio, r[0], r[1], r[2], r[3], r[4]))
print("解读: ① **区域数随 ratio 单调增加**：0.2/0.25 只有 7 个（两对粘连圆各自都不够拆）、"
      "0.3/0.35 是 8 个（只拆开了一对）、**0.4 起才是 9 个**（全部拆开）；"
      "② 但 ratio 太大（0.5）反而**变差**：最小核心从 2455 涨到 2567，"
      "粘连最差从 0.075171 恶化到 0.099423 —— 因为核心缩得太小时，"
      "分水岭的起点会偏向大圆一侧；"
      "③ **这个窗口（0.4~0.45）只能扫出来，不能拍脑袋定** —— "
      "ch10 的 coins 要 0.2~0.3、合成细胞要 0.6~0.8，本章要 0.4~0.45。"
      "同一个算法，三张图三个窗口")
assert core_scan[0.2][:4] == (7, 7, 2731, 0.055913)
assert core_scan[0.3][:4] == (8, 8, 2724, 0.031757)
assert core_scan[0.35][:4] == (8, 8, 2724, 0.031698)
assert core_scan[0.4][:4] == (9, 9, 2473, 0.027164)
assert core_scan[0.45][:4] == (9, 9, 2455, 0.026523)
assert core_scan[0.5][:4] == (9, 9, 2567, 0.030380)
assert near(core_scan[0.45][4], 0.075171)
assert near(core_scan[0.5][4], 0.099423)
assert near(float(dist_r.max()), 47.534882)
'''

E3_CODE = '''ws, dist, n_core = watershed_split(img, cl)
labs, cnts = region_labels(ws)
M = collect(ws)
# @@todo 记下选定档位下的三个量：目标区域数、边界像素数、dist 峰值
# @@hint 两行：n_targets = len(cnts)；n_boundary = int((ws == -1).sum())
# @@hint 一行：dist_max = float(dist.max())
n_targets = len(cnts)
n_boundary = int((ws == -1).sum())
dist_max = float(dist.max())
# @@end

print("选定 core_ratio = %.2f -> 区域 %d 个 | 核心 %d 个 | 边界像素 %d | dist.max %.6f"
      % (CORE_RATIO, n_targets, n_core - 1, n_boundary, dist_max))
print("区域面积:", cnts.tolist())
print("解读: ① 区域从 7 个变 9 个，**和 GT 刚好对上** —— 但先别高兴，"
      "数量对上不代表边界切得对（下一节会看到粘连目标误差高达 7.5%）；"
      "② 分水岭会产出一条**宽 1 px 的边界线**，标签是 **-1** —— "
      "统计区域面积时它既不算前景也不算背景，这也是「区域面积之和 < 前景像素数」的原因；"
      "③ **标签 1 是背景区域、标签 -1 是边界线、标签 >= 2 才是目标** —— "
      "这是 OpenCV 分水岭最容易搞错的一套约定")
assert n_targets == 9 and n_core - 1 == 9
assert n_boundary == 4102
assert near(dist_max, 47.534882)
assert cnts.tolist() == [4859, 4409, 2731, 2455, 7093, 3527, 2724, 5954, 3951]
'''

E4_CODE = '''print("逐目标测量（面积 / 等效直径 / 最小外接圆 / 周长 / 圆度）:")
for gi in sorted(M):
    d = M[gi]
    print("   GT#%d 真值直径 %2d px | 面积 %4d | 等效直径 %8.4f | 最小外接圆 %8.4f | 周长 %7.2f | 圆度 %.6f | 误差 %+8.4f (%+.4f%%)"
          % (gi, 2 * GT_CIRCLES[gi][2], d["area"], d["d_eq"], d["min_circle"], d["perim"],
             d["roundness"], d["d_eq"] - 2 * GT_CIRCLES[gi][2],
             100 * (d["d_eq"] - 2 * GT_CIRCLES[gi][2]) / (2 * GT_CIRCLES[gi][2])))

# @@todo 把 9 个目标分成两组（孤立 / 粘连），各算偏差数组与相对误差数组
# @@hint 四行：iso_err / con_err / iso_rel / con_rel
# @@hint 一行：iso_err = np.array([M[gi]["d_eq"] - 2 * GT_CIRCLES[gi][2] for gi in ISOLATED])
# @@hint 一行：iso_rel = np.array([(M[gi]["d_eq"] - 2 * GT_CIRCLES[gi][2]) / (2 * GT_CIRCLES[gi][2]) for gi in ISOLATED])
iso_err = np.array([M[gi]["d_eq"] - 2 * GT_CIRCLES[gi][2] for gi in ISOLATED])
con_err = np.array([M[gi]["d_eq"] - 2 * GT_CIRCLES[gi][2] for gi in CONTACTED])
iso_rel = np.array([(M[gi]["d_eq"] - 2 * GT_CIRCLES[gi][2]) / (2 * GT_CIRCLES[gi][2]) for gi in ISOLATED])
con_rel = np.array([(M[gi]["d_eq"] - 2 * GT_CIRCLES[gi][2]) / (2 * GT_CIRCLES[gi][2]) for gi in CONTACTED])
# @@end

rnd = np.array([M[gi]["roundness"] for gi in sorted(M)])
print("孤立 %d 个: 平均偏差 %+.6f | 平均|偏差| %.6f | 平均|相对| %.6f | 范围 %.4f%% ~ %.4f%%"
      % (len(ISOLATED), iso_err.mean(), np.abs(iso_err).mean(), np.abs(iso_rel).mean(),
         100 * iso_rel.min(), 100 * iso_rel.max()))
print("粘连 %d 个: 平均偏差 %+.6f | 平均|偏差| %.6f | 平均|相对| %.6f | 范围 %.4f%% ~ %.4f%%"
      % (len(CONTACTED), con_err.mean(), np.abs(con_err).mean(), np.abs(con_rel).mean(),
         100 * con_rel.min(), 100 * con_rel.max()))
print("圆度: min %.6f (GT#2) | max %.6f (GT#6) | mean %.6f" % (rnd.min(), rnd.max(), rnd.mean()))
print("解读: ① **孤立目标的偏差非常一致：5 个全在 -0.93 ~ -1.11 px，平均 -1.019971 px**"
      "（相对 -1.43%）。这是一个**系统性偏置**，不是随机噪声 —— "
      "根因是「面积等效直径」在数字图像上与真实几何直径之间差半个像素量级；"
      "② 粘连目标平均|偏差| 2.832284 px，最差 +7.5171% —— "
      "**分水岭把粘连目标分开了，但切的位置不精确**；"
      "③ **圆度是免费的质检指标**：GT#2 的圆度只有 0.519006（正常圆约 0.85），"
      "一眼就能看出「这个区域被切坏了」—— **不需要 GT 就能发现**。"
      "工业现场没有 GT，圆度/长宽比这类形状指标就是你的自检工具")
assert near(iso_err.mean(), -1.019971)
assert near(np.abs(iso_err).mean(), 1.019971)
assert near(np.abs(iso_rel).mean(), 0.014273, 1e-6)
assert near(con_err.mean(), -0.877843)
assert near(np.abs(con_err).mean(), 2.832284)
assert near(np.abs(con_rel).mean(), 0.041835, 1e-6)
assert near(float(rnd.min()), 0.519006)
assert near(float(rnd.max()), 0.872563)
assert near(float(rnd.mean()), 0.789334)
'''

E5_CODE = '''# @@todo 算「|相对误差|」与「真值直径」的相关系数
# @@hint 两行：iso_d = np.array([2 * GT_CIRCLES[gi][2] for gi in ISOLATED])
# @@hint 一行：corr = float(np.corrcoef(iso_d, np.abs(iso_rel))[0, 1])
iso_d = np.array([2 * GT_CIRCLES[gi][2] for gi in ISOLATED])
corr = float(np.corrcoef(iso_d, np.abs(iso_rel))[0, 1])
# @@end

print("孤立目标的 直径(px) -> |相对误差|:")
for d_px, r in zip(iso_d.tolist(), np.abs(iso_rel).tolist()):
    print("   %3d px -> %.6f" % (d_px, r))
print("相关系数 = %.6f" % corr)
print("解读: **相对误差与目标尺寸强负相关（-0.970140）** —— 直径 60 px 时相对误差 1.72%，"
      "88 px 时只有 1.06%。原因是偏置是**固定的 ~1 px**（绝对量），"
      "所以相对误差 ≈ 1/直径：**目标越小，测量相对误差越大**。"
      "这正是 ch09 讲的「小目标难」在测量上的另一个面孔 —— "
      "如果被测零件只有 20 px 宽，1 px 的偏置就是 5% 误差，很多工业场景直接不合格")
assert near(corr, -0.970140, 1e-6)
assert iso_d.tolist() == [60, 76, 68, 72, 88]
assert abs(np.abs(iso_rel)[0]) > abs(np.abs(iso_rel)[-1])
'''

E6_CODE = '''# @@todo 三种面积口径并排：像素计数 / contourArea / 最小外接圆面积
# @@hint 三行：pix = np.array([M[gi]["area"] for gi in sorted(M)])
# @@hint 一行：cta = np.array([M[gi]["area_ct"] for gi in sorted(M)])
# @@hint 一行：mca = np.array([np.pi * (M[gi]["min_circle"] / 2) ** 2 for gi in sorted(M)])
pix = np.array([M[gi]["area"] for gi in sorted(M)])
cta = np.array([M[gi]["area_ct"] for gi in sorted(M)])
mca = np.array([np.pi * (M[gi]["min_circle"] / 2) ** 2 for gi in sorted(M)])
# @@end

print("GT#0 三种口径: 像素计数 %d | contourArea %.4f | 最小外接圆面积 %.4f"
      % (M[0]["area"], M[0]["area_ct"], np.pi * (M[0]["min_circle"] / 2) ** 2))
print("面积合计: 像素计数 %d | contourArea %.4f | 最小外接圆 %.4f" % (pix.sum(), cta.sum(), mca.sum()))
print("contourArea / 像素计数 范围 %.6f ~ %.6f（平均 %.6f）"
      % ((cta / pix).min(), (cta / pix).max(), (cta / pix).mean()))
print("最小外接圆 / 像素计数 范围 %.6f ~ %.6f（平均 %.6f）"
      % ((mca / pix).min(), (mca / pix).max(), (mca / pix).mean()))
print("解读: **同一块区域，三种「面积」差很多：** "
      "① `contourArea` 用格林公式沿轮廓积分，比像素计数**小 2.74%**（0.972580）—— "
      "它算的是「轮廓围出的连续区域」，而像素计数把边界像素整个算进去了；"
      "② 最小外接圆面积平均是像素计数的 **1.134137 倍**，最差 1.929754 —— "
      "因为它是**外接圆**，对任何非圆形状（尤其被分水岭切歪的区域）都会显著偏大；"
      "③ **报告里写「面积」时必须注明口径** —— 差 2.7% 还是 93%，取决于你用哪个 API")
assert M[0]["area"] == 2731 and near(M[0]["area_ct"], 2645.5)
assert near(np.pi * (M[0]["min_circle"] / 2) ** 2, 2785.5912, 1e-3)
assert pix.sum() == 37703 and near(cta.sum(), 36739.0)
assert near((cta / pix).mean(), 0.972580, 1e-6)
assert near((mca / pix).mean(), 1.134137, 1e-6)
assert near((mca / pix).max(), 1.929754, 1e-6)
'''

E7_CODE = '''def thr_route(clk):
    """纯阈值路线：直接量连通域（不做分水岭分离）。"""
    n, lab = cv2.connectedComponents(clk)
    out = {}
    for i in range(1, n):
        m = (lab == i)
        if m.sum() <= 100:
            continue
        # @@todo 算该连通域的质心、匹配最近的真值圆、记下它的等效直径
        # @@hint 两行：ys, xs = np.nonzero(m)
        # @@hint 一行：out[nearest_gt(float(xs.mean()), float(ys.mean()))] = 2 * np.sqrt(m.sum() / np.pi)
        ys, xs = np.nonzero(m)
        out[nearest_gt(float(xs.mean()), float(ys.mean()))] = 2 * np.sqrt(m.sum() / np.pi)
        # @@end
    return out


def ws_route(bgr, clk):
    """分水岭路线：先分离再量。"""
    ws_r, _, _ = watershed_split(bgr, clk)
    # @@todo 一行：取出每个目标区域的等效直径，按「最近 GT 下标」建字典
    # @@hint 一行：return {gi: r["d_eq"] for gi, r in collect(ws_r).items()}
    return {gi: r["d_eq"] for gi, r in collect(ws_r).items()}
    # @@end


sens = {}
for thr in (116.0, 126.0, 136.0, 146.0, 156.0, 166.0, 176.0):
    _, mk = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY_INV)
    clk = cleanup(mk)[1]
    a = thr_route(clk)
    b = ws_route(img, clk)
    # @@todo 只统计 5 个孤立目标：两条路线各自的「平均直径偏差」
    # @@hint 两行：ea = float(np.mean([a[gi] - 2*GT_CIRCLES[gi][2] for gi in ISOLATED if gi in a]))
    # @@hint 一行：eb = float(np.mean([b[gi] - 2*GT_CIRCLES[gi][2] for gi in ISOLATED if gi in b]))
    ea = float(np.mean([a[gi] - 2 * GT_CIRCLES[gi][2] for gi in ISOLATED if gi in a]))
    eb = float(np.mean([b[gi] - 2 * GT_CIRCLES[gi][2] for gi in ISOLATED if gi in b]))
    # @@end
    sens[thr] = (int((clk > 0).sum()), round(ea, 6), round(eb, 6), len(a), len(b))

print("阈值敏感性（只看 5 个孤立目标）:")
print("   thr    | 前景像素 | 纯阈值 平均偏差 | 分水岭 平均偏差")
for thr in (116.0, 126.0, 136.0, 146.0, 156.0, 166.0, 176.0):
    s = sens[thr]
    print("   %.1f |  %5d   |    %+.6f    |   %+.6f" % (thr, s[0], s[1], s[2]))
d_thr = (sens[176.0][1] - sens[116.0][1]) / 6
d_ws = (sens[176.0][2] - sens[116.0][2]) / 6
print("纯阈值: 116 -> 176 总变化 %+.6f px，每 +10 阈值 %+.6f px" % (sens[176.0][1] - sens[116.0][1], d_thr))
print("分水岭: 116 -> 176 总变化 %+.6f px，每 +10 阈值 %+.6f px" % (sens[176.0][2] - sens[116.0][2], d_ws))
print("解读: **两条路线的性格完全不同。** "
      "① 纯阈值路线**单调漂移**：阈值从 116 抬到 176，直径偏差从 -1.162492 一路涨到 +0.702888，"
      "**每 +10 阈值就变 +0.310897 px** —— 阈值卡不准，量出来就不准；"
      "② 分水岭路线**几乎不动**：全程只在 -1.0155 ~ -1.0239 之间，"
      "**每 +10 阈值只变 -0.000667 px（比纯阈值小 466 倍）**；"
      "③ 为什么？因为 **`cv2.watershed` 的最终边界由图像的梯度脊线决定，不是由二值掩码决定** —— "
      "掩码只提供「标记点」。所以分水岭**顺手帮你把阈值敏感性消掉了**；"
      "④ 代价是那个**固定的 -1.02 px 偏置**（梯度脊线比 50% 灰度穿越点略靠内）。"
      "**「稳但偏」和「准但不稳」，这就是工程上要权衡的东西**")
assert near(sens[116.0][1], -1.162492)
assert near(sens[146.0][1], -0.141305)
assert near(sens[176.0][1], 0.702888)
assert near(d_thr, 0.310897)
assert near(sens[116.0][2], -1.015489)
assert near(sens[176.0][2], -1.019491)
assert near(d_ws, -0.000667)
assert all(sens[t][3] == 7 for t in sens)
assert all(sens[t][4] == 9 for t in sens)
'''

E8_CODE = '''gt0 = 2 * GT_CIRCLES[CALIB_IDX][2]
# @@todo 用标定件求出比例尺，并把它施加到其余孤立目标上，得到校正后的误差数组
# @@hint 一行：scale = gt0 / M[CALIB_IDX]["d_eq"]
# @@hint 一行：corr_err = np.array([M[gi]["d_eq"] * scale - 2 * GT_CIRCLES[gi][2] for gi in ISOLATED])
scale = gt0 / M[CALIB_IDX]["d_eq"]
corr_err = np.array([M[gi]["d_eq"] * scale - 2 * GT_CIRCLES[gi][2] for gi in ISOLATED])
# @@end

print("标定件 GT#%d: 实测 %.6f px | 真值 %d px | 比例 %.6f（等价 1 px = %.6f mm，名义 %.4f）"
      % (CALIB_IDX, M[CALIB_IDX]["d_eq"], gt0, scale, scale / PX_PER_MM, 1 / PX_PER_MM))
print("校正前后（孤立目标）:")
for gi in ISOLATED:
    print("   GT#%d: %.6f -> %.6f px | 误差 %+.6f -> %+.6f" % (gi, M[gi]["d_eq"], M[gi]["d_eq"] * scale,
                                                             M[gi]["d_eq"] - 2 * GT_CIRCLES[gi][2],
                                                             M[gi]["d_eq"] * scale - 2 * GT_CIRCLES[gi][2]))
print("平均|偏差|: %.6f px -> %.6f px（降 %.4f%%）"
      % (np.abs(iso_err).mean(), np.abs(corr_err).mean(),
         100 * (1 - np.abs(corr_err).mean() / np.abs(iso_err).mean())))
print("毫米误差: 名义 %.6f mm -> 校正后 %.6f mm" % (np.abs(iso_err).mean() / PX_PER_MM,
                                                 np.abs(corr_err).mean() / PX_PER_MM))
print("解读: **这是工业测量里最关键的一步：标定。** "
      "① 名义标定（1 px = 0.25 mm）只是「读表」，它不含任何偏差修正，"
      "于是孤立目标平均差 0.254993 mm；"
      "② 拿一个**已知直径的标准件**量一次，就能把「系统性偏置 + 尺度误差」一起折进比例尺"
      "（这里 1.017502）—— 误差立刻从 0.254993 mm 压到 **0.059084 mm**；"
      "③ 注意标定件本身被校正成**误差精确为 0**（60.000000），这不是巧合，是「用 A 标 A」的必然结果 —— "
      "**所以不能用标定件自己来评估精度**，必须用别的目标（本章用的是另外 4 个孤立目标）；"
      "④ 现场做测量时，这正是「先量量块 / 标定板」的意义所在")
assert near(scale, 1.017502)
assert near(M[CALIB_IDX]["d_eq"] * scale, 60.0, 1e-6)
assert near(np.abs(corr_err).mean(), 0.236334, 1e-6)
assert near(100 * (1 - np.abs(corr_err).mean() / np.abs(iso_err).mean()), 76.8293, 1e-3)
assert near(np.abs(corr_err).mean() / PX_PER_MM, 0.059084, 1e-6)
assert np.abs(corr_err).mean() < np.abs(iso_err).mean() / 3
'''

E9_CODE = '''coins = cv2.imread(str(CL / "coins.png"))
cgray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
cblur = cv2.GaussianBlur(cgray, (5, 5), 0)
# @@todo 与 ch10 同口径：Otsu 二值 + 开运算 2 次（注意硬币是**亮**的，阈值类型要用 BINARY）
# @@hint 两行：cot, cbw = cv2.threshold(cblur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# @@hint 一行：cop = cv2.morphologyEx(cbw, cv2.MORPH_OPEN, K3, iterations=2)
cot, cbw = cv2.threshold(cblur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cop = cv2.morphologyEx(cbw, cv2.MORPH_OPEN, K3, iterations=2)
# @@end

cws, cdist, cn = watershed_split(coins, cop, core_ratio=0.3)
clabs, ccnts = region_labels(cws)
# @@todo 逐个区域测量，收集 面积 / 等效直径 / 圆度 三个数组
# @@hint 一行：meas = [measure((cws == lab).astype(np.uint8)) for lab in clabs]
# @@hint 三行：ca = np.array([r["area"] for r in meas])；cd / crn 同理
meas = [measure((cws == lab).astype(np.uint8)) for lab in clabs]
ca = np.array([r["area"] for r in meas])
cd = np.array([r["d_eq"] for r in meas])
crn = np.array([r["roundness"] for r in meas])
# @@end

print("coins %s | Otsu %.1f | 二值前景 %d | 开运算后 %d | 连通域轮廓 %d"
      % (coins.shape, cot, int((cbw > 0).sum()), int((cop > 0).sum()), len(count_cc(cop))))
print("dist.max %.6f | 核心 %d | ws.max %d | 目标区域 %d | 边界像素 %d"
      % (float(cdist.max()), cn - 1, int(cws.max()), len(ccnts), int((cws == -1).sum())))
print("面积(px): min %d max %d mean %.4f" % (ca.min(), ca.max(), ca.mean()))
print("等效直径(px): min %.4f max %.4f mean %.4f" % (cd.min(), cd.max(), cd.mean()))
print("圆度: min %.6f max %.6f mean %.6f" % (crn.min(), crn.max(), crn.mean()))
print("面积 > 1500 的 %d 个 | 圆度 > 0.8 的 %d 个 | 圆度 < 0.4 的 %d 个"
      % (int((ca > 1500).sum()), int((crn > 0.8).sum()), int((crn < 0.4).sum())))
print("解读: ① coins 的连通域是 **24** 个、分水岭分离后是 **25** 个 —— "
      "和 ch10 完全对得上（ch10 §10.4 的 ws.max = 26 含背景标签）；"
      "② **圆度立刻帮你发现异常**：25 个区域里 23 个圆度 > 0.8（正常硬币），"
      "有 **2 个 < 0.4**（0.223429 那个是贴着图像边缘被截断的硬币）—— "
      "不需要 GT，规则就能筛出来「这几个不许当合格品」；"
      "③ 真实图没有精确 GT，所以**能做的交付是「计数 + 尺寸分布 + 形状质检」**，"
      "而不是逐目标误差 —— 这是合成场景与真实场景在报告口径上的根本差别")
assert near(cot, 104.0)
assert int((cbw > 0).sum()) == 48069 and int((cop > 0).sum()) == 47691
assert near(float(cdist.max()), 47.384338)
assert cn - 1 == 25 and len(ccnts) == 25 and int((cws == -1).sum()) == 4596
assert ca.min() == 559 and ca.max() == 6086
assert near(ca.mean(), 1679.36, 1e-4)
assert near(cd.mean(), 44.7470, 1e-4)
assert near(float(crn.mean()), 0.852731)
assert int((crn > 0.8).sum()) == 23 and int((crn < 0.4).sum()) == 2
'''

E10_CODE = '''vis = img.copy()
ov = np.zeros_like(vis)
# @@todo 目标区域涂半透明绿，边界像素涂纯红
# @@hint 两行：ov[ws >= 2] = (0, 200, 0)；vis = cv2.addWeighted(vis, 0.7, ov, 0.3, 0)
# @@hint 一行：vis[ws == -1] = (0, 0, 255)
ov[ws >= 2] = (0, 200, 0)
vis = cv2.addWeighted(vis, 0.7, ov, 0.3, 0)
vis[ws == -1] = (0, 0, 255)
# @@end

for gi in sorted(M):
    r = M[gi]
    # @@todo 在目标中心画红点，并在上方标注「校正后的直径(mm)」保留 1 位小数
    # @@hint 两行：cv2.circle(vis, (int(round(r["cx"])), int(round(r["cy"]))), 3, (0, 0, 255), -1)
    # @@hint 一行：cv2.putText(vis, "%.1fmm" % (r["d_eq"] * scale / PX_PER_MM), (int(r["cx"]) - 20, int(r["cy"]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(vis, (int(round(r["cx"])), int(round(r["cy"]))), 3, (0, 0, 255), -1)
    cv2.putText(vis, "%.1fmm" % (r["d_eq"] * scale / PX_PER_MM), (int(r["cx"]) - 20, int(r["cy"]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    # @@end

changed = int((np.abs(vis.astype(int) - img.astype(int)).sum(axis=2) > 0).sum())
in_gt = int(((ws >= 2) & (GTM > 0)).sum()) / int((ws >= 2).sum())
out_gt = int(((ws >= 2) & (GTM == 0)).sum()) / int((ws >= 2).sum())
print("叠加图: 目标区 %d px | 边界 %d px | 标注目标 %d 个 | 改动像素 %d / %d"
      % (int((ws >= 2).sum()), int((ws == -1).sum()), len(M), changed, W * H))
print("目标区落在 GT 内的比例 %.6f | 落在 GT 外的比例 %.6f" % (in_gt, out_gt))
print("解读: **可视化报告是唯一能让产线的人一眼看懂的分工方式。**"
      "① 绿色 = 「被判为目标的像素」、红色 = 分水岭画的边界线、黄字 = 校正后的直径 —— "
      "三样东西叠在**原图**上，人才能发现「这个边界切歪了」「那个字标到了界外」；"
      "② 注意 `cv2.addWeighted(vis, 0.7, ov, 0.3, 0)` 会改动**整幅图**"
      "（改动像素 = 307200 = 640x480，一个不漏）—— 因为 0.7 倍加权对所有像素都生效，"
      "不只是叠加区。想只改叠加区，得用布尔掩码赋值而不是 addWeighted；"
      "③ 「目标区落在 GT 外只有 0.000477」说明整体是对的，"
      "但这 **0.0477%** 就是前面那些 7.5% 尺寸误差的来源；"
      "④ 报告三件套：**数量 + 尺寸表 + 叠加图**。少任何一个，交付都不完整")
assert int((ws >= 2).sum()) == 37703
assert int((ws == -1).sum()) == 4102
assert len(M) == 9
assert changed == 307200
assert near(in_gt, 0.999523, 1e-6)
assert near(out_gt, 0.000477, 1e-6)
'''

E11_CODE = '''# @@todo 把本章三条口径并排成一张表：(目标数, 平均|直径误差| px)
# @@hint 三行：第一条「纯阈值(连通域)」用 n_cc 与 sens[146.0][1] 的绝对值
# @@hint 一行：第二条「分水岭(Otsu)」用 n_targets 与 np.abs(iso_err).mean()
# @@hint 一行：第三条「分水岭 + 标定校正」用 n_targets 与 np.abs(corr_err).mean()
report = {"纯阈值(连通域)": (n_cc, round(abs(sens[146.0][1]), 6)),
          "分水岭(Otsu)": (n_targets, round(float(np.abs(iso_err).mean()), 6)),
          "分水岭 + 标定校正": (n_targets, round(float(np.abs(corr_err).mean()), 6))}
# @@end

print("=" * 70)
print("final_03 结论表（目标数 / 孤立目标平均|直径误差| px）")
print("-" * 70)
for k, (n_t, e) in report.items():
    print("  %-22s 目标数 %s | 平均|误差| %.6f px" % (k, n_t, e))
print("-" * 70)
print("  像素级 Dice %.6f | IoU %.6f（很高，但只说明像素对）" % (dice, iou))
print("  粘连目标平均|相对误差| %.6f（分水岭也不保证准）" % np.abs(con_rel).mean())
print("=" * 70)
print("交付结论：")
print("  1. **像素级指标和实例级指标必须一起报**（Dice 0.998681 vs 连通域 7/9）。")
print("  2. 分水岭核心阈值只能扫不能拍：0.2->7 个、0.3->8 个、0.4~0.45->9 个、0.5 反而变差。")
print("  3. 孤立目标偏差稳定 -1.019971 px（相对 -1.43%）；粘连目标平均|偏差| 2.832284 px。")
print("  4. 相对误差与尺寸强负相关（-0.970140）—— 小目标测不准是几何必然。")
print("  5. 「面积」有三种口径：contourArea 比像素计数小 2.74%，最小外接圆平均大 13.41%。")
print("  6. 纯阈值路线 +0.310897 px/10 阈值（准但不稳）；分水岭 -0.000667 px/10（稳但偏）。")
print("  7. **标定件把误差从 0.254993 mm 压到 0.059084 mm（降 76.83%）—— 这是工业测量的关键一步。**")
print("  8. 圆度是免费的质检指标：GT#2 被切坏时圆度掉到 0.519006（正常约 0.85）。")
print("  9. 报告三件套：数量 + 尺寸表 + 叠加图。")
assert len(report) == 3
assert report["分水岭 + 标定校正"][1] < report["分水岭(Otsu)"][1] / 3
assert report["纯阈值(连通域)"][0] == 7 and report["分水岭(Otsu)"][0] == 9
'''
# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# final_03 全流程：分割测量

> 03_cv 的第三个综合题。前两个综合题分别在「检测」和「分类」上收口，这一个落在
> **工业测量**上：**阈值 → 形态学 → 距离变换/分水岭分离 → 逐目标测量 → 与 GT 对账 →
> 标定校正 → 可视化报告**。
>
> 它要回答的是：**「我量出来的尺寸能信吗？误差从哪来？怎么把它压下去？」**

## 本章路线

| 小节 | 内容 | 关键结论 |
|---|---|---|
| §F3.1 | 环境与场景（9 个已知直径的圆，两对粘连） | GT 精确已知 |
| §F3.2 | 分割 + **像素级对账** | Dice **0.998681** / IoU 0.997366 |
| §F3.3 | **实例级计数基线** | 连通域 **7** vs GT **9** |
| §F3.4 | **核心阈值扫描** | 0.2→7 · 0.3→8 · **0.4/0.45→9** · 0.5 反而变差 |
| §F3.5 | 分水岭分离 | 9 区域、边界 4102 px |
| §F3.6 | **逐目标测量** | 面积 / 等效直径 / 周长 / 圆度 |
| §F3.7 | **误差分解** | 孤立 **-1.019971 px** vs 粘连最差 **+7.5171%** |
| §F3.8 | 相对误差 ∝ 1/直径 | 相关系数 **-0.970140** |
| §F3.9 | **三种面积口径** | contourArea **-2.74%** / 最小外接圆 **+13.41%** |
| §F3.10 | **阈值敏感性：两条路线** | 纯阈值 **+0.310897 px/10** vs 分水岭 **-0.000667 px/10** |
| §F3.11 | **标定件校正** | 误差 **降 76.83%**（0.254993 → 0.059084 mm） |
| §F3.12 | 真实图 `coins` | 连通域 24 → 分水岭 **25 个目标** |
| §F3.13 | 可视化报告 + 结论 | 数量 + 尺寸表 + 叠加图 |

## 核心矛盾（本章的三句话）

1. **「分割得好」≠「量得准」**：Dice 0.998681，但连通域少 2 个，直径系统性偏小 1.02 px。
2. **「稳」和「准」是两条不同的路线**：分水岭把阈值敏感性消掉了（-0.000667 px/10），
   代价是一个 -1.02 px 的固定偏置；纯阈值反过来。
3. **标定是工业测量的分水岭**：一个标准件就把毫米误差从 0.254993 压到 0.059084。

## 关于数据

- **自造测量场景**（640×480）：9 个深色圆，直径 52~96 px，其中 #1/#2 与 #5/#6 互相重叠。
  **GT 是精确已知的**（自己画的），所以能逐目标算误差。
- **`classic/coins.png`**：真实硬币，与 ch10 §10.4 完全同口径（Otsu 104.0、开运算、core 0.3）。
  真实图没有精确 GT，所以交付口径只能是「计数 + 尺寸分布 + 形状质检」。

> 合成场景和真实场景在报告上的差别，本身就是本章要讲的一个知识点。
"""
    ),
    md(
        """
## §F3.1 环境与场景

先看清楚「待测的不是随机点，而是 9 个有确定直径的圆」。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## §F3.1b 两个核心函数：分水岭与测量

这两个函数在本章只写一次、用很多次（合成场景、阈值扫描、coins 都要用）。

`watershed_split` 的坑全在 **markers 约定**：`0` = 待定、`1` = 背景、`>= 2` = 各目标核心。

`measure` 里最值得注意的是**圆度**：`4πA/P²`。理想圆是 1，但数字图像有阶梯效应，
实际圆的上限约 **0.87** —— 这不是误差，是「数字周长的固有偏大」。
"""
    ),
    code(T1_CODE),
    code(T2_CODE),
    md(
        """
## §F3.2 分割与像素级对账

Otsu + 开闭运算，然后先算最基本的 Dice / IoU。
"""
    ),
    code(E1_CODE),
    md(
        """
## §F3.3 核心阈值扫描：窗口只能扫出来

`core_ratio` 决定「一个目标必须多「胖」才被认为有一个核心」。
这个参数**没有任何理论最优值**，只能扫。
"""
    ),
    code(E2_CODE),
    md(
        """
## §F3.4 分水岭分离

选定 0.45 档。注意三个标签的含义。
"""
    ),
    code(E3_CODE),
    md(
        """
## §F3.5 逐目标测量

把每个区域的面积换算成等效直径，与 GT 逐个对账。
"""
    ),
    code(E4_CODE),
    md(
        """
## §F3.6 相对误差 ∝ 1/直径

偏置是固定 1 px 量级，所以**目标越小相对误差越大**。
"""
    ),
    code(E5_CODE),
    md(
        """
## §F3.7 三种面积口径

`mask.sum()` / `cv2.contourArea` / `πr²(最小外接圆)` —— 同一块区域，三个不同的「面积」。
"""
    ),
    code(E6_CODE),
    md(
        """
## §F3.8 阈值敏感性：纯阈值 vs 分水岭

**本章最有价值的一节。** 两条路线跑同一组阈值，性格完全不同。
"""
    ),
    code(E7_CODE),
    md(
        """
## §F3.9 标定件校正

名义标定只是「读表」。真正的精度来自**用已知尺寸的标准件校正**。
"""
    ),
    code(E8_CODE),
    md(
        """
## §F3.10 真实图：coins

与 ch10 同口径走一遍，看真实图的报告口径有什么不同。
"""
    ),
    code(E9_CODE),
    md(
        """
## §F3.11 可视化报告
"""
    ),
    code(E10_CODE),
    md(
        """
## §F3.12 结论表
"""
    ),
    code(E11_CODE),
    md(
        """
## 自查清单

- [ ] 知道**像素级指标（Dice/IoU）和实例级指标（计数）必须一起报**
- [ ] 知道高 Dice 可能掩盖漏检（0.998681 vs 7/9）
- [ ] 知道 `DIST_MASK` 与 `distanceTransform` 的用法
- [ ] 会写分水岭的**三个区域**（核心 / 背景 / 待定）
- [ ] **记得 markers 约定：0 = 待定、1 = 背景、≥2 = 目标；输出里 -1 是边界线**
- [ ] 知道 `cv2.subtract` 处理 0/255 图（不要用 numpy 相减）
- [ ] 知道 `watershed` 会**就地改** markers，要传 `.copy()`
- [ ] **知道核心阈值只能扫不能拍**（本章 0.4~0.45、ch10 coins 0.2~0.3）
- [ ] 会算**等效直径** `2√(A/π)`
- [ ] 会算**圆度** `4πA/P²`，并知道数字图像上限约 0.87（不是 1）
- [ ] **知道圆度是免费的质检指标**（不需要 GT 就能发现切坏的区域）
- [ ] 知道「面积」有三种口径，差 2.74% ~ 93%
- [ ] **知道分水岭的边界由图像梯度决定，不由二值掩码决定**
- [ ] 知道纯阈值路线「准但不稳」，分水岭路线「稳但偏」
- [ ] 理解**相对误差 ∝ 1/直径**（小目标测不准是几何必然）
- [ ] **理解标定的意义**，以及「不能用标定件自己评估精度」
- [ ] 知道真实图（无 GT）与合成图（有 GT）的**报告口径不同**
- [ ] 知道报告三件套：**数量 + 尺寸表 + 叠加图**

## 交付物

| 文件 | 内容 |
|---|---|
| `final_03_measure_pipeline.ipynb` | 讲解版（含结论与解读） |
| `final_03_measure_pipeline_practice.ipynb` | 练习版（挖空 + 提示） |
| `final_03_measure_pipeline_solution.ipynb` | 答案版（完整可跑） |
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# final_03 全流程：分割测量（练习版）

> 按提示补全 `____`，跑通所有 assert。
> `img` / `g` / `GTM` / `GT_CIRCLES` / `ISOLATED` / `CONTACTED` 以及
> `seg_otsu` / `cleanup` / `count_cc` / `nearest_gt` / `region_labels` / `collect`
> 脚手架可用；`watershed_split` / `measure` 需要你补全函数体。
>
> 注意：挖空块内每条语句保持**单行**；控制流写在挖空块之外
> （所有循环骨架都已给好，你只填里面的几行）。
> 题目真值都来自实跑（opencv 5.0.0 / numpy 2.5.3）。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：`watershed_split`（三区 + markers）

核心区 / 背景区 / 待定区，然后按约定造 markers。
"""
    ),
    code(T1_CODE),
    md(
        """
## 任务 2：`measure`（等效直径与圆度）

`d_eq = 2√(A/π)`、`圆度 = 4πA/P²`。
"""
    ),
    code(T2_CODE),
    md(
        """
## 任务 3：分割 + 像素级与实例级对账
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 4：核心阈值扫描
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 5：选定档位的分水岭结果
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 6：孤立 / 粘连两组误差
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 7：相对误差与直径的相关系数
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 8：三种面积口径
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 9：阈值敏感性（纯阈值 vs 分水岭）
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 10：标定件校正
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 11：真实图 coins
"""
    ),
    code(E9_CODE),
    md(
        """
## 任务 12：可视化报告
"""
    ),
    code(E10_CODE),
    md(
        """
## 任务 13：结论表
"""
    ),
    code(E11_CODE),
    md(
        """
## 自查清单

- [ ] 像素级与实例级指标一起报
- [ ] 分水岭三区 + markers 约定（0/1/≥2，-1 是边界）
- [ ] `cv2.subtract` 处理 0/255 图
- [ ] `watershed` 会就地改 markers，要 `.copy()`
- [ ] 核心阈值只能扫不能拍
- [ ] 等效直径与圆度公式；数字圆度上限约 0.87
- [ ] 圆度是免费的质检指标
- [ ] 三种面积口径
- [ ] 分水岭边界由梯度决定
- [ ] 纯阈值「准但不稳」vs 分水岭「稳但偏」
- [ ] 相对误差 ∝ 1/直径
- [ ] 标定的意义（不能用标定件自评）
- [ ] 报告三件套：数量 + 尺寸表 + 叠加图
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
