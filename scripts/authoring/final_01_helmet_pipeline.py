#!/usr/bin/env python3
"""cv final_01 全流程（安全帽视觉检测）：坏图体检 / 标注质检 / HSV 阈值分割 ->
形态学 -> 轮廓计数 -> 与 GT 对账 -> 参数扫描 -> 色带融合 -> 合成对照场景 -> 报告。

数据：`helmet/` 的 4 张现场图 + `labels.json`（VOC xyxy + helmet/head 两类），
外加本章**自造的**两个白底合成场景（颜色单一可控）作为对照组。
真值全部实跑（opencv 5.0.0 / numpy 2.5.3）。

> 本章的核心**不是一个漂亮的结果，而是一个诚实的失败**：
> 黄带 HSV 阈值在 4 张现场图上 **TP = 0 / 检出 64 框**，
> 15 组 (面积阈值 × 开运算次数) 全部 **F1 = 0**，
> 6 种色带融合最好也只有 **F1 = 0.063492**。
> 根因是 21 个 GT 头盔的**中位色相从 7 跨到 116**（黄 11 / 蓝 7 / 红 3）——
> 现场根本没有统一颜色。而同一条流水线在**白底合成场景**上 P/R/F1 = **1.0 / 1.0 / 1.0**。
> 所以本章的结论是：**阈值分割只在「背景可控、颜色可预测」时成立**；
> 一旦进入真实现场，就必须换成学习式方法（ch07 特征 + 分类器 / ch09 检测框 + NMS）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/final_01_helmet_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv" / "final"
NAME = "final_01_helmet_pipeline"

IMPORTS = '''from pathlib import Path

import json

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

DATA = Path("../data")
HL = DATA / "helmet"
IOU_THR = 0.3
K3 = np.ones((3, 3), np.uint8)
'''

SETUP = '''lab = json.loads((HL / "labels.json").read_text(encoding="utf-8"))
NAMES = ["site_01.jpg", "site_03.jpg", "site_05.jpg", "site_07.jpg"]

print("标注类别:", lab["classes"], "（本章只做二分类：helmet 算正类，head 不当目标）")
print("进场图:", NAMES)
for nm in NAMES:
    meta = lab["images"][nm]
    print(f"   {nm}: {meta['width']:4d}x{meta['height']:4d} | 目标 {len(meta['objects'])} 个")
print("IOU_THR =", IOU_THR, "（IoU >= 0.3 记命中 —— 头盔是小目标，阈值给松一点）")
print("提示: 本章有两套数据 —— ① 4 张真实现场图（带 GT，但**颜色不可控**）；"
      "② 自造的白底合成场景（颜色可控，用来验证流水线本身对不对）")
'''

SCAFFOLD = '''# 脚手架：色带定义 + 合成场景 + 5 个工具函数（缺口都在 @@todo 里）
REAL_BANDS = {
    "红低": (np.array([0, 90, 70]), np.array([10, 255, 255])),
    "红高": (np.array([170, 90, 70]), np.array([180, 255, 255])),
    "黄": (np.array([15, 80, 80]), np.array([40, 255, 255])),
    "蓝": (np.array([90, 60, 80]), np.array([130, 255, 255])),
    "白": (np.array([0, 0, 180]), np.array([180, 60, 255])),
}
SYN_BANDS = {
    "橙黄": (np.array([10, 120, 120]), np.array([40, 255, 255])),
    "蓝": (np.array([90, 120, 120]), np.array([130, 255, 255])),
    "白": (np.array([0, 0, 200]), np.array([180, 50, 255])),
}

S = 480
CENTERS = [(80, 80), (215, 80), (350, 80), (80, 245), (215, 245), (350, 245), (140, 400), (310, 400)]
SYN_ORANGE = (0, 165, 255)
SYN_BLUE = (200, 90, 20)


def make_scene(n_blue):
    """造一张白底合成图：8 个圆「安全帽」，末 n_blue 个是蓝色，其余橙色。返回 (图, GT 框)。"""
    rng = np.random.default_rng(3)
    img = np.full((S, S, 3), 245, np.uint8)
    boxes = []
    for i, (cx, cy) in enumerate(CENTERS):
        col = SYN_BLUE if i >= len(CENTERS) - n_blue else SYN_ORANGE
        cv2.circle(img, (cx, cy), 28, col, -1)
        boxes.append((cx - 28, cy - 28, cx + 28, cy + 28))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return img, boxes


def gt_helmet(nm):
    """取某张图的 helmet 框（VOC xyxy），head 不算目标。"""
    return [o["bbox_xyxy"] for o in lab["images"][nm]["objects"] if o["class_name"] == "helmet"]


def near(a, b, tol=1e-6):
    """浮点容差比较。真值都实跑得到，用容差避免 == 在末位翻车。"""
    return abs(float(a) - float(b)) <= tol


def iou(a, b):
    """手撕 IoU：交集左上取 max、右下取 min，宽高夹到 0，再除以并集。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match_counts(dets, gts, thr=IOU_THR):
    """贪心匹配，返回 (TP, FP, FN)。每个 GT 最多被认领一次。"""
    matched, tp = set(), 0
    for d in dets:
        for j, g in enumerate(gts):
            if j in matched:
                continue
            if iou(d, g) >= thr:
                tp += 1
                matched.add(j)
                break
    return tp, len(dets) - tp, len(gts) - tp


def prf(tp, fp, fn):
    """由 TP/FP/FN 算 (Precision, Recall, F1)，保留 6 位小数。"""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 6), round(r, 6), round(f1, 6)


def hsv_pipeline(img, bands, area_thr=200, it=1):
    """HSV 阈值 -> 多色带按位或 -> 开运算 -> 闭运算 -> 外轮廓 -> 面积过滤，返回 (框表, 掩码, 开后, 闭后)。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], np.uint8)
    for b in bands:
        lo, hi = REAL_BANDS.get(b, SYN_BANDS.get(b))
        mask |= cv2.inRange(hsv, lo, hi)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K3, iterations=it)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, K3, iterations=2)
    cs = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = []
    for c in cs:
        if cv2.contourArea(c) > area_thr:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
    return boxes, mask, opened, closed
'''

E1_CODE = '''health = {}
for nm in NAMES:
    img = cv2.imread(str(HL / nm))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # @@todo 算清晰度指标（Laplacian 方差）并把这 6 个指标写进 health[nm]
    # @@hint 一行：lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    # @@hint 六元组顺序：img.shape / 灰度均值 / 灰度标准差 / 过曝占比 / 欠曝占比 / lap
    # @@hint 一元组：health[nm] = (img.shape, round(float(g.mean()), 4), round(float(g.std()), 4),
    # @@hint              round(float((g > 250).mean()), 6), round(float((g < 5).mean()), 6), round(lap, 4))
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    health[nm] = (img.shape, round(float(g.mean()), 4), round(float(g.std()), 4),
                  round(float((g > 250).mean()), 6), round(float((g < 5).mean()), 6), round(lap, 4))
    # @@end

print("坏图体检（尺寸 / 均值 / 标准差 / 过曝 / 欠曝 / 清晰度）:")
for nm in NAMES:
    sh, mu, sd, over, under, lap = health[nm]
    print(f"   {nm}: {sh[1]}x{sh[0]} | 均值 {mu:8.4f} 标准差 {sd:7.4f} | 过曝 {over:.6f} 欠曝 {under:.6f} | LapVar {lap:9.4f}")
print("解读: ① **Laplacian 方差 ≈ 清晰度** —— 值越小越糊（运动模糊 / 失焦）；"
      "② site_07 欠曝像素占 0.047629（其他三张都在 1.4e-04 以下）—— **这张图整体偏暗**，"
      "而 HSV 的 V 通道直接受亮度影响，暗图会让阈值分割整体掉一批；"
      "③ 巡检场景里「先体检再算指标」是必须的：把坏图混进统计，指标会无端变差，"
      "而你会误以为是算法不行")
assert health["site_01.jpg"][5] == 1189.0867
assert health["site_03.jpg"][5] == 2788.226
assert health["site_05.jpg"][5] == 2218.9136
assert health["site_07.jpg"][5] == 943.093
assert health["site_07.jpg"][4] == 0.047629
assert health["site_01.jpg"][4] == 4.5e-05
assert health["site_01.jpg"][0] == (768, 1024, 3)
'''

E2_CODE = '''ALL_BOXES = np.array([o["bbox_xyxy"] for nm in NAMES for o in lab["images"][nm]["objects"]], dtype=np.float64)
# @@todo 从 ALL_BOXES 一次算出三件东西：面积、宽高比、小目标（面积<500）占比
# @@hint 三行：areas = (ALL_BOXES[:, 2] - ALL_BOXES[:, 0]) * (ALL_BOXES[:, 3] - ALL_BOXES[:, 1])
# @@hint aspects = (ALL_BOXES[:, 2] - ALL_BOXES[:, 0]) / (ALL_BOXES[:, 3] - ALL_BOXES[:, 1])
# @@hint small_ratio = float((areas < 500).mean())
areas = (ALL_BOXES[:, 2] - ALL_BOXES[:, 0]) * (ALL_BOXES[:, 3] - ALL_BOXES[:, 1])
aspects = (ALL_BOXES[:, 2] - ALL_BOXES[:, 0]) / (ALL_BOXES[:, 3] - ALL_BOXES[:, 1])
small_ratio = float((areas < 500).mean())
# @@end

# @@todo 分别数出 helmet 与 head 的框数
# @@hint 两行：n_helmet = sum(1 for nm in NAMES for o in lab["images"][nm]["objects"] if o["class_name"] == "helmet")
# @@hint n_head = sum(1 for nm in NAMES for o in lab["images"][nm]["objects"] if o["class_name"] == "head")
n_helmet = sum(1 for nm in NAMES for o in lab["images"][nm]["objects"] if o["class_name"] == "helmet")
n_head = sum(1 for nm in NAMES for o in lab["images"][nm]["objects"] if o["class_name"] == "head")
# @@end

print("框总数:", ALL_BOXES.shape[0], "| ALL_BOXES.shape =", ALL_BOXES.shape)
print("面积: min %.1f  max %.1f  mean %.6f" % (areas.min(), areas.max(), float(areas.mean())))
print("宽高比: min %.6f  max %.6f  mean %.6f" % (aspects.min(), aspects.max(), float(aspects.mean())))
print("小目标(面积<500)占比: %.6f（%d 个）" % (small_ratio, int((areas < 500).sum())))
print("类别分布: helmet %d / head %d" % (n_helmet, n_head))
print("解读: ① 标注质检要在**建模之前**做 —— 面积跨度 238~3264（相差 13.7 倍）说明"
      "目标尺度很不统一，全靠一个面积阈值去卡是危险的；"
      "② 宽高比 0.95~1.461538、均值 1.18128 —— 都比较接近 1，"
      "说明这些头盔/人头基本是**近方形**的，可以用方形先验做后处理（但别当硬约束）；"
      "③ **小目标占 0.206897（6/29）** —— 五分之一的目标面积小于 500 像素，"
      "这正是 ch09 讲过的「IoU 对偏移敏感度与面积成反比」的重灾区")
assert ALL_BOXES.shape == (29, 4)
assert areas.min() == 238.0 and areas.max() == 3264.0
assert near(areas.mean(), 1451.655172, 1e-6)
assert near(aspects.min(), 0.95, 1e-6) and near(aspects.max(), 1.461538, 1e-6)
assert near(aspects.mean(), 1.18128, 1e-6)
assert near(small_ratio, 0.206897, 1e-6)
assert int((areas < 500).sum()) == 6
assert n_helmet == 21 and n_head == 8
'''

E3_CODE = '''yellow_runs = {}
for nm in NAMES:
    img = cv2.imread(str(HL / nm))
    # @@todo 跑「黄」带流水线，记下 (掩码像素数, 开运算后, 闭运算后, 检出框数)
    # @@hint 一行：det, mask, op, cl = hsv_pipeline(img, ["黄"])
    # @@hint 一行：yellow_runs[nm] = (int((mask > 0).sum()), int((op > 0).sum()), int((cl > 0).sum()), len(det))
    det, mask, op, cl = hsv_pipeline(img, ["黄"])
    yellow_runs[nm] = (int((mask > 0).sum()), int((op > 0).sum()), int((cl > 0).sum()), len(det))
    # @@end

print("黄带阈值路线（掩码像素 / 开运算后 / 闭运算后 / 检出框数）:")
for nm in NAMES:
    m, o, c, n = yellow_runs[nm]
    print(f"   {nm}: 掩码 {m:7d} -> 开 {o:7d} -> 闭 {c:7d} | 检出 {n:3d}")
print("解读: ① 掩码像素从 4807（site_03）到 99367（site_05），**相差 20 倍** —— "
      "同一组 H/S/V 阈值在不同曝光、不同背景的图上覆盖面积完全不同；"
      "② 开运算把掩码砍掉约 10%~30%（消掉椒盐噪点），闭运算只补回一点点；"
      "③ **检出框数（28 / 1 / 25 / 10）和真实头盔数（0 / 6 / 10 / 5）对不上** —— "
      "框多不代表对，框少也不代表漏。要判对错，只能和 GT 对账")
assert yellow_runs["site_01.jpg"] == (77556, 69626, 72993, 28)
assert yellow_runs["site_03.jpg"] == (4807, 2843, 3128, 1)
assert yellow_runs["site_05.jpg"] == (99367, 69745, 74967, 25)
assert yellow_runs["site_07.jpg"] == (13275, 11454, 12370, 10)
'''

E4_CODE = '''per_img = {}
for nm in NAMES:
    img = cv2.imread(str(HL / nm))
    det, _, _, _ = hsv_pipeline(img, ["黄"])
    # @@todo 与 GT 做 IoU 匹配，把 (检出数, TP, FP, FN, GT 数) 记进 per_img
    # @@hint 两行：tp, fp, fn = match_counts(det, gt_helmet(nm))
    # @@hint 一字典：per_img[nm] = (len(det), tp, fp, fn, len(gt_helmet(nm)))
    tp, fp, fn = match_counts(det, gt_helmet(nm))
    per_img[nm] = (len(det), tp, fp, fn, len(gt_helmet(nm)))
    # @@end

tot_det = sum(v[0] for v in per_img.values())
tot_tp = sum(v[1] for v in per_img.values())
tot_fp = sum(v[2] for v in per_img.values())
tot_fn = sum(v[3] for v in per_img.values())

print("黄带 vs GT 对账:")
for nm in NAMES:
    nd, t, f, n, g = per_img[nm]
    print(f"   {nm}: 检出 {nd:3d} | TP {t:2d} FP {f:3d} FN {n:2d} | GT {g:2d} | P/R/F1 {prf(t, f, n)}")
print("合计: 检出 %d | TP %d FP %d FN %d -> P/R/F1 %s" % (tot_det, tot_tp, tot_fp, tot_fn, prf(tot_tp, tot_fp, tot_fn)))
print("解读: **这是一个彻头彻尾的失败：64 个检出框，0 个命中。**"
      "注意「FP 64」不是说这 64 个框都在瞎指，而是说**它们和任何一个头盔的 IoU 都 < 0.3**；"
      "site_01 甚至检出 28 个框而 GT 为 0 —— 这张图里根本没人戴黄帽子，"
      "阈值却把大片背景/泥地判成了黄色。**F1 = 0 不是代码 bug，是方法失效**")
assert per_img["site_01.jpg"] == (28, 0, 28, 0, 0)
assert per_img["site_03.jpg"] == (1, 0, 1, 6, 6)
assert per_img["site_05.jpg"] == (25, 0, 25, 10, 10)
assert per_img["site_07.jpg"] == (10, 0, 10, 5, 5)
assert (tot_det, tot_tp, tot_fp, tot_fn) == (64, 0, 64, 21)
assert prf(tot_tp, tot_fp, tot_fn) == (0.0, 0.0, 0.0)
'''

E5_CODE = '''sweep = {}
for area_thr in (100, 200, 400, 800, 1500):
    for it in (1, 2, 3):
        dt = tp = 0
        for nm in NAMES:
            img = cv2.imread(str(HL / nm))
            # @@todo 用当前 (area_thr, it) 重跑流水线，累加「总检出数 dt」与「总命中 tp」
            # @@hint 一行：det, _, _, _ = hsv_pipeline(img, ["黄"], area_thr, it)
            # @@hint 一行：t, _, _ = match_counts(det, gt_helmet(nm))
            # @@hint 一行：dt, tp = dt + len(det), tp + t
            det, _, _, _ = hsv_pipeline(img, ["黄"], area_thr, it)
            t, _, _ = match_counts(det, gt_helmet(nm))
            dt, tp = dt + len(det), tp + t
            # @@end
        sweep[(area_thr, it)] = (dt, tp) + prf(tp, dt - tp, 21 - tp)

print("参数扫描（面积阈值 × 开运算次数）:")
for area_thr in (100, 200, 400, 800, 1500):
    for it in (1, 2, 3):
        dt, tp, p, r, f1 = sweep[(area_thr, it)]
        print(f"   area>{area_thr:4d} open{it}: 检出 {dt:4d} TP {tp:2d} -> P {p:.6f} R {r:.6f} F1 {f1:.6f}")
best = max(sweep.items(), key=lambda kv: kv[1][4])
print("最佳 F1:", best[1][2:], "出现在 area>%d open%d" % best[0])
print("解读: **15 组参数全部 F1 = 0**。这说明失败不是「阈值没调好」—— "
      "面积从 100 拉到 1500（检出从 96 个砍到 20 个）、开运算从 1 次加到 3 次，"
      "都改不出一个真阳性。**调参只能优化一个方向正确的方法，救不了一个方向就错的方法**")
assert sweep[(100, 1)] == (96, 0, 0.0, 0.0, 0.0)
assert sweep[(200, 1)] == (64, 0, 0.0, 0.0, 0.0)
assert sweep[(800, 2)] == (32, 0, 0.0, 0.0, 0.0)
assert sweep[(1500, 1)] == (22, 0, 0.0, 0.0, 0.0)
assert sweep[(1500, 3)] == (20, 0, 0.0, 0.0, 0.0)
assert all(v[4] == 0.0 for v in sweep.values())
assert len(sweep) == 15
'''

E6_CODE = '''hs = []
for nm in NAMES:
    img = cv2.imread(str(HL / nm))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for o in lab["images"][nm]["objects"]:
        if o["class_name"] != "helmet":
            continue
        x1, y1, x2, y2 = [int(v) for v in o["bbox_xyxy"]]
        p = hsv[y1:y2, x1:x2]
        sel = p[p[..., 1] > 80]
        if len(sel):
            # @@todo 把该头盔「高饱和像素」的中位色相（保留 4 位小数）追加进 hs
            # @@hint 一行：hs.append(round(float(np.median(sel[..., 0])), 4))
            hs.append(round(float(np.median(sel[..., 0])), 4))
            # @@end
hs = np.array(hs)

# @@todo 一行算出六个统计量：(最小色相, 最大色相, 平均色相, 黄带数, 蓝带数, 红带数)
# @@hint 一元组：h_min, h_max, h_mean, n_yellow, n_blue, n_red = (
# @@hint   float(hs.min()), float(hs.max()), round(float(hs.mean()), 6),
# @@hint   int(((hs >= 15) & (hs <= 40)).sum()), int(((hs > 90) & (hs <= 130)).sum()),
# @@hint   int(((hs < 15) | (hs > 130)).sum()))
h_min, h_max, h_mean, n_yellow, n_blue, n_red = (
    float(hs.min()), float(hs.max()), round(float(hs.mean()), 6),
    int(((hs >= 15) & (hs <= 40)).sum()), int(((hs > 90) & (hs <= 130)).sum()),
    int(((hs < 15) | (hs > 130)).sum()))
# @@end

print("21 个 GT 头盔的中位色相:", hs.tolist())
print("最小 %.1f  最大 %.1f  平均 %.6f" % (h_min, h_max, h_mean))
print("落在各色带的个数: 黄 %d / 蓝 %d / 红 %d" % (n_yellow, n_blue, n_red))
print("解读: **这就是 F1 = 0 的根因 —— 现场头盔根本没有统一颜色。** "
      "色相从 7（红）一路跨到 116（蓝），中间被 40~90 的空白区隔开；"
      "黄带只覆盖 11/21，剩下 10 个（蓝 7 + 红 3）**从第一天起就不可能被黄带检出**。"
      "换任何一组黄带阈值都救不回来 —— 因为漏掉的不是「黄的一点边角」，是**整顶帽子**")
assert len(hs) == 21
assert h_min == 7.0 and h_max == 116.0
assert near(h_mean, 49.0, 1e-6)
assert (n_yellow, n_blue, n_red) == (11, 7, 3)
assert n_yellow + n_blue + n_red == 21
'''

E7_CODE = '''combos = [["黄"], ["黄", "白"], ["黄", "蓝"],
          ["红低", "红高", "黄"], ["红低", "红高", "黄", "蓝"],
          ["红低", "红高", "黄", "蓝", "白"]]
combo_res = {}
for cb in combos:
    dt = tp = 0
    for nm in NAMES:
        img = cv2.imread(str(HL / nm))
        # @@todo 用当前色带组合 cb 跑流水线，累加「总检出 dt」与「总命中 tp」
        # @@hint 一行：det, _, _, _ = hsv_pipeline(img, cb)
        # @@hint 一行：t, _, _ = match_counts(det, gt_helmet(nm))
        # @@hint 一行：dt, tp = dt + len(det), tp + t
        det, _, _, _ = hsv_pipeline(img, cb)
        t, _, _ = match_counts(det, gt_helmet(nm))
        dt, tp = dt + len(det), tp + t
        # @@end
    combo_res[tuple(cb)] = (dt, tp) + prf(tp, dt - tp, 21 - tp)

print("色带融合尝试（现场图）:")
for cb, v in combo_res.items():
    print(f"   {'+'.join(cb):22s}: 检出 {v[0]:4d} TP {v[1]:2d} -> P {v[2]:.6f} R {v[3]:.6f} F1 {v[4]:.6f}")
best_combo = max(combo_res.items(), key=lambda kv: kv[1][4])
print("最佳色带组合:", "+".join(best_combo[0]), "->", best_combo[1][2:])
print("解读: ① 加色带确实把 TP 从 0 拉到 1~4 个，但代价是检出框从 64 涨到 96~105 —— "
      "**TP 涨 4 个、FP 涨 41 个**，F1 只从 0 爬到 0.063492；"
      "② 为什么会这样？因为多色带的**并集**同时放大了背景里所有接近这些颜色的东西"
      "（安全网的黄、反光背心的橙、路面的灰白）；"
      "③ 「黄+白」看起来最好，其实是**巧合式的好**（白带把 V>180 的大片亮区吞进来，"
      "恰好框住了一个头盔）。**不要被这种数字骗了** —— 下一节会揭穿它")
assert combo_res[("黄",)] == (64, 0, 0.0, 0.0, 0.0)
assert combo_res[("黄", "白")] == (105, 4, 0.038095, 0.190476, 0.063492)
assert combo_res[("黄", "蓝")] == (87, 1, 0.011494, 0.047619, 0.018519)
assert combo_res[("红低", "红高", "黄")] == (66, 0, 0.0, 0.0, 0.0)
assert combo_res[("红低", "红高", "黄", "蓝")] == (83, 1, 0.012048, 0.047619, 0.019231)
assert combo_res[("红低", "红高", "黄", "蓝", "白")] == (96, 1, 0.010417, 0.047619, 0.017094)
assert best_combo[0] == ("黄", "白")
'''

E8_CODE = '''max_iou = {}
for nm in NAMES:
    img = cv2.imread(str(HL / nm))
    det, _, _, _ = hsv_pipeline(img, ["黄"])
    gts = gt_helmet(nm)
    # @@todo 一行：算「所有 检出框 × GT 框」里的最大 IoU；没有检出或没有 GT 时取 0.0
    # @@hint 一内置：max_iou[nm] = max((iou(d, g) for d in det for g in gts), default=0.0)
    max_iou[nm] = max((iou(d, g) for d in det for g in gts), default=0.0)
    # @@end

print("逐图最大 IoU 诊断:")
for nm in NAMES:
    print(f"   {nm}: 检出 {per_img[nm][0]:3d} GT {len(gt_helmet(nm)):2d} | 最大 IoU {max_iou[nm]:.6f}")
print("解读: ① site_01 的 GT 是 0，所以最大 IoU 必然是 0.0 —— **这张图不该参与召回统计**，"
      "把它算进 FN 分母会低估召回；"
      "② site_05 最大 0.189696、site_07 最大 0.068702，**都远低于 0.3 的命中线** —— "
      "说明这 64 个检出框和头盔在**空间上也不沾边**（不只是分类错，是位置也错）；"
      "③ 「最大 IoU」是诊断阈值方案的**低成本探针**：如果最大 IoU 已经有 0.5，"
      "那问题在阈值/后处理；如果最大 IoU 长期 < 0.2，那**整个颜色假设就不成立**，"
      "不用再调了")
assert max_iou["site_01.jpg"] == 0.0
assert max_iou["site_03.jpg"] == 0.0
assert near(max_iou["site_05.jpg"], 0.189696, 1e-6)
assert near(max_iou["site_07.jpg"], 0.068702, 1e-6)
assert max(v for v in max_iou.values()) < IOU_THR
'''

E9_CODE = '''imgA, gtA = make_scene(0)
detA, maskA, opA, clA = hsv_pipeline(imgA, ["橙黄"])
# @@todo 记下三类像素数，再与 gtA 匹配算 P / R / F1
# @@hint 一元组：pixA = (int((maskA > 0).sum()), int((opA > 0).sum()), int((clA > 0).sum()))
# @@hint 两行：tpA, fpA, fnA = match_counts(detA, gtA)
# @@hint 一行：prfA = prf(tpA, fpA, fnA)
pixA = (int((maskA > 0).sum()), int((opA > 0).sum()), int((clA > 0).sum()))
tpA, fpA, fnA = match_counts(detA, gtA)
prfA = prf(tpA, fpA, fnA)
# @@end

print("合成场景 A（白底 + 8 顶全橙，颜色单一可控）")
print("  掩码 %d -> 开 %d -> 闭 %d | 检出 %d 框" % (pixA[0], pixA[1], pixA[2], len(detA)))
print("  TP %d FP %d FN %d -> P/R/F1 %s" % (tpA, fpA, fnA, prfA))
print("  检出框:", sorted(detA))
print("  真值框:", sorted(gtA))
print("解读: **同一条流水线，在合成图上 P/R/F1 全是 1.0。** 这说明"
      "① 代码没问题、色带选得也没问题；② 阈值分割**本身**是有效的方法；"
      "③ 它失效的条件是「背景复杂 + 颜色不可控」，而不是「方法过时」。"
      "**做 CV 项目的正确姿势就是先造一个可控小场景把它跑通，再上真实数据** —— "
      "否则你分不清是「算法错」还是「数据难」")
assert pixA == (19624, 19592, 19592)
assert len(detA) == 8 and len(gtA) == 8
assert prfA == (1.0, 1.0, 1.0)
assert sorted(detA) == [(53, 53, 108, 108), (53, 218, 108, 273), (113, 373, 168, 428), (188, 53, 243, 108),
                        (188, 218, 243, 273), (283, 373, 338, 428), (323, 53, 378, 108), (323, 218, 378, 273)]
assert sorted(gtA) == [(52, 52, 108, 108), (52, 217, 108, 273), (112, 372, 168, 428), (187, 52, 243, 108),
                       (187, 217, 243, 273), (282, 372, 338, 428), (322, 52, 378, 108), (322, 217, 378, 273)]
'''

E10_CODE = '''imgB, gtB = make_scene(2)
# @@todo 分别用「橙黄」与「橙黄+蓝」两条色带跑，各算一次 P / R / F1
# @@hint 两行：detB1, _, _, _ = hsv_pipeline(imgB, ["橙黄"])；detB2, _, _, _ = hsv_pipeline(imgB, ["橙黄", "蓝"])
# @@hint 两行：prfB1 = prf(*match_counts(detB1, gtB))；prfB2 = prf(*match_counts(detB2, gtB))
detB1, _, _, _ = hsv_pipeline(imgB, ["橙黄"])
detB2, _, _, _ = hsv_pipeline(imgB, ["橙黄", "蓝"])
prfB1 = prf(*match_counts(detB1, gtB))
prfB2 = prf(*match_counts(detB2, gtB))
# @@end

print("合成场景 B（白底 + 6 顶橙 + 2 顶蓝）")
print("  只用「橙黄」: 检出 %d -> P/R/F1 %s（漏 2 顶蓝帽）" % (len(detB1), prfB1))
print("  「橙黄+蓝」  : 检出 %d -> P/R/F1 %s" % (len(detB2), prfB2))
print("解读: ① 单色带漏掉 2 顶蓝帽 -> 召回 0.75；**补上蓝带立刻回到 1.0**；"
      "② 这恰好是现场图做不到的事 —— 现场有**黄/蓝/红三种**帽子，"
      "而且每种帽子的实际色相还会被光照、阴影、相机白平衡拉偏，"
      "你没法用有限几条色带把它们圈全；"
      "③ 更本质的问题：**真实场景里「头盔」这个概念不是颜色，是形状 + 位置 + 上下文**。"
      "用颜色去定义它，一开始就在解一个错的问题")
assert len(detB1) == 6 and prfB1 == (1.0, 0.75, 0.857143)
assert len(detB2) == 8 and prfB2 == (1.0, 1.0, 1.0)
'''

E11_CODE = '''# @@todo 用 SYN_BANDS 的「白」带在场景 A 上做 inRange，算它吃掉的像素数与占比（占比保留 6 位）
# @@hint 两行：maskW = cv2.inRange(cv2.cvtColor(imgA, cv2.COLOR_BGR2HSV), *SYN_BANDS["白"])
# @@hint 一行：white_share = round(int((maskW > 0).sum()) / (S * S), 6)
maskW = cv2.inRange(cv2.cvtColor(imgA, cv2.COLOR_BGR2HSV), *SYN_BANDS["白"])
white_share = round(int((maskW > 0).sum()) / (S * S), 6)
# @@end

# @@todo 用「白」带走一遍完整流水线，看它检出几个框
# @@hint 一行：detW, _, _, _ = hsv_pipeline(imgA, ["白"])
detW, _, _, _ = hsv_pipeline(imgA, ["白"])
# @@end

print("「白」色带的坑（在合成场景 A 上）:")
print("  白带掩码 %d / %d = %.6f（吃掉了 %.2f%% 的画面）" % (int((maskW > 0).sum()), S * S, white_share, 100 * white_share))
print("  白带检出 %d 个框: %s" % (len(detW), detW))
print("解读: **「白色」不是一个可靠的色相判据。** 白/灰/黑的 S 都很低（近乎无彩），"
      "H 通道对它们只是噪声。于是 `H 任意 + S<50 + V>200` 这条「白带」把"
      "**91.02% 的白色背景**全吞了进来，最后吐出一个 `(0, 0, 480, 480)` 的全图框 —— "
      "一个「检出」但毫无意义的框。这就是为什么上一节里「黄+白」的 F1 好看得可疑："
      "**它的功劳全来自误吞背景，不是来自真正的头盔**")
assert int((maskW > 0).sum()) == 209716
assert near(white_share, 0.910226, 1e-6)
assert len(detW) == 1 and detW == [(0, 0, 480, 480)]
'''

E12_CODE = '''# @@todo 把三条口径并排成一张表：现场图最佳色带 / 合成 A / 合成 B(橙黄+蓝)
# @@hint 一字典：summary = {"现场图(红低+红高+黄+蓝)": combo_res[("红低", "红高", "黄", "蓝")][2:],
# @@hint            "合成A(橙黄)": prfA, "合成B(橙黄+蓝)": prfB2}
summary = {"现场图(红低+红高+黄+蓝)": combo_res[("红低", "红高", "黄", "蓝")][2:],
           "合成A(橙黄)": prfA, "合成B(橙黄+蓝)": prfB2}
# @@end

print("=" * 62)
print("final_01 结论表（P / R / F1）")
print("-" * 62)
for k, v in summary.items():
    print(f"  {k:24s} P {v[0]:.6f}  R {v[1]:.6f}  F1 {v[2]:.6f}")
print("-" * 62)
print("  现场图黄带（baseline）          P 0.000000  R 0.000000  F1 0.000000")
print("=" * 62)
print("交付结论：")
print("  1. 阈值分割**只在背景可控、颜色可预测**时成立；白底合成场景 F1 = 1.0 即为证。")
print("  2. 现场图的失败不是调参问题：15 组参数全 F1 = 0，6 种色带最好 0.063492。")
print("  3. 根因是**目标颜色不统一**（色相 7~116，黄 11 / 蓝 7 / 红 3）+ 背景同色干扰。")
print("  4. 换方法：ch07 的 HOG/LBP 特征 + 分类器（把「是不是头盔」交给数据学），")
print("     或 ch09 的候选框 + NMS（把「在哪里」交给检测器）。")
print("  5. 工程上先跑「合成可控场景」验证流水线，再上真实数据 —— 这样才不会把")
print("     「数据难」误判成「代码错」。")
'''
# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# final_01 全流程：安全帽视觉检测

> 这是 03_cv 的**第一个综合题**。它把 ch02/ch03/ch06/ch07/ch09 的东西串成一条流水线：
> **读图 → 坏图体检 → 标注质检 → HSV 阈值分割 → 形态学 → 轮廓计数 → 与 GT 对账 → 报告**。
>
> 但它最重要的产出**不是一个漂亮指标，而是一次诚实的失败复盘**。
> 做完这一章你会明白一件事：**「方法本身对不对」和「在这个数据上能不能用」是两个问题**。

## 本章路线

| 小节 | 内容 | 关键结论 |
|---|---|---|
| §F1.0 | 坏图体检（Laplacian 方差 / 过曝欠曝） | 先证明图能看，再谈指标 |
| §F1.1 | 标注质检（面积 / 宽高比 / 小目标占比） | 小目标占 20.7%，尺度跨 13.7 倍 |
| §F1.2 | 单图 HSV 阈值分割 | 掩码覆盖面积跨 20 倍 |
| §F1.3 | **与 GT 对账** | **检出 64 / TP 0 —— F1 = 0** |
| §F1.4 | **参数扫描** | 15 组 (面积 × 开运算) **全部 F1 = 0** |
| §F1.5 | **GT 色相跨度** | 中位色相 **7~116**，黄 11 / 蓝 7 / 红 3 |
| §F1.6 | 色带融合 | 最好 0.063492，且是巧合 |
| §F1.7 | 最大 IoU 诊断 | 全图最大 IoU < 0.2，远低于命中线 |
| §F1.8 | **合成对照场景 A** | 同一流水线 **P/R/F1 = 1.0** |
| §F1.9 | 合成场景 B（6 橙 + 2 蓝） | 单色带 R 0.75 → 补色带 1.0 |
| §F1.10 | 「白」色带的坑 | 吞掉 91.02% 背景，吐出全图框 |
| §F1.11 | 三口径汇总报告 | 交付结论五条 |

## 关于数据

- **4 张现场图**（`helmet/site_0X.jpg`）：真实工地照片，带 `labels.json` 的 VOC xyxy 标注，
  但**颜色完全不可控**（黄/蓝/红混在一起，还有大量同色背景）。
- **2 个自造合成场景**（`make_scene(n_blue)`）：480×480 白底 + 8 个纯色圆，
  颜色单一可控、GT 精确。**它存在的唯一目的，是证明「代码没错」**。

> 少了合成场景，你拿到现场图 F1 = 0 时无法判断是「算法写错了」还是「数据太难」。
> **这是所有 CV 项目都该有的对照实验设计。**
"""
    ),
    md(
        """
## §F1.0 环境与数据准备
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## §F1.0b 坏图体检

**先别急着算指标。** 如果图本身就是糊的/欠曝的，后面所有数字都会被污染，
而你会误以为是算法不行。
"""
    ),
    code(E1_CODE),
    md(
        """
## §F1.1 标注质检

GT 也不是天然可信的。面积跨度、宽高比、小目标占比 —— 这三项决定了**你的指标下限**。
"""
    ),
    code(E2_CODE),
    md(
        """
## §F1.2 单图阈值分割：黄带路线

按 ch03 的套路走一遍：HSV → `inRange` → 开运算 → 闭运算 → 外轮廓 → 面积过滤。
"""
    ),
    code(E3_CODE),
    md(
        """
## §F1.3 与 GT 对账

**框多不代表对。** 唯一能判对错的方法是和 GT 做 IoU 匹配。
"""
    ),
    code(E4_CODE),
    md(
        """
## §F1.4 参数扫描：是不是阈值没调好？

最自然的怀疑是「面积阈值是不是卡太狠了」。把两个参数扫一遍试试。
"""
    ),
    code(E5_CODE),
    md(
        """
## §F1.5 根因定位：GT 头盔到底是什么颜色？

参数救不回来，那就去看**被测对象本身**。逐个头盔取中位色相 —— 结果会让人清醒。
"""
    ),
    code(E6_CODE),
    md(
        """
## §F1.6 补救尝试：多色带融合

既然现场有黄/蓝/红三种帽子，那就把色带加全。这一节会展示**「补得上」和「补干净」是两回事**。
"""
    ),
    code(E7_CODE),
    md(
        """
## §F1.7 最大 IoU 诊断：一个便宜的探针

不用画图、不用肉眼比对 —— 只要看「所有检出框 × 所有 GT 框」的最大 IoU，
就能判断问题出在分类还是定位。
"""
    ),
    code(E8_CODE),
    md(
        """
## §F1.8 对照实验：白底合成场景

**这是本章最重要的一节。** 同一条流水线，换到一个颜色可控的场景上跑。
"""
    ),
    code(E9_CODE),
    md(
        """
## §F1.9 合成场景 B：6 顶橙 + 2 顶蓝

在可控场景里做一次「单色带 vs 多色带」的受控对比。
"""
    ),
    code(E10_CODE),
    md(
        """
## §F1.10 「白」色带的坑

上一节 §F1.6 里「黄+白」的 F1 好看得可疑。这一节拆开它看看代价。
"""
    ),
    code(E11_CODE),
    md(
        """
## §F1.11 三口径汇总报告

把三条口径并排，给出最终交付结论。
"""
    ),
    code(E12_CODE),
    md(
        """
## 自查清单

- [ ] 知道**动手算指标之前**要先做坏图体检与标注质检
- [ ] 知道 Laplacian 方差 ≈ 清晰度，越小越糊
- [ ] 知道 HSV 的 V 通道受亮度影响，暗图会整体掉一批像素
- [ ] 知道「检出框数」和「检出对不对」是两回事
- [ ] 能写出 TP/FP/FN → P/R/F1 的换算
- [ ] **知道 F1 = 0 时，第一件事是排查「方法假设」而不是「参数」**
- [ ] 知道 15 组参数全 0 说明方向错了，不是没调好
- [ ] 会算 GT 目标的中位色相，用它判断「颜色可不可控」
- [ ] 知道多色带并集**同时放大背景干扰**，TP 和 FP 一起涨
- [ ] **会用「最大 IoU」做低成本诊断探针**
- [ ] **知道 CV 项目必须有一个「合成可控场景」作对照**
- [ ] 知道「白色」色相不可靠（S 低时 H 是噪声）
- [ ] 知道阈值分割的适用边界：背景可控、颜色可预测
- [ ] 知道换方法的两条路：ch07 特征 + 分类器 / ch09 检测框 + NMS

## 交付物

| 文件 | 内容 |
|---|---|
| `final_01_helmet_pipeline.ipynb` | 讲解版（含结论与解读） |
| `final_01_helmet_pipeline_practice.ipynb` | 练习版（挖空 + 提示） |
| `final_01_helmet_pipeline_solution.ipynb` | 答案版（完整可跑） |
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# final_01 全流程：安全帽视觉检测（练习版）

> 按提示补全 `____`，跑通所有 assert。
> `lab` / `NAMES` / `REAL_BANDS` / `SYN_BANDS` / `make_scene` / `gt_helmet`
> 以及 `near` / `iou` / `match_counts` / `prf` / `hsv_pipeline` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；控制流写在挖空块之外
> （循环骨架都已给好，你只填循环体里的几行）。
> 题目真值都来自实跑（opencv 5.0.0 / numpy 2.5.3）。
>
> **提示**：本章结果是「失败」的 —— 别怀疑自己写错了，请读每次 print 后面的解读。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：坏图体检

算每张图的 Laplacian 方差与过曝/欠曝占比。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：标注质检

面积、宽高比、小目标占比，以及类别分布。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：单图阈值分割

黄带流水线走一遍，看掩码覆盖面积与检出框数。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：与 GT 对账

IoU 匹配，算 TP / FP / FN。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：参数扫描

面积阈值 × 开运算次数，共 15 组。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：GT 色相跨度

逐个头盔算中位色相，看颜色到底统不统一。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：色带融合

把黄/蓝/红/白各色带分别加成组合。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：最大 IoU 诊断

一个便宜的探针：问题出在分类还是定位？
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 9：合成对照场景 A

同一条流水线换到白底可控场景。
"""
    ),
    code(E9_CODE),
    md(
        """
## 任务 10：合成场景 B

6 顶橙 + 2 顶蓝，看单色带与多色带的差距。
"""
    ),
    code(E10_CODE),
    md(
        """
## 任务 11：「白」色带的坑

拆穿 §任务 7 里「黄+白」的好看数字。
"""
    ),
    code(E11_CODE),
    md(
        """
## 任务 12：汇总报告

三口径并排 + 交付结论。
"""
    ),
    code(E12_CODE),
    md(
        """
## 自查清单

- [ ] 算指标前先做坏图体检 / 标注质检
- [ ] Laplacian 方差 ≈ 清晰度
- [ ] 「检出框数」≠「检出对不对」
- [ ] F1 = 0 先查方法假设，再查参数
- [ ] 会用中位色相判断「颜色可不可控」
- [ ] 会用最大 IoU 做诊断探针
- [ ] CV 项目要有合成可控场景作对照
- [ ] 阈值分割的边界：背景可控、颜色可预测
- [ ] 换方法：ch07 特征 + 分类器 / ch09 检测框 + NMS
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
