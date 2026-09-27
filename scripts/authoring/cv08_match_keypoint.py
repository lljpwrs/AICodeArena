#!/usr/bin/env python3
"""cv08 模板匹配与关键点：`matchTemplate` 六法 / 扰动鲁棒性 / 多目标贪心抑制 /
多尺度扫描 / Harris / Shi-Tomasi / ORB / BFMatcher + 比率测试 + RANSAC 单应。

数据：`classic/box.png`（模板来源）、`home.jpg`（场景）、`board.jpg`（角点）。
真值全部实跑（opencv 5.0.0）。

> 本章有三条「反直觉」是最值钱的：① `matchTemplate` 的六种方法**极值口径不同**
> （SQDIFF 系列看 min，其余看 max）；② 未归一化的 `CCORR` 会**定位到错误位置**；
> ③ `np.where` 是**行优先**返回的，直接取前 5 个拿到的不是响应最强的 5 个。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv08_match_keypoint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch08_match_keypoint"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

DATA = Path("data")
CL = DATA / "classic"
'''

SETUP = '''box = cv2.imread(str(CL / "box.png"))
box_g = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
home = cv2.imread(str(CL / "home.jpg"))
home_g = cv2.cvtColor(home, cv2.COLOR_BGR2GRAY)
board = cv2.cvtColor(cv2.imread(str(CL / "board.jpg")), cv2.COLOR_BGR2GRAY)

T_Y, T_X, T_S = 120, 200, 48
tmpl = box_g[60:60 + T_S, 80:80 + T_S].copy()
scene = home_g.copy()
scene[T_Y:T_Y + T_S, T_X:T_X + T_S] = tmpl

print("box", box_g.shape, "| home", home_g.shape, "| board", board.shape)
print("模板", tmpl.shape, "灰度范围", int(tmpl.min()), "~", int(tmpl.max()), "std", round(float(tmpl.std()), 4))
print("场景", scene.shape, "| 模板左上角真值 (x, y) =", (T_X, T_Y))
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
METHODS = [("SQDIFF", cv2.TM_SQDIFF), ("SQDIFF_NORMED", cv2.TM_SQDIFF_NORMED),
           ("CCORR", cv2.TM_CCORR), ("CCORR_NORMED", cv2.TM_CCORR_NORMED),
           ("CCOEFF", cv2.TM_CCOEFF), ("CCOEFF_NORMED", cv2.TM_CCOEFF_NORMED)]
MIN_METHODS = ("SQDIFF", "SQDIFF_NORMED")


def best_loc(name, rmap):
    """按方法语义返回「最佳位置」：SQDIFF 系列看最小值，其余看最大值。"""
    mn, mx, mnl, mxl = cv2.minMaxLoc(rmap)
    return mnl if name in MIN_METHODS else mxl


def paste(scene_g, patch, y, x):
    """把 patch 贴到灰度场景的 (y, x) 处（返回新图，不改原图）。"""
    out = scene_g.copy()
    out[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
    return out


def nms_harris(har, th, ksize=3):
    """Harris 局部极大抑制：ksize x ksize 的膨胀图与原响应逐点相等处留下来。"""
    dil = cv2.dilate(har, np.ones((ksize, ksize), np.uint8))
    return (har == dil) & (har > th)


print("脚手架就绪：METHODS / MIN_METHODS / best_loc / paste / nms_harris")'''

# =========================================================================== #
# 练习（答案版内容，practice 由 builder 自动挖空）
# =========================================================================== #

E1_CODE = '''res_all = {}
for name, m in METHODS:
    # @@todo(1) 六种方法各跑一遍 matchTemplate，记下 map 形状 / min / max / min 位置 / max 位置
    # @@hint cv2.matchTemplate(scene, tmpl, m) 得到响应图，再 cv2.minMaxLoc 一次拿四个值
    # @@hint 数值统一 round(float(x), 6)，否则 float32 会带来一堆尾数
    r = cv2.matchTemplate(scene, tmpl, m)
    mn, mx, mnl, mxl = cv2.minMaxLoc(r)
    res_all[name] = (r.shape, round(float(mn), 6), round(float(mx), 6), mnl, mxl)
    # @@end

# @@todo(2) 每个方法挑出「最佳位置」：SQDIFF 系列取最小值处，其余取最大值处
# @@hint 一条字典推导就够；判据就是脚手架里的 name in MIN_METHODS
best = {name: (res_all[name][3] if name in MIN_METHODS else res_all[name][4]) for name, _ in METHODS}
# @@end
locs = tuple(best[name] for name, _ in METHODS)
map_shape = res_all["SQDIFF"][0]
normed_ok = all(res_all[n][2] <= 1.0 for n, _ in METHODS if n.endswith("_NORMED"))
print("响应图形状:", map_shape, "（预期 = 场景 - 模板 + 1 =",
      (scene.shape[0] - tmpl.shape[0] + 1, scene.shape[1] - tmpl.shape[1] + 1), "）")
print(f"{'方法':15s} {'min':>16s} {'@位置':>12s} {'max':>16s} {'@位置':>12s}   最佳位置")
for name, _ in METHODS:
    sh, mn, mx, mnl, mxl = res_all[name]
    print(f"   {name:15s} {mn:>16.6f} {str(mnl):>12s} {mx:>16.6f} {str(mxl):>12s}   {best[name]}")
print("所有 *_NORMED 方法的 max 都 <= 1:", normed_ok)
print("六个方法的「最佳位置」依次是:", locs)
print("解读: 三个要点 —— ① 响应图尺寸 = 场景 - 模板 + 1（这里", map_shape, "），"
      "因为模板必须完整落在场景内；② **极值口径不统一**：SQDIFF 系列是「越小越像」，"
      "其余四种是「越大越像」，写通用代码时必须区分；③ `CCORR` 未归一化时 max 跑到 (244, 284)，"
      "**那是错误位置** —— 它只是找「最亮的一团」，根本不管像不像。"
      "加上 NORM 之后（`*_NORMED`）最大值恒 <= 1，位置也回到了 (200, 120)")
assert map_shape == (337, 465)
assert normed_ok is True
assert locs == ((200, 120), (200, 120), (244, 284), (200, 120), (200, 120), (200, 120))
assert res_all["SQDIFF"][1] == 0.0 and res_all["SQDIFF"][3] == (200, 120)
assert res_all["SQDIFF_NORMED"][2] == 0.940459 and res_all["SQDIFF_NORMED"][4] == (285, 0)
assert res_all["CCORR"][3] == (285, 0) and res_all["CCORR"][4] == (244, 284)
assert res_all["CCORR_NORMED"][2] == 1.0 and res_all["CCORR_NORMED"][4] == (200, 120)
assert res_all["CCOEFF"][2] == 9122235.0 and res_all["CCOEFF"][3] == (239, 239)
assert res_all["CCOEFF_NORMED"][1] == -0.413884 and res_all["CCOEFF_NORMED"][4] == (200, 120)
'''

E2_CODE = '''sc_b = paste(home_g, np.clip(tmpl.astype(np.int16) + 40, 0, 255).astype(np.uint8), T_Y, T_X)
sc_c = paste(home_g, np.clip(tmpl.astype(np.float32) * 0.6 + 60, 0, 255).astype(np.uint8), T_Y, T_X)
robust = {}
for tag, sc in (("原样", scene), ("亮度+40", sc_b), ("对比度*0.6+60", sc_c)):
    row = {}
    for name, m in METHODS:
        # @@todo(3) 跑一次匹配，记下「最佳位置」和该位置上的分数
        # @@hint r = cv2.matchTemplate(sc, tmpl, m)；loc = best_loc(name, r)
        # @@hint 分数取 r[loc[1], loc[0]]（注意索引是 [y, x]），再 round(..., 6)
        r = cv2.matchTemplate(sc, tmpl, m)
        loc = best_loc(name, r)
        row[name] = (loc, round(float(r[loc[1], loc[0]]), 6))
        # @@end
    robust[tag] = row
hit = {tag: tuple(n for n, _ in METHODS if robust[tag][n][0] == (T_X, T_Y)) for tag in robust}
print("场景被改动的方式: 场景 2 = 模板 +40；场景 3 = 模板 × 0.6 + 60（都是线性灰度变换）")
for tag in robust:
    print(f"  {tag:12s}", [(n, robust[tag][n][0]) for n, _ in METHODS])
print("定位正确的：")
for tag in hit:
    print(f"  {tag:12s} 命中 {len(hit[tag])}/6 ->", hit[tag])
print("解读: 结论很干脆 —— ① **未归一化的 CCORR 在两种扰动下都可能跑偏**"
      "（对比度 ×0.6+60 时仍然落在 (244, 284)）；② **归一化方法在任何扰动下都回到 (200, 120)**，"
      "而且分数几乎不变（CCORR_NORMED 1.0 -> 0.997 / 0.989；CCOEFF_NORMED 1.0 -> 0.9999 / 0.9999）；"
      "③ SQDIFF 虽然位置也对，但**分数完全不可比**（0 -> 3674344 / 1461197），"
      "阈值型判断会立刻失效。所以工程里**一律用 `*_NORMED`**；"
      "如果只要快速定位且场景内没有大面积强光，CCOEFF_NORMED 是最稳的选择")
assert robust["原样"]["SQDIFF"] == ((200, 120), 0.0)
assert robust["原样"]["CCORR"] == ((244, 284), 67659192.0)
assert robust["原样"]["CCOEFF_NORMED"] == ((200, 120), 1.0)
assert robust["亮度+40"]["SQDIFF"] == ((200, 120), 3674344.0)
assert robust["亮度+40"]["SQDIFF_NORMED"] == ((200, 120), 0.04872)
assert robust["亮度+40"]["CCORR"] == ((200, 120), 75191872.0)
assert robust["亮度+40"]["CCORR_NORMED"] == ((200, 120), 0.997017)
assert robust["亮度+40"]["CCOEFF_NORMED"] == ((200, 120), 0.999952)
assert robust["对比度*0.6+60"]["SQDIFF"] == ((200, 120), 1461197.0)
assert robust["对比度*0.6+60"]["CCORR"] == ((244, 284), 67659192.0)
assert robust["对比度*0.6+60"]["CCOEFF_NORMED"] == ((200, 120), 0.999971)
assert hit["原样"] == ("SQDIFF", "SQDIFF_NORMED", "CCORR_NORMED", "CCOEFF", "CCOEFF_NORMED")
assert hit["亮度+40"] == ("SQDIFF", "SQDIFF_NORMED", "CCORR", "CCORR_NORMED", "CCOEFF", "CCOEFF_NORMED")
assert hit["对比度*0.6+60"] == ("SQDIFF", "SQDIFF_NORMED", "CCORR_NORMED", "CCOEFF", "CCOEFF_NORMED")
'''

E3_CODE = '''T_S3 = 40
t3 = box_g[40:40 + T_S3, 60:60 + T_S3].copy()
locs3 = [(60, 90), (200, 320), (330, 40)]
sc4 = home_g.copy()
for yy, xx in locs3:
    sc4[yy:yy + T_S3, xx:xx + T_S3] = t3
r4 = cv2.matchTemplate(sc4, t3, cv2.TM_CCOEFF_NORMED)
r4_shape = r4.shape
th_counts = {}
for th in (0.99, 0.95, 0.9, 0.8, 0.7):
    # @@todo(4) 统计每一档阈值下「命中像素数」，看阈值降到多少开始冒出假目标
    # @@hint np.where(r4 >= th) 返回 (行数组, 列数组)，命中数就是 len(行)
    # @@hint 一个真目标会命中**一整片相邻像素**，所以命中数不是目标数
    ys, xs = np.where(r4 >= th)
    th_counts[th] = int(len(ys))
    # @@end
peak_top5 = tuple(round(float(v), 6) for v in np.sort(r4.ravel())[::-1][:5])
print("同一块 40x40 模板被贴到 3 个位置 (y, x):", locs3, "| 响应图", r4_shape)
print("响应值最高的 5 个像素:", peak_top5)
for th in sorted(th_counts, reverse=True):
    print(f"  阈值 {th}: 命中 {th_counts[th]} 个像素")
print("解读: 贴进去的是**同一块像素**，所以三个真位置的响应精确等于 1.0 附近"
      "（前三名 1.0 / 0.999999 / 0.999999）；第 4 名直接掉到 0.714325 —— 峰值非常「尖」。"
      "阈值从 0.99 一路降到 0.8，命中数都是 3，说明**每个真目标只命中 1 个像素**；"
      "再降到 0.7 就冒出 7 个，假目标出现。"
      "工程含义：**阈值要在「真目标峰」和「背景次峰」之间取**（这里 0.8 附近很安全）；"
      "另外不能拿「命中像素数」当目标个数 —— 真实场景里相关性是平滑过渡的，"
      "一个目标会命中一片，必须再做局部极大抑制（下一步）")
assert r4_shape == (345, 473)
assert th_counts == {0.99: 3, 0.95: 3, 0.9: 3, 0.8: 3, 0.7: 7}
assert peak_top5 == (1.0, 0.999999, 0.999999, 0.714325, 0.710718)
'''

E4_CODE = '''found = []
work = r4.copy()
for _ in range(8):
    _, mx, _, mxl = cv2.minMaxLoc(work)
    if mx < 0.9:
        break
    # @@todo(5) 贪心抑制：记下峰值 -> 把该位置的邻域涂成 -1，下一轮自然找下一个峰
    # @@hint found.append((mxl, round(float(mx), 6)))；再取 y0, x0 = mxl[1], mxl[0]
    # @@hint 邻域半径取 T_S3 // 2，涂 -1 才能让 minMaxLoc 不再选到同一片区域
    found.append((mxl, round(float(mx), 6)))
    y0, x0 = mxl[1], mxl[0]
    work[max(0, y0 - T_S3 // 2):y0 + T_S3 // 2, max(0, x0 - T_S3 // 2):x0 + T_S3 // 2] = -1
    # @@end
suppressed = int((work == -1).sum())
print("抑制抑制后找到", len(found), "个目标（按响应从高到低）:")
for loc, score in found:
    print(f"   位置 (x, y) = {loc}  分数 {score}")
print("被涂成 -1 的像素数:", suppressed)
print("解读: 这就是 **NMS（非极大值抑制）的雏形** —— 反复取全局最大、命中后把邻域「屏蔽」，"
      "直到分数低于阈值。三个位置 (x, y) = (40, 330) / (90, 60) / (320, 200) 正好对应"
      "真值 (y, x) = (330, 40) / (60, 90) / (200, 320)，一个不漏、一个不多。"
      "两个必须记住的细节：① **matchTemplate 返回的坐标是 (x, y)**，而 numpy 索引是 [y, x]，"
      "写画框代码时极容易搞反；② 抑制半径不能小于目标尺寸，否则同一个目标会被数两次")
assert found == [((40, 330), 1.0), ((90, 60), 0.999999), ((320, 200), 0.999999)]
assert suppressed == 4600
'''

E5_CODE = '''BIG_Y, BIG_X = 150, 250
big = cv2.resize(t3, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_LINEAR)
sc5 = paste(home_g, big, BIG_Y, BIG_X)
r5 = cv2.matchTemplate(sc5, t3, cv2.TM_CCOEFF_NORMED)
naive_scale = (round(float(r5.max()), 6), cv2.minMaxLoc(r5)[3])
scan = []
for s in [round(0.6 + 0.05 * i, 2) for i in range(22)]:
    # @@todo(6) 多尺度扫描：模板按 s 缩放后再匹配，把 (分数, 缩放, 峰值位置, 模板尺寸) 收进 scan
    # @@hint cv2.resize(t3, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    # @@hint 分数取 CCOEFF_NORMED 响应图的 max（越大越像）
    ts = cv2.resize(t3, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    _, mx, _, mxl = cv2.minMaxLoc(cv2.matchTemplate(sc5, ts, cv2.TM_CCOEFF_NORMED))
    scan.append((round(float(mx), 6), s, mxl, ts.shape))
    # @@end
best_scale = max(scan, key=lambda t: t[0])
scan_top3 = tuple(sorted(scan, reverse=True)[:3])
print("场景里贴的是放大 1.25 倍后的模板，实际尺寸", big.shape)
print("直接用原尺寸模板匹配: 最高分", naive_scale[0], "@ 位置", naive_scale[1])
print("多尺度扫描的最佳 (分数, 缩放, 位置, 模板尺寸):", best_scale)
print("前三名:", scan_top3)
print("解读: **模板匹配不具备尺度不变性**。目标被放大 1.25 倍后，用原尺寸模板去匹配，"
      "最高分只有 0.303955，而且位置 (386, 61) 完全是错的 —— 它只是找到了一个「碰巧局部相似」的地方。"
      "把模板按 0.6~1.65 共 22 档缩放后再匹配，1.25 档拿到满分 1.0，位置 (250, 150) 精确命中。"
      "代价是**耗时线性增长**（22 档就是 22 倍），所以工程上要先用"
      "「金字塔粗扫 + 细扫」或者「关键点匹配」来避免全尺度暴力扫描 —— 这就是本章后半部分"
      "要学 Harris / ORB 的原因")
assert big.shape == (50, 50)
assert naive_scale == (0.303955, (386, 61))
assert best_scale == (1.0, 1.25, (250, 150), (50, 50))
assert scan_top3 == ((1.0, 1.25, (250, 150), (50, 50)), (0.876258, 1.2, (251, 151), (48, 48)),
                     (0.79025, 1.3, (249, 149), (52, 52)))
'''

E6_CODE = '''har = cv2.cornerHarris(np.float32(board), 2, 3, 0.04)
har_map = (har.shape, str(har.dtype), round(float(har.min()), 6), round(float(har.max()), 6))
th_counts_h = {}
for k in (0.01, 0.02, 0.05, 0.1):
    # @@todo(7) 阈值取 k * R.max()，统计响应超过它的像素个数
    # @@hint (har > k * har.max()) 是布尔矩阵，.sum() 就是 True 的个数
    th_counts_h[k] = int((har > k * har.max()).sum())
    # @@end
nms = nms_harris(har, 0.01 * har.max(), 3)
nms_count = int(nms.sum())
# @@todo(8) 按**响应强度**取 top5，再顺手记下「行优先」的前 5 个位置做对照
# @@hint ys, xs = np.where(nms) —— 注意 np.where 是**行优先**返回的！
# @@hint order = np.argsort(har[ys, xs])[::-1]；再按 order[:5] 取 (x, y, R) 三元组
ys, xs = np.where(nms)
order = np.argsort(har[ys, xs])[::-1]
har_top5 = tuple((int(xs[i]), int(ys[i]), round(float(har[ys[i], xs[i]]), 4)) for i in order[:5])
har_rowmajor5 = tuple((int(xs[i]), int(ys[i])) for i in range(5))
# @@end
print("board", board.shape, "| Harris 响应图", har_map, "（shape, dtype, min, max）")
for k in sorted(th_counts_h, reverse=True):
    print(f"    阈值 {k} * R.max: 角点像素 {th_counts_h[k]}")
print("3x3 局部极大抑制后角点数:", nms_count, "（抑制前 27380）")
print("按响应强度 top5 (x, y, R):", har_top5)
print("np.where 行优先的前 5 个 (x, y):", har_rowmajor5, " <- 全在第 0 行")
print("解读: Harris 的响应 R = det(M) - k·trace(M)^2，**可以为负**"
      "（边缘/平坦区为负，角点为正），所以阈值必须写成「k × R.max()」而不是绝对值。"
      "两个关键坑：① 抑制前有 27380 个像素超过 0.01·R.max，"
      "因为它们**连成一片**（一个角点周围一圈都超阈值）；做 3x3 局部极大抑制后剩 5934 个。"
      "② **`np.where` 是行优先返回的**，直接取前 5 个拿到的全是最上面一行（y = 0）的角点，"
      "响应值只有 6e6 量级；真正最强的 5 个都在 y = 185 那条棋盘格线上，响应 3e8 量级 —— "
      "**差了 50 倍**。要取「最强的 K 个」必须显式 `np.argsort` 排序")
assert har_map == ((480, 640), "float32", -82010912.0, 307304896.0)
assert th_counts_h == {0.01: 27380, 0.02: 17667, 0.05: 7441, 0.1: 2802}
assert nms_count == 5934
assert har_top5 == ((314, 185, 307304896.0), (327, 185, 305193408.0), (350, 185, 287178688.0),
                    (340, 185, 279111360.0), (337, 185, 277100384.0))
assert har_rowmajor5 == ((269, 0), (280, 0), (292, 0), (303, 0), (337, 0))
'''

E7_CODE = '''har_cfg = [("blockSize", 2), ("blockSize", 3), ("blockSize", 5),
           ("ksize", 3), ("ksize", 5), ("ksize", 7),
           ("k", 0.02), ("k", 0.04), ("k", 0.06), ("k", 0.1)]
har_sweep = {}
for key, val in har_cfg:
    # @@todo(9) 每次只改一个参数（其余固定 blockSize=2 / ksize=3 / k=0.04）
    # @@hint kw = dict(blockSize=2, ksize=3, k=0.04)；kw[key] = val；再 cv2.cornerHarris(np.float32(board), **kw)
    # @@hint 存 (round(R.max(), 6), round(R.min(), 6), int((R > 0.01 * R.max()).sum())) 三元组
    kw = dict(blockSize=2, ksize=3, k=0.04)
    kw[key] = val
    h2 = cv2.cornerHarris(np.float32(board), **kw)
    har_sweep[(key, val)] = (round(float(h2.max()), 6), round(float(h2.min()), 6),
                             int((h2 > 0.01 * h2.max()).sum()))
    # @@end
print(f"{'参数':22s} {'R.max':>18s} {'R.min':>18s} {'角点数':>8s}")
for key, val in har_cfg:
    print(f"   {key + '=' + str(val):19s} {har_sweep[(key, val)][0]:>18.1f} "
          f"{har_sweep[(key, val)][1]:>18.1f} {har_sweep[(key, val)][2]:>8d}")
print("解读: 三个参数的作用完全不同 —— ① **blockSize**（Sobel 算 M 的窗口）越大，"
      "响应被平均得越厉害（R.max 从 3.07e8 掉到 1.23e8），但**角点反而变多**"
      "（27380 -> 102027），因为大窗口让更多像素「看起来像角点」；"
      "② **ksize**（Sobel 孔径）越大，微分越锐利，R 的量级**暴涨**（ksize 3 -> 7 涨了 1800 倍："
      "3.07e8 -> 5.55e11），所以**换 ksize 必须重新定阈值**；"
      "③ **k** 越大，`det(M) - k·trace^2` 里负项权重越大，R.min 越来越负"
      "（-4.06e7 -> -2.06e8），过阈值的角点越来越少（32612 -> 16302）。"
      "工程默认值 `blockSize=2, ksize=3, k=0.04` 是几十年调出来的平衡点")
assert har_sweep[("blockSize", 2)] == (307304896.0, -82010912.0, 27380)
assert har_sweep[("blockSize", 5)] == (122746256.0, -26624974.0, 102027)
assert har_sweep[("ksize", 3)] == (307304896.0, -82010912.0, 27380)
assert har_sweep[("ksize", 7)] == (555186520064.0, -766063738880.0, 26756)
assert har_sweep[("k", 0.02)] == (339269568.0, -40581280.0, 32612)
assert har_sweep[("k", 0.1)] == (217774784.0, -206299792.0, 16302)
'''

E8_CODE = '''gf_cfg = [("qualityLevel", 0.001), ("qualityLevel", 0.01), ("qualityLevel", 0.05),
          ("minDistance", 5), ("minDistance", 10), ("minDistance", 20), ("minDistance", 40),
          ("maxCorners", 10), ("maxCorners", 50), ("maxCorners", 100), ("maxCorners", 500)]
gf_counts = {}
for key, val in gf_cfg:
    # @@todo(10) 每次只改一个参数（其余固定 maxCorners=200 / qualityLevel=0.01 / minDistance=10）
    # @@hint kw = dict(maxCorners=200, qualityLevel=0.01, minDistance=10)；kw[key] = val
    # @@hint goodFeaturesToTrack 可能返回 None（一个都没找到），要兼容
    kw = dict(maxCorners=200, qualityLevel=0.01, minDistance=10)
    kw[key] = val
    p = cv2.goodFeaturesToTrack(board, **kw)
    gf_counts[(key, val)] = 0 if p is None else int(len(p))
    # @@end
pts = cv2.goodFeaturesToTrack(board, maxCorners=200, qualityLevel=0.01, minDistance=10)
gf_top5 = tuple(sorted((round(float(p[0][0]), 1), round(float(p[0][1]), 1)) for p in pts)[:5])
gf_vs_harris = (nms_count, int(len(pts)))
print(f"{'参数':22s} {'角点数':>8s}")
for key, val in gf_cfg:
    print(f"   {key + '=' + str(val):19s} {gf_counts[(key, val)]:>8d}")
print("qualityLevel=0.01 / minDistance=10 / maxCorners=200 时: 角点", len(pts), "个")
print("  前 5 个 (x, y):", gf_top5)
print("与手写 Harris + NMS 对比: Harris+NMS =", gf_vs_harris[0], " vs goodFeaturesToTrack =", gf_vs_harris[1])
print("解读: **Shi-Tomasi 是 Harris 的改良版** —— 不用 `det - k·trace^2`，"
      "而是直接取 M 的**较小特征值** min(λ1, λ2) 当响应（少一个要调的 k）。"
      "三个参数里，`maxCorners` 是硬上限（10/50/100/500 完全线性），"
      "`minDistance` 管「角点之间至少隔多远」（40 时才掉到 130），"
      "而 `qualityLevel` 在 0.001~0.05 之间**完全没起作用**（都是 200）—— "
      "因为候选角点数远超 maxCorners，上限先到了。"
      "**注意 5934 vs 200 差 30 倍**：`goodFeaturesToTrack` 内部还做了"
      "「按响应排序 + minDistance 去重」，比朴素的「阈值 + 3x3 抑制」严格得多。"
      "做「角点数量」类指标时，**必须写清楚用的是哪一套口径和参数**，否则数字没有可比性")
assert gf_counts[("qualityLevel", 0.001)] == 200
assert gf_counts[("qualityLevel", 0.01)] == 200
assert gf_counts[("qualityLevel", 0.05)] == 200
assert gf_counts[("minDistance", 5)] == 200
assert gf_counts[("minDistance", 40)] == 130
assert gf_counts[("maxCorners", 10)] == 10
assert gf_counts[("maxCorners", 500)] == 500
assert gf_top5 == ((6.0, 153.0), (9.0, 329.0), (13.0, 270.0), (21.0, 440.0), (21.0, 454.0))
assert gf_vs_harris == (5934, 200)
'''

E9_CODE = '''orb = cv2.ORB_create(nfeatures=500)
kp, des = orb.detectAndCompute(board, None)
# @@todo(11) 从关键点里提取：描述子 shape/dtype、响应 top5、尺度种类前 8、角度种类前 8、层(octave)
# @@hint kp 里每个点有 .pt / .size / .angle / .response / .octave 属性
# @@hint 排序统一 sorted(..., reverse=True)；去重排序后取前 8 个用 [:8]
orb_stat = (len(kp), des.shape, str(des.dtype))
orb_resp_top5 = tuple(sorted((round(float(k.response), 4) for k in kp), reverse=True)[:5])
orb_sizes = tuple(sorted(set(round(k.size, 2) for k in kp))[:8])
orb_angles = tuple(sorted(set(round(float(k.angle), 2) for k in kp))[:8])
orb_oct = tuple(sorted(set(int(k.octave) for k in kp)))
# @@end
orb_cfg = [("nfeatures", 100), ("nfeatures", 500), ("nfeatures", 1000), ("nfeatures", 2000),
           ("fastThreshold", 5), ("fastThreshold", 10), ("fastThreshold", 20), ("fastThreshold", 40)]
orb_sweep = {}
for key, val in orb_cfg:
    # @@todo(12) nfeatures 是硬上限、fastThreshold 是候选门槛：每次只改一个
    # @@hint 基础配置给 nfeatures=5000 / fastThreshold=20（这样 fastThreshold 才看得出差别）
    # @@hint 只要关键点个数就行，用 ORB_create(**kw).detect(board, None) 再 len(...)
    kw2 = dict(nfeatures=5000, fastThreshold=20)
    kw2[key] = val
    orb_sweep[(key, val)] = int(len(cv2.ORB_create(**kw2).detect(board, None)))
    # @@end
print("ORB(500) 在 board 上: 关键点", orb_stat[0], "个 | 描述子", orb_stat[1], orb_stat[2],
      "=", orb_stat[1][1] * 8, "bit")
print("响应 top5:", orb_resp_top5, "| 尺度种类前 8:", orb_sizes)
print("角度种类前 8:", orb_angles, "| 所在层:", orb_oct)
for key, val in orb_cfg:
    print(f"   {key + '=' + str(val):20s} -> {orb_sweep[(key, val)]} 个关键点")
print("解读: ORB = **oFAST（带方向的 FAST 角点）+ rBRIEF（旋转后的二进制描述子）**，"
      "描述子是 32 字节 = **256 bit 的 uint8**，所以匹配要用 `NORM_HAMMING`（汉明距离）"
      "而不是欧氏距离。看几个属性：尺度是 **1.2 的等比序列**（31.0, 37.2, 44.64, 53.57…，"
      "每档 ×1.2），这就是**尺度金字塔**；8 个不同 octave 说明关键点分布在 8 层；"
      "角度是 0~360° 的 ICM 主轴方向，这就是**旋转不变性**的来源。"
      "两个参数：`nfeatures` 是**硬上限**（100/500/1000/2000 完全线性）；"
      "`fastThreshold` 是 FAST 的亮度差门槛，只有在 `nfeatures` **大于**候选数时才起作用 —— "
      "基础配 nfeatures=5000 时，阈值提到 40 才让候选从 5000 掉到 4654，"
      "而 nfeatures=500 时不管阈值取多少都是 500（上限先到）")
assert orb_stat == (500, (500, 32), "uint8")
assert orb_resp_top5 == (0.0112, 0.0111, 0.011, 0.0097, 0.0096)
assert orb_sizes == (31.0, 37.2, 44.64, 53.57, 64.28, 77.14, 92.57, 111.08)
assert orb_angles == (3.94, 5.59, 5.81, 6.3, 6.33, 6.89, 7.4, 7.83)
assert orb_oct == (0, 1, 2, 3, 4, 5, 6, 7)
assert orb_sweep[("nfeatures", 100)] == 100 and orb_sweep[("nfeatures", 2000)] == 2000
assert orb_sweep[("fastThreshold", 20)] == 5000 and orb_sweep[("fastThreshold", 40)] == 4654
'''

E10_CODE = '''M = cv2.getRotationMatrix2D((box_g.shape[1] / 2, box_g.shape[0] / 2), 25, 1.0)
rot = cv2.warpAffine(box_g, M, (box_g.shape[1], box_g.shape[0]))
kp1, d1 = cv2.ORB_create(nfeatures=500).detectAndCompute(box_g, None)
kp2, d2 = cv2.ORB_create(nfeatures=500).detectAndCompute(rot, None)
bf_cc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
cc_stat = (len(bf_cc), int(min(m.distance for m in bf_cc)), int(max(m.distance for m in bf_cc)))
knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
knn_len = len(knn)
ratio_pass = {}
for r in (0.6, 0.7, 0.75, 0.8, 0.9):
    # @@todo(13) Lowe 比率测试：最近距离 < r × 次近距离 才算「好匹配」
    # @@hint knn 里每个元素是 [最近, 次近] 两个 DMatch；用生成器求和即可
    # @@hint r 越大越宽松，通过数应该单调不降
    ratio_pass[r] = int(sum(1 for m, n in knn if m.distance < r * n.distance))
    # @@end
good = [m for m, n in knn if m.distance < 0.75 * n.distance]
src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
Mrot = np.vstack([M, [0, 0, 1]])
hom_stat = (int(inliers.sum()), len(good), round(float(np.linalg.norm(H - Mrot)), 6))
hom_angle = round(float(np.degrees(np.arctan2(H[1, 0], H[0, 0]))), 4)
print("A: ", len(kp1), d1.shape, "| B: ", len(kp2), d2.shape, "（B = A 逆时针旋转 25°）")
print("crossCheck=True 匹配数:", cc_stat[0], "| 距离范围", cc_stat[1], "~", cc_stat[2])
print("knnMatch(k=2) 对数:", knn_len)
for r in sorted(ratio_pass):
    print(f"    比率 {r}: 通过 {ratio_pass[r]} 对")
print("比率 0.75 的匹配做 RANSAC 单应: 内点", hom_stat[0], "/", hom_stat[1],
      "| 与真值矩阵的 Frobenius 差", hom_stat[2])
print("从 H 反推旋转角:", hom_angle, "°（施加的是 +25°）")
print("估计 H =\\n", np.round(H, 4))
print("真值 M =\\n", np.round(Mrot, 4))
print("解读: 两条匹配策略 —— `crossCheck=True`（互为最近邻）给出 285 对，"
      "胜在**几乎不需要调参**但漏召回多；`knnMatch` + **Lowe 比率测试**（最近距离 < r × 次近）"
      "是从「这个匹配有多独特」的角度筛选，r 越大越宽松（0.6 -> 193 对，0.9 -> 322 对），"
      "**常用 0.75**。注意：`BFMatcher(crossCheck=True)` **不能再调 knnMatch**"
      "（OpenCV 会直接断言失败），两者必须各建一个匹配器。"
      "最后用 244 对做 `findHomography(..., RANSAC, 3.0)`，236 个内点、8 个外点被剔掉，"
      "恢复的 H 与真值矩阵 Frobenius 差只有 1.009325（差主要来自平移项 tx：-32.867 vs -31.944）。"
      "**从 H 反推的角度是 -24.09° 而不是 +25°** —— 因为图像坐标 y 轴向下，"
      "逆时针旋转在数学意义上是**负角**，这个符号坑在做机器人/云台标定时一定会遇到")
assert cc_stat == (285, 10, 78)
assert knn_len == 453
assert ratio_pass == {0.6: 193, 0.7: 222, 0.75: 244, 0.8: 266, 0.9: 322}
assert hom_stat == (236, 244, 1.009325)
assert hom_angle == -24.0895
'''

# =========================================================================== #
# 讲解用代码块（与练习内容一一对应，去掉挖空标记）
# =========================================================================== #

S1_CODE = '''res_all = {}
for name, m in METHODS:
    r = cv2.matchTemplate(scene, tmpl, m)
    mn, mx, mnl, mxl = cv2.minMaxLoc(r)
    res_all[name] = (r.shape, round(float(mn), 6), round(float(mx), 6), mnl, mxl)
best = {name: (res_all[name][3] if name in MIN_METHODS else res_all[name][4]) for name, _ in METHODS}
locs = tuple(best[name] for name, _ in METHODS)
map_shape = res_all["SQDIFF"][0]
normed_ok = all(res_all[n][2] <= 1.0 for n, _ in METHODS if n.endswith("_NORMED"))
print("响应图形状:", map_shape, "（预期 = 场景 - 模板 + 1 =",
      (scene.shape[0] - tmpl.shape[0] + 1, scene.shape[1] - tmpl.shape[1] + 1), "）")
print(f"{'方法':15s} {'min':>16s} {'@位置':>12s} {'max':>16s} {'@位置':>12s}   最佳位置")
for name, _ in METHODS:
    sh, mn, mx, mnl, mxl = res_all[name]
    print(f"   {name:15s} {mn:>16.6f} {str(mnl):>12s} {mx:>16.6f} {str(mxl):>12s}   {best[name]}")
print("所有 *_NORMED 方法的 max 都 <= 1:", normed_ok)
print("六个方法的「最佳位置」依次是:", locs)
print("解读: 三个要点 —— ① 响应图尺寸 = 场景 - 模板 + 1（这里", map_shape, "），"
      "因为模板必须完整落在场景内；② **极值口径不统一**：SQDIFF 系列是「越小越像」，"
      "其余四种是「越大越像」，写通用代码时必须区分；③ `CCORR` 未归一化时 max 跑到 (244, 284)，"
      "**那是错误位置** —— 它只是找「最亮的一团」，根本不管像不像。"
      "加上 NORM 之后（`*_NORMED`）最大值恒 <= 1，位置也回到了 (200, 120)")
'''

S2_CODE = '''sc_b = paste(home_g, np.clip(tmpl.astype(np.int16) + 40, 0, 255).astype(np.uint8), T_Y, T_X)
sc_c = paste(home_g, np.clip(tmpl.astype(np.float32) * 0.6 + 60, 0, 255).astype(np.uint8), T_Y, T_X)
robust = {}
for tag, sc in (("原样", scene), ("亮度+40", sc_b), ("对比度*0.6+60", sc_c)):
    row = {}
    for name, m in METHODS:
        r = cv2.matchTemplate(sc, tmpl, m)
        loc = best_loc(name, r)
        row[name] = (loc, round(float(r[loc[1], loc[0]]), 6))
    robust[tag] = row
hit = {tag: tuple(n for n, _ in METHODS if robust[tag][n][0] == (T_X, T_Y)) for tag in robust}
print("场景被改动的方式: 场景 2 = 模板 +40；场景 3 = 模板 × 0.6 + 60（都是线性灰度变换）")
for tag in robust:
    print(f"  {tag:12s}", [(n, robust[tag][n][0]) for n, _ in METHODS])
print("定位正确的：")
for tag in hit:
    print(f"  {tag:12s} 命中 {len(hit[tag])}/6 ->", hit[tag])
print("解读: 结论很干脆 —— ① **未归一化的 CCORR 在两种扰动下都可能跑偏**"
      "（对比度 ×0.6+60 时仍然落在 (244, 284)）；② **归一化方法在任何扰动下都回到 (200, 120)**，"
      "而且分数几乎不变（CCORR_NORMED 1.0 -> 0.997 / 0.989；CCOEFF_NORMED 1.0 -> 0.9999 / 0.9999）；"
      "③ SQDIFF 虽然位置也对，但**分数完全不可比**（0 -> 3674344 / 1461197），"
      "阈值型判断会立刻失效。所以工程里**一律用 `*_NORMED`**；"
      "如果只要快速定位且场景内没有大面积强光，CCOEFF_NORMED 是最稳的选择")

assert robust["原样"]["SQDIFF"] == ((200, 120), 0.0)
assert robust["原样"]["CCORR"] == ((244, 284), 67659192.0)
assert robust["原样"]["CCOEFF_NORMED"] == ((200, 120), 1.0)
assert robust["亮度+40"]["CCORR"] == ((200, 120), 75191872.0)
assert robust["亮度+40"]["CCORR_NORMED"] == ((200, 120), 0.997017)
assert robust["对比度*0.6+60"]["CCORR"] == ((244, 284), 67659192.0)
assert robust["对比度*0.6+60"]["CCOEFF_NORMED"] == ((200, 120), 0.999971)
assert hit["原样"] == ("SQDIFF", "SQDIFF_NORMED", "CCORR_NORMED", "CCOEFF", "CCOEFF_NORMED")
assert hit["亮度+40"] == ("SQDIFF", "SQDIFF_NORMED", "CCORR", "CCORR_NORMED", "CCOEFF", "CCOEFF_NORMED")
'''

S3_CODE = '''T_S3 = 40
t3 = box_g[40:40 + T_S3, 60:60 + T_S3].copy()
locs3 = [(60, 90), (200, 320), (330, 40)]
sc4 = home_g.copy()
for yy, xx in locs3:
    sc4[yy:yy + T_S3, xx:xx + T_S3] = t3
r4 = cv2.matchTemplate(sc4, t3, cv2.TM_CCOEFF_NORMED)
r4_shape = r4.shape
th_counts = {}
for th in (0.99, 0.95, 0.9, 0.8, 0.7):
    ys, xs = np.where(r4 >= th)
    th_counts[th] = int(len(ys))
peak_top5 = tuple(round(float(v), 6) for v in np.sort(r4.ravel())[::-1][:5])
print("同一块 40x40 模板被贴到 3 个位置 (y, x):", locs3, "| 响应图", r4_shape)
print("响应值最高的 5 个像素:", peak_top5)
for th in sorted(th_counts, reverse=True):
    print(f"  阈值 {th}: 命中 {th_counts[th]} 个像素")
print("解读: 贴进去的是**同一块像素**，所以三个真位置的响应精确等于 1.0 附近"
      "（前三名 1.0 / 0.999999 / 0.999999）；第 4 名直接掉到 0.714325 —— 峰值非常「尖」。"
      "阈值从 0.99 一路降到 0.8，命中数都是 3，说明**每个真目标只命中 1 个像素**；"
      "再降到 0.7 就冒出 7 个，假目标出现。"
      "工程含义：**阈值要在「真目标峰」和「背景次峰」之间取**（这里 0.8 附近很安全）；"
      "另外不能拿「命中像素数」当目标个数 —— 真实场景里相关性是平滑过渡的，"
      "一个目标会命中一片，必须再做局部极大抑制（下一步）")

assert r4_shape == (345, 473)
assert th_counts == {0.99: 3, 0.95: 3, 0.9: 3, 0.8: 3, 0.7: 7}
assert peak_top5 == (1.0, 0.999999, 0.999999, 0.714325, 0.710718)
'''

S4_CODE = '''found = []
work = r4.copy()
for _ in range(8):
    _, mx, _, mxl = cv2.minMaxLoc(work)
    if mx < 0.9:
        break
    found.append((mxl, round(float(mx), 6)))
    y0, x0 = mxl[1], mxl[0]
    work[max(0, y0 - T_S3 // 2):y0 + T_S3 // 2, max(0, x0 - T_S3 // 2):x0 + T_S3 // 2] = -1
suppressed = int((work == -1).sum())
print("抑制抑制后找到", len(found), "个目标（按响应从高到低）:")
for loc, score in found:
    print(f"   位置 (x, y) = {loc}  分数 {score}")
print("被涂成 -1 的像素数:", suppressed)
print("解读: 这就是 **NMS（非极大值抑制）的雏形** —— 反复取全局最大、命中后把邻域「屏蔽」，"
      "直到分数低于阈值。三个位置 (x, y) = (40, 330) / (90, 60) / (320, 200) 正好对应"
      "真值 (y, x) = (330, 40) / (60, 90) / (200, 320)，一个不漏、一个不多。"
      "两个必须记住的细节：① **matchTemplate 返回的坐标是 (x, y)**，而 numpy 索引是 [y, x]，"
      "写画框代码时极容易搞反；② 抑制半径不能小于目标尺寸，否则同一个目标会被数两次")

assert found == [((40, 330), 1.0), ((90, 60), 0.999999), ((320, 200), 0.999999)]
assert suppressed == 4600
'''

S5_CODE = '''BIG_Y, BIG_X = 150, 250
big = cv2.resize(t3, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_LINEAR)
sc5 = paste(home_g, big, BIG_Y, BIG_X)
r5 = cv2.matchTemplate(sc5, t3, cv2.TM_CCOEFF_NORMED)
naive_scale = (round(float(r5.max()), 6), cv2.minMaxLoc(r5)[3])
scan = []
for s in [round(0.6 + 0.05 * i, 2) for i in range(22)]:
    ts = cv2.resize(t3, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    _, mx, _, mxl = cv2.minMaxLoc(cv2.matchTemplate(sc5, ts, cv2.TM_CCOEFF_NORMED))
    scan.append((round(float(mx), 6), s, mxl, ts.shape))
best_scale = max(scan, key=lambda t: t[0])
scan_top3 = tuple(sorted(scan, reverse=True)[:3])
print("场景里贴的是放大 1.25 倍后的模板，实际尺寸", big.shape)
print("直接用原尺寸模板匹配: 最高分", naive_scale[0], "@ 位置", naive_scale[1])
print("多尺度扫描的最佳 (分数, 缩放, 位置, 模板尺寸):", best_scale)
print("前三名:", scan_top3)
print("解读: **模板匹配不具备尺度不变性**。目标被放大 1.25 倍后，用原尺寸模板去匹配，"
      "最高分只有 0.303955，而且位置 (386, 61) 完全是错的 —— 它只是找到了一个「碰巧局部相似」的地方。"
      "把模板按 0.6~1.65 共 22 档缩放后再匹配，1.25 档拿到满分 1.0，位置 (250, 150) 精确命中。"
      "代价是**耗时线性增长**（22 档就是 22 倍），所以工程上要先用"
      "「金字塔粗扫 + 细扫」或者「关键点匹配」来避免全尺度暴力扫描 —— 这就是本章后半部分"
      "要学 Harris / ORB 的原因")

assert big.shape == (50, 50)
assert naive_scale == (0.303955, (386, 61))
assert best_scale == (1.0, 1.25, (250, 150), (50, 50))
assert scan_top3 == ((1.0, 1.25, (250, 150), (50, 50)), (0.876258, 1.2, (251, 151), (48, 48)),
                     (0.79025, 1.3, (249, 149), (52, 52)))
'''

S6_CODE = '''har = cv2.cornerHarris(np.float32(board), 2, 3, 0.04)
har_map = (har.shape, str(har.dtype), round(float(har.min()), 6), round(float(har.max()), 6))
th_counts_h = {}
for k in (0.01, 0.02, 0.05, 0.1):
    th_counts_h[k] = int((har > k * har.max()).sum())
nms = nms_harris(har, 0.01 * har.max(), 3)
nms_count = int(nms.sum())
ys, xs = np.where(nms)
order = np.argsort(har[ys, xs])[::-1]
har_top5 = tuple((int(xs[i]), int(ys[i]), round(float(har[ys[i], xs[i]]), 4)) for i in order[:5])
har_rowmajor5 = tuple((int(xs[i]), int(ys[i])) for i in range(5))
print("board", board.shape, "| Harris 响应图", har_map, "（shape, dtype, min, max）")
for k in sorted(th_counts_h, reverse=True):
    print(f"    阈值 {k} * R.max: 角点像素 {th_counts_h[k]}")
print("3x3 局部极大抑制后角点数:", nms_count, "（抑制前 27380）")
print("按响应强度 top5 (x, y, R):", har_top5)
print("np.where 行优先的前 5 个 (x, y):", har_rowmajor5, " <- 全在第 0 行")
print("解读: Harris 的响应 R = det(M) - k·trace(M)^2，**可以为负**"
      "（边缘/平坦区为负，角点为正），所以阈值必须写成「k × R.max()」而不是绝对值。"
      "两个关键坑：① 抑制前有 27380 个像素超过 0.01·R.max，"
      "因为它们**连成一片**（一个角点周围一圈都超阈值）；做 3x3 局部极大抑制后剩 5934 个。"
      "② **`np.where` 是行优先返回的**，直接取前 5 个拿到的全是最上面一行（y = 0）的角点，"
      "响应值只有 6e6 量级；真正最强的 5 个都在 y = 185 那条棋盘格线上，响应 3e8 量级 —— "
      "**差了 50 倍**。要取「最强的 K 个」必须显式 `np.argsort` 排序")

assert har_map == ((480, 640), "float32", -82010912.0, 307304896.0)
assert th_counts_h == {0.01: 27380, 0.02: 17667, 0.05: 7441, 0.1: 2802}
assert nms_count == 5934
assert har_top5 == ((314, 185, 307304896.0), (327, 185, 305193408.0), (350, 185, 287178688.0),
                    (340, 185, 279111360.0), (337, 185, 277100384.0))
assert har_rowmajor5 == ((269, 0), (280, 0), (292, 0), (303, 0), (337, 0))
'''

S7_CODE = '''har_cfg = [("blockSize", 2), ("blockSize", 3), ("blockSize", 5),
           ("ksize", 3), ("ksize", 5), ("ksize", 7),
           ("k", 0.02), ("k", 0.04), ("k", 0.06), ("k", 0.1)]
har_sweep = {}
for key, val in har_cfg:
    kw = dict(blockSize=2, ksize=3, k=0.04)
    kw[key] = val
    h2 = cv2.cornerHarris(np.float32(board), **kw)
    har_sweep[(key, val)] = (round(float(h2.max()), 6), round(float(h2.min()), 6),
                             int((h2 > 0.01 * h2.max()).sum()))
print(f"{'参数':22s} {'R.max':>18s} {'R.min':>18s} {'角点数':>8s}")
for key, val in har_cfg:
    print(f"   {key + '=' + str(val):19s} {har_sweep[(key, val)][0]:>18.1f} "
          f"{har_sweep[(key, val)][1]:>18.1f} {har_sweep[(key, val)][2]:>8d}")
print("解读: 三个参数的作用完全不同 —— ① **blockSize**（Sobel 算 M 的窗口）越大，"
      "响应被平均得越厉害（R.max 从 3.07e8 掉到 1.23e8），但**角点反而变多**"
      "（27380 -> 102027），因为大窗口让更多像素「看起来像角点」；"
      "② **ksize**（Sobel 孔径）越大，微分越锐利，R 的量级**暴涨**（ksize 3 -> 7 涨了 1800 倍："
      "3.07e8 -> 5.55e11），所以**换 ksize 必须重新定阈值**；"
      "③ **k** 越大，`det(M) - k·trace^2` 里负项权重越大，R.min 越来越负"
      "（-4.06e7 -> -2.06e8），过阈值的角点越来越少（32612 -> 16302）。"
      "工程默认值 `blockSize=2, ksize=3, k=0.04` 是几十年调出来的平衡点")

assert har_sweep[("blockSize", 2)] == (307304896.0, -82010912.0, 27380)
assert har_sweep[("blockSize", 5)] == (122746256.0, -26624974.0, 102027)
assert har_sweep[("ksize", 7)] == (555186520064.0, -766063738880.0, 26756)
assert har_sweep[("k", 0.1)] == (217774784.0, -206299792.0, 16302)
'''

S8_CODE = '''gf_cfg = [("qualityLevel", 0.001), ("qualityLevel", 0.01), ("qualityLevel", 0.05),
          ("minDistance", 5), ("minDistance", 10), ("minDistance", 20), ("minDistance", 40),
          ("maxCorners", 10), ("maxCorners", 50), ("maxCorners", 100), ("maxCorners", 500)]
gf_counts = {}
for key, val in gf_cfg:
    kw = dict(maxCorners=200, qualityLevel=0.01, minDistance=10)
    kw[key] = val
    p = cv2.goodFeaturesToTrack(board, **kw)
    gf_counts[(key, val)] = 0 if p is None else int(len(p))
pts = cv2.goodFeaturesToTrack(board, maxCorners=200, qualityLevel=0.01, minDistance=10)
gf_top5 = tuple(sorted((round(float(p[0][0]), 1), round(float(p[0][1]), 1)) for p in pts)[:5])
gf_vs_harris = (nms_count, int(len(pts)))
print(f"{'参数':22s} {'角点数':>8s}")
for key, val in gf_cfg:
    print(f"   {key + '=' + str(val):19s} {gf_counts[(key, val)]:>8d}")
print("qualityLevel=0.01 / minDistance=10 / maxCorners=200 时: 角点", len(pts), "个")
print("  前 5 个 (x, y):", gf_top5)
print("与手写 Harris + NMS 对比: Harris+NMS =", gf_vs_harris[0], " vs goodFeaturesToTrack =", gf_vs_harris[1])
print("解读: **Shi-Tomasi 是 Harris 的改良版** —— 不用 `det - k·trace^2`，"
      "而是直接取 M 的**较小特征值** min(λ1, λ2) 当响应（少一个要调的 k）。"
      "三个参数里，`maxCorners` 是硬上限（10/50/100/500 完全线性），"
      "`minDistance` 管「角点之间至少隔多远」（40 时才掉到 130），"
      "而 `qualityLevel` 在 0.001~0.05 之间**完全没起作用**（都是 200）—— "
      "因为候选角点数远超 maxCorners，上限先到了。"
      "**注意 5934 vs 200 差 30 倍**：`goodFeaturesToTrack` 内部还做了"
      "「按响应排序 + minDistance 去重」，比朴素的「阈值 + 3x3 抑制」严格得多。"
      "做「角点数量」类指标时，**必须写清楚用的是哪一套口径和参数**，否则数字没有可比性")

assert gf_counts[("qualityLevel", 0.05)] == 200
assert gf_counts[("minDistance", 40)] == 130
assert gf_counts[("maxCorners", 500)] == 500
assert gf_top5 == ((6.0, 153.0), (9.0, 329.0), (13.0, 270.0), (21.0, 440.0), (21.0, 454.0))
assert gf_vs_harris == (5934, 200)
'''

S9_CODE = '''orb = cv2.ORB_create(nfeatures=500)
kp, des = orb.detectAndCompute(board, None)
orb_stat = (len(kp), des.shape, str(des.dtype))
orb_resp_top5 = tuple(sorted((round(float(k.response), 4) for k in kp), reverse=True)[:5])
orb_sizes = tuple(sorted(set(round(k.size, 2) for k in kp))[:8])
orb_angles = tuple(sorted(set(round(float(k.angle), 2) for k in kp))[:8])
orb_oct = tuple(sorted(set(int(k.octave) for k in kp)))
orb_cfg = [("nfeatures", 100), ("nfeatures", 500), ("nfeatures", 1000), ("nfeatures", 2000),
           ("fastThreshold", 5), ("fastThreshold", 10), ("fastThreshold", 20), ("fastThreshold", 40)]
orb_sweep = {}
for key, val in orb_cfg:
    kw2 = dict(nfeatures=5000, fastThreshold=20)
    kw2[key] = val
    orb_sweep[(key, val)] = int(len(cv2.ORB_create(**kw2).detect(board, None)))
print("ORB(500) 在 board 上: 关键点", orb_stat[0], "个 | 描述子", orb_stat[1], orb_stat[2],
      "=", orb_stat[1][1] * 8, "bit")
print("响应 top5:", orb_resp_top5, "| 尺度种类前 8:", orb_sizes)
print("角度种类前 8:", orb_angles, "| 所在层:", orb_oct)
for key, val in orb_cfg:
    print(f"   {key + '=' + str(val):20s} -> {orb_sweep[(key, val)]} 个关键点")
print("解读: ORB = **oFAST（带方向的 FAST 角点）+ rBRIEF（旋转后的二进制描述子）**，"
      "描述子是 32 字节 = **256 bit 的 uint8**，所以匹配要用 `NORM_HAMMING`（汉明距离）"
      "而不是欧氏距离。看几个属性：尺度是 **1.2 的等比序列**（31.0, 37.2, 44.64, 53.57…，"
      "每档 ×1.2），这就是**尺度金字塔**；8 个不同 octave 说明关键点分布在 8 层；"
      "角度是 0~360° 的 ICM 主轴方向，这就是**旋转不变性**的来源。"
      "两个参数：`nfeatures` 是**硬上限**（100/500/1000/2000 完全线性）；"
      "`fastThreshold` 是 FAST 的亮度差门槛，只有在 `nfeatures` **大于**候选数时才起作用 —— "
      "基础配 nfeatures=5000 时，阈值提到 40 才让候选从 5000 掉到 4654，"
      "而 nfeatures=500 时不管阈值取多少都是 500（上限先到）")

assert orb_stat == (500, (500, 32), "uint8")
assert orb_sizes == (31.0, 37.2, 44.64, 53.57, 64.28, 77.14, 92.57, 111.08)
assert orb_oct == (0, 1, 2, 3, 4, 5, 6, 7)
assert orb_sweep[("nfeatures", 100)] == 100 and orb_sweep[("nfeatures", 2000)] == 2000
assert orb_sweep[("fastThreshold", 20)] == 5000 and orb_sweep[("fastThreshold", 40)] == 4654
'''

S10_CODE = '''M = cv2.getRotationMatrix2D((box_g.shape[1] / 2, box_g.shape[0] / 2), 25, 1.0)
rot = cv2.warpAffine(box_g, M, (box_g.shape[1], box_g.shape[0]))
kp1, d1 = cv2.ORB_create(nfeatures=500).detectAndCompute(box_g, None)
kp2, d2 = cv2.ORB_create(nfeatures=500).detectAndCompute(rot, None)
bf_cc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
cc_stat = (len(bf_cc), int(min(m.distance for m in bf_cc)), int(max(m.distance for m in bf_cc)))
knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
knn_len = len(knn)
ratio_pass = {}
for r in (0.6, 0.7, 0.75, 0.8, 0.9):
    ratio_pass[r] = int(sum(1 for m, n in knn if m.distance < r * n.distance))
good = [m for m, n in knn if m.distance < 0.75 * n.distance]
src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
Mrot = np.vstack([M, [0, 0, 1]])
hom_stat = (int(inliers.sum()), len(good), round(float(np.linalg.norm(H - Mrot)), 6))
hom_angle = round(float(np.degrees(np.arctan2(H[1, 0], H[0, 0]))), 4)
print("A: ", len(kp1), d1.shape, "| B: ", len(kp2), d2.shape, "（B = A 逆时针旋转 25°）")
print("crossCheck=True 匹配数:", cc_stat[0], "| 距离范围", cc_stat[1], "~", cc_stat[2])
print("knnMatch(k=2) 对数:", knn_len)
for r in sorted(ratio_pass):
    print(f"    比率 {r}: 通过 {ratio_pass[r]} 对")
print("比率 0.75 的匹配做 RANSAC 单应: 内点", hom_stat[0], "/", hom_stat[1],
      "| 与真值矩阵的 Frobenius 差", hom_stat[2])
print("从 H 反推旋转角:", hom_angle, "°（施加的是 +25°）")
print("估计 H =\\n", np.round(H, 4))
print("真值 M =\\n", np.round(Mrot, 4))
print("解读: 两条匹配策略 —— `crossCheck=True`（互为最近邻）给出 285 对，"
      "胜在**几乎不需要调参**但漏召回多；`knnMatch` + **Lowe 比率测试**（最近距离 < r × 次近）"
      "是从「这个匹配有多独特」的角度筛选，r 越大越宽松（0.6 -> 193 对，0.9 -> 322 对），"
      "**常用 0.75**。注意：`BFMatcher(crossCheck=True)` **不能再调 knnMatch**"
      "（OpenCV 会直接断言失败），两者必须各建一个匹配器。"
      "最后用 244 对做 `findHomography(..., RANSAC, 3.0)`，236 个内点、8 个外点被剔掉，"
      "恢复的 H 与真值矩阵 Frobenius 差只有 1.009325（差主要来自平移项 tx：-32.867 vs -31.944）。"
      "**从 H 反推的角度是 -24.09° 而不是 +25°** —— 因为图像坐标 y 轴向下，"
      "逆时针旋转在数学意义上是**负角**，这个符号坑在做机器人/云台标定时一定会遇到")

assert cc_stat == (285, 10, 78)
assert knn_len == 453
assert ratio_pass == {0.6: 193, 0.7: 222, 0.75: 244, 0.8: 266, 0.9: 322}
assert hom_stat == (236, 244, 1.009325)
assert hom_angle == -24.0895
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch08 模板匹配与关键点

> 数据：`classic/box.png`（模板来源）、`home.jpg`（场景）、`board.jpg`（角点）。
> 真值全部实跑（opencv 5.0.0）。

**本章在解决什么问题**

前一章我们把图像变成了**特征向量**去分类。本章回答另一类问题：
**「这个目标在哪」**（定位）。两条技术路线：

| 路线 | 方法 | 特点 |
|---|---|---|
| 稠密搜索 | `matchTemplate` | 直接、可解释；但**不具备尺度/旋转不变性**，且要给定模板 |
| 稀疏匹配 | Harris / Shi-Tomasi / ORB + 描述子匹配 | 天然支持尺度（金字塔）与旋转（方向）；但需要足够纹理 |

**本章考点**

1. `matchTemplate` 六种方法 —— **极值口径不统一**（SQDIFF 看 min，其余看 max）
2. 未归一化的 `CCORR` 会**定位到错误位置**；归一化方法对亮度/对比度扰动免疫
3. 阈值选取与「命中像素数 ≠ 目标数」
4. **贪心抑制**（NMS 雏形）与 `matchTemplate` 的 **(x, y) vs [y, x]** 坐标坑
5. **尺度失配**：原尺寸模板匹配放大后的目标只有 0.304 分；多尺度扫描的代价
6. Harris 角点：R 可负、阈值扫描、3x3 局部极大抑制
7. **`np.where` 行优先** —— 「前 5 个」不是「最强的 5 个」
8. Harris 三个参数（`blockSize` / `ksize` / `k`）的不同作用
9. Shi-Tomasi（`goodFeaturesToTrack`）与手写 Harris+NMS 的数量口径差异
10. ORB 关键点属性（尺度金字塔 / 角度 / octave）与 `nfeatures` vs `fastThreshold`
11. `crossCheck` vs `knnMatch` + Lowe 比率测试；RANSAC 单应与**角度符号坑**

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| 极值口径不统一 | SQDIFF 越小越像，其余越大越像 | §8.1 |
| CCORR 定位跑偏 | 未归一化时只找「最亮的一团」 | §8.1 / §8.2 |
| 分数不可比 | SQDIFF 从 0 变到 3674344 | §8.2 |
| 命中像素 ≠ 目标数 | 相关性是平滑的，一个目标命中一片 | §8.3 |
| (x, y) vs [y, x] | `matchTemplate` 返回 (x, y)，numpy 索引是 [y, x] | §8.4 |
| 尺度失配 | 1.25 倍目标用原尺寸模板只有 0.304 分 | §8.5 |
| R 可为负 | 阈值必须写 `k * R.max()` | §8.6 |
| **`np.where` 行优先** | 「前 5 个」响应差 50 倍 | §8.6 |
| 换 ksize 阈值失效 | R 量级从 3.07e8 涨到 5.55e11 | §8.7 |
| qualityLevel 不起作用 | 被 maxCorners 上限掩盖 | §8.8 |
| 角点数口径不一致 | Harris+NMS 5934 vs goodFeatures 200 | §8.8 |
| fastThreshold 看不出差别 | 被 nfeatures 上限掩盖 | §8.9 |
| crossCheck 与 knnMatch 互斥 | OpenCV 直接断言失败 | §8.10 |
| **旋转角符号** | y 轴向下 → 逆时针是负角（-24.09° vs +25°） | §8.10 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 8.1 `matchTemplate` 六种方法

我们先把 `box.png` 里的一块 48×48 区域**贴到** `home.jpg` 的 (x, y) = (200, 120)，
这样就有了**确定的真值位置**，再看六种方法各自表现。

`cv2.matchTemplate(scene, tmpl, method)` 返回一张响应图，尺寸是
`(场景高 − 模板高 + 1, 场景宽 − 模板宽 + 1)` —— 因为模板必须完整落在场景内。

| 方法 | 语义 | 极值口径 |
|---|---|---|
| `TM_SQDIFF` | 差的平方和 | **越小越像** |
| `TM_SQDIFF_NORMED` | 归一化 SQDIFF | **越小越像**，值域 [0, 1] |
| `TM_CCORR` | 互相关 | 越大越像 |
| `TM_CCORR_NORMED` | 归一化互相关 | 越大越像，值域 [0, 1] |
| `TM_CCOEFF` | 去均值互相关 | 越大越像 |
| `TM_CCOEFF_NORMED` | 归一化 CCOEFF | 越大越像，值域 [−1, 1] |

实测（响应图 337×465，真值位置 (200, 120)）：

| 方法 | min @ 位置 | max @ 位置 | 最佳位置 |
|---|---|---|---|
| `SQDIFF` | **0.0** @ (200, 120) | 34639148.0 @ (397, 58) | ✅ (200, 120) |
| `SQDIFF_NORMED` | **0.0** @ (200, 120) | 0.940459 @ (285, 0) | ✅ (200, 120) |
| `CCORR` | 23958996.0 @ (285, 0) | 67659192.0 @ **(244, 284)** | ❌ (244, 284) |
| `CCORR_NORMED` | 0.659850 @ (397, 58) | **1.0** @ (200, 120) | ✅ (200, 120) |
| `CCOEFF` | −2321679.5 @ (239, 239) | 9122235.0 @ (200, 120) | ✅ (200, 120) |
| `CCOEFF_NORMED` | −0.413884 @ (219, 165) | **1.0** @ (200, 120) | ✅ (200, 120) |

> **`CCORR` 是唯一跑偏的** —— 它只关心「模板和这块区域是不是都很亮」，
> 完全不管形状。加上归一化之后（`CCORR_NORMED`）才回到正确位置。
"""
    ),
    code(S1_CODE),
    md(
        """
## 8.2 亮度 / 对比度扰动下的定位稳定性

真实的巡检图像受曝光、逆光、补光灯影响，**同一块目标在不同照片里亮度差很大**。
我们模拟两种线性灰度变换：

- 场景 2：目标区域 **+40**（整体提亮）
- 场景 3：目标区域 **× 0.6 + 60**（压低对比度）

| 场景 | 定位正确的方法 | 说明 |
|---|---|---|
| 原样 | 5 / 6（`CCORR` 跑偏） | |
| 亮度 +40 | **6 / 6** | 巧合：提亮后 `CCORR` 也找对了 |
| 对比度 ×0.6+60 | 5 / 6（`CCORR` 跑偏） | |

归一化方法的**分数几乎不受影响**：

| 方法 | 原样 | 亮度 +40 | 对比度 ×0.6+60 |
|---|---|---|---|
| `CCORR_NORMED` | 1.0 | 0.997017 | 0.988887 |
| `CCOEFF_NORMED` | 1.0 | 0.999952 | 0.999971 |
| `SQDIFF` | 0.0 | **3674344.0** | **1461197.0** |
| `SQDIFF_NORMED` | 0.0 | 0.04872 | 0.025137 |

三条结论：

1. **未归一化的 `CCORR` 不可用**（扰动下会跑偏）。
2. **归一化方法对亮度/对比度扰动免疫**，分数稳定在 0.99 以上。
3. **`SQDIFF` 的分数完全不可比**（0 → 3674344）—— 想设「分数 > 0.8 就算命中」
   这类阈值判断，**必须用 `*_NORMED`**。
"""
    ),
    code(S2_CODE),
    md(
        """
## 8.3 多目标与阈值选取

把同一块 40×40 模板贴到 **3 个位置**：(y, x) = (60, 90)、(200, 320)、(330, 40)，
再做 `TM_CCOEFF_NORMED`。

响应值最高的 5 个像素：

```
1.0  /  0.999999  /  0.999999  /  0.714325  /  0.710718
```

阈值扫描：

| 阈值 | 命中像素数 |
|---|---|
| 0.99 | 3 |
| 0.95 | 3 |
| 0.9 | 3 |
| 0.8 | 3 |
| 0.7 | **7** |

两个要点：

- 三个真位置的响应**精确等于 1.0 附近**（贴的是同一块像素），
  第 4 名直接掉到 **0.714325** —— 峰值非常「尖」，阈值在 **0.8 附近**很安全。
- **命中像素数 ≠ 目标个数**：真实场景里相关性是平滑过渡的，一个目标会命中一整片，
  必须再做局部极大抑制。这里因为贴的是同一块像素才恰好一个目标一个像素。
"""
    ),
    code(S3_CODE),
    md(
        """
## 8.4 贪心抑制（NMS 雏形）

反复取响应图的**全局最大**，命中后把该位置的**邻域涂成 −1**，直到分数低于阈值：

```python
for _ in range(8):
    _, mx, _, mxl = cv2.minMaxLoc(work)
    if mx < 0.9:
        break
    found.append((mxl, mx))
    work[邻域] = -1          # 屏蔽掉，下一轮自然找下一个峰
```

结果与真值完全对上：

| 找到的 (x, y) | 分数 | 对应的真值 (y, x) |
|---|---|---|
| (40, 330) | 1.0 | (330, 40) |
| (90, 60) | 0.999999 | (60, 90) |
| (320, 200) | 0.999999 | (200, 320) |

两个必须记住的细节：

1. **`matchTemplate` 返回的坐标是 (x, y)**，而 numpy 索引是 **[y, x]**。
   写画框 `cv2.rectangle(scene, loc, ...)` 时用的是 (x, y)，写切片时用 [y, x] ——
   这是本行最高频的 bug 之一。
2. **抑制半径不能小于目标尺寸**，否则同一个目标会被数两次。
"""
    ),
    code(S4_CODE),
    md(
        """
## 8.5 尺度失配与多尺度扫描

把目标**放大 1.25 倍**（40×40 → 50×50）贴进场景，然后：

| 做法 | 最高分 | 位置 |
|---|---|---|
| 直接用**原尺寸**模板匹配 | **0.303955** | (386, 61) ❌ 完全错 |
| 模板按 0.6~1.65 扫 **22 档尺度** | **1.0**（s = 1.25） | (250, 150) ✅ |

多尺度扫描前三名：

| 分数 | 缩放 | 位置 | 模板尺寸 |
|---|---|---|---|
| 1.0 | 1.25 | (250, 150) | (50, 50) |
| 0.876258 | 1.2 | (251, 151) | (48, 48) |
| 0.79025 | 1.3 | (249, 149) | (52, 52) |

> **模板匹配不具备尺度不变性**。而且多尺度扫描的代价是**耗时线性增长**
> （22 档就是 22 倍）。这就是为什么要引入**尺度金字塔**和**关键点匹配** ——
> 也就是下面 Harris / ORB 的部分。
"""
    ),
    code(S5_CODE),
    md(
        """
## 8.6 Harris 角点

Harris 响应：`R = det(M) − k·trace(M)²`，其中 `M` 是窗口内的梯度二阶矩矩阵。

**R 可以为负** —— 边缘区为负、平坦区接近 0、角点为正。所以阈值只能写成
`k × R.max()`，不能给绝对值。

`board.jpg`（480×640 棋盘格）实测：

| 项 | 值 |
|---|---|
| R 的 min / max | **−82010912.0** / **307304896.0** |
| dtype | float32 |
| 阈值 0.01·R.max | 27380 像素 |
| 阈值 0.02·R.max | 17667 像素 |
| 阈值 0.05·R.max | 7441 像素 |
| 阈值 0.1·R.max | 2802 像素 |
| **3×3 局部极大抑制后** | **5934** |

抑制前有 27380 个像素超阈值，因为一个角点周围**连成一片**都超阈值。

**⚠️ 一个隐蔽的坑**：`np.where(nms)` 是**行优先**返回的，
直接取前 5 个拿到的是：

```
(269, 0) (280, 0) (292, 0) (303, 0) (337, 0)     ← 全在 y = 0，响应只有 6e6 量级
```

而**真正响应最强**的 5 个是：

```
(314, 185, 307304896.0)  (327, 185, 305193408.0)  (350, 185, 287178688.0)
(340, 185, 279111360.0)  (337, 185, 277100384.0)  ← 都在 y = 185 的棋盘格线上，3e8 量级
```

**差了 50 倍。** 要取「最强的 K 个」，必须显式 `np.argsort` 排序。
"""
    ),
    code(S6_CODE),
    md(
        """
## 8.7 Harris 的三个参数

`cv2.cornerHarris(src, blockSize, ksize, k)` 每次只改一个：

| 参数 | R.max | R.min | 过 0.01·R.max 的角点数 |
|---|---|---|---|
| `blockSize=2` | 307304896.0 | −82010912.0 | 27380 |
| `blockSize=3` | 233273712.0 | −52867512.0 | 54295 |
| `blockSize=5` | 122746256.0 | −26624974.0 | **102027** |
| `ksize=3` | 307304896.0 | −82010912.0 | 27380 |
| `ksize=5` | 7121759744.0 | −6370684928.0 | 31301 |
| `ksize=7` | **555186520064.0** | −766063738880.0 | 26756 |
| `k=0.02` | 339269568.0 | −40581280.0 | 32612 |
| `k=0.04` | 307304896.0 | −82010912.0 | 27380 |
| `k=0.06` | 277461504.0 | −123440536.0 | 23132 |
| `k=0.1` | 217774784.0 | **−206299792.0** | 16302 |

- **`blockSize` 越大** → 响应被平均（R.max 3.07e8 → 1.23e8），
  但**角点反而变多**（27380 → 102027）：大窗口让更多像素「看起来像角点」。
- **`ksize` 越大** → 微分越锐利，R 的**量级暴涨**（3 → 7 涨了 1800 倍：3.07e8 → 5.55e11）。
  **换 ksize 必须重新定阈值。**
- **`k` 越大** → `−k·trace²` 权重越大，R.min 越来越负（−4.06e7 → −2.06e8），
  过阈值的角点越来越少（32612 → 16302）。

> 工程默认 `blockSize=2, ksize=3, k=0.04` 是几十年调出来的平衡点，别随手改。
"""
    ),
    code(S7_CODE),
    md(
        """
## 8.8 Shi-Tomasi（`goodFeaturesToTrack`）

Shi-Tomasi 是 Harris 的改良版：不用 `det − k·trace²`，
而是直接取 `M` 的**较小特征值** `min(λ1, λ2)` 作响应 —— **少一个要调的 k**。
OpenCV 里就是 `cv2.goodFeaturesToTrack(...)`。

| 参数 | 角点数 | 说明 |
|---|---|---|
| `qualityLevel=0.001` | 200 | 撞到 `maxCorners` 上限 |
| `qualityLevel=0.01` | 200 | 同上 |
| `qualityLevel=0.05` | 200 | 同上 |
| `minDistance=5` | 200 | |
| `minDistance=10` | 200 | |
| `minDistance=20` | 200 | |
| `minDistance=40` | **130** | 终于起作用 |
| `maxCorners=10 / 50 / 100 / 500` | 10 / 50 / 100 / 500 | **完全线性**（硬上限） |

默认参数下前 5 个角点 (x, y)：
`(6.0, 153.0)`、`(9.0, 329.0)`、`(13.0, 270.0)`、`(21.0, 440.0)`、`(21.0, 454.0)`

**与手写 Harris + NMS 的数量口径对比：5934 vs 200 —— 差 30 倍。**

原因：`goodFeaturesToTrack` 内部还做了「**按响应排序 + minDistance 去重**」，
比朴素的「阈值 + 3×3 抑制」严格得多。

> 所以做「角点数量」这类指标时，**必须写清楚用的是哪一套口径、哪些参数**，
> 否则数字之间完全没有可比性。
>
> 另外注意 `qualityLevel` 在 0.001~0.05 之间**完全没起作用** ——
> 因为候选数远超 `maxCorners`，**上限先到了**。这是调参时最容易被骗的地方：
> 「我改了参数但结果没变」往往不是参数无效，而是被另一个约束掩盖了。
"""
    ),
    code(S8_CODE),
    md(
        """
## 8.9 ORB 关键点与描述子

**ORB = oFAST（带方向的 FAST 角点）+ rBRIEF（旋转后的二进制描述子）**

在 `board` 上（`nfeatures=500`）：

| 项 | 值 |
|---|---|
| 关键点数 | 500 |
| 描述子 | shape **(500, 32)**、dtype **uint8** → 32 字节 = **256 bit** |
| 响应 top5 | 0.0112 / 0.0111 / 0.011 / 0.0097 / 0.0096 |
| 尺度种类前 8 | 31.0, 37.2, 44.64, 53.57, 64.28, 77.14, 92.57, 111.08 |
| 角度种类前 8 | 3.94, 5.59, 5.81, 6.3, 6.33, 6.89, 7.4, 7.83 |
| 所在层（octave） | 0 ~ 7（**8 层**） |

**读出来的三件事：**

1. 尺度是 **1.2 的等比序列**（31.0 → 37.2 → 44.64 …每档 ×1.2）——
   这就是**尺度金字塔**，也是「匹配放大后的目标」能成功的原因（对比 §8.5！）。
2. 每个关键点带 **角度**（ICM 主轴方向）→ 描述子可以按这个角度旋转对齐 →
   **旋转不变性**。
3. 描述子是 **256 bit 的二进制串**，所以匹配必须用 `NORM_HAMMING`（汉明距离），
   **不能用欧氏距离**。

**参数扫描**（基础配 `nfeatures=5000 / fastThreshold=20`）：

| 参数 | 关键点数 |
|---|---|
| `nfeatures=100 / 500 / 1000 / 2000` | 100 / 500 / 1000 / 2000（**完全线性 = 硬上限**） |
| `fastThreshold=5 / 10 / 20` | 5000 / 5000 / 5000 |
| `fastThreshold=40` | **4654** |

> **`fastThreshold` 只有在 `nfeatures` 大于候选数时才起作用**。
> 如果配 `nfeatures=500`，不管 `fastThreshold` 取 5 还是 40 都是 500 ——
> 又一次「被另一个约束掩盖」。
"""
    ),
    code(S9_CODE),
    md(
        """
## 8.10 `BFMatcher` + 比率测试 + RANSAC 单应

把 `box.png` **逆时针旋转 25°** 得到 B 图，各自提 ORB 特征再匹配。

**两条匹配策略：**

| 策略 | 匹配数 | 说明 |
|---|---|---|
| `BFMatcher(NORM_HAMMING, crossCheck=True).match` | **285** | 互为最近邻；不用调参，但漏召回多 |
| `BFMatcher(NORM_HAMMING).knnMatch(k=2)` | 453 对 | 每对给「最近 + 次近」 |

**Lowe 比率测试**（`最近距离 < r × 次近距离`）：

| r | 通过对数 |
|---|---|
| 0.6 | 193 |
| 0.7 | 222 |
| **0.75** | **244** |
| 0.8 | 266 |
| 0.9 | 322 |

**RANSAC 单应**（用 r = 0.75 的 244 对）：

| 项 | 值 |
|---|---|
| 内点 / 总数 | **236 / 244**（剔掉 8 个外点） |
| 与真值矩阵 Frobenius 差 | **1.009325** |
| 从 H 反推的旋转角 | **−24.0895°** |
| 实际施加的旋转 | **+25°** |

**两个坑：**

1. `BFMatcher(crossCheck=True)` **不能再调 `knnMatch`** —— OpenCV 直接断言失败
   （`Assertion failed: K == 1 && update == 0 && mask.empty()`）。
   两条策略必须各建一个匹配器。
2. **旋转角的符号**：反推得到的是 **−24.09°** 而不是 +25°。
   因为**图像坐标的 y 轴向下**，数学意义上的「逆时针」在这里对应**负角**。
   误差 0.91° 来自 RANSAC 只用 236 个点的拟合噪声。
   做机器人 / 云台标定时这个符号坑一定会遇到。

> Frobenius 差 1.009325 看起来不小，但**差主要来自平移项**
> （tx：−32.867 vs −31.944）；旋转项 `cos ≈ 0.9244 vs 0.9063`、
> `sin ≈ 0.4254 vs 0.4226` 都在千分位级别。
> 所以评估单应质量时**不能只看一个整体范数**，要分项看。
"""
    ),
    code(S10_CODE),
    md(
        """
## 小结

1. `matchTemplate` 响应图尺寸 = **场景 − 模板 + 1**。
2. **极值口径不统一**：SQDIFF 系列看 **min**，其余四种看 **max**。
3. 未归一化的 **`CCORR` 会定位到错误位置**（(244, 284) vs 真值 (200, 120)）——
   它只找「最亮的一团」。
4. **工程上一律用 `*_NORMED`**：亮度/对比度扰动下分数稳定在 0.99 以上；
   `SQDIFF` 的分数从 0 变成 3674344，**完全不可比**。
5. **命中像素数 ≠ 目标个数**；阈值要取在「真目标峰」与「背景次峰」之间。
6. **贪心抑制 = NMS 雏形**：取全局最大 → 记位置 → 邻域涂 −1。
7. **`matchTemplate` 返回 (x, y)**，numpy 索引是 **[y, x]**。
8. **模板匹配没有尺度不变性**：1.25 倍目标用原尺寸模板只有 0.304 分；
   多尺度扫描能救但**耗时线性增长**。
9. Harris 的 **R 可以为负**，阈值必须写 `k * R.max()`。
10. 一个角点在响应图上**连成一片**：27380 → NMS 后 **5934**。
11. **`np.where` 是行优先的**：「前 5 个」和「最强的 5 个」相差 **50 倍**，
    取 top-K 必须显式排序。
12. `blockSize` 越大角点**越多**；`ksize` 越大 R 量级**暴涨**（要重定阈值）；
    `k` 越大 R.min 越负。
13. Shi-Tomasi 取 `min(λ1, λ2)`，**比 Harris 少一个参数**。
14. **角点数口径必须写清楚**：Harris+NMS = 5934 vs `goodFeaturesToTrack` = 200（差 30 倍）。
15. **参数「不起作用」常常是被另一个约束掩盖**：`qualityLevel` 被 `maxCorners` 掩盖、
    `fastThreshold` 被 `nfeatures` 掩盖。
16. ORB 描述子是 **256 bit uint8** → 匹配必须用 **`NORM_HAMMING`**。
17. 尺度 **×1.2 等比序列** + 每个关键点带**角度** = 尺度 + 旋转不变性。
18. `crossCheck=True` 与 `knnMatch` **互斥**（断言失败）。
19. Lowe 比率测试常用 **0.75**；RANSAC 单应剔掉外点后 236/244 内点。
20. **旋转角符号**：y 轴向下 → 逆时针是 **负角**（−24.09° vs +25°）。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch08 模板匹配与关键点（练习版）

> 按提示补全 `____`，跑通所有 assert。`box_g` / `home_g` / `scene` / `tmpl` / `board`
> 已在 setup 里备好；`METHODS` / `MIN_METHODS` / `best_loc` / `paste` / `nms_harris`
> 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；控制流写在挖空块之外
> （要写函数时，`def` 与 `return` 已经给好，你只填函数体，且**不许用 for**）。
> 题目真值都来自实跑（opencv 5.0.0）。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：`matchTemplate` 六种方法

逐个跑六种方法，记下 min / max 和它们的位置，再按各自的口径挑出「最佳位置」。
注意 `SQDIFF` 系列是**越小越像**。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：亮度 / 对比度扰动

模拟「整体提亮 +40」和「压低对比度 ×0.6+60」，看六种方法还找不找得到 (200, 120)。
注意分数会变得完全不可比。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：多目标与阈值

同一块模板贴到 3 个位置，做阈值扫描。想清楚「为什么命中像素数不能当目标数」。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：贪心抑制

循环骨架已给（含 `if mx < 0.9: break`）。你只需要写「记峰值 + 屏蔽邻域」两件事。
注意返回坐标是 (x, y)。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：尺度失配与多尺度扫描

先用原尺寸模板去匹配放大 1.25 倍的目标，再扫 22 档尺度把它找回来。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：Harris 角点

先做四档阈值扫描，再做 3×3 局部极大抑制，最后**按响应强度**取 top5。
小心 `np.where` 的行优先顺序。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：Harris 三个参数

`blockSize` / `ksize` / `k` 每次只改一个，记下 R.max / R.min / 角点数。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：Shi-Tomasi

`goodFeaturesToTrack` 的三个参数扫描，并和手写 Harris+NMS 的数量做对比。
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 9：ORB 关键点与描述子

先从关键点里提取属性（尺度 / 角度 / octave），再扫 `nfeatures` 与 `fastThreshold`。
"""
    ),
    code(E9_CODE),
    md(
        """
## 任务 10：BFMatcher + 比率测试 + 单应

两条匹配策略、Lowe 比率测试、RANSAC 单应。最后留意反推旋转角的**符号**。
"""
    ),
    code(E10_CODE),
    md(
        """
## 自查清单

- [ ] 知道响应图尺寸 = 场景 − 模板 + 1
- [ ] 说得清六种方法哪些看 min、哪些看 max
- [ ] 知道未归一化 `CCORR` 会定位到错误位置及原因
- [ ] 能解释为什么工程上一律用 `*_NORMED`
- [ ] 知道 `SQDIFF` 的分数为什么不能设阈值
- [ ] 知道命中像素数 ≠ 目标数
- [ ] 能默写贪心抑制（NMS 雏形）的循环
- [ ] 记得 `matchTemplate` 返回 (x, y)、numpy 索引是 [y, x]
- [ ] 知道模板匹配没有尺度不变性，以及多尺度扫描的代价
- [ ] 知道 Harris 的 R 可以为负，阈值必须写 `k * R.max()`
- [ ] 知道一个角点在响应图上连成一片（27380 → 5934）
- [ ] **记得 `np.where` 行优先**，取 top-K 必须显式排序
- [ ] 说得清 `blockSize` / `ksize` / `k` 各自的影响方向
- [ ] 知道 Shi-Tomasi 比 Harris 少一个参数（取 min(λ1, λ2)）
- [ ] 知道角点数口径不一致（5934 vs 200）及原因
- [ ] 会识别「参数被另一个约束掩盖」的假象
- [ ] 知道 ORB 描述子是 256 bit uint8，必须用 `NORM_HAMMING`
- [ ] 能从尺度序列看出 1.2 金字塔、从角度看出旋转不变性
- [ ] 知道 `crossCheck=True` 与 `knnMatch` 互斥
- [ ] 会用 Lowe 比率测试，并知道常用 0.75
- [ ] **记得旋转角符号坑**（y 轴向下 → 逆时针是负角）
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
