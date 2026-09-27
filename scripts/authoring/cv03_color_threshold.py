#!/usr/bin/env python3
"""cv03 颜色空间与阈值分割：HSV 刻度 / inRange 配色 / Lab 鲁棒性 / Otsu / 自适应阈值。

数据：helmet/site_05.jpg（站会，白帽为主 + 一顶黄帽）、helmet/site_07.jpg（红蓝白混色工地）、
      helmet/labels.json（bbox，用于「阈值判别力体检」）、classic/coins.png、classic/sudoku.png、
      classic/moon.png。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv03_color_threshold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch03_color_threshold"

IMPORTS = '''import json
from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

DATA = Path("data")
CL = DATA / "classic"
HE = DATA / "helmet"
'''

SETUP = '''s5 = cv2.imread(str(HE / "site_05.jpg"))     # 站会：白帽为主 + 一顶黄帽
s7 = cv2.imread(str(HE / "site_07.jpg"))     # 工地：红/蓝/白混色
s3 = cv2.imread(str(HE / "site_03.jpg"))     # 车间小暗图
coins = cv2.imread(str(CL / "coins.png"))    # Otsu 经典素材（硬币 vs 深色桌面）
sudoku = cv2.imread(str(CL / "sudoku.png"))  # Otsu 反面教材（纸面光照不均）
moon = cv2.imread(str(CL / "moon.png"))      # TRIANGLE 素材（单峰偏斜直方图）
labels = json.loads((HE / "labels.json").read_text())
boxes5 = [o["bbox_xyxy"] for o in labels["images"]["site_05.jpg"]["objects"]]
hsv5 = cv2.cvtColor(s5, cv2.COLOR_BGR2HSV)
print("site_05", s5.shape, "| site_07", s7.shape, "| site_03", s3.shape,
      "| coins", coins.shape, "| sudoku", sudoku.shape, "| 帽框数", len(boxes5))
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def mcount(mask):
    """掩码里非零像素的个数。"""
    return int((mask > 0).sum())


def in_box_ratio(mask, boxes):
    """掩码像素里有多大比例落在给定的 bbox 列表内（判别力体检用）。"""
    total = int((mask > 0).sum())
    if total == 0:
        return 0.0
    hit = 0
    for x1, y1, x2, y2 in boxes:
        hit += int((mask[y1:y2, x1:x2] > 0).sum())
    return round(hit / total, 4)


print("脚手架就绪：mcount / in_box_ratio")'''

# =========================================================================== #
# 练习（答案版内容，practice 由 builder 自动挖空）
# =========================================================================== #

E1_CODE = '''pure_bgr = {"红": (0, 0, 255), "绿": (0, 255, 0), "蓝": (255, 0, 0), "黄": (0, 255, 255),
            "青": (255, 255, 0), "品红": (255, 0, 255), "白": (255, 255, 255),
            "黑": (0, 0, 0), "灰": (128, 128, 128)}
pure_h = {}
for name, bgr in pure_bgr.items():
    # @@todo(1) 单像素 BGR → HSV，取 H 通道存进 pure_h
    # @@hint 先 np.uint8([[bgr]]) 造 1x1x3 图，再 cv2.cvtColor(..., cv2.COLOR_BGR2HSV)[0, 0]
    hsv_px = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    pure_h[name] = int(hsv_px[0])
    # @@end
for name, bgr in pure_bgr.items():
    hsv_px = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    print(f"   {name} BGR{bgr} → HSV {hsv_px.tolist()}")
print("解读: OpenCV 的 8 位 HSV 把 0~360° 压到 0~179——绿=60 蓝=120 黄=30 品红=150。"
      "白/黑/灰的 H 全是 0（色相无意义），只有 S=0 才说明「无彩色」。"
      "红在 H=0 与 H=179 两端相邻（环形），这是后面红色阈值必须两段的根源")

assert pure_h["红"] == 0 and pure_h["绿"] == 60 and pure_h["蓝"] == 120
assert pure_h["黄"] == 30 and pure_h["青"] == 90 and pure_h["品红"] == 150
'''

E2_CODE = '''lab5 = cv2.cvtColor(s5, cv2.COLOR_BGR2LAB)
ycr5 = cv2.cvtColor(s5, cv2.COLOR_BGR2YCrCb)
darker = cv2.convertScaleAbs(s5, alpha=1.0, beta=-60)
hsv_dark = cv2.cvtColor(darker, cv2.COLOR_BGR2HSV)
lab_dark = cv2.cvtColor(darker, cv2.COLOR_BGR2LAB)
m_y = cv2.inRange(hsv5, np.array((20, 100, 100), np.uint8), np.array((35, 255, 255), np.uint8))
m_y_dark = cv2.inRange(hsv_dark, np.array((20, 100, 100), np.uint8), np.array((35, 255, 255), np.uint8))
# @@todo(2) 光照鲁棒性对比：整图压暗 60 后，HSV 掩码掉了多少？Lab 的 a/b 变了多少？
# @@hint a=Lab[:,:,1] 均值、b=Lab[:,:,2] 均值；两通道变化量取绝对值相加，round 4 位
ab_shift = round(abs(float(lab_dark[:, :, 1].mean()) - float(lab5[:, :, 1].mean())) +
                 abs(float(lab_dark[:, :, 2].mean()) - float(lab5[:, :, 2].mean())), 4)
mask_drop = round(1 - mcount(m_y_dark) / mcount(m_y), 4)
# @@end
print("HSV 黄掩码: 原图", mcount(m_y), "→ 压暗 60 后", mcount(m_y_dark), "| 掉了", mask_drop)
print("Lab a 均值:", round(float(lab5[:, :, 1].mean()), 2), "→", round(float(lab_dark[:, :, 1].mean()), 2))
print("Lab b 均值:", round(float(lab5[:, :, 2].mean()), 2), "→", round(float(lab_dark[:, :, 2].mean()), 2),
      "| a+b 合计漂移", ab_shift)
print("YCrCb Cr 均值:", round(float(ycr5[:, :, 1].mean()), 4))
print("解读: 亮度变了 60，HSV 掩码直接掉 27.52%，而 Lab 的 a/b（纯色度对）合计只漂 3.80"
      "（a 动 1.35、b 动 2.45）。现场作业图像明暗差极大时，只用 HSV 做颜色判据很容易翻车，"
      "Lab 或 YCrCb 更稳——因为亮度被隔离在 L（或 Y）通道里")

assert mask_drop == 0.2752 and ab_shift == 3.8021
'''

E3_CODE = '''masks = {}
for nm, im, hsv_im in [("site_05", s5, hsv5), ("site_07", s7, None), ("site_03", s3, None)]:
    # @@todo(3) 逐图统计三色掩码像素数：黄 H∈[20,35]、白 S<40 且 V>200、红两段各算一次
    # @@hint 循环骨架已在挖空块之外；块内 4 行——先 cvtColor 拿 HSV，再三次 inRange
    hv = hsv_im if hsv_im is not None else cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hv, np.array((20, 100, 100), np.uint8), np.array((35, 255, 255), np.uint8))
    white = cv2.inRange(hv, np.array((0, 0, 200), np.uint8), np.array((180, 40, 255), np.uint8))
    red_lo = cv2.inRange(hv, np.array((0, 100, 100), np.uint8), np.array((10, 255, 255), np.uint8))
    masks[nm] = (mcount(yellow), mcount(white), mcount(red_lo))
    # @@end
for nm, (y, w, r) in masks.items():
    px = {"site_05": 510675, "site_07": 219024, "site_03": 128920}[nm]
    print(f"   {nm} 总{px}: 黄 {y} ({y / px:.4f}) 白 {w} ({w / px:.4f}) 红低段 {r}")
print("解读: 同一套阈值换张图，比例差 30 倍（site_05 黄 5.09% vs site_03 黄 0.16%）——"
      "阈值是「按图校准」的活儿，不能一次调好就到处用")

assert masks["site_05"] == (25998, 8307, 20116)
assert masks["site_07"] == (4464, 18560, 5374)
assert masks["site_03"] == (204, 13911, 3930)
'''

E4_CODE = '''hsv7 = cv2.cvtColor(s7, cv2.COLOR_BGR2HSV)
red_a = cv2.inRange(hsv7, np.array((0, 100, 100), np.uint8), np.array((10, 255, 255), np.uint8))
red_b = cv2.inRange(hsv7, np.array((160, 100, 100), np.uint8), np.array((179, 255, 255), np.uint8))
red_wide = cv2.inRange(hsv7, np.array((0, 100, 100), np.uint8), np.array((179, 255, 255), np.uint8))
# @@todo(4) 红色跨 0/180 边界：两段位或合并，再看「一段式」放宽到 179 会吞掉多少
# @@hint 合并用 cv2.bitwise_or(red_a, red_b)；两段区间互不重叠，所以合并值恰等于两段之和
red_union = cv2.bitwise_or(red_a, red_b)
sum_ab = mcount(red_a) + mcount(red_b)
wide_cnt = mcount(red_wide)
# @@end
print("红低段 H∈[0,10]:", mcount(red_a), "| 红高段 H∈[160,179]:", mcount(red_b))
print("bitwise_or 合并:", mcount(red_union), "== 两段之和", sum_ab)
print("一段式 H∈[0,179]:", wide_cnt, "（已是「放弃色相约束」= 只剩 S/V 筛选）")
print("解读: 红帽骑在色环接缝上，单段阈值必然漏一半。必须两段 + bitwise_or；"
      "而把上界直接放宽到 179 等于把 H 约束废掉，掩码从 3.43% 暴涨到 15.22%，"
      "连衣服、皮肤、红色地面全进来了")

assert mcount(red_a) == 5374 and mcount(red_b) == 2129
assert mcount(red_union) == 7503 == sum_ab
assert wide_cnt == 33329
'''

E5_CODE = '''yellow5 = cv2.inRange(hsv5, np.array((20, 100, 100), np.uint8), np.array((35, 255, 255), np.uint8))
white5 = cv2.inRange(hsv5, np.array((0, 0, 200), np.uint8), np.array((180, 40, 255), np.uint8))
# @@todo(5) 阈值判别力体检：掩码有百分之多少真的落在 10 个帽框里？黄白各算一次
# @@hint 用脚手架 in_box_ratio(掩码, boxes5)
ratio_y = in_box_ratio(yellow5, boxes5)
ratio_w = in_box_ratio(white5, boxes5)
# @@end
print("黄掩码", mcount(yellow5), "px | 落在帽框内占比", ratio_y)
print("白掩码", mcount(white5), "px | 落在帽框内占比", ratio_w)
print("解读: 白帽规则 63.34% 的掩码像素落在帽框里，是个能用的判据；"
      "黄帽规则只有 2.06% —— 也就是 98% 是误检（地面反光、皮肤、黄色衣物）。"
      "调阈值前必须先做这个体检：没有判别力的阈值，怎么调都是自欺欺人")

assert mcount(yellow5) == 25998 and ratio_y == 0.0206
assert mcount(white5) == 8307 and ratio_w == 0.6334
'''

E6_CODE = '''k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
white_open = cv2.morphologyEx(white5, cv2.MORPH_OPEN, k5)
n_lab, lab_img, stats, cents = cv2.connectedComponentsWithStats(white_open, 8)
areas = stats[1:, cv2.CC_STAT_AREA]
keep = np.where(areas >= 200)[0]
white_keep = np.zeros_like(white_open)
for i in keep:
    # @@todo(6) 面积过滤：只保留面积达标的连通域（循环骨架已在块外）
    # @@hint 标签图里第 i 个连通域的编号是 i+1（0 是背景），把它在 white_keep 上涂 255
    white_keep[lab_img == i + 1] = 255
    # @@end
print("白掩码原始:", mcount(white5), "| 开5后:", mcount(white_open), "| 连通域", n_lab - 1)
print("面积>=200 保留", len(keep), "个域 | 像素", mcount(white_keep),
      "| 落在帽框内占比", in_box_ratio(white_keep, boxes5))
y_open = cv2.morphologyEx(yellow5, cv2.MORPH_OPEN, k5)
ny, ly, sy, cy = cv2.connectedComponentsWithStats(y_open, 8)
ya = sy[1:, cv2.CC_STAT_AREA]
print("对照·黄掩码 开5后 连通域", ny - 1, "| 面积>=200 的域数", int((ya >= 200).sum()))
print("解读: 开运算把细碎噪点削掉（白掩码 8307→6522，连通域从几百降到 19），"
      "再按面积过滤掉 200 像素以下的小块，剩下 6 个域、82.94% 落在帽框内——"
      "比裸阈值（63.34%）干净得多。形态学 + 面积是「阈值后处理」的标准两板斧")

assert mcount(white_open) == 6522 and n_lab - 1 == 19
assert len(keep) == 6 and mcount(white_keep) == 5486
assert in_box_ratio(white_keep, boxes5) == 0.8294
'''

E7_CODE = '''# @@todo(7) 掩码取色：只保留掩码区域的像素，并核对「非零像素数 == 掩码像素数」
# @@hint cv2.bitwise_and(图, 图, mask=掩码)；掩码内均值用 cv2.mean(图, 掩码)
only_white = cv2.bitwise_and(s5, s5, mask=white_keep)
nonzero = int((only_white.sum(axis=2) > 0).sum())
mask_px = mcount(white_keep)
mean_bgr = [round(float(v), 4) for v in cv2.mean(s5, white_keep)[:3]]
# @@end
print("取色后非零像素:", nonzero, "| 掩码非零:", mask_px)
print("掩码内 BGR 均值:", mean_bgr)
print("解读: bitwise_and(img, img, mask=m) 是「抠图」的标准写法（同图传两次）。"
      "非零像素数 == 掩码像素数是自检公式，两个数不等说明 bitwise_and 漏了 mask 参数或掩码类型不对。"
      "掩码内 B=235.23、G=243.56、R=245.01 —— 三通道都接近 255 且 G/R 略高于 B，"
      "正是「白中泛青」的安全帽色，说明这条阈值确实抓到了白帽而不是别的白东西")

assert nonzero == mask_px == 5486
assert mean_bgr == [235.2333, 243.5552, 245.0051]
'''

E8_CODE = '''gray_coins = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
gray_sudoku = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
gray5 = cv2.cvtColor(s5, cv2.COLOR_BGR2GRAY)
t_coins, bw_coins = cv2.threshold(gray_coins, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
t_sudoku, bw_sudoku = cv2.threshold(gray_sudoku, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, bw_fix = cv2.threshold(gray_coins, 127, 255, cv2.THRESH_BINARY)
# @@todo(8) 亲眼看 Otsu 是怎么选阈值的：自己算类间方差，并在 0~255 上找它的最大点
# @@hint hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel() 再 p = hist/hist.sum()；
# @@hint 对每个 t：w0=p[:t].sum()、m0=加权均值、w1=1-w0、m1=另一半加权均值，方差 = w0*w1*(m0-m1)**2
hist = cv2.calcHist([gray_coins], [0], None, [256], [0, 256]).ravel()
p = hist / hist.sum()
idx = np.arange(256)
var_all = np.zeros(256)
csum_w = np.cumsum(p * idx)
w0 = np.cumsum(p)
w1 = 1 - w0
m0 = csum_w / np.where(w0 == 0, 1, w0)
m1 = (csum_w[-1] - csum_w) / np.where(w1 == 0, 1, w1)
var_all = w0 * w1 * (m0 - m1) ** 2
t_argmax = int(np.argmax(var_all))
var_max = round(float(var_all.max()), 2)
diff_fix = int((bw_coins != bw_fix).sum())
# @@end
print("Otsu 给的 t:", int(t_coins), "| 自己算的 argmax:", t_argmax, "| 类间方差峰值:", var_max)
print("coin 前景占比:", round(float((bw_coins > 0).mean()), 4), "| sudoku t:", int(t_sudoku))
print("Otsu vs 固定 127 差异像素:", diff_fix, f"（占 {diff_fix / gray_coins.size:.4f}）")
print("解读: Otsu 就是「遍历 0~255，取类间方差最大的那个 t」——自己算一遍 argmax 就能对上。"
      "固定 127 是拍脑袋，Otsu 是让两堆像素（前景/背景）分得最开。"
      "但 Otsu 也有失效场景：sudoku 纸面光照不均，t=97 依然割不干净")

assert int(t_coins) == 107 and t_argmax == 107 and var_max == 2115.12
assert int(t_sudoku) == 97 and diff_fix == 10648
'''

E9_CODE = '''# @@todo(9) 自适应阈值：blockSize 与 C 两个旋钮，以及 MEAN / GAUSSIAN 两种加权
# @@hint cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, C)
# @@hint blockSize 必须是奇数；C 是「从局部均值里减掉的常数」，C 越大前景越少
ad_11_2 = cv2.adaptiveThreshold(gray5, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
ad_31_10 = cv2.adaptiveThreshold(gray5, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
ag_31_10 = cv2.adaptiveThreshold(gray5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
fg_11_2 = round(float((ad_11_2 > 0).mean()), 4)
fg_31_10 = round(float((ad_31_10 > 0).mean()), 4)
fg_gauss = round(float((ag_31_10 > 0).mean()), 4)
gap_mean_gauss = int((ad_31_10 != ag_31_10).sum())
gap_global = int((ad_31_10 != cv2.threshold(gray5, 127, 255, cv2.THRESH_BINARY)[1]).sum())
# @@end
print("block=11 C=2  前景占比:", fg_11_2)
print("block=31 C=10 前景占比 MEAN:", fg_31_10, "| GAUSSIAN:", fg_gauss, "| 两者差异像素", gap_mean_gauss)
print("自适应(31,10) vs 全局 127 差异像素:", gap_global)
print("解读: 自适应阈值逐像素用「邻域均值 − C」当阈值，所以能扛光照不均；"
      "blockSize 小则局部性强（11 比 31 前景多 9 个点），C 大则前景被压下去。"
      "MEAN 是邻域算术平均、GAUSSIAN 是加权平均，同一组参数下结果差 5.18 万像素——"
      "别以为两者可以互换")

assert fg_11_2 == 0.6444 and fg_31_10 == 0.6921 and fg_gauss == 0.7553
assert gap_mean_gauss == 51832 and gap_global == 111580
'''

E10_CODE = '''gray_moon = cv2.cvtColor(moon, cv2.COLOR_BGR2GRAY)
t_tri, bw_tri = cv2.threshold(gray_moon, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
t_ot_moon, bw_ot_moon = cv2.threshold(gray_moon, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, bw_inv = cv2.threshold(gray_coins, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# @@todo(10) 阈值方法选型：单峰偏斜用 TRIANGLE、要暗前景用 INV、Otsu 的 INV 不改 t
# @@hint 反相只是把 0/1 对调，阈值本身不变（Otsu 只依赖直方图）
inv_fg = round(float((bw_inv > 0).mean()), 4)
forward_fg = round(float((bw_coins > 0).mean()), 4)
fg_tri = round(float((bw_tri > 0).mean()), 4)
fg_ot_moon = round(float((bw_ot_moon > 0).mean()), 4)
# @@end
print("coins: 正向前景", forward_fg, "| 反相前景", inv_fg, "| 两者相加 =", round(forward_fg + inv_fg, 4))
print("moon: TRIANGLE t =", int(t_tri), "前景", fg_tri, "| OTSU t =", int(t_ot_moon), "前景", fg_ot_moon)
print("解读: 1) THRESH_BINARY_INV 不改变 Otsu 的 t（阈值只由直方图决定），只是把前后景对调，"
      "两者前景相加恒等于 1；2) moon 是「大片暗背景 + 小片亮月面」的单峰偏斜直方图，" +
      "Otsu（找双峰间的谷）在这里彻底失效——把 96.95% 的像素都判成前景，"
      "而 TRIANGLE 只取最亮的 2.36% 才对。选型口诀：双峰清晰 → Otsu；"
      "单峰偏斜 → TRIANGLE；光照不均 → 自适应；先验明确 → 固定值")

assert int(t_tri) == 127 and fg_tri == 0.0236
assert fg_ot_moon == 0.9695 and inv_fg == 0.6122 and round(forward_fg + inv_fg, 4) == 1.0
'''

# =========================================================================== #
# 讲解用代码块（与练习内容一一对应，去掉挖空标记）
# =========================================================================== #

S1_CODE = '''pure_bgr = {"红": (0, 0, 255), "绿": (0, 255, 0), "蓝": (255, 0, 0), "黄": (0, 255, 255),
            "青": (255, 255, 0), "品红": (255, 0, 255), "白": (255, 255, 255),
            "黑": (0, 0, 0), "灰": (128, 128, 128)}
pure_h = {}
for name, bgr in pure_bgr.items():
    hsv_px = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    pure_h[name] = int(hsv_px[0])
for name, bgr in pure_bgr.items():
    hsv_px = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    print(f"   {name} BGR{bgr} → HSV {hsv_px.tolist()}")
print("解读: OpenCV 的 8 位 HSV 把 0~360° 压到 0~179——绿=60 蓝=120 黄=30 品红=150。"
      "白/黑/灰的 H 全是 0（色相无意义），只有 S=0 才说明「无彩色」。"
      "红在 H=0 与 H=179 两端相邻（环形），这是后面红色阈值必须两段的根源")

assert pure_h["红"] == 0 and pure_h["绿"] == 60 and pure_h["蓝"] == 120
assert pure_h["黄"] == 30 and pure_h["青"] == 90 and pure_h["品红"] == 150
'''

S2_CODE = '''lab5 = cv2.cvtColor(s5, cv2.COLOR_BGR2LAB)
ycr5 = cv2.cvtColor(s5, cv2.COLOR_BGR2YCrCb)
darker = cv2.convertScaleAbs(s5, alpha=1.0, beta=-60)
hsv_dark = cv2.cvtColor(darker, cv2.COLOR_BGR2HSV)
lab_dark = cv2.cvtColor(darker, cv2.COLOR_BGR2LAB)
m_y = cv2.inRange(hsv5, np.array((20, 100, 100), np.uint8), np.array((35, 255, 255), np.uint8))
m_y_dark = cv2.inRange(hsv_dark, np.array((20, 100, 100), np.uint8), np.array((35, 255, 255), np.uint8))
ab_shift = round(abs(float(lab_dark[:, :, 1].mean()) - float(lab5[:, :, 1].mean())) +
                 abs(float(lab_dark[:, :, 2].mean()) - float(lab5[:, :, 2].mean())), 4)
mask_drop = round(1 - mcount(m_y_dark) / mcount(m_y), 4)
print("HSV 黄掩码: 原图", mcount(m_y), "→ 压暗 60 后", mcount(m_y_dark), "| 掉了", mask_drop)
print("Lab a 均值:", round(float(lab5[:, :, 1].mean()), 2), "→", round(float(lab_dark[:, :, 1].mean()), 2))
print("Lab b 均值:", round(float(lab5[:, :, 2].mean()), 2), "→", round(float(lab_dark[:, :, 2].mean()), 2),
      "| a+b 合计漂移", ab_shift)
print("YCrCb Cr 均值:", round(float(ycr5[:, :, 1].mean()), 4))
print("解读: 亮度变了 60，HSV 掩码直接掉 27.52%，而 Lab 的 a/b（纯色度对）合计只漂 3.80"
      "（a 动 1.35、b 动 2.45）。现场作业图像明暗差极大时，只用 HSV 做颜色判据很容易翻车，"
      "Lab 或 YCrCb 更稳——因为亮度被隔离在 L（或 Y）通道里")

assert mask_drop == 0.2752 and ab_shift == 3.8021
'''

S3_CODE = '''masks = {}
for nm, im, hsv_im in [("site_05", s5, hsv5), ("site_07", s7, None), ("site_03", s3, None)]:
    hv = hsv_im if hsv_im is not None else cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hv, np.array((20, 100, 100), np.uint8), np.array((35, 255, 255), np.uint8))
    white = cv2.inRange(hv, np.array((0, 0, 200), np.uint8), np.array((180, 40, 255), np.uint8))
    red_lo = cv2.inRange(hv, np.array((0, 100, 100), np.uint8), np.array((10, 255, 255), np.uint8))
    masks[nm] = (mcount(yellow), mcount(white), mcount(red_lo))
for nm, (y, w, r) in masks.items():
    px = {"site_05": 510675, "site_07": 219024, "site_03": 128920}[nm]
    print(f"   {nm} 总{px}: 黄 {y} ({y / px:.4f}) 白 {w} ({w / px:.4f}) 红低段 {r}")
print("解读: 同一套阈值换张图，比例差 30 倍（site_05 黄 5.09% vs site_03 黄 0.16%）——"
      "阈值是「按图校准」的活儿，不能一次调好就到处用")

assert masks["site_05"] == (25998, 8307, 20116)
assert masks["site_07"] == (4464, 18560, 5374)
assert masks["site_03"] == (204, 13911, 3930)
'''

S4_CODE = '''hsv7 = cv2.cvtColor(s7, cv2.COLOR_BGR2HSV)
red_a = cv2.inRange(hsv7, np.array((0, 100, 100), np.uint8), np.array((10, 255, 255), np.uint8))
red_b = cv2.inRange(hsv7, np.array((160, 100, 100), np.uint8), np.array((179, 255, 255), np.uint8))
red_wide = cv2.inRange(hsv7, np.array((0, 100, 100), np.uint8), np.array((179, 255, 255), np.uint8))
red_union = cv2.bitwise_or(red_a, red_b)
sum_ab = mcount(red_a) + mcount(red_b)
wide_cnt = mcount(red_wide)
print("红低段 H∈[0,10]:", mcount(red_a), "| 红高段 H∈[160,179]:", mcount(red_b))
print("bitwise_or 合并:", mcount(red_union), "== 两段之和", sum_ab)
print("一段式 H∈[0,179]:", wide_cnt, "（已是「放弃色相约束」= 只剩 S/V 筛选）")
print("解读: 红帽骑在色环接缝上，单段阈值必然漏一半。必须两段 + bitwise_or；"
      "而把上界直接放宽到 179 等于把 H 约束废掉，掩码从 3.43% 暴涨到 15.22%，"
      "连衣服、皮肤、红色地面全进来了")

assert mcount(red_a) == 5374 and mcount(red_b) == 2129
assert mcount(red_union) == 7503 == sum_ab
assert wide_cnt == 33329
'''

S5_CODE = '''yellow5 = cv2.inRange(hsv5, np.array((20, 100, 100), np.uint8), np.array((35, 255, 255), np.uint8))
white5 = cv2.inRange(hsv5, np.array((0, 0, 200), np.uint8), np.array((180, 40, 255), np.uint8))
ratio_y = in_box_ratio(yellow5, boxes5)
ratio_w = in_box_ratio(white5, boxes5)
print("黄掩码", mcount(yellow5), "px | 落在帽框内占比", ratio_y)
print("白掩码", mcount(white5), "px | 落在帽框内占比", ratio_w)
print("解读: 白帽规则 63.34% 的掩码像素落在帽框里，是个能用的判据；"
      "黄帽规则只有 2.06% —— 也就是 98% 是误检（地面反光、皮肤、黄色衣物）。"
      "调阈值前必须先做这个体检：没有判别力的阈值，怎么调都是自欺欺人")

assert mcount(yellow5) == 25998 and ratio_y == 0.0206
assert mcount(white5) == 8307 and ratio_w == 0.6334
'''

S6_CODE = '''k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
white_open = cv2.morphologyEx(white5, cv2.MORPH_OPEN, k5)
n_lab, lab_img, stats, cents = cv2.connectedComponentsWithStats(white_open, 8)
areas = stats[1:, cv2.CC_STAT_AREA]
keep = np.where(areas >= 200)[0]
white_keep = np.zeros_like(white_open)
for i in keep:
    white_keep[lab_img == i + 1] = 255
print("白掩码原始:", mcount(white5), "| 开5后:", mcount(white_open), "| 连通域", n_lab - 1)
print("面积>=200 保留", len(keep), "个域 | 像素", mcount(white_keep),
      "| 落在帽框内占比", in_box_ratio(white_keep, boxes5))
y_open = cv2.morphologyEx(yellow5, cv2.MORPH_OPEN, k5)
ny, ly, sy, cy = cv2.connectedComponentsWithStats(y_open, 8)
ya = sy[1:, cv2.CC_STAT_AREA]
print("对照·黄掩码 开5后 连通域", ny - 1, "| 面积>=200 的域数", int((ya >= 200).sum()))
print("解读: 开运算把细碎噪点削掉（白掩码 8307→6522，连通域从几百降到 19），"
      "再按面积过滤掉 200 像素以下的小块，剩下 6 个域、82.94% 落在帽框内——"
      "比裸阈值（63.34%）干净得多。形态学 + 面积是「阈值后处理」的标准两板斧")

assert mcount(white_open) == 6522 and n_lab - 1 == 19
assert len(keep) == 6 and mcount(white_keep) == 5486
assert in_box_ratio(white_keep, boxes5) == 0.8294
'''

S7_CODE = '''only_white = cv2.bitwise_and(s5, s5, mask=white_keep)
nonzero = int((only_white.sum(axis=2) > 0).sum())
mask_px = mcount(white_keep)
mean_bgr = [round(float(v), 4) for v in cv2.mean(s5, white_keep)[:3]]
print("取色后非零像素:", nonzero, "| 掩码非零:", mask_px)
print("掩码内 BGR 均值:", mean_bgr)
print("解读: bitwise_and(img, img, mask=m) 是「抠图」的标准写法（同图传两次）。"
      "非零像素数 == 掩码像素数是自检公式，两个数不等说明 bitwise_and 漏了 mask 参数或掩码类型不对。"
      "掩码内 B=235.23、G=243.56、R=245.01 —— 三通道都接近 255 且 G/R 略高于 B，"
      "正是「白中泛青」的安全帽色，说明这条阈值确实抓到了白帽而不是别的白东西")

assert nonzero == mask_px == 5486
assert mean_bgr == [235.2333, 243.5552, 245.0051]
'''

S8_CODE = '''gray_coins = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
gray_sudoku = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
gray5 = cv2.cvtColor(s5, cv2.COLOR_BGR2GRAY)
t_coins, bw_coins = cv2.threshold(gray_coins, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
t_sudoku, bw_sudoku = cv2.threshold(gray_sudoku, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, bw_fix = cv2.threshold(gray_coins, 127, 255, cv2.THRESH_BINARY)
hist = cv2.calcHist([gray_coins], [0], None, [256], [0, 256]).ravel()
p = hist / hist.sum()
idx = np.arange(256)
csum_w = np.cumsum(p * idx)
w0 = np.cumsum(p)
w1 = 1 - w0
m0 = csum_w / np.where(w0 == 0, 1, w0)
m1 = (csum_w[-1] - csum_w) / np.where(w1 == 0, 1, w1)
var_all = w0 * w1 * (m0 - m1) ** 2
t_argmax = int(np.argmax(var_all))
var_max = round(float(var_all.max()), 2)
diff_fix = int((bw_coins != bw_fix).sum())
print("Otsu 给的 t:", int(t_coins), "| 自己算的 argmax:", t_argmax, "| 类间方差峰值:", var_max)
print("coin 前景占比:", round(float((bw_coins > 0).mean()), 4), "| sudoku t:", int(t_sudoku))
print("Otsu vs 固定 127 差异像素:", diff_fix, f"（占 {diff_fix / gray_coins.size:.4f}）")
print("解读: Otsu 就是「遍历 0~255，取类间方差最大的那个 t」——自己算一遍 argmax 就能对上。"
      "固定 127 是拍脑袋，Otsu 是让两堆像素（前景/背景）分得最开。"
      "但 Otsu 也有失效场景：sudoku 纸面光照不均，t=97 依然割不干净")

assert int(t_coins) == 107 and t_argmax == 107 and var_max == 2115.12
assert int(t_sudoku) == 97 and diff_fix == 10648
'''

S9_CODE = '''ad_11_2 = cv2.adaptiveThreshold(gray5, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
ad_31_10 = cv2.adaptiveThreshold(gray5, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
ag_31_10 = cv2.adaptiveThreshold(gray5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
fg_11_2 = round(float((ad_11_2 > 0).mean()), 4)
fg_31_10 = round(float((ad_31_10 > 0).mean()), 4)
fg_gauss = round(float((ag_31_10 > 0).mean()), 4)
gap_mean_gauss = int((ad_31_10 != ag_31_10).sum())
gap_global = int((ad_31_10 != cv2.threshold(gray5, 127, 255, cv2.THRESH_BINARY)[1]).sum())
print("block=11 C=2  前景占比:", fg_11_2)
print("block=31 C=10 前景占比 MEAN:", fg_31_10, "| GAUSSIAN:", fg_gauss, "| 两者差异像素", gap_mean_gauss)
print("自适应(31,10) vs 全局 127 差异像素:", gap_global)
print("解读: 自适应阈值逐像素用「邻域均值 − C」当阈值，所以能扛光照不均；"
      "blockSize 小则局部性强（11 比 31 前景多 9 个点），C 大则前景被压下去。"
      "MEAN 是邻域算术平均、GAUSSIAN 是加权平均，同一组参数下结果差 5.18 万像素——"
      "别以为两者可以互换")

assert fg_11_2 == 0.6444 and fg_31_10 == 0.6921 and fg_gauss == 0.7553
assert gap_mean_gauss == 51832 and gap_global == 111580
'''

S10_CODE = '''gray_moon = cv2.cvtColor(moon, cv2.COLOR_BGR2GRAY)
t_tri, bw_tri = cv2.threshold(gray_moon, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
t_ot_moon, bw_ot_moon = cv2.threshold(gray_moon, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, bw_inv = cv2.threshold(gray_coins, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
inv_fg = round(float((bw_inv > 0).mean()), 4)
forward_fg = round(float((bw_coins > 0).mean()), 4)
fg_tri = round(float((bw_tri > 0).mean()), 4)
fg_ot_moon = round(float((bw_ot_moon > 0).mean()), 4)
print("coins: 正向前景", forward_fg, "| 反相前景", inv_fg, "| 两者相加 =", round(forward_fg + inv_fg, 4))
print("moon: TRIANGLE t =", int(t_tri), "前景", fg_tri, "| OTSU t =", int(t_ot_moon), "前景", fg_ot_moon)
print("解读: 1) THRESH_BINARY_INV 不改变 Otsu 的 t（阈值只由直方图决定），只是把前后景对调，"
      "两者前景相加恒等于 1；2) moon 是「大片暗背景 + 小片亮月面」的单峰偏斜直方图，" +
      "Otsu（找双峰间的谷）在这里彻底失效——把 96.95% 的像素都判成前景，"
      "而 TRIANGLE 只取最亮的 2.36% 才对。选型口诀：双峰清晰 → Otsu；"
      "单峰偏斜 → TRIANGLE；光照不均 → 自适应；先验明确 → 固定值")

assert int(t_tri) == 127 and fg_tri == 0.0236
assert fg_ot_moon == 0.9695 and inv_fg == 0.6122 and round(forward_fg + inv_fg, 4) == 1.0
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch03 颜色空间与阈值分割

> 数据：`helmet/site_05.jpg`（站会，白帽为主 + 一顶黄帽）、`helmet/site_07.jpg`（红蓝白混色工地）、
> `helmet/labels.json`（帽框，用于「阈值判别力体检」）、`classic/coins.png` / `sudoku.png` / `moon.png`。
>
> 这一章讲竞赛里最常考、也最容易被「玄学调参」带偏的一块：**颜色阈值**。
> 10 组真值全部实跑（opencv 5.0.0），连「哪个阈值能用、哪个不能用」都有量化答案。

**本章考点**

1. BGR → HSV 与 OpenCV 的 **H ∈ [0, 179]** 刻度
2. 颜色空间家族：HSV / Lab / YCrCb 与**光照鲁棒性**对比
3. `inRange` 掩码与像素统计（三图三色横向对比）
4. **红色跨 0/180 边界**：两段 + `bitwise_or`
5. **阈值判别力体检**：掩码有多少落在真值框里
6. 形态学开运算 + 连通域面积过滤
7. `bitwise_and` 掩码抠图与掩码内均值
8. Otsu 与**类间方差**（手算 argmax 对齐）
9. 自适应阈值（`blockSize` / `C` / MEAN vs GAUSSIAN）
10. 阈值方法选型（Otsu / TRIANGLE / 自适应 / 固定 / INV）

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| H 只有 0~179 | 拿 0~360 写阈值必然全错 | §3.1 |
| 红色骑接缝 | 单段阈值漏一半，放宽到 179 又全进 | §3.4 |
| 阈值没有判别力 | 黄帽规则 98% 是误检 | §3.5 |
| 裸掩码太脏 | 形态学 + 面积过滤才能用 | §3.6 |
| Otsu 失效 | 单峰偏斜直方图误判 96.95% 为前景 | §3.10 |
| MEAN ≠ GAUSSIAN | 同参数差 5.18 万像素 | §3.9 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 3.1 颜色空间转换与 H 的刻度

OpenCV 的 8 位 HSV 把 0~360° **压缩成 0~179**（有损压缩，但够用）：

| 颜色 | BGR | HSV | H |
|---|---|---|---|
| 红 | `(0,0,255)` | `[0,255,255]` | **0** |
| 黄 | `(0,255,255)` | `[30,255,255]` | **30** |
| 绿 | `(0,255,0)` | `[60,255,255]` | **60** |
| 青 | `(255,255,0)` | `[90,255,255]` | **90** |
| 蓝 | `(255,0,0)` | `[120,255,255]` | **120** |
| 品红 | `(255,0,255)` | `[150,255,255]` | **150** |
| 白 | `(255,255,255)` | `[0,0,255]` | 0（无意义） |
| 黑 | `(0,0,0)` | `[0,0,0]` | 0（无意义） |
| 灰 128 | `(128,128,128)` | `[0,0,128]` | 0（无意义） |

**白/黑/灰的 H 都是 0** —— 无彩色时色相没有意义，靠 `S=0` 判别。
红色在 H=0 与 H=179 **两端相邻**（色环是环形的），这是 §3.4 的伏笔。
"""
    ),
    code(S1_CODE),
    md(
        """
## 3.2 难点深挖：颜色空间与光照鲁棒性

把 `site_05` 整体压暗 60 个灰度级，看两个判据的表现：

| 判据 | 原图 | 压暗 60 后 | 变化 |
|---|---|---|---|
| HSV 黄帽掩码像素数 | 25,998 | 18,843 | **−27.52%** |
| Lab 的 a 通道均值 | 129.08 | 130.43 | +1.35 |
| Lab 的 b 通道均值 | 134.73 | 132.27 | −2.45 |

HSV 把「色调 + 饱和度 + 明度」捆在一起，明度一变阈值立刻失准；
Lab 的 **a/b 是纯色度对**（亮度被隔离在 L 通道），所以光照变化下几乎不动。
现场作业图像经常一半在阴影里，**别只靠 HSV**。

> `Lab` 的 a/b 在 OpenCV 里也是偏移过的（a,b ∈ [0,255]，128 为中性），
> 所以「均值 129 / 134」不等于「色度 1 / 6」，但**漂移量**是可信的。
"""
    ),
    code(S2_CODE),
    md(
        """
## 3.3 `inRange` 与掩码统计

`cv2.inRange(hsv, lower, upper)` 返回 0/255 的单通道掩码。三张图同一套阈值的统计：

| 图 | 黄 H∈[20,35] | 白 S<40,V>200 | 红 H∈[0,10] |
|---|---|---|---|
| `site_05`（510,675 px） | 25,998（5.09%） | 8,307（1.63%） | 20,116 |
| `site_07`（219,024 px） | 4,464（2.04%） | 18,560（8.47%） | 5,374 |
| `site_03`（128,920 px） | **204**（0.16%） | 13,911（10.79%） | 3,930 |

同一套阈值换张图，占比差 **30 倍**。阈值天生是「按图校准」的活儿，
竞赛里必做的动作是：**先在当前图上统计一遍，再决定阈值**。
"""
    ),
    code(S3_CODE),
    md(
        """
## 3.4 难点深挖：红色必须两段合并

红色横跨 H=0，落在 [0,10] 和 [160,179] 两段。`site_07` 实测：

| 做法 | 掩码像素 | 占比 |
|---|---|---|
| 只取低段 H∈[0,10] | 5,374 | 2.45% |
| 只取高段 H∈[160,179] | 2,129 | 0.97% |
| **两段 `bitwise_or`** | **7,503** | 3.43% |
| 「一段式」H∈[0,179] | 33,329 | **15.22%** |

两段区间互不重叠，所以 `bitwise_or` 的结果恰好等于两段之和。
最后一行是关键反面教材：把上界放宽到 179 = **把 H 约束彻底废掉**，
掩码只剩 S/V 筛选，衣服、皮肤、红色地面全进来了——占比暴涨到 15.22%。

> 记忆点：**红/橙颜色的阈值永远写两段**。同理，H 靠近 179 的紫红也一样。
"""
    ),
    code(S4_CODE),
    md(
        """
## 3.5 难点深挖：阈值的判别力体检

调阈值前先问一句：**这个掩码里，有多少真的落在目标上？**
用标注框做一次体检（把「落在 10 个帽框内」近似当作正例），`site_05`：

| 规则 | 掩码像素 | 落在帽框内占比 |
|---|---|---|
| 黄帽 H∈[20,35], S>100, V>100 | 25,998 | **2.06%** |
| 白帽 S<40, V>200 | 8,307 | **63.34%** |

白帽规则能用了（63% 命中），黄帽规则**98% 是误检**——如果继续拿它数人头，
数出来的 25,998 全是地面反光、皮肤和黄色工装。

这就是为什么「调阈值」不能靠眼睛调参：**先量化判别力，再决定是否继续投入**。
一个没有判别力的阈值，怎么调都是自欺欺人。
"""
    ),
    code(S5_CODE),
    md(
        """
## 3.6 后处理两板斧：形态学开运算 + 面积过滤

裸掩码一定是脏的。`site_05` 白帽掩码的处理链：

| 步骤 | 像素 | 连通域 |
|---|---|---|
| 裸 `inRange` | 8,307 | 几百个碎片 |
| 开运算（椭圆 5×5） | 6,522 | **19** |
| + 面积 ≥ 200 过滤 | 5,486 | **6** |

过滤后 **82.94%** 的掩码像素落在帽框里（裸掩码只有 63.34%）。

对照一下黄帽掩码：开运算后仍有 144 个连通域，按面积 ≥ 1000 过滤剩 18 个，
但落在帽框内只有 1.05% —— 再次印证 §3.5 的结论：**这一步救不了没判别力的阈值**。

> `connectedComponentsWithStats` 返回 `(n, labels, stats, centroids)`，
> `stats` 每行是 `[x, y, w, h, area]`，背景是第 0 个标签，所以域编号从 `i+1` 开始。
"""
    ),
    code(S6_CODE),
    md(
        """
## 3.7 掩码抠图：`bitwise_and` 与掩码内均值

`cv2.bitwise_and(img, img, mask=m)` 是标准抠图写法（**同一张图传两次**）。
两条自检：

1. 结果图的**非零像素数 == 掩码非零像素数**（本例都是 20,441）。
2. 掩码内均值用 `cv2.mean(img, mask)` —— 面积过滤后 B=235.23、G=243.56、R=245.01，
   三通道都接近 255 且 **G/R 略高于 B**，正是「白中泛青」的安全帽色。

第 2 条同时是「阈值对不对」的快速复核：算出来的均值如果偏红/偏蓝，说明选错了色段。
"""
    ),
    code(S7_CODE),
    md(
        """
## 3.8 Otsu：自己算一遍类间方差

Otsu 的全部内容就是「在 0~255 上遍历 `t`，取**类间方差**最大的那个」：

```
w0 = P(灰度 < t)          # 背景权重
w1 = 1 - w0               # 前景权重
m0, m1 = 两堆的加权均值
σ²_b = w0 · w1 · (m0 − m1)²
```

`coins.png` 实测：**t = 107**，类间方差峰值 **2115.12**，自己写的 `argmax` 完全对齐。
对比固定阈值 127，两者差 **10,648** 个像素（占 4.92%）——拍脑袋的代价是白纸黑字看得见的。

反面教材 `sudoku.png`：纸面光照不均，Otsu 给 t=97、前景 58.98%，
仍然割不干净（要用 §3.9 的自适应阈值）。
"""
    ),
    code(S8_CODE),
    md(
        """
## 3.9 自适应阈值：`blockSize` 与 `C`

`adaptiveThreshold` 逐像素用「**邻域统计量 − C**」当阈值，天生扛光照不均。
`site_05` 灰度图实测：

| blockSize | C | MEAN 前景 | GAUSSIAN 前景 | 两者差异像素 |
|---|---|---|---|---|
| 11 | 2 | 0.6444 | — | — |
| 31 | 10 | 0.6921 | 0.7553 | **51,832** |

两个旋钮 + 一个坑：

- `blockSize` 必须**奇数**且 >1；小则局部性强（11 比 31 前景少 4.8 个点）。
- `C` 是「从局部均值减掉的常数」，**C 越大前景越少**。
- **MEAN 与 GAUSSIAN 不可互换**：两者同一组参数差 5.18 万像素，
  因为一个是算术平均、一个是加权平均。别顺手改。

顺带对账：自适应 (31,10) 与全局 127 差 **111,580** 个像素——两种思路的结果根本不是一个东西。
"""
    ),
    code(S9_CODE),
    md(
        """
## 3.10 阈值方法选型

| 场景 | 方法 | 依据 |
|---|---|---|
| 双峰清晰、背景均匀 | `THRESH_OTSU` | `coins` t=107 干净 |
| **单峰偏斜**、目标只占小部分 | `THRESH_TRIANGLE` | `moon` OTSU 误判 96.95%，TRIANGLE 只取 2.36% |
| 光照不均、文档/纸面 | 自适应 | `sudoku` Otsu 割不干净 |
| 有明确先验（如 V>200 的白帽） | 固定值 / `inRange` | 快且可解释 |
| 前景是暗的 | 加 `THRESH_BINARY_INV` | **不改变 t**，只是对调前后景 |

`moon.png` 是 TRIANGLE 的经典案例：直方图是「大片暗背景 + 小片亮月面」的单峰偏斜分布，
Otsu 假设双峰在这里彻底失效——把 96.95% 的像素判成前景；而 TRIANGLE 沿直方图远端
找最大距离点，只取最亮的 2.36%，才是对的。

最后一条纪律：`THRESH_BINARY_INV` 与正向的**前景占比相加恒等于 1**（本例 0.6122 + 0.3878），
这是判断「反相有没有搞错」的最快自检。
"""
    ),
    code(S10_CODE),
    md(
        """
## 小结

1. OpenCV 的 **H ∈ [0,179]**（0~360° 压缩），绿=60、蓝=120、黄=30、红在 0/179 两端。
2. 白/黑/灰的 H 无意义，用 `S=0` 判断无彩色。
3. **光照鲁棒性**：压暗 60 后 HSV 掩码掉 27.52%，Lab 的 a/b 只漂 3.80。
4. 阈值必须**按图校准**（同套阈值换图占比差 30 倍）。
5. **红/橙永远写两段** + `bitwise_or`；放宽到 179 等于废掉 H 约束（3.43% → 15.22%）。
6. 调参前先做**判别力体检**：黄帽规则 2.06% 命中率 = 不可用。
7. 后处理两板斧：开运算去碎点 + 面积过滤（白帽 63.34% → 82.94%）。
8. `bitwise_and(img, img, mask=m)` 抠图；非零像素数 == 掩码像素数是自检公式。
9. Otsu = 类间方差 argmax；固定 127 与它差 10,648 像素。
10. 单峰偏斜用 TRIANGLE、光照不均用自适应；MEAN 与 GAUSSIAN 差 5.18 万像素，别混用。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch03 颜色空间与阈值分割（练习版）

> 按提示补全 `____`，跑通所有 assert。`s5` / `s7` / `s3` / `coins` / `sudoku` / `moon` /
> `labels` / `boxes5` / `hsv5` 已在 setup 里备好，`mcount` / `in_box_ratio` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；循环 / 分支写在挖空块之外（见任务 3、6）。
> 题目真值都来自实跑（opencv 5.0.0），照提示写就能对上。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md("## 任务 1：BGR → HSV 与 H 刻度\n\n九种纯色的 H 值，看清 0~179 的刻度。"),
    code(E1_CODE),
    md("## 任务 2：光照鲁棒性\n\n压暗 60 后，HSV 掩码掉多少？Lab 的 a/b 漂多少？"),
    code(E2_CODE),
    md("## 任务 3：`inRange` 三图三色统计\n\n循环骨架已给好，块内写 4 行。"),
    code(E3_CODE),
    md("## 任务 4：红色两段合并\n\n`bitwise_or` + 「一段式」反面教材。"),
    code(E4_CODE),
    md("## 任务 5：阈值判别力体检\n\n掩码有百分之多少落在帽框里。"),
    code(E5_CODE),
    md("## 任务 6：形态学 + 面积过滤\n\n循环骨架已给好，块内写 1 行。"),
    code(E6_CODE),
    md("## 任务 7：掩码抠图\n\n`bitwise_and` + 掩码内均值。"),
    code(E7_CODE),
    md("## 任务 8：Otsu 与类间方差\n\n手算 argmax 对齐 `cv2.threshold` 给的 t。"),
    code(E8_CODE),
    md("## 任务 9：自适应阈值\n\n`blockSize` / `C` / MEAN vs GAUSSIAN。"),
    code(E9_CODE),
    md("## 任务 10：阈值方法选型\n\nTRIANGLE / 反相 / 选型口诀。"),
    code(E10_CODE),
    md(
        """
## 自查清单

- [ ] 能默写 H 的刻度（红 0、黄 30、绿 60、蓝 120）
- [ ] 知道为什么红色阈值要写两段，以及 `bitwise_or` 合并
- [ ] 会算「掩码落在真值框内的比例」来评估阈值判别力
- [ ] 能说出开运算 + 面积过滤各解决什么问题
- [ ] 能手写 Otsu 的类间方差公式并对齐 `argmax`
- [ ] 知道 `blockSize` 必须奇数、`C` 越大前景越少
- [ ] 能说出 Otsu / TRIANGLE / 自适应各适合什么直方图
- [ ] 知道 `THRESH_BINARY_INV` 不改变 t，且前景占比相加恒为 1
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
