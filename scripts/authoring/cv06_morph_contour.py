#!/usr/bin/env python3
"""cv06 形态学与轮廓：腐蚀膨胀几何 / 结构元形状 / 迭代=大核 / 开闭 / 噪声实战 /
开闭尺度边界 / 梯度顶帽黑帽 / findContours 三件套 / 轮廓几何量 / 连通域 / 硬币计数口径。

数据：classic/coins.png（主用例）。
真值全部实跑（opencv 5.0.0）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv06_morph_contour.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch06_morph_contour"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

DATA = Path("data")
CL = DATA / "classic"
HE = DATA / "helmet"
'''

SETUP = '''coins = cv2.imread(str(CL / "coins.png"))
cg = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
cg_blur = cv2.GaussianBlur(cg, (5, 5), 0)
otsu_t, cbw = cv2.threshold(cg_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 400 个白噪点（固定种子，结果可复现）
rng = np.random.default_rng(0)
noisy = cbw.copy()
ys = rng.integers(0, noisy.shape[0], 400)
xs = rng.integers(0, noisy.shape[1], 400)
noisy[ys, xs] = 255

print("coins", coins.shape, "| gray", cg.shape, "| Otsu T", otsu_t, "| cbw 前景", int((cbw > 0).sum()))
print("cg 灰度范围", int(cg.min()), "~", int(cg.max()), "-> 全局单一阈值在这张图上有光照不均风险")
print("noisy 前景", int((noisy > 0).sum()), "| 外轮廓",
      len(cv2.findContours(noisy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]))
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)


def ascii_bin(m, mark="##"):
    """把小尺寸二值图渲染成文本（0 -> ".."，非 0 -> mark）。"""
    return "\\n".join("".join(mark if v else ".." for v in row) for row in m)


def biggest(m):
    """返回面积最大的外轮廓（没有轮廓则返回 None）。"""
    cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return max(cs, key=cv2.contourArea) if cs else None


def bbox_of(m):
    """返回最大外轮廓的 boundingRect（空图返回 None）。"""
    c = biggest(m)
    return None if c is None else cv2.boundingRect(c)


def diff(a, b):
    """两张二值图的逐像素最大差（0 表示完全相同）。"""
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())


def close_fill(m, k):
    """用 k x k 全 1 核做闭运算后的前景像素数（用来观察「洞有没有被填上」）。"""
    return int((cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0).sum())


print("脚手架就绪：K3 / K5 / ascii_bin / biggest / bbox_of / diff / close_fill")'''

# =========================================================================== #
# 练习（答案版内容，practice 由 builder 自动挖空）
# =========================================================================== #

E1_CODE = '''sq = np.zeros((9, 9), np.uint8)
sq[3:6, 3:6] = 255
line = np.zeros((9, 9), np.uint8)
line[:, 4] = 255
# @@todo(1) 腐蚀 = 邻域取 min（白区被啃瘦），膨胀 = 邻域取 max（白区变胖）
# @@hint 都用 K3；各量 (原图, 腐蚀后, 膨胀后) 三个前景像素数，注意 1px 细线会被啃光
sq_px = (int((sq > 0).sum()), int((cv2.erode(sq, K3) > 0).sum()), int((cv2.dilate(sq, K3) > 0).sum()))
line_px = (int((line > 0).sum()), int((cv2.erode(line, K3) > 0).sum()), int((cv2.dilate(line, K3) > 0).sum()))
bb_sq, bb_er, bb_di = bbox_of(sq), bbox_of(cv2.erode(sq, K3)), bbox_of(cv2.dilate(sq, K3))
# @@end
print("3x3 方块 (原, 腐蚀, 膨胀) 前景:", sq_px)
print("1px 竖线 (原, 腐蚀, 膨胀) 前景:", line_px)
print("方块外框: 原", bb_sq, "| 腐蚀后", bb_er, "| 膨胀后", bb_di)
print("腐蚀后剩下的像素坐标:", np.argwhere(cv2.erode(sq, K3) > 0).tolist())
print("解读: 3x3 方块腐蚀后就剩中心 1 个点（9 -> 1），膨胀成 5x5（25）—— 核 3x3 把边界各吃掉 1 圈。"
      "1px 细线的腐蚀结果是 **0**（整条消失），膨胀成 3x9=27。"
      "所以「腐蚀掉噪点」和「腐蚀掉细线」是同一个机制，选核大小就是在选你能忍受多细的损失")

assert sq_px == (9, 1, 25) and line_px == (9, 0, 27)
assert (bb_sq, bb_er, bb_di) == ((3, 3, 3, 3), (4, 4, 1, 1), (2, 2, 5, 5))
'''

E2_CODE = '''k_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k_ell3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k_cross3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
k_ell5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_rect5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# @@todo(2) 结构元形状也是旋钮：RECT / ELLIPSE / CROSS 的「有效面积」并不一样
# @@hint 数每个核的 1 元素个数；再判断 ELLIPSE3 与 CROSS3 是否逐元素相同；最后看方块用 ELLIPSE3 膨胀的面积
k_sizes = (int(k_rect3.sum()), int(k_ell3.sum()), int(k_ell5.sum()), int(k_rect5.sum()))
k_same = bool((k_ell3 == k_cross3).all())
di_ell = int((cv2.dilate(sq, k_ell3) > 0).sum())
# @@end
print("核的 1 元素个数 (RECT3, ELLIPSE3, ELLIPSE5, RECT5):", k_sizes)
print("ELLIPSE3 与 CROSS3 逐元素相同?", k_same)
print("ELLIPSE5 行内元素数:", k_ell5.sum(axis=1).tolist())
print("方块用 ELLIPSE3 膨胀后面积:", di_ell, "（RECT3 是 25）")
print("解读: 3x3 时 ELLIPSE 退化成 CROSS（只有 5 个 1），比 RECT 的 9 个少 —— "
      "所以「膨胀 3x3」的默认含义取决于核形状，用 ELLIPSE 得到的是十字形邻域，"
      "方块只能长成 21 像素的「圆角方块」而不是 25。Ellipse 核在圆目标上更自然，"
      "但形状一变，同样的 ksize 结果就不可比")

assert k_sizes == (9, 5, 17, 25) and k_same is True and di_ell == 21
'''

E3_CODE = '''# @@todo(3) 迭代 n 次 3x3 是否等价于「一次 (2n+1)x(2n+1) 矩形核」？（要逐像素为 0 的证据）
# @@hint 比 erode(cbw, K3, iterations=2) 与 erode(cbw, K5)；再比 iterations=3 与 7x7 全 1 核；用 diff() 对账
er_32 = cv2.erode(cbw, K3, iterations=2)
er_51 = cv2.erode(cbw, K5)
di_32 = cv2.dilate(cbw, K3, iterations=2)
di_51 = cv2.dilate(cbw, K5)
er_33 = cv2.erode(cbw, K3, iterations=3)
d_er2, d_di2 = diff(er_32, er_51), diff(di_32, di_51)
d_er3 = diff(er_33, cv2.erode(cbw, np.ones((7, 7), np.uint8)))
px_iter = (int((er_32 > 0).sum()), int((di_32 > 0).sum()), int((er_33 > 0).sum()))
# @@end
print("erode 3x3x2 vs 5x5x1: 前景", int((er_32 > 0).sum()), "vs", int((er_51 > 0).sum()), "| 逐像素差", d_er2)
print("dilate 3x3x2 vs 5x5x1: 前景", int((di_32 > 0).sum()), "vs", int((di_51 > 0).sum()), "| 逐像素差", d_di2)
print("erode 3x3x3 vs 7x7x1 逐像素差:", d_er3)
print("(er_32, di_32, er_33) 前景:", px_iter)
print("解读: 矩形核下 **迭代 2 次 3x3 == 一次 5x5**，3 次 == 一次 7x7（差全为 0）—— "
      "因为 5x5 方块 = 3x3 方块 ⊕ 3x3 方块，而「核的膨胀」对应「运算的叠加」。"
      "实战价值：想开很大的核，用 iterations 比造大核便宜（大核是 O(k²) 元素，"
      "迭代是 O(n) 次 3x3）。但换成 ELLIPSE 核这个等式就不成立了")

assert d_er2 == 0 and d_di2 == 0 and d_er3 == 0
assert px_iter == (37867, 58379, 33150)
'''

E4_CODE = '''# @@todo(4) 开运算 = 先腐蚀后膨胀（去小白点）；闭运算 = 先膨胀后腐蚀（填小黑洞）
# @@hint cv2.morphologyEx(cbw, cv2.MORPH_OPEN / MORPH_CLOSE, K3)，再与手写的 erode→dilate / dilate→erode 对账
op3 = cv2.morphologyEx(cbw, cv2.MORPH_OPEN, K3)
cl3 = cv2.morphologyEx(cbw, cv2.MORPH_CLOSE, K3)
d_op = diff(op3, cv2.dilate(cv2.erode(cbw, K3), K3))
d_cl = diff(cl3, cv2.erode(cv2.dilate(cbw, K3), K3))
px_morph = (int((cbw > 0).sum()), int((op3 > 0).sum()), int((cl3 > 0).sum()))
d_opcl = diff(op3, cl3)
# @@end
print("(cbw, open3, close3) 前景:", px_morph)
print("open3 == dilate(erode) 逐像素差:", d_op, "| close3 == erode(dilate) 逐像素差:", d_cl)
print("open3 与 close3 的最大差:", d_opcl)
print("解读: 开运算与手写 erode→dilate 逐像素差 0，闭运算与 dilate→erode 差 0 —— 定义完全对得上。"
      "但 open3 与 close3 彼此差 255：**它们是两个方向相反的算子**，"
      "一个往「小」的方向修（侵蚀掉小的白结构），一个往「大」的方向修（填掉小的黑结构），不能互换")

assert px_morph == (48069, 47961, 48300)
assert d_op == 0 and d_cl == 0 and d_opcl == 255
'''

E5_CODE = '''# @@todo(5) 实战：开运算能不能把 400 个白噪点的轮廓数压回去？闭运算呢？
# @@hint 对 noisy 做 OPEN / CLOSE，分别数 RETR_EXTERNAL 轮廓数与前景像素数（都用 K3）
n_noisy = len(cv2.findContours(noisy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
noisy_op = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, K3)
noisy_cl = cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, K3)
noisy_res = (len(cv2.findContours(noisy_op, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]),
             int((noisy_op > 0).sum()),
             len(cv2.findContours(noisy_cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]),
             int((noisy_cl > 0).sum()))
# @@end
print("噪声图轮廓数 原:", n_noisy, "| open 后 (轮廓, 前景):", noisy_res[:2], "| close 后:", noisy_res[2:])
print("解读: 开运算用来**去掉比核小的白点**（400 个噪点把轮廓从 34 顶到 252，"
      "open3 后回到 28，比原始 cbw 的 34 还干净 —— 顺带把硬币内部的亮斑也修了）；"
      "闭运算用来**填掉比核小的黑洞**，但它同样会把孤立的噪点**连成一片**"
      "（close 后轮廓 224，前景反涨到 48625）。"
      "记住口诀：**开去白噪点、闭填黑洞**，用反了就是把噪声放大")

assert n_noisy == 252 and noisy_res == (28, 47965, 224, 48625)
'''

E6_CODE = '''plate = np.zeros((25, 25), np.uint8)
plate[5:20, 5:20] = 255
hole3 = plate.copy()
hole3[11:14, 11:14] = 0
hole5 = plate.copy()
hole5[10:15, 10:15] = 0
thin1 = np.zeros((60, 60), np.uint8)
thin1[30, 5:55] = 255
thin1[10:50, 30] = 255
thin5 = np.zeros((60, 60), np.uint8)
thin5[28:33, 5:55] = 255
thin5[10:50, 28:33] = 255
# @@todo(6) 闭运算能填多大的洞？开运算会干掉多细的线？（边界条件比结论重要）
# @@hint 洞的对照：3x3 洞 / 5x5 洞 分别用 close3 / close5 / close7，看前景能否回到 225（=plate 面积）
# @@hint 线的对照：1px 十字与 5px 十字各做一次 open3，看前景还剩多少
fill3 = (close_fill(hole3, 3), close_fill(hole3, 5), close_fill(hole3, 7))
fill5 = (close_fill(hole5, 3), close_fill(hole5, 5), close_fill(hole5, 7))
thin_px = (int((thin1 > 0).sum()), int((cv2.morphologyEx(thin1, cv2.MORPH_OPEN, K3) > 0).sum()),
           int((thin5 > 0).sum()), int((cv2.morphologyEx(thin5, cv2.MORPH_OPEN, K3) > 0).sum()))
# @@end
print("plate 前景:", int((plate > 0).sum()))
print("3x3 洞 + close3/5/7 后前景:", fill3)
print("5x5 洞 + close3/5/7 后前景:", fill5)
print("(1px 十字原, open3, 5px 十字原, open3):", thin_px)
print("解读: 结论很干脆 —— **洞的尺寸必须严格小于核尺寸才填得上**："
      "3x3 洞要 close5、5x5 洞要 close7，同尺寸的 close3/close5 都填不满。"
      "线也一样：1px 十字被 open3 **整条抹掉**（89 -> 0），5px 十字毫发无伤（425 -> 425）。"
      "所以形态学的所有「去噪」都是**按尺度筛结构**，没有一个既能保细节又只去噪声的核")

assert fill3 == (216, 225, 225) and fill5 == (200, 200, 225)
assert thin_px == (89, 0, 425, 425)
'''

E7_CODE = '''# @@todo(7) 三个派生算子：形态学梯度 / 顶帽 / 黑帽，分别等于哪两个基础运算的差？
# @@hint MORPH_GRADIENT / MORPH_TOPHAT / MORPH_BLACKHAT，再用 diff() 验证等式
mg = cv2.morphologyEx(cbw, cv2.MORPH_GRADIENT, K3)
th = cv2.morphologyEx(cbw, cv2.MORPH_TOPHAT, K3)
bh = cv2.morphologyEx(cbw, cv2.MORPH_BLACKHAT, K3)
px_ops = (int((mg > 0).sum()), int((th > 0).sum()), int((bh > 0).sum()))
d_mg = diff(mg, cv2.absdiff(cv2.dilate(cbw, K3), cv2.erode(cbw, K3)))
d_th = diff(th, cv2.subtract(cbw, op3))
d_bh = diff(bh, cv2.subtract(cl3, cbw))
th_max = int(th.max())
# @@end
print("(gradient, tophat, blackhat) 前景:", px_ops)
print("gradient == dilate - erode 差:", d_mg)
print("tophat == src - open 差:", d_th, "| blackhat == close - src 差:", d_bh)
print("tophat 最大值:", th_max)
print("解读: 三个都是「差出来的」——梯度 = 膨胀 - 腐蚀（前景的边界带，等于对二值图做了一遍轮廓描边）；"
      "顶帽 = 原图 - 开运算（留下被开运算抹掉的**亮的细结构**：噪点、细线、高光）；"
      "黑帽 = 闭运算 - 原图（留下被填掉的**暗的细结构**：裂纹、划痕、孔洞）。"
      "**顶帽最实用的地方是「光照校正」**：开运算 ≈ 低频背景，原图减掉它就只剩高频细节")

assert px_ops == (10423, 108, 231) and th_max == 255
assert d_mg == 0 and d_th == 0 and d_bh == 0
'''

E8_CODE = '''# @@todo(8) findContours 的三件套：返回值个数 / 检索模式 / 点近似方法
# @@hint 老版本返回 (image, contours, hierarchy) 三个值、新版本只有两个，先用 len() 确认你手上是几个
# @@hint 再数 RETR_EXTERNAL / RETR_LIST / RETR_TREE 的轮廓数，最后数三种 CHAIN_APPROX 的点总数
fc_ret = cv2.findContours(cl3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_ret = len(fc_ret)
cnts_ext, hier_ext = fc_ret
n_ext, hier_shape = len(cnts_ext), hier_ext.shape
n_list = len(cv2.findContours(cl3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
n_tree = len(cv2.findContours(cl3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
ext_none = cv2.findContours(cl3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
ext_simp = cv2.findContours(cl3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
ext_tc89 = cv2.findContours(cl3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[0]
pts = (sum(len(c) for c in ext_none), sum(len(c) for c in ext_simp), sum(len(c) for c in ext_tc89))
# @@end
print("cv2 版本:", cv2.__version__, "| findContours 返回元组长度:", n_ret)
print("RETR_EXTERNAL 轮廓数:", n_ext, "| hierarchy 形状:", hier_shape)
print("RETR_LIST:", n_list, "| RETR_TREE:", n_tree)
print("点总数 (APPROX_NONE, SIMPLE, TC89_L1):", pts)
print("解读: 1) OpenCV 3 返回 (image, contours, hierarchy) 3 个值，4/5 只返回 2 个 —— "
      "老教程里 `img, cnts, hier = cv2.findContours(...)` 在新版会直接 ValueError，"
      "跨版本代码要写 `cnts, hier = cv2.findContours(...)[-2:]`；"
      "2) hierarchy 恒为 (1, N, 4) 的 [next, prev, first_child, parent]；"
      "3) **APPROX_SIMPLE 省掉了共线的中间点**（3703 -> 1729 点，省 53%），"
      "算面积/周长时三种近似给出的答案是同一量级，但点数差 4 倍直接影响后续处理速度")

assert n_ret == 2 and n_ext == 30 and hier_shape == (1, 30, 4)
assert n_list == 33 and n_tree == 33 and pts == (3703, 1729, 833)
'''

E9_CODE = '''# @@todo(9) 轮廓的几何量：面积 / 周长 / 外接矩形 / 外接圆 / 凸包 / 多边形逼近
# @@hint 按面积排序取**第 2 大**（第 1 大是顶行与过曝背景粘连的块），它就是一枚独立硬币
# @@hint 周长用 cv2.arcLength(c, True)；圆度用 4πA/P²；凸性用 A / hull 面积
cs_size = sorted((c for c in cnts_ext if cv2.contourArea(c) > 1000), key=cv2.contourArea, reverse=True)
coin = cs_size[1]
c_bbox = cv2.boundingRect(coin)
c_area = round(cv2.contourArea(coin), 1)
c_fill = round(c_area / (c_bbox[2] * c_bbox[3]), 4)
c_per, c_per_open = round(cv2.arcLength(coin, True), 4), round(cv2.arcLength(coin, False), 4)
c_circ = round(4 * np.pi * c_area / c_per ** 2, 4)
(cx, cy), c_r = cv2.minEnclosingCircle(coin)
hull = cv2.convexHull(coin)
c_solidity = round(c_area / cv2.contourArea(hull), 4)
c_pt = tuple(len(cv2.approxPolyDP(coin, e * c_per, True)) for e in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05))
# @@end
print("这枚硬币的 bbox:", c_bbox, "| 矩形面积:", c_bbox[2] * c_bbox[3])
print("contourArea:", c_area, "| 填充率 area/(w*h):", c_fill)
print("arcLength closed:", c_per, "| open:", c_per_open, "| 圆度 4πA/P²:", c_circ)
print("minEnclosingCircle 圆心:", (round(cx, 2), round(cy, 2)), "半径:", round(c_r, 4))
print("凸包点数:", len(hull), "（原轮廓", len(coin), "点）| 凸性 solidity:", c_solidity)
print("approxPolyDP 点数 (eps=0.001/0.002/0.005/0.01/0.02/0.05 * 周长):", c_pt)
print("解读: 硬币是圆，所以填充率只有 0.7585（圆内切于边长 65x61 的矩形，π/4≈0.785），"
      "圆度 4πA/P² 0.8871 已经接近 1；solidity 0.9836 说明轮廓几乎全是凸的（缺口很小）。"
      "arcLength(closed=True) 比 False 多 6.0 —— 它把首尾闭合的那段也算进去了，算周长必须 True。"
      "**approxPolyDP 的 eps 是「能容忍的最大偏离」**：0.005*P（约 1 px）就把 90 个点压成 18 个，"
      "0.05*P 只剩 5 个 —— 这就是「用几个点描述一个形状」的精度-体积旋钮")

assert c_bbox == (315, 156, 65, 61) and c_area == 3007.5 and c_fill == 0.7585
assert c_per == 206.4092 and c_per_open == 200.4092 and c_circ == 0.8871
assert (round(cx, 2), round(cy, 2), round(c_r, 4)) == (347.21, 186.8, 32.4343)
assert len(hull) == 34 and len(coin) == 90 and c_solidity == 0.9836
assert c_pt == (90, 62, 18, 13, 8, 5)
'''

E10_CODE = '''# @@todo(10) connectedComponentsWithStats：连通域计数与轮廓计数对不对得上？
# @@hint 返回 (n, labels, stats, centroids)，n 把背景也算一个；面积列用 stats[:, cv2.CC_STAT_AREA]
n_cc, cc_lbl, cc_stats, cc_cent = cv2.connectedComponentsWithStats(cl3, 8)
cc_area = cc_stats[1:, cv2.CC_STAT_AREA]
cc_kept = int((cc_area > 1000).sum())
cc_top6 = sorted(cc_area.tolist(), reverse=True)[:6]
cc_first_bbox = cc_stats[1, :4].tolist()
# @@end
print("connectedComponentsWithStats 返回 n =", n_cc, "（含背景），所以前景连通域", n_cc - 1, "个")
print("面积 >1000 的连通域:", cc_kept,
      "| 而轮廓（面积>1000）也是", sum(1 for c in cnts_ext if cv2.contourArea(c) > 1000), "个")
print("面积前 6:", cc_top6)
print("第 1 个前景连通域的 bbox:", cc_first_bbox, "| 质心:", [round(v, 2) for v in cc_cent[1].tolist()])
print("解读: 两者**在这种「实心块、无嵌套」的图上是等价的**，但语义不同："
      "连通域给的是**像素级**的标签图（labels 可以逐像素索引）+ 现成的面积/外接框/质心；"
      "轮廓给的是**边界点序列**（能画、能算周长、能逼近多边形）。"
      "所以「只要个数和面积」用 connectedComponentsWithStats 更快更省内存，"
      "「要形状」才用 findContours。注意 n 含背景，别把 n 当成目标个数")

assert n_cc == 31 and cc_kept == 22
assert cc_top6 == [13319, 3097, 2665, 2423, 2136, 1949]
assert cc_first_bbox == [0, 0, 331, 79]
'''

E11_CODE = '''# @@todo(11) 综合：硬币到底几枚？—— 三种口径各数一遍，再和人工目检对账
# @@hint 口径 A：按面积阈值扫描；口径 B：先用顶帽（51x51 椭圆核）校正光照再数；口径 C：霍夫圆交叉验证
cs_all = sorted((c for c in cnts_ext), key=cv2.contourArea, reverse=True)
n_by_thr = tuple(sum(cv2.contourArea(c) > t for c in cs_all) for t in (10, 1000, 1500))
k_top = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
top51 = cv2.morphologyEx(cg, cv2.MORPH_TOPHAT, k_top)
top_bw = cv2.threshold(top51, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
top_mask = cv2.morphologyEx(top_bw, cv2.MORPH_CLOSE, K3)
top_cs = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
n_tophat = sum(1 for c in top_cs if cv2.contourArea(c) > 800)
cir = cv2.HoughCircles(cg, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=100, param2=25,
                       minRadius=15, maxRadius=45)
n_hough = int(cir.shape[1])
n_manual = 24
top_max, top_mean = int(top51.max()), round(float(top51.mean()), 4)
# @@end
print("口径 A（全局 Otsu + close3，面积阈值 >10 / >1000 / >1500）:", n_by_thr)
print("面积最大的块 bbox:", cv2.boundingRect(cs_all[0]), "面积:", round(cv2.contourArea(cs_all[0]), 1))
print("口径 B（顶帽 51x51 校正光照 + 面积>800）:", n_tophat, "| 顶帽图 max", top_max, "mean", top_mean)
print("口径 C（霍夫圆 param2=25）:", n_hough)
print("人工目检:", n_manual)
print("解读: 同一张图给出 22 / 23 / 24 三个数 —— 这就是「计数」的真相："
      "**数字是「预处理 + 形态学 + 面积阈值」三者的联合产物**。"
      "22 的根源是最大块 bbox(0,0,331,79)、面积 13113.5 ≈ 4 枚硬币：顶行硬币与过曝背景"
      "（图上方背景整体亮于 Otsu 阈值 104）粘成一片；顶帽把低频背景先减掉，就修正到 23；"
      "霍夫圆带「圆」的几何先验，给出正确的 24。"
      "**所以任何计数结论都要写清口径、并且必须目检**——只交一个数字、不交口径，"
      "在评审/答辩时是站不住的")

assert n_by_thr == (22, 22, 10)
assert (n_tophat, n_hough, n_manual) == (23, 24, 24)
assert top_max == 226 and top_mean == 47.2838
'''

# =========================================================================== #
# 讲解用代码块（与练习内容一一对应，去掉挖空标记）
# =========================================================================== #

S1_CODE = '''sq = np.zeros((9, 9), np.uint8)
sq[3:6, 3:6] = 255
line = np.zeros((9, 9), np.uint8)
line[:, 4] = 255
sq_px = (int((sq > 0).sum()), int((cv2.erode(sq, K3) > 0).sum()), int((cv2.dilate(sq, K3) > 0).sum()))
line_px = (int((line > 0).sum()), int((cv2.erode(line, K3) > 0).sum()), int((cv2.dilate(line, K3) > 0).sum()))
bb_sq, bb_er, bb_di = bbox_of(sq), bbox_of(cv2.erode(sq, K3)), bbox_of(cv2.dilate(sq, K3))
print("3x3 方块 (原, 腐蚀, 膨胀) 前景:", sq_px)
print("1px 竖线 (原, 腐蚀, 膨胀) 前景:", line_px)
print("方块外框: 原", bb_sq, "| 腐蚀后", bb_er, "| 膨胀后", bb_di)
print("腐蚀后剩下的像素坐标:", np.argwhere(cv2.erode(sq, K3) > 0).tolist())
print("解读: 3x3 方块腐蚀后就剩中心 1 个点（9 -> 1），膨胀成 5x5（25）—— 核 3x3 把边界各吃掉 1 圈。"
      "1px 细线的腐蚀结果是 **0**（整条消失），膨胀成 3x9=27。"
      "所以「腐蚀掉噪点」和「腐蚀掉细线」是同一个机制，选核大小就是在选你能忍受多细的损失")

assert sq_px == (9, 1, 25) and line_px == (9, 0, 27)
assert (bb_sq, bb_er, bb_di) == ((3, 3, 3, 3), (4, 4, 1, 1), (2, 2, 5, 5))
'''

S2_CODE = '''k_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k_ell3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k_cross3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
k_ell5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_rect5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
k_sizes = (int(k_rect3.sum()), int(k_ell3.sum()), int(k_ell5.sum()), int(k_rect5.sum()))
k_same = bool((k_ell3 == k_cross3).all())
di_ell = int((cv2.dilate(sq, k_ell3) > 0).sum())
print("核的 1 元素个数 (RECT3, ELLIPSE3, ELLIPSE5, RECT5):", k_sizes)
print("ELLIPSE3 与 CROSS3 逐元素相同?", k_same)
print("ELLIPSE5 行内元素数:", k_ell5.sum(axis=1).tolist())
print("方块用 ELLIPSE3 膨胀后面积:", di_ell, "（RECT3 是 25）")
print("解读: 3x3 时 ELLIPSE 退化成 CROSS（只有 5 个 1），比 RECT 的 9 个少 —— "
      "所以「膨胀 3x3」的默认含义取决于核形状，用 ELLIPSE 得到的是十字形邻域，"
      "方块只能长成 21 像素的「圆角方块」而不是 25。Ellipse 核在圆目标上更自然，"
      "但形状一变，同样的 ksize 结果就不可比")

assert k_sizes == (9, 5, 17, 25) and k_same is True and di_ell == 21
'''

S3_CODE = '''er_32 = cv2.erode(cbw, K3, iterations=2)
er_51 = cv2.erode(cbw, K5)
di_32 = cv2.dilate(cbw, K3, iterations=2)
di_51 = cv2.dilate(cbw, K5)
er_33 = cv2.erode(cbw, K3, iterations=3)
d_er2, d_di2 = diff(er_32, er_51), diff(di_32, di_51)
d_er3 = diff(er_33, cv2.erode(cbw, np.ones((7, 7), np.uint8)))
px_iter = (int((er_32 > 0).sum()), int((di_32 > 0).sum()), int((er_33 > 0).sum()))
print("erode 3x3x2 vs 5x5x1: 前景", int((er_32 > 0).sum()), "vs", int((er_51 > 0).sum()), "| 逐像素差", d_er2)
print("dilate 3x3x2 vs 5x5x1: 前景", int((di_32 > 0).sum()), "vs", int((di_51 > 0).sum()), "| 逐像素差", d_di2)
print("erode 3x3x3 vs 7x7x1 逐像素差:", d_er3)
print("(er_32, di_32, er_33) 前景:", px_iter)
print("解读: 矩形核下 **迭代 2 次 3x3 == 一次 5x5**，3 次 == 一次 7x7（差全为 0）—— "
      "因为 5x5 方块 = 3x3 方块 ⊕ 3x3 方块，而「核的膨胀」对应「运算的叠加」。"
      "实战价值：想开很大的核，用 iterations 比造大核便宜（大核是 O(k²) 元素，"
      "迭代是 O(n) 次 3x3）。但换成 ELLIPSE 核这个等式就不成立了")

assert d_er2 == 0 and d_di2 == 0 and d_er3 == 0
assert px_iter == (37867, 58379, 33150)
'''

S4_CODE = '''op3 = cv2.morphologyEx(cbw, cv2.MORPH_OPEN, K3)
cl3 = cv2.morphologyEx(cbw, cv2.MORPH_CLOSE, K3)
d_op = diff(op3, cv2.dilate(cv2.erode(cbw, K3), K3))
d_cl = diff(cl3, cv2.erode(cv2.dilate(cbw, K3), K3))
px_morph = (int((cbw > 0).sum()), int((op3 > 0).sum()), int((cl3 > 0).sum()))
d_opcl = diff(op3, cl3)
print("(cbw, open3, close3) 前景:", px_morph)
print("open3 == dilate(erode) 逐像素差:", d_op, "| close3 == erode(dilate) 逐像素差:", d_cl)
print("open3 与 close3 的最大差:", d_opcl)
print("解读: 开运算与手写 erode→dilate 逐像素差 0，闭运算与 dilate→erode 差 0 —— 定义完全对得上。"
      "但 open3 与 close3 彼此差 255：**它们是两个方向相反的算子**，"
      "一个往「小」的方向修（侵蚀掉小的白结构），一个往「大」的方向修（填掉小的黑结构），不能互换")

assert px_morph == (48069, 47961, 48300)
assert d_op == 0 and d_cl == 0 and d_opcl == 255
'''

S5_CODE = '''n_noisy = len(cv2.findContours(noisy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
noisy_op = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, K3)
noisy_cl = cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, K3)
noisy_res = (len(cv2.findContours(noisy_op, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]),
             int((noisy_op > 0).sum()),
             len(cv2.findContours(noisy_cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]),
             int((noisy_cl > 0).sum()))
print("噪声图轮廓数 原:", n_noisy, "| open 后 (轮廓, 前景):", noisy_res[:2], "| close 后:", noisy_res[2:])
print("解读: 开运算用来**去掉比核小的白点**（400 个噪点把轮廓从 34 顶到 252，"
      "open3 后回到 28，比原始 cbw 的 34 还干净 —— 顺带把硬币内部的亮斑也修了）；"
      "闭运算用来**填掉比核小的黑洞**，但它同样会把孤立的噪点**连成一片**"
      "（close 后轮廓 224，前景反涨到 48625）。"
      "记住口诀：**开去白噪点、闭填黑洞**，用反了就是把噪声放大")

assert n_noisy == 252 and noisy_res == (28, 47965, 224, 48625)
'''

S6_CODE = '''plate = np.zeros((25, 25), np.uint8)
plate[5:20, 5:20] = 255
hole3 = plate.copy()
hole3[11:14, 11:14] = 0
hole5 = plate.copy()
hole5[10:15, 10:15] = 0
thin1 = np.zeros((60, 60), np.uint8)
thin1[30, 5:55] = 255
thin1[10:50, 30] = 255
thin5 = np.zeros((60, 60), np.uint8)
thin5[28:33, 5:55] = 255
thin5[10:50, 28:33] = 255
fill3 = (close_fill(hole3, 3), close_fill(hole3, 5), close_fill(hole3, 7))
fill5 = (close_fill(hole5, 3), close_fill(hole5, 5), close_fill(hole5, 7))
thin_px = (int((thin1 > 0).sum()), int((cv2.morphologyEx(thin1, cv2.MORPH_OPEN, K3) > 0).sum()),
           int((thin5 > 0).sum()), int((cv2.morphologyEx(thin5, cv2.MORPH_OPEN, K3) > 0).sum()))
print("plate 前景:", int((plate > 0).sum()))
print("3x3 洞 + close3/5/7 后前景:", fill3)
print("5x5 洞 + close3/5/7 后前景:", fill5)
print("(1px 十字原, open3, 5px 十字原, open3):", thin_px)
print("解读: 结论很干脆 —— **洞的尺寸必须严格小于核尺寸才填得上**："
      "3x3 洞要 close5、5x5 洞要 close7，同尺寸的 close3/close5 都填不满。"
      "线也一样：1px 十字被 open3 **整条抹掉**（89 -> 0），5px 十字毫发无伤（425 -> 425）。"
      "所以形态学的所有「去噪」都是**按尺度筛结构**，没有一个既能保细节又只去噪声的核")

assert fill3 == (216, 225, 225) and fill5 == (200, 200, 225)
assert thin_px == (89, 0, 425, 425)
'''

S7_CODE = '''mg = cv2.morphologyEx(cbw, cv2.MORPH_GRADIENT, K3)
th = cv2.morphologyEx(cbw, cv2.MORPH_TOPHAT, K3)
bh = cv2.morphologyEx(cbw, cv2.MORPH_BLACKHAT, K3)
px_ops = (int((mg > 0).sum()), int((th > 0).sum()), int((bh > 0).sum()))
d_mg = diff(mg, cv2.absdiff(cv2.dilate(cbw, K3), cv2.erode(cbw, K3)))
d_th = diff(th, cv2.subtract(cbw, op3))
d_bh = diff(bh, cv2.subtract(cl3, cbw))
th_max = int(th.max())
print("(gradient, tophat, blackhat) 前景:", px_ops)
print("gradient == dilate - erode 差:", d_mg)
print("tophat == src - open 差:", d_th, "| blackhat == close - src 差:", d_bh)
print("tophat 最大值:", th_max)
print("解读: 三个都是「差出来的」——梯度 = 膨胀 - 腐蚀（前景的边界带，等于对二值图做了一遍轮廓描边）；"
      "顶帽 = 原图 - 开运算（留下被开运算抹掉的**亮的细结构**：噪点、细线、高光）；"
      "黑帽 = 闭运算 - 原图（留下被填掉的**暗的细结构**：裂纹、划痕、孔洞）。"
      "**顶帽最实用的地方是「光照校正」**：开运算 ≈ 低频背景，原图减掉它就只剩高频细节")

assert px_ops == (10423, 108, 231) and th_max == 255
assert d_mg == 0 and d_th == 0 and d_bh == 0
'''

S8_CODE = '''fc_ret = cv2.findContours(cl3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_ret = len(fc_ret)
cnts_ext, hier_ext = fc_ret
n_ext, hier_shape = len(cnts_ext), hier_ext.shape
n_list = len(cv2.findContours(cl3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
n_tree = len(cv2.findContours(cl3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
ext_none = cv2.findContours(cl3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
ext_simp = cv2.findContours(cl3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
ext_tc89 = cv2.findContours(cl3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[0]
pts = (sum(len(c) for c in ext_none), sum(len(c) for c in ext_simp), sum(len(c) for c in ext_tc89))
print("cv2 版本:", cv2.__version__, "| findContours 返回元组长度:", n_ret)
print("RETR_EXTERNAL 轮廓数:", n_ext, "| hierarchy 形状:", hier_shape)
print("RETR_LIST:", n_list, "| RETR_TREE:", n_tree)
print("点总数 (APPROX_NONE, SIMPLE, TC89_L1):", pts)
print("解读: 1) OpenCV 3 返回 (image, contours, hierarchy) 3 个值，4/5 只返回 2 个 —— "
      "老教程里 `img, cnts, hier = cv2.findContours(...)` 在新版会直接 ValueError，"
      "跨版本代码要写 `cnts, hier = cv2.findContours(...)[-2:]`；"
      "2) hierarchy 恒为 (1, N, 4) 的 [next, prev, first_child, parent]；"
      "3) **APPROX_SIMPLE 省掉了共线的中间点**（3703 -> 1729 点，省 53%），"
      "算面积/周长时三种近似给出的答案是同一量级，但点数差 4 倍直接影响后续处理速度")

assert n_ret == 2 and n_ext == 30 and hier_shape == (1, 30, 4)
assert n_list == 33 and n_tree == 33 and pts == (3703, 1729, 833)
'''

S9_CODE = '''cs_size = sorted((c for c in cnts_ext if cv2.contourArea(c) > 1000), key=cv2.contourArea, reverse=True)
coin = cs_size[1]
c_bbox = cv2.boundingRect(coin)
c_area = round(cv2.contourArea(coin), 1)
c_fill = round(c_area / (c_bbox[2] * c_bbox[3]), 4)
c_per, c_per_open = round(cv2.arcLength(coin, True), 4), round(cv2.arcLength(coin, False), 4)
c_circ = round(4 * np.pi * c_area / c_per ** 2, 4)
(cx, cy), c_r = cv2.minEnclosingCircle(coin)
hull = cv2.convexHull(coin)
c_solidity = round(c_area / cv2.contourArea(hull), 4)
c_pt = tuple(len(cv2.approxPolyDP(coin, e * c_per, True)) for e in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05))
print("这枚硬币的 bbox:", c_bbox, "| 矩形面积:", c_bbox[2] * c_bbox[3])
print("contourArea:", c_area, "| 填充率 area/(w*h):", c_fill)
print("arcLength closed:", c_per, "| open:", c_per_open, "| 圆度 4πA/P²:", c_circ)
print("minEnclosingCircle 圆心:", (round(cx, 2), round(cy, 2)), "半径:", round(c_r, 4))
print("凸包点数:", len(hull), "（原轮廓", len(coin), "点）| 凸性 solidity:", c_solidity)
print("approxPolyDP 点数 (eps=0.001/0.002/0.005/0.01/0.02/0.05 * 周长):", c_pt)
print("解读: 硬币是圆，所以填充率只有 0.7585（圆内切于边长 65x61 的矩形，π/4≈0.785），"
      "圆度 4πA/P² 0.8871 已经接近 1；solidity 0.9836 说明轮廓几乎全是凸的（缺口很小）。"
      "arcLength(closed=True) 比 False 多 6.0 —— 它把首尾闭合的那段也算进去了，算周长必须 True。"
      "**approxPolyDP 的 eps 是「能容忍的最大偏离」**：0.005*P（约 1 px）就把 90 个点压成 18 个，"
      "0.05*P 只剩 5 个 —— 这就是「用几个点描述一个形状」的精度-体积旋钮")

assert c_bbox == (315, 156, 65, 61) and c_area == 3007.5 and c_fill == 0.7585
assert c_per == 206.4092 and c_per_open == 200.4092 and c_circ == 0.8871
assert (round(cx, 2), round(cy, 2), round(c_r, 4)) == (347.21, 186.8, 32.4343)
assert len(hull) == 34 and len(coin) == 90 and c_solidity == 0.9836
assert c_pt == (90, 62, 18, 13, 8, 5)
'''

S10_CODE = '''n_cc, cc_lbl, cc_stats, cc_cent = cv2.connectedComponentsWithStats(cl3, 8)
cc_area = cc_stats[1:, cv2.CC_STAT_AREA]
cc_kept = int((cc_area > 1000).sum())
cc_top6 = sorted(cc_area.tolist(), reverse=True)[:6]
cc_first_bbox = cc_stats[1, :4].tolist()
print("connectedComponentsWithStats 返回 n =", n_cc, "（含背景），所以前景连通域", n_cc - 1, "个")
print("面积 >1000 的连通域:", cc_kept,
      "| 而轮廓（面积>1000）也是", sum(1 for c in cnts_ext if cv2.contourArea(c) > 1000), "个")
print("面积前 6:", cc_top6)
print("第 1 个前景连通域的 bbox:", cc_first_bbox, "| 质心:", [round(v, 2) for v in cc_cent[1].tolist()])
print("解读: 两者**在这种「实心块、无嵌套」的图上是等价的**，但语义不同："
      "连通域给的是**像素级**的标签图（labels 可以逐像素索引）+ 现成的面积/外接框/质心；"
      "轮廓给的是**边界点序列**（能画、能算周长、能逼近多边形）。"
      "所以「只要个数和面积」用 connectedComponentsWithStats 更快更省内存，"
      "「要形状」才用 findContours。注意 n 含背景，别把 n 当成目标个数")

assert n_cc == 31 and cc_kept == 22
assert cc_top6 == [13319, 3097, 2665, 2423, 2136, 1949]
assert cc_first_bbox == [0, 0, 331, 79]
'''

S11_CODE = '''cs_all = sorted((c for c in cnts_ext), key=cv2.contourArea, reverse=True)
n_by_thr = tuple(sum(cv2.contourArea(c) > t for c in cs_all) for t in (10, 1000, 1500))
k_top = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
top51 = cv2.morphologyEx(cg, cv2.MORPH_TOPHAT, k_top)
top_bw = cv2.threshold(top51, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
top_mask = cv2.morphologyEx(top_bw, cv2.MORPH_CLOSE, K3)
top_cs = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
n_tophat = sum(1 for c in top_cs if cv2.contourArea(c) > 800)
cir = cv2.HoughCircles(cg, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=100, param2=25,
                       minRadius=15, maxRadius=45)
n_hough = int(cir.shape[1])
n_manual = 24
top_max, top_mean = int(top51.max()), round(float(top51.mean()), 4)
print("口径 A（全局 Otsu + close3，面积阈值 >10 / >1000 / >1500）:", n_by_thr)
print("面积最大的块 bbox:", cv2.boundingRect(cs_all[0]), "面积:", round(cv2.contourArea(cs_all[0]), 1))
print("口径 B（顶帽 51x51 校正光照 + 面积>800）:", n_tophat, "| 顶帽图 max", top_max, "mean", top_mean)
print("口径 C（霍夫圆 param2=25）:", n_hough)
print("人工目检:", n_manual)
print("解读: 同一张图给出 22 / 23 / 24 三个数 —— 这就是「计数」的真相："
      "**数字是「预处理 + 形态学 + 面积阈值」三者的联合产物**。"
      "22 的根源是最大块 bbox(0,0,331,79)、面积 13113.5 ≈ 4 枚硬币：顶行硬币与过曝背景"
      "（图上方背景整体亮于 Otsu 阈值 104）粘成一片；顶帽把低频背景先减掉，就修正到 23；"
      "霍夫圆带「圆」的几何先验，给出正确的 24。"
      "**所以任何计数结论都要写清口径、并且必须目检**——只交一个数字、不交口径，"
      "在评审/答辩时是站不住的")

assert n_by_thr == (22, 22, 10)
assert (n_tophat, n_hough, n_manual) == (23, 24, 24)
assert top_max == 226 and top_mean == 47.2838
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch06 形态学与轮廓

> 数据：`classic/coins.png`（全局 Otsu 阈值 104，前景 48069 px），外加合成小图做对照。
> 真值全部实跑（opencv 5.0.0）。

**本章考点**

1. **腐蚀 / 膨胀的几何本质**：邻域取 min / max，1px 结构会被直接啃光
2. **结构元形状**（`RECT` / `ELLIPSE` / `CROSS`）——3×3 时 ELLIPSE 退化成 CROSS
3. **迭代 n 次 3×3 ≡ 一次 (2n+1)×(2n+1) 矩形核**（逐像素差 0）
4. **开运算去白噪点 / 闭运算填黑洞**，以及用反了的后果
5. 开闭的**尺度边界**：洞必须严格小于核才填得上；细线小于核就直接消失
6. **形态学梯度 / 顶帽 / 黑帽**三者的差式定义，顶帽做光照校正
7. **`findContours` 三件套**：返回值版本坑 / `RETR_*` 检索模式 / `CHAIN_APPROX_*` 点数
8. **轮廓几何量**：面积、周长（`closed=True`）、填充率、圆度、外接圆、凸包、`approxPolyDP`
9. **连通域 vs 轮廓**：像素标签 vs 边界点序列，什么时候用哪个
10. **综合：硬币计数**——同一张图能数出 22 / 23 / 24，数字必须连着口径一起汇报

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| 1px 结构消失 | 腐蚀 3×3 直接把细线抹掉（9 → 0） | §6.1 |
| 结构元形状 | 3×3 ELLIPSE == CROSS，膨胀只长 21 px 不是 25 | §6.2 |
| 迭代 vs 大核 | 只在矩形核下等价，换 ELLIPSE 就不成立 | §6.3 |
| 开闭用反 | 闭运算会把噪点连成片（224 轮廓且前景变多） | §6.5 |
| 洞的尺度 | 3×3 洞必须用 close5，同尺寸核填不满 | §6.6 |
| `findContours` 返回值 | 3 值 vs 2 值，跨版本必炸 | §6.8 |
| `arcLength` 的 closed | 不传 True 少算 6.0（闭合段） | §6.9 |
| 计数口径 | 22 / 23 / 24 三个数都"对" | §6.11 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 6.1 形态学的两个原子操作：腐蚀与膨胀

形态学只有两个原子操作：**腐蚀**（邻域取 min）与**膨胀**（邻域取 max）。

| 结构 | 原前景 | `erode(K3)` | `dilate(K3)` | 外框（原 → 腐蚀 → 膨胀） |
|---|---|---|---|---|
| 3×3 实心方块 | 9 | **1**（只剩中心 (4,4)） | **25**（5×5） | (3,3,3,3) → (4,4,1,1) → (2,2,5,5) |
| 1px 竖线 | 9 | **0**（整条消失） | **27**（3×9） | — |

一句话记住：**核 3×3 就把边界各吃掉/长出 1 圈**。所以「腐蚀去噪点」和
「腐蚀掉细线」是同一个机制 —— 选核尺寸，本质是在选「你能忍受多细的结构被牺牲」。
"""
    ),
    code(S1_CODE),
    md(
        """
## 6.2 结构元的形状也是旋钮

`getStructuringElement(shape, ksize)` 的 `shape` 决定核的有效面积：

| 结构元 | 1 元素个数 | 说明 |
|---|---|---|
| `RECT (3,3)` | **9** | 全 1 |
| `ELLIPSE (3,3)` | **5** | 退化成十字！与 `CROSS (3,3)` **逐元素相同** |
| `ELLIPSE (5,5)` | **17** | 行内元素数 `[1, 5, 5, 5, 1]` |
| `RECT (5,5)` | **25** | 全 1 |

后果是实打实的：同一个 3×3 方块用 `ELLIPSE3` 膨胀只得 **21** 像素（圆角方块），
用 `RECT3` 是 **25**。**ksize 相同但 shape 不同，结果不可比。**
"""
    ),
    code(S2_CODE),
    md(
        """
## 6.3 迭代 n 次 ≡ 一次大核

在**矩形核**下，`erode(cbw, K3, iterations=2)` 与 `erode(cbw, K5)` 逐像素差 **0**；
`iterations=3` 与 7×7 全 1 核也差 **0**：

| 组合 | 前景 | 逐像素最大差 |
|---|---|---|
| erode 3×3×2 vs 5×5×1 | 37867 vs 37867 | **0** |
| dilate 3×3×2 vs 5×5×1 | 58379 vs 58379 | **0** |
| erode 3×3×3 vs 7×7×1 | 33150 vs 33150 | **0** |

数学上因为 `5×5 方块 = 3×3 方块 ⊕ 3×3 方块`，而「核的膨胀」恰好对应「运算的叠加」。
工程意义：**开大核时用 `iterations` 比造大核便宜**（大核 O(k²) 个元素，迭代是 n 次 3×3）。
换成 `ELLIPSE` 核，这个等式就不成立了。
"""
    ),
    code(S3_CODE),
    md(
        """
## 6.4 开运算与闭运算

```
开运算 open  = 先腐蚀后膨胀   →  去掉比核小的白（亮）结构
闭运算 close = 先膨胀后腐蚀   →  填掉比核小的黑（暗）结构
```

| 掩码 | 前景 |
|---|---|
| `cbw` | 48069 |
| `open3`（= `dilate(erode)`，差 0） | 47961 |
| `close3`（= `erode(dilate)`，差 0） | 48300 |

`open3` 与 `close3` 彼此差 **255** —— 它们是**两个方向相反**的算子，
一个往「小」的方向修，一个往「大」的方向修，**不能互换**。
"""
    ),
    code(S4_CODE),
    md(
        """
## 6.5 实战：400 个白噪点

| 掩码 | 外轮廓数 | 前景像素 |
|---|---|---|
| 原 `cbw` | 34 | 48069 |
| 加 400 个白噪点（`noisy`） | **252** | 48306 |
| 噪点 + `open3` | **28** | 47965 |
| 噪点 + `close3` | 224 | **48625** |

开运算把噪点清得比原始 `cbw` 还干净（28 < 34，顺带修掉了硬币内部的亮斑）；
闭运算则把孤立噪点**连成片** —— 轮廓数降了，但前景反而变多。

> **口诀：开去白噪点、闭填黑洞。用反了就是把噪声放大。**
"""
    ),
    code(S5_CODE),
    md(
        """
## 6.6 开闭的尺度边界

「闭运算能填洞」是有条件的。在白方块上挖正方形洞，用不同尺寸的核去填：

| 洞尺寸 | close3 | close5 | close7 |
|---|---|---|---|
| 3×3 | 216（填不满） | **225 ✅** | 225 ✅ |
| 5×5 | 200 | 200（填不满） | **225 ✅** |

**规则：洞的尺寸必须严格小于核尺寸。** 3×3 洞要 close5，5×5 洞要 close7。

线也一样，而且更凶：

| 线宽 | 原前景 | `open3` 后 |
|---|---|---|
| 1px 十字 | 89 | **0**（整条抹掉） |
| 5px 十字 | 425 | 425（毫发无伤） |

所以形态学的「去噪」全是**按尺度筛结构**；不存在「既保细节又只去噪声」的核。
"""
    ),
    code(S6_CODE),
    md(
        """
## 6.7 形态学梯度 / 顶帽 / 黑帽

三个派生算子全是**差出来的**，用 `diff()` 验证等式逐像素差都是 **0**：

| 算子 | 定义 | 留下什么 | 前景像素 |
|---|---|---|---|
| `MORPH_GRADIENT` | `dilate - erode` | 前景的**边界带**（等效描边） | 10423 |
| `MORPH_TOPHAT` | `src - open` | 被开运算抹掉的**亮细结构** | 108 |
| `MORPH_BLACKHAT` | `close - src` | 被闭运算填掉的**暗细结构** | 231 |

**顶帽最实用的用法是「光照校正」**：开运算 ≈ 低频背景（大核），原图减掉它
就只剩高频细节 —— §6.11 就靠它把硬币从「过曝背景粘连」里救回来。
"""
    ),
    code(S7_CODE),
    md(
        """
## 6.8 `findContours` 的三件套

**① 返回值版本坑**

| 版本 | 返回 |
|---|---|
| OpenCV 3 | `(image, contours, hierarchy)` **3 个值** |
| OpenCV 4 / 5（本仓 5.0.0） | `(contours, hierarchy)` **2 个值** |

老教程里的 `img, cnts, hier = cv2.findContours(...)` 在新版会直接 `ValueError`；
跨版本写法是 `cnts, hier = cv2.findContours(...)[-2:]`。
`hierarchy` 形状恒为 `(1, N, 4)`，每行是 `[next, prev, first_child, parent]`。

**② 检索模式**

| 模式 | 轮廓数（`cl3`） | 含义 |
|---|---|---|
| `RETR_EXTERNAL` | **30** | 只要最外层 |
| `RETR_LIST` | 33 | 全部，不分层级 |
| `RETR_TREE` | 33 | 全部 + 完整父子树 |

（原始 `cbw` 上 EXTERNAL 34 vs LIST/TREE 61 —— 差出来的 27 个正是硬币内部的**孔洞**轮廓。）

**③ 点近似方法**

| 方法 | 点总数 |
|---|---|
| `CHAIN_APPROX_NONE` | 3703 |
| `CHAIN_APPROX_SIMPLE` | **1729**（省掉共线中间点，省 53%） |
| `CHAIN_APPROX_TC89_L1` | 833 |

面积/周长对三种近似是同一量级，但点数差 4 倍 —— 直接影响后续处理速度。
"""
    ),
    code(S8_CODE),
    md(
        """
## 6.9 轮廓的几何量

拿一枚独立硬币（面积第 2 大的轮廓，第 1 大是粘连块）：

| 量 | API | 真值 |
|---|---|---|
| 外接矩形 | `boundingRect` | `(315, 156, 65, 61)`，矩形面积 3965 |
| 面积 | `contourArea` | **3007.5** |
| 填充率 | `area / (w*h)` | **0.7585** |
| 周长（闭合） | `arcLength(c, True)` | **206.4092** |
| 周长（不闭合） | `arcLength(c, False)` | 200.4092（**少 6.0**） |
| 圆度 | `4πA / P²` | **0.8871** |
| 最小外接圆 | `minEnclosingCircle` | 圆心 (347.21, 186.8)，r **32.4343** |
| 凸包 | `convexHull` | **34** 点（原轮廓 90 点） |
| 凸性 | `area / hull_area` | **0.9836** |

`approxPolyDP` 的 `eps` 是「能容忍的最大偏离」，实测曲线：

| `eps`（× 周长） | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|---|---|
| 点数 | 90 | 62 | **18** | 13 | 8 | **5** |

`0.005 * P`（约 1 px）就把 90 个点压成 18 个 —— 这就是「几个点描述一个形状」的旋钮。
"""
    ),
    code(S9_CODE),
    md(
        """
## 6.10 连通域 vs 轮廓

`connectedComponentsWithStats` 返回 `(n, labels, stats, centroids)`，**`n` 含背景**
（本图 31 → 前景 30 个）。

| 口径 | 个数 | 面积前 6 |
|---|---|---|
| 轮廓（`area > 1000`） | 22 | 13113.5 / 3007.5 / 2582 / 2344 / 2088 / 1877.5 |
| 连通域（`area > 1000`） | **22** | 13319 / 3097 / 2665 / 2423 / 2136 / 1949 |

面积极接近但不完全相同（**轮廓面积用格林公式算边界，连通域面积直接数像素**）。

- 只要**个数 / 面积 / 外接框 / 质心** → 用 `connectedComponentsWithStats`（更快、更省内存）
- 要**形状**（画轮廓、算周长、多边形逼近、凸包） → 用 `findContours`

**别把 `n` 当成目标个数**（它含背景）。
"""
    ),
    code(S10_CODE),
    md(
        """
## 6.11 综合：硬币计数——口径决定数字

同一张 `coins.png`，三种口径给出三个答案：

| 口径 | 做法 | 结果 |
|---|---|---|
| A | 全局 Otsu + `close3` + 面积过滤 | **22**（>10 / >1000 都是 22；>1500 只剩 10） |
| B | **顶帽 51×51 校正光照** + Otsu + `close3` + `area > 800` | **23** |
| C | 霍夫圆（`param2=25`, `minDist=25`, r∈[15,45]） | **24** |
| — | **人工目检** | **24** |

口径 A 为什么少？最大那块 `bbox = (0,0,331,79)`、面积 **13113.5** ≈ 4 枚硬币 ——
图的**上方背景整体亮于 Otsu 阈值 104**（照度不均），顶行硬币与背景粘成一片，
按面积过滤时它们被当成「一个目标」。

结论（也是本章最该带走的一句）：

> **「计数」的数字是「预处理 + 形态学 + 面积阈值」三者的联合产物。**
> 报数的时候必须连着口径一起报，而且**必须目检**。

在生产里这就对应着两件事：① 用顶帽 / 自适应阈值先解决照度不均；
② 用形状先验（圆度、solidity、长宽比）替代纯面积过滤。
"""
    ),
    code(S11_CODE),
    md(
        """
## 小结

1. 腐蚀 = 邻域 min、膨胀 = 邻域 max；3×3 核把边界各动 1 圈。
2. **1px 结构会被 `erode(K3)` 整条抹掉**（9 → 0）——去噪与丢细节是同一件事。
3. `ELLIPSE(3,3)` == `CROSS(3,3)`（5 个元素），膨胀方块只长到 21 而不是 25。
4. 矩形核下 **迭代 2 次 3×3 == 5×5**，3 次 == 7×7（差 0）；换成 ELLIPSE 不等价。
5. **开去白噪点**（252 → 28 轮廓）、**闭填黑洞**，但闭会把孤立噪点连成片。
6. **洞的尺寸必须严格小于核尺寸**才填得上；细线小于核就直接消失。
7. `gradient = dilate - erode`、`tophat = src - open`、`blackhat = close - src`（差全为 0）。
8. 顶帽 51×51 是最实用的**光照校正**手段。
9. `findContours` 在 OpenCV 4/5 只返回 **2** 个值；`hierarchy` 形状 `(1, N, 4)`。
10. `CHAIN_APPROX_SIMPLE` 比 `NONE` 省 53% 的点；算面积/周长结果同量级。
11. `arcLength(c, True)` 才是周长；`closed=False` 会少算闭合段（6.0）。
12. 圆度 `4πA/P²`、凸性 `area/hull_area` 是判定「像不像圆/有没有缺角」的两把尺子。
13. 只要个数和面积用 `connectedComponentsWithStats`；要形状才用 `findContours`。
14. **计数必须连着口径汇报**：同一张图 22 / 23 / 24 都能"算"出来，目检是 24。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch06 形态学与轮廓（练习版）

> 按提示补全 `____`，跑通所有 assert。`coins` / `cg` / `cbw`（全局 Otsu 二值图）/
> `noisy`（400 个白噪点）已在 setup 里备好；`K3` / `K5` / `ascii_bin` / `biggest` /
> `bbox_of` / `diff` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；控制流写在挖空块之外。
> 题目真值都来自实跑（opencv 5.0.0），照提示写就能对上。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md("## 任务 1：腐蚀 / 膨胀的几何\n\n方块、细线、外框各变化多少？"),
    code(E1_CODE),
    md("## 任务 2：结构元的形状\n\n`RECT` / `ELLIPSE` / `CROSS` 的有效面积。"),
    code(E2_CODE),
    md("## 任务 3：迭代 n 次 ≡ 一次大核\n\n矩形核下的等价性，要逐像素为 0 的证据。"),
    code(E3_CODE),
    md("## 任务 4：开运算与闭运算\n\n先与手写组合对账，再看两者之间的差异。"),
    code(E4_CODE),
    md("## 任务 5：实战 400 个白噪点\n\n开运算 vs 闭运算，谁的轮廓数更低、谁的前景更多？"),
    code(E5_CODE),
    md("## 任务 6：开闭的尺度边界\n\n洞要多小才填得上？线要多粗才不被抹掉？"),
    code(E6_CODE),
    md("## 任务 7：梯度 / 顶帽 / 黑帽\n\n三个差式定义 + 等式验证。"),
    code(E7_CODE),
    md("## 任务 8：`findContours` 三件套\n\n返回值个数 / 检索模式 / 点近似方法。"),
    code(E8_CODE),
    md("## 任务 9：轮廓的几何量\n\n取一枚独立硬币，量面积、周长、圆度、凸性、`approxPolyDP`。"),
    code(E9_CODE),
    md("## 任务 10：连通域 vs 轮廓\n\n两种口径能不能对上？差在哪？"),
    code(E10_CODE),
    md("## 任务 11：综合——硬币到底几枚\n\n三种口径各数一遍，再和目检对账。"),
    code(E11_CODE),
    md(
        """
## 自查清单

- [ ] 能说出腐蚀 / 膨胀各是邻域的 min 还是 max
- [ ] 知道为什么 1px 细线会被 `erode(K3)` 抹掉
- [ ] 记得 `ELLIPSE(3,3)` 就是 `CROSS(3,3)`，膨胀方块只长 21 px
- [ ] 能解释「迭代 2 次 3×3 == 5×5」成立的前提
- [ ] 能用一句话说清开运算和闭运算各干什么、用反了什么后果
- [ ] 知道闭运算填洞的尺寸条件（严格小于核）
- [ ] 能默写 `gradient / tophat / blackhat` 的差式定义
- [ ] 知道顶帽怎么用来做光照校正
- [ ] 记得 `findContours` 在 OpenCV 4/5 返回 2 个值
- [ ] 说得清 `RETR_EXTERNAL` 与 `RETR_TREE` 差在哪些轮廓
- [ ] 知道算周长必须 `arcLength(c, True)`
- [ ] 能背出圆度与凸性的公式
- [ ] 知道「个数/面积」用连通域、「形状」用轮廓
- [ ] 能解释为什么同一张图能数出 22 / 23 / 24
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
