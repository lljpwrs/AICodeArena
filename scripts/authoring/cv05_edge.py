#!/usr/bin/env python3
"""cv05 边缘检测与梯度：Sobel CV_64F 坑 / 梯度幅值与方向 / Scharr / Laplacian / Canny 双阈值。

数据：classic/home.jpg、classic/camera.png、classic/coins.png、helmet/site_03.jpg（灰度）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv05_edge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch05_edge"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

DATA = Path("data")
CL = DATA / "classic"
HE = DATA / "helmet"
'''

SETUP = '''gray = cv2.cvtColor(cv2.imread(str(CL / "home.jpg")), cv2.COLOR_BGR2GRAY)
cam = cv2.cvtColor(cv2.imread(str(CL / "camera.png")), cv2.COLOR_BGR2GRAY)
coins = cv2.cvtColor(cv2.imread(str(CL / "coins.png")), cv2.COLOR_BGR2GRAY)
s3 = cv2.cvtColor(cv2.imread(str(HE / "site_03.jpg")), cv2.COLOR_BGR2GRAY)
print("home", gray.shape, "| camera", cam.shape, "| coins", coins.shape, "| site_03", s3.shape)
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float64)
SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float64)


def correlate_edge_pad(img, k):
    """边缘复制补边 + 逐位置点积（复刻 cv2 的 BORDER_REPLICATE 口径）。"""
    r = k.shape[0] // 2
    pad = np.pad(np.asarray(img, np.float64), r, mode="edge")
    h, w = img.shape
    out = np.zeros((h, w), np.float64)
    for i in range(h):
        for j in range(w):
            out[i, j] = float((pad[i:i + k.shape[0], j:j + k.shape[1]] * k).sum())
    return out


def max_abs_diff(a, b):
    return round(float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()), 6)


print("脚手架就绪：SOBEL_X / SOBEL_Y / SCHARR_X / correlate_edge_pad / max_abs_diff")'''

# =========================================================================== #
# 练习（答案版内容，practice 由 builder 自动挖空）
# =========================================================================== #

E1_CODE = '''# @@todo(1) Sobel 的 ddepth 坑：不写 CV_64F，负梯度会被整片截成 0
# @@hint 直接 cv2.Sobel(img, -1, 1, 0, ksize=3) 与 cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) 对比
# @@hint 关注三点：uint8 输出的最大值、被压成 0 的像素数、|float| 均值 vs uint8 均值
sx_u8 = cv2.Sobel(gray, -1, 1, 0, ksize=3)
sx_f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
u8_max = int(sx_u8.max())
u8_zero = int((sx_u8 == 0).sum())
f_neg = int((sx_f < 0).sum())
f_min, f_max = round(float(sx_f.min()), 2), round(float(sx_f.max()), 2)
abs_f_mean = round(float(np.abs(sx_f).mean()), 4)
u8_mean = round(float(sx_u8.mean()), 4)
# @@end
print("uint8 输出: max", u8_max, "| 值为 0 的像素", u8_zero, "| 均值", u8_mean)
print("CV_64F 输出:", f_min, "~", f_max, "| 负值像素", f_neg, "| |响应|均值", abs_f_mean)
print("三张图的 uint8 零值数 vs float 负值数：")
for nm, im in [("home", gray), ("camera", cam), ("coins", coins)]:
    a = cv2.Sobel(im, -1, 1, 0, ksize=3)
    b = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    print(f"   {nm}: uint8 零值 {int((a == 0).sum())} | float 负值 {int((b < 0).sum())} "
          f"| |f|均值 {np.abs(b).mean():.4f} vs uint8 均值 {a.mean():.4f}")
print("解读: 一阶导有正有负，uint8 把负的一律截成 0（home 有 81773 个负值像素），"
      "结果只剩半边边缘。|响应| 均值 32.7893 vs uint8 的 16.0151 —— 信息量差近一倍。"
      "**结论：Sobel / Scharr / Laplacian 一律写 cv2.CV_64F 或 cv2.CV_32F**，"
      "之后再取绝对值或归一化显示")

assert u8_max == 255 and u8_zero == 120015 and f_neg == 81773
assert (f_min, f_max) == (-585.0, 764.0)
assert abs_f_mean == 32.7893 and u8_mean == 16.0151
'''

E2_CODE = '''k_parsed = cv2.getDerivKernels(1, 0, 3)
row_k = np.asarray(k_parsed[0], np.float64).ravel()
col_k = np.asarray(k_parsed[1], np.float64).ravel()
small = gray[:60, :60]
cv_sx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
cv_sy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
cv_sc = cv2.Scharr(small, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE)
# @@todo(2) 用现成的 SOBEL_X / SOBEL_Y / SCHARR_X 核做手写相关，与 cv2 对账
# @@hint 手写用脚手架 correlate_edge_pad(small, 核)；三行分别对 Sobel-x / Sobel-y / Scharr-x
d_sx = max_abs_diff(correlate_edge_pad(small, SOBEL_X), cv_sx)
d_sy = max_abs_diff(correlate_edge_pad(small, SOBEL_Y), cv_sy)
d_sc = max_abs_diff(correlate_edge_pad(small, SCHARR_X), cv_sc)
# @@end
print("getDerivKernels(1,0,3) 解析出的行核/列核:", row_k.tolist(), col_k.tolist())
print("手写 Sobel-x 核 vs cv2.Sobel(dx=1) 最大差:", d_sx)
print("手写 Sobel-y 核 vs cv2.Sobel(dy=1) 最大差:", d_sy)
print("手写 Scharr-x 核 vs cv2.Scharr(dx=1) 最大差:", d_sc)
print("解读: 三个核逐像素完全对上（差 0），说明 Sobel / Scharr 就是「能分离的差分核」。"
      "取核的土办法：造一张只有一个 1 的 delta 图，套上去看输出——"
      "输出就是核的镜像（因为 filter2D 是相关）。这样任何自定义算子的核都能反推出来")

assert d_sx == 0.0 and d_sy == 0.0 and d_sc == 0.0
assert row_k.tolist() == [-1.0, 0.0, 1.0] and col_k.tolist() == [1.0, 2.0, 1.0]
'''

E3_CODE = '''sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
vb = np.zeros((40, 40), np.uint8)
vb[:, 20:] = 255
hb = np.zeros((40, 40), np.uint8)
hb[20:, :] = 255
# @@todo(3) 梯度幅值与方向：mag = √(gx²+gy²)，方向 = atan2(gy, gx)
# @@hint 幅值用 np.sqrt(sx**2 + sy**2)；方向用 np.degrees(np.arctan2(sy, sx))
mag = np.sqrt(sx ** 2 + sy ** 2)
vb_gx = np.abs(cv2.Sobel(vb, cv2.CV_64F, 1, 0, ksize=3)).max()
vb_gy = np.abs(cv2.Sobel(vb, cv2.CV_64F, 0, 1, ksize=3)).max()
hb_gx = np.abs(cv2.Sobel(hb, cv2.CV_64F, 1, 0, ksize=3)).max()
hb_gy = np.abs(cv2.Sobel(hb, cv2.CV_64F, 0, 1, ksize=3)).max()
mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
ang = np.degrees(np.arctan2(sy, sx))
strong = mag > 200
n_strong = int(strong.sum())
# @@end
print("|dx| 均值:", round(float(np.abs(sx).mean()), 4), "| |dy| 均值:", round(float(np.abs(sy).mean()), 4))
print("mag: 均值", round(float(mag.mean()), 4), "| max", round(float(mag.max()), 2),
      "| min", round(float(mag.min()), 2))
print("归一化到 0~255 后唯一值数:", len(np.unique(mag_norm)), "| 原始 mag 唯一值数:", len(np.unique(mag)))
print("垂直边缘: |gx| max", int(vb_gx), "| |gy| max", int(vb_gy))
print("水平边缘: |gx| max", int(hb_gx), "| |gy| max", int(hb_gy))
print("强边缘(mag>200) 像素:", n_strong, "| 方向范围:", round(float(ang[strong].min()), 1),
      "~", round(float(ang[strong].max()), 1))
print("解读: 垂直边缘只让 gx 响应（|gy| 恰好 0），水平边缘只让 gy 响应 —— "
      "**梯度方向垂直于边缘走向**。幅值图 13907 个灰阶被归一化成 246 级，"
      "所以「显示归一化」只看形状，量化分析必须用原始浮点幅值")

assert (int(vb_gx), int(vb_gy)) == (1020, 0)
assert (int(hb_gx), int(hb_gy)) == (0, 1020)
assert n_strong == 15163
assert round(float(mag.mean()), 4) == 54.6795 and len(np.unique(mag_norm)) == 246
'''

E4_CODE = '''sx3 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sch = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
sch_via_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
# @@todo(4) Scharr vs Sobel：同阶导，幅度差多少倍？
# @@hint 比值 = |Scharr| 均值 / |Sobel3| 均值；再看 ksize=-1（=FILTER_SCHARR）是否等价
r_sch = round(float(np.abs(sch).mean()), 4)
r_sob = round(float(np.abs(sx3).mean()), 4)
ratio = round(r_sch / r_sob, 4)
d_equiv = max_abs_diff(sch, sch_via_sobel)
# @@end
print("|Scharr| 均值:", r_sch, "| |Sobel ksize=3| 均值:", r_sob, "| 比值:", ratio)
print("ksize=cv2.FILTER_SCHARR 与 cv2.Scharr 最大差:", d_equiv)
print("解读: Scharr 是 Sobel 的「修正版」——3x3 的 Sobel 核在旋转方向上不够对称，"
      "Scharr 把列核从 [1,2,1] 换成 [3,10,3]，|响应| 整体放大 4.2501 倍，"
      "方向精度更高。代价是幅值不再是「相邻像素差」的直觉量级，"
      "所以用 Scharr 时**不要**拿绝对值当阈值，要么归一化，要么直接用 Canny（内部自带）")

assert r_sch == 139.3566 and r_sob == 32.7893 and ratio == 4.2501
assert d_equiv == 0.0
'''

E5_CODE = '''ramp = np.tile(np.arange(64, dtype=np.uint8), (32, 1))
# @@todo(5) Laplacian 三种 ksize：ksize 越大核越大、响应越猛；对线性斜坡应几乎为 0
# @@hint cv2.Laplacian(img, cv2.CV_64F, ksize=ksz)；ksize 取值 1/3/5
l1 = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
l3 = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
l5 = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
abs_mean = {1: round(float(np.abs(l1).mean()), 4), 3: round(float(np.abs(l3).mean()), 4),
            5: round(float(np.abs(l5).mean()), 4)}
d13 = round(float(np.abs(l1 - l3).max()), 2)
ramp_max = int(np.abs(cv2.Laplacian(ramp, cv2.CV_64F, ksize=1)).max())
ramp_dx = int(cv2.Sobel(ramp, cv2.CV_64F, 1, 0, ksize=3)[16, 16])
# @@end
print("|Laplacian| 均值: ksize=1", abs_mean[1], "| ksize=3", abs_mean[3], "| ksize=5", abs_mean[5])
print("ksize=1 与 ksize=3 的最大差:", d13)
print("线性斜坡图: |Laplacian| max =", ramp_max, "| Sobel dx 恒为", ramp_dx)
print("解读: ksize=1 是 4 邻域核 [[0,1,0],[1,-4,1],[0,1,0]]，ksize=3 是 8 邻域（对角权重 2），"
      "ksize=5 是更大的二阶导核——|响应| 从 21.16 → 53.04 → 316.94，量级差 15 倍，"
      "**ksize 不同结果根本不可比**。"
      "斜坡实验是拉普拉斯的本质验证：线性斜坡上二阶导为零（|L| 仅 2，来自边界），"
      "而一阶导 Sobel 恒定 8 —— 所以拉普拉斯对「渐变的照明」不敏感，只抓「拐弯的地方」")

assert abs_mean == {1: 21.1641, 3: 53.036, 5: 316.9449}
assert d13 == 667.0 and ramp_max == 2 and ramp_dx == 8
'''

E6_CODE = '''pairs = [(30, 90), (50, 150), (100, 200), (100, 300), (150, 200), (200, 300)]
edge_cnt = {}
for lo, hi in pairs:
    # @@todo(6) Canny 双阈值扫描：逐个 (lo, hi) 算边缘图，统计边缘像素数
    # @@hint 循环骨架已在挖空块之外；块内只写 2 行 —— cv2.Canny(gray, lo, hi) 与 edge_cnt[(lo, hi)] = 计数
    e = cv2.Canny(gray, lo, hi)
    edge_cnt[(lo, hi)] = int((e > 0).sum())
    # @@end
# @@todo(7) 固定低阈值 100，看 1:2 / 1:3 / 1:4 三档比例；顺便确认 Canny 输出的取值集合
# @@hint 一条字典推导式 {r: 统计 cv2.Canny(gray, 100, 100 * r)}；取值集合用 np.unique(cv2.Canny(gray, 100, 200)).tolist()
ratios = {r: int((cv2.Canny(gray, 100, 100 * r) > 0).sum()) for r in (2, 3, 4)}
uniq = np.unique(cv2.Canny(gray, 100, 200)).tolist()
# @@end
for (lo, hi), c in edge_cnt.items():
    print(f"   Canny({lo},{hi}) 比例 {hi / lo:.2f}: 边缘像素 {c} ({c / gray.size:.4f})")
print("   固定 lo=100，比例 1:2 / 1:3 / 1:4 →", ratios)
print("   Canny 输出的唯一值:", uniq)
print("解读: Canny 是二值输出（只有 0 和 255），不是灰度梯度图。"
      "高阈值越高边缘越少（100→300 时从 20630 掉到 18107）；"
      "常用的高:低 = 3:1 是个经验值（1:2 给 20630、1:3 给 18107、1:4 给 15745）。"
      "注意「比例」比「绝对值」更重要：换个图亮度，绝对值全变但 1:3 的手感还能用")

assert edge_cnt == {(30, 90): 26901, (50, 150): 24539, (100, 200): 20630,
                    (100, 300): 18107, (150, 200): 17330, (200, 300): 12768}
assert ratios == {2: 20630, 3: 18107, 4: 15745} and uniq == [0, 255]
'''

E7_CODE = '''# @@todo(8) Canny 的旋钮一：梯度幅值的口径 —— L2gradient=True 用 √(gx²+gy²)，False 用 |gx|+|gy|
# @@hint 两行：cv2.Canny(gray, 100, 200, L2gradient=False/True)，再数 (edges > 0).sum()
e_l1 = cv2.Canny(gray, 100, 200, L2gradient=False)
e_l2 = cv2.Canny(gray, 100, 200, L2gradient=True)
n_l1, n_l2 = int((e_l1 > 0).sum()), int((e_l2 > 0).sum())
# @@end
ap_cnt = {}
for ap in (3, 5, 7):
    # @@todo(9) Canny 的旋钮二：内部 Sobel 的核大小 apertureSize —— 循环骨架在块外，块内补 1 行
    # @@hint ap_cnt[ap] = int((cv2.Canny(gray, 100, 200, apertureSize=ap) > 0).sum())
    ap_cnt[ap] = int((cv2.Canny(gray, 100, 200, apertureSize=ap) > 0).sum())
    # @@end
print("L2gradient=False（L1 范数）:", n_l1, "| True（L2 范数）:", n_l2,
      "| 差额", abs(n_l2 - n_l1))
print("apertureSize 3 / 5 / 7 →", ap_cnt)
print("解读: 1) L1 范数 |gx|+|gy| 会高估对角方向的梯度（最大 √2 倍），"
      "所以 L2gradient=True 更严谨、边缘更少（17831 vs 20630）；"
      "2) apertureSize 是内部 Sobel 的核大小，5x5/7x7 的差分核更平滑也更宽，"
      "同样的阈值下边缘数暴涨到 35207 / 50329 —— **换 apertureSize 必须重调阈值**")

assert n_l1 == 20630 and n_l2 == 17831
assert ap_cnt == {3: 20630, 5: 35207, 7: 50329}
'''

E8_CODE = '''e = cv2.Canny(gray, 100, 200)
sx2 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy2 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
mg = np.sqrt(sx2 ** 2 + sy2 ** 2)
# @@todo(10) Canny 与梯度幅值的关系：边缘像素的 mag 一定比非边缘大吗？极端阈值会怎样？
# @@hint 用 e>0 做掩码取 mg 的均值；再试 Canny(0,0) 与 Canny(255,255) 这两个极端
mag_on = round(float(mg[e > 0].mean()), 4)
mag_off = round(float(mg[e == 0].mean()), 4)
low_frac = round(float((mg[e > 0] < 100).mean()), 4)
n_00 = int((cv2.Canny(gray, 0, 0) > 0).sum())
n_255 = int((cv2.Canny(gray, 255, 255) > 0).sum())
# @@end
print("边缘像素的 mag 均值:", mag_on, "| 非边缘:", mag_off, "| 比值:", round(mag_on / mag_off, 4))
print("边缘像素里 mag<100（低于低阈值）的占比:", low_frac)
print("极端阈值: Canny(0,0) →", n_00, "| Canny(255,255) →", n_255)
print("解读: 边缘像素的梯度幅值平均是背景的 6.9 倍，但仍有 5.51% 的边缘像素 mag<100 ——"
      "因为**滞回（hysteresis）**：弱边缘只要连到强边缘就连带保留，这是 Canny 比"
      "「简单阈值化梯度图」强的地方（边缘更连续、断点更少）。"
      "另外 Canny(0,0) 仍给出 54891 个边缘（非极大值抑制后剩下的脊线），"
      "Canny(255,255) 也有 10984（梯度幅值能超过 255 的像素），所以「阈值设 0 就没边缘」是错的")

assert mag_on == 233.1316 and mag_off == 33.7594
assert low_frac == 0.0551 and n_00 == 54891 and n_255 == 10984
'''

# =========================================================================== #
# 讲解用代码块（与练习内容一一对应，去掉挖空标记）
# =========================================================================== #

S1_CODE = '''sx_u8 = cv2.Sobel(gray, -1, 1, 0, ksize=3)
sx_f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
u8_max = int(sx_u8.max())
u8_zero = int((sx_u8 == 0).sum())
f_neg = int((sx_f < 0).sum())
f_min, f_max = round(float(sx_f.min()), 2), round(float(sx_f.max()), 2)
abs_f_mean = round(float(np.abs(sx_f).mean()), 4)
u8_mean = round(float(sx_u8.mean()), 4)
print("uint8 输出: max", u8_max, "| 值为 0 的像素", u8_zero, "| 均值", u8_mean)
print("CV_64F 输出:", f_min, "~", f_max, "| 负值像素", f_neg, "| |响应|均值", abs_f_mean)
print("三张图的 uint8 零值数 vs float 负值数：")
for nm, im in [("home", gray), ("camera", cam), ("coins", coins)]:
    a = cv2.Sobel(im, -1, 1, 0, ksize=3)
    b = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    print(f"   {nm}: uint8 零值 {int((a == 0).sum())} | float 负值 {int((b < 0).sum())} "
          f"| |f|均值 {np.abs(b).mean():.4f} vs uint8 均值 {a.mean():.4f}")
print("解读: 一阶导有正有负，uint8 把负的一律截成 0（home 有 81773 个负值像素），"
      "结果只剩半边边缘。|响应| 均值 32.7893 vs uint8 的 16.0151 —— 信息量差近一倍。"
      "**结论：Sobel / Scharr / Laplacian 一律写 cv2.CV_64F 或 cv2.CV_32F**，"
      "之后再取绝对值或归一化显示")

assert u8_max == 255 and u8_zero == 120015 and f_neg == 81773
assert (f_min, f_max) == (-585.0, 764.0)
assert abs_f_mean == 32.7893 and u8_mean == 16.0151
'''

S2_CODE = '''k_parsed = cv2.getDerivKernels(1, 0, 3)
row_k = np.asarray(k_parsed[0], np.float64).ravel()
col_k = np.asarray(k_parsed[1], np.float64).ravel()
small = gray[:60, :60]
cv_sx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
cv_sy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
cv_sc = cv2.Scharr(small, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE)
d_sx = max_abs_diff(correlate_edge_pad(small, SOBEL_X), cv_sx)
d_sy = max_abs_diff(correlate_edge_pad(small, SOBEL_Y), cv_sy)
d_sc = max_abs_diff(correlate_edge_pad(small, SCHARR_X), cv_sc)
print("getDerivKernels(1,0,3) 解析出的行核/列核:", row_k.tolist(), col_k.tolist())
print("手写 Sobel-x 核 vs cv2.Sobel(dx=1) 最大差:", d_sx)
print("手写 Sobel-y 核 vs cv2.Sobel(dy=1) 最大差:", d_sy)
print("手写 Scharr-x 核 vs cv2.Scharr(dx=1) 最大差:", d_sc)
print("解读: 三个核逐像素完全对上（差 0），说明 Sobel / Scharr 就是「能分离的差分核」。"
      "取核的土办法：造一张只有一个 1 的 delta 图，套上去看输出——"
      "输出就是核的镜像（因为 filter2D 是相关）。这样任何自定义算子的核都能反推出来")

assert d_sx == 0.0 and d_sy == 0.0 and d_sc == 0.0
assert row_k.tolist() == [-1.0, 0.0, 1.0] and col_k.tolist() == [1.0, 2.0, 1.0]
'''

S3_CODE = '''sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
vb = np.zeros((40, 40), np.uint8)
vb[:, 20:] = 255
hb = np.zeros((40, 40), np.uint8)
hb[20:, :] = 255
mag = np.sqrt(sx ** 2 + sy ** 2)
vb_gx = np.abs(cv2.Sobel(vb, cv2.CV_64F, 1, 0, ksize=3)).max()
vb_gy = np.abs(cv2.Sobel(vb, cv2.CV_64F, 0, 1, ksize=3)).max()
hb_gx = np.abs(cv2.Sobel(hb, cv2.CV_64F, 1, 0, ksize=3)).max()
hb_gy = np.abs(cv2.Sobel(hb, cv2.CV_64F, 0, 1, ksize=3)).max()
mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
ang = np.degrees(np.arctan2(sy, sx))
strong = mag > 200
n_strong = int(strong.sum())
print("|dx| 均值:", round(float(np.abs(sx).mean()), 4), "| |dy| 均值:", round(float(np.abs(sy).mean()), 4))
print("mag: 均值", round(float(mag.mean()), 4), "| max", round(float(mag.max()), 2),
      "| min", round(float(mag.min()), 2))
print("归一化到 0~255 后唯一值数:", len(np.unique(mag_norm)), "| 原始 mag 唯一值数:", len(np.unique(mag)))
print("垂直边缘: |gx| max", int(vb_gx), "| |gy| max", int(vb_gy))
print("水平边缘: |gx| max", int(hb_gx), "| |gy| max", int(hb_gy))
print("强边缘(mag>200) 像素:", n_strong, "| 方向范围:", round(float(ang[strong].min()), 1),
      "~", round(float(ang[strong].max()), 1))
print("解读: 垂直边缘只让 gx 响应（|gy| 恰好 0），水平边缘只让 gy 响应 —— "
      "**梯度方向垂直于边缘走向**。幅值图 13907 个灰阶被归一化成 246 级，"
      "所以「显示归一化」只看形状，量化分析必须用原始浮点幅值")

assert (int(vb_gx), int(vb_gy)) == (1020, 0)
assert (int(hb_gx), int(hb_gy)) == (0, 1020)
assert n_strong == 15163
assert round(float(mag.mean()), 4) == 54.6795 and len(np.unique(mag_norm)) == 246
'''

S4_CODE = '''sx3 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sch = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
sch_via_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
r_sch = round(float(np.abs(sch).mean()), 4)
r_sob = round(float(np.abs(sx3).mean()), 4)
ratio = round(r_sch / r_sob, 4)
d_equiv = max_abs_diff(sch, sch_via_sobel)
print("|Scharr| 均值:", r_sch, "| |Sobel ksize=3| 均值:", r_sob, "| 比值:", ratio)
print("ksize=cv2.FILTER_SCHARR 与 cv2.Scharr 最大差:", d_equiv)
print("解读: Scharr 是 Sobel 的「修正版」——3x3 的 Sobel 核在旋转方向上不够对称，"
      "Scharr 把列核从 [1,2,1] 换成 [3,10,3]，|响应| 整体放大 4.2501 倍，"
      "方向精度更高。代价是幅值不再是「相邻像素差」的直觉量级，"
      "所以用 Scharr 时**不要**拿绝对值当阈值，要么归一化，要么直接用 Canny（内部自带）")

assert r_sch == 139.3566 and r_sob == 32.7893 and ratio == 4.2501
assert d_equiv == 0.0
'''

S5_CODE = '''ramp = np.tile(np.arange(64, dtype=np.uint8), (32, 1))
l1 = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
l3 = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
l5 = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
abs_mean = {1: round(float(np.abs(l1).mean()), 4), 3: round(float(np.abs(l3).mean()), 4),
            5: round(float(np.abs(l5).mean()), 4)}
d13 = round(float(np.abs(l1 - l3).max()), 2)
ramp_max = int(np.abs(cv2.Laplacian(ramp, cv2.CV_64F, ksize=1)).max())
ramp_dx = int(cv2.Sobel(ramp, cv2.CV_64F, 1, 0, ksize=3)[16, 16])
print("|Laplacian| 均值: ksize=1", abs_mean[1], "| ksize=3", abs_mean[3], "| ksize=5", abs_mean[5])
print("ksize=1 与 ksize=3 的最大差:", d13)
print("线性斜坡图: |Laplacian| max =", ramp_max, "| Sobel dx 恒为", ramp_dx)
print("解读: ksize=1 是 4 邻域核 [[0,1,0],[1,-4,1],[0,1,0]]，ksize=3 是 8 邻域（对角权重 2），"
      "ksize=5 是更大的二阶导核——|响应| 从 21.16 → 53.04 → 316.94，量级差 15 倍，"
      "**ksize 不同结果根本不可比**。"
      "斜坡实验是拉普拉斯的本质验证：线性斜坡上二阶导为零（|L| 仅 2，来自边界），"
      "而一阶导 Sobel 恒定 8 —— 所以拉普拉斯对「渐变的照明」不敏感，只抓「拐弯的地方」")

assert abs_mean == {1: 21.1641, 3: 53.036, 5: 316.9449}
assert d13 == 667.0 and ramp_max == 2 and ramp_dx == 8
'''

S6_CODE = '''pairs = [(30, 90), (50, 150), (100, 200), (100, 300), (150, 200), (200, 300)]
edge_cnt = {}
for lo, hi in pairs:
    e = cv2.Canny(gray, lo, hi)
    edge_cnt[(lo, hi)] = int((e > 0).sum())
ratios = {r: int((cv2.Canny(gray, 100, 100 * r) > 0).sum()) for r in (2, 3, 4)}
uniq = np.unique(cv2.Canny(gray, 100, 200)).tolist()
for (lo, hi), c in edge_cnt.items():
    print(f"   Canny({lo},{hi}) 比例 {hi / lo:.2f}: 边缘像素 {c} ({c / gray.size:.4f})")
print("   固定 lo=100，比例 1:2 / 1:3 / 1:4 →", ratios)
print("   Canny 输出的唯一值:", uniq)
print("解读: Canny 是二值输出（只有 0 和 255），不是灰度梯度图。"
      "高阈值越高边缘越少（100→300 时从 20630 掉到 18107）；"
      "常用的高:低 = 3:1 是个经验值（1:2 给 20630、1:3 给 18107、1:4 给 15745）。"
      "注意「比例」比「绝对值」更重要：换个图亮度，绝对值全变但 1:3 的手感还能用")

assert edge_cnt == {(30, 90): 26901, (50, 150): 24539, (100, 200): 20630,
                    (100, 300): 18107, (150, 200): 17330, (200, 300): 12768}
assert ratios == {2: 20630, 3: 18107, 4: 15745} and uniq == [0, 255]
'''

S7_CODE = '''e_l1 = cv2.Canny(gray, 100, 200, L2gradient=False)
e_l2 = cv2.Canny(gray, 100, 200, L2gradient=True)
n_l1, n_l2 = int((e_l1 > 0).sum()), int((e_l2 > 0).sum())
ap_cnt = {}
for ap in (3, 5, 7):
    ap_cnt[ap] = int((cv2.Canny(gray, 100, 200, apertureSize=ap) > 0).sum())
print("L2gradient=False（L1 范数）:", n_l1, "| True（L2 范数）:", n_l2,
      "| 差额", abs(n_l2 - n_l1))
print("apertureSize 3 / 5 / 7 →", ap_cnt)
print("解读: 1) L1 范数 |gx|+|gy| 会高估对角方向的梯度（最大 √2 倍），"
      "所以 L2gradient=True 更严谨、边缘更少（17831 vs 20630）；"
      "2) apertureSize 是内部 Sobel 的核大小，5x5/7x7 的差分核更平滑也更宽，"
      "同样的阈值下边缘数暴涨到 35207 / 50329 —— **换 apertureSize 必须重调阈值**")

assert n_l1 == 20630 and n_l2 == 17831
assert ap_cnt == {3: 20630, 5: 35207, 7: 50329}
'''

S8_CODE = '''e = cv2.Canny(gray, 100, 200)
sx2 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy2 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
mg = np.sqrt(sx2 ** 2 + sy2 ** 2)
mag_on = round(float(mg[e > 0].mean()), 4)
mag_off = round(float(mg[e == 0].mean()), 4)
low_frac = round(float((mg[e > 0] < 100).mean()), 4)
n_00 = int((cv2.Canny(gray, 0, 0) > 0).sum())
n_255 = int((cv2.Canny(gray, 255, 255) > 0).sum())
print("边缘像素的 mag 均值:", mag_on, "| 非边缘:", mag_off, "| 比值:", round(mag_on / mag_off, 4))
print("边缘像素里 mag<100（低于低阈值）的占比:", low_frac)
print("极端阈值: Canny(0,0) →", n_00, "| Canny(255,255) →", n_255)
print("解读: 边缘像素的梯度幅值平均是背景的 6.9 倍，但仍有 5.51% 的边缘像素 mag<100 ——"
      "因为**滞回（hysteresis）**：弱边缘只要连到强边缘就连带保留，这是 Canny 比"
      "「简单阈值化梯度图」强的地方（边缘更连续、断点更少）。"
      "另外 Canny(0,0) 仍给出 54891 个边缘（非极大值抑制后剩下的脊线），"
      "Canny(255,255) 也有 10984（梯度幅值能超过 255 的像素），所以「阈值设 0 就没边缘」是错的")

assert mag_on == 233.1316 and mag_off == 33.7594
assert low_frac == 0.0551 and n_00 == 54891 and n_255 == 10984
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch05 边缘检测与梯度

> 数据：`classic/home.jpg`、`camera.png`、`coins.png`、`helmet/site_03.jpg` 的灰度图。
> 真值全部实跑（opencv 5.0.0）。

**本章考点**

1. **Sobel 的 `ddepth` 坑**：不写 `CV_64F` 负梯度全被截成 0
2. Sobel / Scharr 的等效核，以及「delta 法反推核」
3. 梯度幅值 `√(gx²+gy²)` 与方向 `atan2(gy, gx)`
4. **Scharr vs Sobel**（幅度差 4.2501 倍）
5. `Laplacian` 三种 ksize + 线性斜坡上二阶导为零
6. **Canny 双阈值扫描**与 1:2 / 1:3 / 1:4 比例
7. Canny 的 `L2gradient` 与 `apertureSize` 旋钮
8. Canny 的**滞回**机制与极端阈值行为

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| ddepth 默认 uint8 | 负梯度全截成 0，只剩半边边缘 | §5.1 |
| Scharr 幅度放大 4.25 倍 | 拿绝对值当阈值必然翻车 | §5.4 |
| Laplacian 的 ksize | 量级差 15 倍，结果不可比 | §5.5 |
| apertureSize 换核 | 同阈值边缘数暴涨 2.4 倍 | §5.7 |
| Canny 输出是二值 | 别当灰度梯度图用 | §5.6 |
| 阈值设 0/255 的直觉 | `Canny(0,0)` 仍有 54891 条边缘 | §5.8 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 5.1 难点深挖：Sobel 的 `ddepth` 坑

`cv2.Sobel(src, ddepth, dx, dy, ksize)` 的第二个参数是**输出深度**。
不写（或用 `-1` = 与输入同类型）就会拿 `uint8` 接一阶导 —— 而一阶导有正有负：

| 素材 | uint8 输出 max | uint8 中为 0 的像素 | `CV_64F` 负值像素 | `\\|float\\|` 均值 vs uint8 均值 |
|---|---|---|---|---|
| home | 255 | 120,015 | **81,773** | **32.7893** vs 16.0151 |
| camera | 255 | 141,251 | 118,800 | 32.5966 vs 14.9505 |
| coins | 255 | 60,697 | 57,567 | 44.3711 vs 19.8289 |

负响应被一律截成 0 ⇒ **边缘只剩一半**，`|响应|` 均值几乎腰斩。
所以 Sobel / Scharr / Laplacian **一律写 `cv2.CV_64F` 或 `cv2.CV_32F`**，
最后再取绝对值或归一化显示。
"""
    ),
    code(S1_CODE),
    md(
        """
## 5.2 Sobel / Scharr 的等效核

`getDerivKernels(1, 0, 3)` 给出可分离的行核 `[-1, 0, 1]` 与列核 `[1, 2, 1]`，
外积就是经典 Sobel-x：

```
SOBEL_X  = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBEL_Y  = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
SCHARR_X = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
```

用脚手架做手写相关与 `cv2.Sobel` / `cv2.Scharr` 对账，**三者最大差均为 0.0**。

> **delta 法取核**：造一张只有一个 `1` 的图，套上算子，输出的就是核的镜像
> （因为 `filter2D` 是相关）。任何自定义算子的核都能这样反推出来。
"""
    ),
    code(S2_CODE),
    md(
        """
## 5.3 梯度幅值与方向

```
gx = Sobel(dx=1)   gy = Sobel(dy=1)
mag  = √(gx² + gy²)
ang  = atan2(gy, gx)
```

`home` 实测：`|dx|` 均值 32.7893、`|dy|` 均值 36.9827，`mag` 均值 **54.6795**、最大 786.16。

用合成图验证「方向与边缘垂直」：

| 图像 | `\\|gx\\|` max | `\\|gy\\|` max |
|---|---|---|
| 垂直边缘（左黑右白） | **1020** | **0** |
| 水平边缘（上黑下白） | **0** | **1020** |

另外注意：`mag` 有 13,907 个不同取值，归一化到 0~255 后只剩 **246** 级。
**归一化只看形状，量化分析必须用原始 float32 幅值。**
"""
    ),
    code(S3_CODE),
    md(
        """
## 5.4 难点深挖：Scharr 与 Sobel 不可直接比

`cv2.Scharr(gray, cv2.CV_64F, 1, 0)` 等价于 `cv2.Sobel(..., ksize=cv2.FILTER_SCHARR)`
（两者最大差 **0.0**）。区别在列核：

| 算子 | 列核 | `\\|dx\\|` 均值 |
|---|---|---|
| Sobel ksize=3 | `[1, 2, 1]` | 32.7893 |
| **Scharr** | `[3, 10, 3]` | **139.3566**（**4.2501 倍**）|

Scharr 的 3×3 核旋转对称性更好、方向精度更高，但**幅度被整体放大 4.25 倍** ——
拿绝对幅值当阈值必然翻车。要么归一化，要么直接用 Canny（内部会自己处理）。
"""
    ),
    code(S4_CODE),
    md(
        """
## 5.5 Laplacian 的 ksize 与「二阶导」本质

| ksize | 等效核 | `\\|L\\|` 均值 | min/max |
|---|---|---|---|
| 1 | `[[0,1,0],[1,-4,1],[0,1,0]]`（4 邻域） | 21.1641 | −402 / 399 |
| 3 | 8 邻域（对角权重 2） | 53.0360 | −968 / 978 |
| 5 | 更大的二阶导核 | **316.9449** | −6998 / 5676 |

ksize=1 与 ksize=3 的最大差 **667.0**，`|L|` 均值差 15 倍 —— **ksize 不同结果根本不可比**。

线性斜坡实验最能说明拉普拉斯的本质：

| 算子 | 线性斜坡上的响应 |
|---|---|
| `Laplacian`（二阶导） | `\\|L\\| max` = **2**（≈0） |
| `Sobel dx`（一阶导） | 恒为 **8** |

所以拉普拉斯**对渐变照明不敏感**，只抓「拐弯的地方」——这也是它在文档/缺陷检测里
比 Sobel 更常用于「细线/斑点」检测的原因。
"""
    ),
    code(S5_CODE),
    md(
        """
## 5.6 Canny 双阈值

| `Canny(lo, hi)` | 比例 | 边缘像素 | 占比 |
|---|---|---|---|
| (30, 90) | 3.00 | 26,901 | 13.68% |
| (50, 150) | 3.00 | 24,539 | 12.48% |
| (100, 200) | 2.00 | 20,630 | 10.49% |
| (100, 300) | 3.00 | 18,107 | 9.21% |
| (150, 200) | 1.33 | 17,330 | 8.81% |
| (200, 300) | 1.50 | 12,768 | 6.49% |

固定低阈值 100 时，比例 1:2 / 1:3 / 1:4 → **20,630 / 18,107 / 15,745**。

两个要点：

1. **Canny 输出是二值的**（唯一值只有 `[0, 255]`），不是灰度梯度图 —— 别拿它当特征图。
2. 高:低 ≈ **3:1** 是工程上的常用起点；**比例**比绝对值更可移植
   （换张图亮度变，绝对值全废，比例的手感还在）。
"""
    ),
    code(S6_CODE),
    md(
        """
## 5.7 难点深挖：`L2gradient` 与 `apertureSize`

| 参数 | 取值 | 边缘像素 |
|---|---|---|
| `L2gradient` | `False`（`\\|gx\\|+\\|gy\\|`） | 20,630 |
| `L2gradient` | `True`（`√(gx²+gy²)`） | **17,831** |
| `apertureSize` | 3 | 20,630 |
| `apertureSize` | 5 | **35,207** |
| `apertureSize` | 7 | **50,329** |

- L1 范数 `|gx|+|gy|` 会**高估对角方向的梯度**（最大 √2 倍），所以 L2 更严谨、边缘更少。
- `apertureSize` 是内部 Sobel 的核大小：换成 5×5 / 7×7 后差分核更平滑也更宽，
  同一套阈值下边缘数暴涨到 2.4 倍 —— **换 apertureSize 必须重调阈值**。
"""
    ),
    code(S7_CODE),
    md(
        """
## 5.8 难点深挖：滞回（hysteresis）与极端阈值

`Canny(100, 200)` 的边缘像素 vs 全图背景：

| | `mag` 均值 |
|---|---|
| 边缘像素 | **233.1316** |
| 非边缘像素 | 33.7594 |
| 比值 | 6.9× |

即使如此，边缘像素里仍有 **5.51%** 的 `mag < 100`（低于低阈值）。
原因就是 Canny 的**滞回**：弱边缘只要与强边缘连通就一并保留，
这带来更连续、断点更少的边缘 —— 这正是 Canny 比「梯度图直接阈值化」强的地方。

两个反直觉的极端：

| 调用 | 边缘像素 |
|---|---|
| `Canny(0, 0)` | **54,891** |
| `Canny(255, 255)` | **10,984** |

阈值设 0 不会「全是边缘」（非极大值抑制仍会挑出脊线），设 255 也不会「没有边缘」
（梯度幅值本来就能超过 255，最高到 786）。**别用直觉猜阈值，永远扫一遍。**
"""
    ),
    code(S8_CODE),
    md(
        """
## 小结

1. **Sobel/Scharr/Laplacian 必须写 `CV_64F`**：uint8 会把 81,773 个负响应截成 0。
2. Sobel-x = `[[-1,0,1],[-2,0,2],[-1,0,1]]`，Sobel-y 是其转置；手写相关差 0.0。
3. 「delta 法」可以反推任意算子的核（输出是核的镜像，因为 `filter2D` 是相关）。
4. 梯度方向**垂直于边缘**：垂直边缘 `|gx|`=1020 / `|gy|`=0，水平边缘反之。
5. 幅值图归一化后只剩 246 级（原 13,907 级），量化分析要用原始浮点。
6. **Scharr 幅度是 Sobel 的 4.2501 倍**，别混用阈值。
7. Laplacian 的 ksize 决定量级（21 → 53 → 317），结果不可比。
8. 线性斜坡上二阶导 ≈ 0：拉普拉斯只抓「拐弯」，对渐变照明免疫。
9. Canny 输出二值（0/255）；高:低 ≈ 3:1 是常用起点。
10. `L2gradient=True` 边缘更少（17,831）；`apertureSize=7` 边缘暴涨到 50,329。
11. Canny 的滞回让 5.51% 的边缘像素低于低阈值；阈值必须扫，不能猜。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch05 边缘检测与梯度（练习版）

> 按提示补全 `____`，跑通所有 assert。`gray` / `cam` / `coins` / `s3` 已在 setup 里备好；
> `SOBEL_X` / `SOBEL_Y` / `SCHARR_X` / `correlate_edge_pad` / `max_abs_diff` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；循环 / 分支写在挖空块之外（见任务 6、7）。
> 题目真值都来自实跑（opencv 5.0.0），照提示写就能对上。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md("## 任务 1：`ddepth` 坑\n\nuint8 截断 vs `CV_64F` 保负。"),
    code(E1_CODE),
    md("## 任务 2：等效核与 delta 法\n\n手写相关对账 Sobel-x / Sobel-y / Scharr-x。"),
    code(E2_CODE),
    md("## 任务 3：梯度幅值与方向\n\n合成条带 + 归一化损失。"),
    code(E3_CODE),
    md("## 任务 4：Scharr vs Sobel\n\n幅度比值与 `FILTER_SCHARR` 等价性。"),
    code(E4_CODE),
    md("## 任务 5：Laplacian\n\n三种 ksize + 线性斜坡二阶导为零。"),
    code(E5_CODE),
    md("## 任务 6：Canny 双阈值\n\n循环骨架已给好，块内写 2 行。"),
    code(E6_CODE),
    md("## 任务 7：`L2gradient` 与 `apertureSize`\n\n两个旋钮各试一遍。"),
    code(E7_CODE),
    md("## 任务 8：滞回与极端阈值\n\n边缘像素的 mag 分布 + `Canny(0,0)` / `Canny(255,255)`。"),
    code(E8_CODE),
    md(
        """
## 自查清单

- [ ] 能说出 Sobel 第二个参数写错会有什么后果
- [ ] 能默写 Sobel-x / Sobel-y 核
- [ ] 会用 delta 法反推一个自定义算子的核
- [ ] 知道垂直边缘只让 gx 响应、水平边缘只让 gy 响应
- [ ] 知道 Scharr 与 Sobel 的幅度差几倍
- [ ] 能解释线性斜坡上 Laplacian 响应为 0、Sobel 恒定
- [ ] 知道 Canny 输出只有 0/255
- [ ] 能说出 `L2gradient` 与 `apertureSize` 各自影响什么
- [ ] 能解释 Canny 的滞回为什么能保住弱边缘
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
