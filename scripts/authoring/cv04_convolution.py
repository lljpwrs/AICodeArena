#!/usr/bin/env python3
"""cv04 卷积与滤波：手写相关 / 相关≠卷积 / 均值·高斯核 / 边界模式 / 可分离性 / 锐化。

数据：classic/home.jpg（灰度图作滤波素材，384×512）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv04_convolution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch04_convolution"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

DATA = Path("data")
CL = DATA / "classic"
'''

SETUP = '''home = cv2.imread(str(CL / "home.jpg"))
gray = cv2.cvtColor(home, cv2.COLOR_BGR2GRAY)   # (384, 512) uint8，滤波素材
f32 = gray.astype(np.float32)                   # 线性滤波统一在 float32 上做
print("home", home.shape, "| 灰度", gray.shape, gray.dtype,
      "| 均值", round(float(gray.mean()), 4), "| std", round(float(gray.std()), 4))
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def correlate_zero_pad(img, k):
    """手写 2D 相关（零填充 + 逐位置点积），与 cv2.filter2D 同口径。

    注意：这是「相关」（correlation）——核不翻转。要算真正的卷积，把 k 翻转再传进来。
    """
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(np.asarray(img, np.float64), ((ph, ph), (pw, pw)), mode="constant")
    h, w = img.shape
    out = np.zeros((h, w), np.float64)
    for i in range(h):
        for j in range(w):
            out[i, j] = float((pad[i:i + kh, j:j + kw] * k).sum())
    return out


def max_abs_diff(a, b):
    return round(float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()), 6)


def gap_to_clean(a, clean):
    """与干净图的平均绝对差（去噪效果的第二把尺子）。"""
    return round(float(np.abs(np.asarray(a, np.int64) - np.asarray(clean, np.int64)).mean()), 4)


print("脚手架就绪：correlate_zero_pad / max_abs_diff / gap_to_clean")'''

# =========================================================================== #
# 练习（答案版内容，practice 由 builder 自动挖空）
# =========================================================================== #

E1_CODE = '''ks = np.ones((3, 3), np.float32) / 9.0
ka = np.array([[1, 2], [3, 4]], np.float32)
tiny = gray[:40, :40]
cv_sym = cv2.filter2D(tiny.astype(np.float32), -1, ks, borderType=cv2.BORDER_CONSTANT)
cv_asym = cv2.filter2D(tiny.astype(np.float32), -1, ka, borderType=cv2.BORDER_CONSTANT)
# @@todo(1) 三连对账：手写相关 vs filter2D（对称核 / 非对称核），以及「真卷积」差多少
# @@hint 手写相关用脚手架 correlate_zero_pad(图, 核)；「真卷积」= 把核翻转 ka[::-1, ::-1] 再相关
d_sym = max_abs_diff(correlate_zero_pad(tiny, ks), cv_sym)
d_asym_corr = max_abs_diff(correlate_zero_pad(tiny, ka), cv_asym)
d_asym_conv = max_abs_diff(correlate_zero_pad(tiny, ka[::-1, ::-1]), cv_asym)
# @@end
print("对称核   手写相关 vs filter2D 最大差:", d_sym)
print("非对称核 手写相关 vs filter2D 最大差:", d_asym_corr)
print("非对称核 手写卷积(翻核) vs filter2D 最大差:", d_asym_conv)
print("解读: 前两行几乎为 0（浮点误差），第三行差出 352 —— 说明 cv2.filter2D 做的是「相关」"
      "（核不翻转）而不是数学意义的卷积。对称核看不出区别，一旦用非对称核"
      "（如 Sobel、Gabor、方向性核）就必须想清楚自己是哪个口径")

assert d_sym <= 1e-4 and d_asym_corr == 0.0
assert d_asym_conv == 352.0
'''

E2_CODE = '''k5 = np.ones((5, 5), np.float32) / 25.0
box_norm = cv2.blur(gray, (5, 5))
box_unnorm = cv2.boxFilter(gray, -1, (5, 5), normalize=False)
f2d_blur = cv2.filter2D(gray, -1, k5)
kk = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
# @@todo(2) 核的「和」决定亮度：sum=1 保亮度，sum=25 直接饱和到 255
# @@hint boxFilter(normalize=False) 就是「核没除以元素个数」；再对比归一化核与 filter2D/blur
ksum = float(kk.sum())
ksum_norm = round(float((kk / kk.sum()).sum()), 6)
m_norm = round(float(box_norm.mean()), 4)
m_unnorm = round(float(box_unnorm.mean()), 4)
d_blur_f2d = int(np.abs(box_norm.astype(int) - f2d_blur.astype(int)).max())
# @@end
print("原图均值:", round(float(gray.mean()), 4))
print("blur(5,5) 均值:", m_norm, "| filter2D(均值核) 均值:", round(float(f2d_blur.mean()), 4),
      "| 两者最大差:", d_blur_f2d)
print("boxFilter 未归一化 均值:", m_unnorm, "| 唯一值数:", len(np.unique(box_unnorm)))
print("3x3 高斯样核 sum:", ksum, "→ 归一化后 sum:", ksum_norm)
print("解读: blur == filter2D(均值核)，逐像素最大差 0，两套写法完全等价。"
      "核的所有元素之和就是「直流增益」：sum=1 亮度不变；boxFilter 不归一化时 sum=25，"
      "结果被放大 25 倍直接饱和到 255（唯一值只剩 64 个）——这是新手最常踩的坑")

assert d_blur_f2d == 0 and m_norm == 116.1955
assert m_unnorm == 254.9622 and ksum == 16.0 and ksum_norm == 1.0
'''

E3_CODE = '''k1 = cv2.getGaussianKernel(5, 1.0)
k2 = cv2.getGaussianKernel(5, 2.0)
k5s = cv2.getGaussianKernel(5, 5.0)
k0 = cv2.getGaussianKernel(5, 0)
# @@todo(3) 高斯核：sigma 影响、1D 外积 = 2D、可分离滤波与全 2D 滤波对账
# @@hint 外积写 k1 @ k1.T；sepFilter2D(图, -1, 横核, 纵核) 与 filter2D(图, -1, 外积核) 应等价
g2d = k1 @ k1.T
sum1, sum2 = round(float(k1.sum()), 6), round(float(g2d.sum()), 6)
d_sep = max_abs_diff(cv2.filter2D(f32, -1, g2d), cv2.sepFilter2D(f32, -1, k1, k1))
first1 = round(float(k1.ravel()[0]), 6)
first5 = round(float(k5s.ravel()[0]), 6)
auto0 = [round(float(v), 6) for v in k0.ravel()]
# @@end
print("sigma=1 的 1D 核:", [round(float(v), 6) for v in k1.ravel()], "sum", sum1)
print("sigma=2 的 1D 核:", [round(float(v), 6) for v in k2.ravel()])
print("sigma=5 的 1D 核:", [round(float(v), 6) for v in k5s.ravel()])
print("外积核 sum:", sum2, "| 2D 滤波 vs sepFilter2D 最大差:", d_sep)
print("sigma=0（自动推算，即二项式系数）:", auto0)
print("解读: sigma 越大核越「平」（sigma=1 中心权重 0.4026，sigma=5 只有 0.2080）——"
      "平滑更强但更糊。1D 高斯的外积恰好等于 2D 高斯核，所以可以拆成横竖两趟做，"
      "这正是 sepFilter2D 快 15.5 倍的原理。sigma=0 时 OpenCV 按 ksize 反推，"
      "ksize=5 给出的是标准二项式系数 [0.0625, 0.25, 0.375, 0.25, 0.0625]")

assert sum1 == 1.0 and sum2 == 1.0 and d_sep <= 1e-4
assert first1 == 0.054489 and first5 == 0.192051
assert auto0 == [0.0625, 0.25, 0.375, 0.25, 0.0625]
'''

E4_CODE = '''k3e = np.ones((3, 3), np.float32) / 9.0
k5e = np.ones((5, 5), np.float32) / 25.0
base5 = cv2.filter2D(f32, -1, k5e, borderType=cv2.BORDER_REFLECT_101)
modes = {"REFLECT": cv2.BORDER_REFLECT, "REPLICATE": cv2.BORDER_REPLICATE,
         "CONSTANT": cv2.BORDER_CONSTANT}
border_stats = {}
for nm, bt in modes.items():
    # @@todo(4) 逐个边界模式与默认 REFLECT_101 对账：差异有多大、是否只出现在外圈
    # @@hint 循环骨架在块外；块内 3 行——算输出、算绝对差、记录 (最大差, 差异像素数, 内圈是否一致)
    out = cv2.filter2D(f32, -1, k5e, borderType=bt)
    d = np.abs(out - base5)
    border_stats[nm] = (round(float(d.max()), 4), int((d > 1e-6).sum()), bool((d[2:-2, 2:-2] < 1e-6).all()))
    # @@end
# 半径 1 时 REFLECT 与 REPLICATE 是否等价
r1_reflect = cv2.filter2D(f32, -1, k3e, borderType=cv2.BORDER_REFLECT)
r1_replicate = cv2.filter2D(f32, -1, k3e, borderType=cv2.BORDER_REPLICATE)
r1_same = int((np.abs(r1_reflect - r1_replicate) > 1e-6).sum())
# copyMakeBorder 直观展示（含 filter2D 不支持的 WRAP）
bor = {nm: cv2.copyMakeBorder(gray, 2, 2, 2, 2, bt)[:2, :4] for nm, bt in
       [("REFLECT_101", cv2.BORDER_REFLECT_101), ("REFLECT", cv2.BORDER_REFLECT),
        ("REPLICATE", cv2.BORDER_REPLICATE), ("CONSTANT", cv2.BORDER_CONSTANT),
        ("WRAP", cv2.BORDER_WRAP)]}
print("原图左上 2x4:", gray[:2, :4].tolist())
for nm, im in bor.items():
    print(f"   copyMakeBorder {nm:12s} 左上 2x4:", im.tolist())
for nm, (mx, cnt, inner) in border_stats.items():
    print(f"   {nm:10s} vs 默认: 最大差 {mx} | 差异像素 {cnt} | 内圈完全一致 {inner}")
print("半径 1 时 REFLECT vs REPLICATE 差异像素:", r1_same)
print("解读: 边界模式只影响最外 k 圈（内圈逐像素一致），但对边缘目标影响很大——"
      "CONSTANT 把边缘当 0，最大差能到 112.2。反直觉的是半径 1 时 REFLECT 与 REPLICATE "
      "完全等价（差异 0）：REFLECT 会把边缘像素本身也镜像一次，一步之内就是复制。"
      "另外 BORDER_WRAP / TRANSPARENT 在 filter2D 里不支持（会抛异常），"
      "要手工用 copyMakeBorder 铺边")

assert border_stats["REPLICATE"] == (16.88, 2143, True)
assert border_stats["CONSTANT"] == (112.2, 3568, True)
assert r1_same == 0
'''

E5_CODE = '''rs = np.random.RandomState(3)
m = rs.rand(*gray.shape)
sp = gray.copy()
sp[m < 0.02] = 0
sp[m > 0.98] = 255
gn = np.clip(gray + rs.normal(0, 20, gray.shape), 0, 255).astype(np.uint8)
# @@todo(5) 两把尺子评估去噪：std（越小越"平"）与 gap_to_clean（越小越接近真值）
# @@hint 对椒盐和高斯两种噪声，各算 中值5 / 均值5 / 双边 的 std 与 gap_to_clean
filters = lambda src: {"中值5": cv2.medianBlur(src, 5),
                       "均值5": cv2.blur(src, (5, 5)),
                       "双边": cv2.bilateralFilter(src, 5, 50, 50)}
sp_gap = {k: gap_to_clean(v, gray) for k, v in filters(sp).items()}
gn_gap = {k: gap_to_clean(v, gray) for k, v in filters(gn).items()}
sp_std = {k: round(float(v.std()), 4) for k, v in filters(sp).items()}
gn_std = {k: round(float(v.std()), 4) for k, v in filters(gn).items()}
# @@end
print("椒盐 2%: gap 原", gap_to_clean(sp, gray), "|", sp_gap, "| std 原", round(float(sp.std()), 4), sp_std)
print("高斯 σ=20: gap 原", gap_to_clean(gn, gray), "|", gn_gap, "| std 原", round(float(gn.std()), 4), gn_std)
print("解读: std 是「图有多平」，gap 是「离真值多远」，两把尺子会打架——"
      "椒盐图上均值滤波的 std 最低（41.1108 < 中值 43.7865），但 gap 反而最大（9.2347 > 5.8994），"
      "因为它把 0/255 抹成中间灰，看着平其实是糊。"
      "更反直觉的是：这个噪声密度下，「不做任何滤波」的 gap 只有 5.0841，比中值滤波还小——"
      "原图只被污染 2%，滤波却把整幅图都模糊了一遍，去噪收益抵不过模糊代价")

assert sp_gap == {"中值5": 5.8994, "均值5": 9.2347, "双边": 8.02}
assert gn_gap == {"中值5": 8.7138, "均值5": 8.5429, "双边": 7.5279}
assert sp_std["均值5"] == 41.1108 and sp_std["中值5"] == 43.7865
assert gap_to_clean(sp, gray) == 5.0841
'''

E6_CODE = '''sharp_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
lapl_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
sharped = cv2.filter2D(gray, -1, sharp_k)
flat = np.full((10, 10), 128, np.uint8)
flat_out = cv2.filter2D(flat, -1, lapl_k)
lap_f = cv2.Laplacian(f32, cv2.CV_32F)
blurred = cv2.GaussianBlur(gray, (0, 0), 3)
usm = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
# @@todo(6) 锐化三件事：sum=1 保亮度、sum=0 抹掉常量区、unsharp mask 的权重和也守 1
# @@hint 核的 sum 用 float(k.sum())；addWeighted(gray, a, blurred, b, 0) 里 a+b 应等于 1
s_sharp, s_lapl = float(sharp_k.sum()), float(lapl_k.sum())
flat_max = int(np.abs(flat_out.astype(int)).max())
sharp_std = round(float(sharped.std()), 4)
sharp_mean = round(float(sharped.mean()), 4)
usm_w = round(1.6 + (-0.6), 6)
usm_std = round(float(usm.std()), 4)
# @@end
print("锐化核 sum:", s_sharp, "| 锐化后 std:", sharp_std, "（原", round(float(gray.std()), 4), "）",
      "| 均值:", sharp_mean, "（原", round(float(gray.mean()), 4), "）")
print("拉普拉斯核 sum:", s_lapl, "| 纯常量图经它的输出绝对最大值:", flat_max)
print("Laplacian(CV_32F) 均值:", round(float(lap_f.mean()), 6),
      "min/max:", round(float(lap_f.min()), 2), round(float(lap_f.max()), 2))
print("unsharp 权重和:", usm_w, "| 后 std:", usm_std, "（原", round(float(gray.std()), 4), "）")
print("解读: 1) 锐化核 sum=1，所以亮度守恒（均值 116.1954 → 116.5697 只差 0.37），"
      "只是把 std 从 45.59 拉到 60.71；2) 拉普拉斯核 sum=0，常量区域输出恒 0——"
      "这是「边缘检测」的数学本质：对平坦区域不响应；3) unsharp mask 的权重和也是 1，"
      "（1.6 − 0.6 = 1）所以亮度不漂，只是把高频放大")

assert s_sharp == 1.0 and s_lapl == 0.0 and flat_max == 0
assert sharp_std == 60.7078 and sharp_mean == 116.5697
assert usm_w == 1.0 and usm_std == 50.7266
'''

E7_CODE = '''edge_k = np.array([[1, 0, -1]], np.float32)
dx_u8 = cv2.filter2D(gray, -1, edge_k)
dx_f = cv2.filter2D(f32, -1, edge_k)
# @@todo(7) 输出类型决定你能看到什么：uint8 把负响应截成 0，float32 保留符号
# @@hint 直接对比 min/max、负值像素数，再用 abs(响应).mean() 对比「梯度总量」
u8_min, u8_max = int(dx_u8.min()), int(dx_u8.max())
f_min, f_max = round(float(dx_f.min()), 2), round(float(dx_f.max()), 2)
n_neg = int((dx_f < 0).sum())
n_zero = int((dx_u8 == 0).sum())
abs_mean_f = round(float(np.abs(dx_f).mean()), 4)
mean_u8 = round(float(dx_u8.mean()), 4)
# @@end
print("dx 核 {1,0,-1}: uint8 输出", u8_min, "~", u8_max, "| float32 输出", f_min, "~", f_max)
print("float 里负值像素:", n_neg, "| uint8 输出里等于 0 的:", n_zero)
print("abs(float).mean():", abs_mean_f, "vs uint8.mean():", mean_u8)
print("解读: 同一组卷积，uint8 把 −202 直接截成 0（负梯度全丢），输出均值只剩 uint8 的 5.4454；"
      "float32 的 |响应| 均值是 10.971，才是真实的梯度总量。"
      "所以方向性滤波（Sobel/Laplacian/自定义核）一律先转 float32，"
      "或者用 cv2.filter2D(gray, cv2.CV_32F, k) 让输出保持浮点")

assert u8_min == 0 and u8_max == 196 and f_min == -202.0 and f_max == 196.0
assert n_neg == 70003 and n_zero == 123276
assert abs_mean_f == 10.971 and mean_u8 == 5.4454
'''

E8_CODE = '''sizes = [3, 7, 15, 31]
blur_std = {}
blur_gap = {}
for ks in sizes:
    # @@todo(8) 核越大越平滑：记录每个核的 std 与「与干净图的差」
    # @@hint 循环骨架在块外；块内 2 行——cv2.blur(f32, (ks, ks)) 后记 std 和 gap
    b = cv2.blur(f32, (ks, ks))
    blur_std[ks] = round(float(b.std()), 4)
    # @@end
for ks in sizes:
    print(f"   blur({ks:2d},{ks:2d}) std: {blur_std[ks]}")
d = round(blur_std[3] - blur_std[31], 4)
print("   std 从 3x3 到 31x31 降了", d, "| 核面积涨了", (31 * 31) / (3 * 3), "倍")
print("解读: 核大小与平滑强度不是线性关系——面积涨 106.8 倍，std 只降 7.37。"
      "因为均值滤波的方差按 1/核面积 递减，收益是开根号级的，"
      "所以「再糊一点」的边际代价很高、边际收益很低。选核大小要跟目标尺度匹配："
      "小目标用 3x3，去大块噪声才考虑 15 以上")

assert blur_std == {3: 43.4368, 7: 41.4624, 15: 39.0128, 31: 36.0681}
assert d == 7.3687
'''

# =========================================================================== #
# 讲解用代码块（与练习内容一一对应，去掉挖空标记）
# =========================================================================== #

S1_CODE = '''ks = np.ones((3, 3), np.float32) / 9.0
ka = np.array([[1, 2], [3, 4]], np.float32)
tiny = gray[:40, :40]
cv_sym = cv2.filter2D(tiny.astype(np.float32), -1, ks, borderType=cv2.BORDER_CONSTANT)
cv_asym = cv2.filter2D(tiny.astype(np.float32), -1, ka, borderType=cv2.BORDER_CONSTANT)
d_sym = max_abs_diff(correlate_zero_pad(tiny, ks), cv_sym)
d_asym_corr = max_abs_diff(correlate_zero_pad(tiny, ka), cv_asym)
d_asym_conv = max_abs_diff(correlate_zero_pad(tiny, ka[::-1, ::-1]), cv_asym)
print("对称核   手写相关 vs filter2D 最大差:", d_sym)
print("非对称核 手写相关 vs filter2D 最大差:", d_asym_corr)
print("非对称核 手写卷积(翻核) vs filter2D 最大差:", d_asym_conv)
print("解读: 前两行几乎为 0（浮点误差），第三行差出 352 —— 说明 cv2.filter2D 做的是「相关」"
      "（核不翻转）而不是数学意义的卷积。对称核看不出区别，一旦用非对称核"
      "（如 Sobel、Gabor、方向性核）就必须想清楚自己是哪个口径")

assert d_sym <= 1e-4 and d_asym_corr == 0.0
assert d_asym_conv == 352.0
'''

S2_CODE = '''k5 = np.ones((5, 5), np.float32) / 25.0
box_norm = cv2.blur(gray, (5, 5))
box_unnorm = cv2.boxFilter(gray, -1, (5, 5), normalize=False)
f2d_blur = cv2.filter2D(gray, -1, k5)
kk = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
ksum = float(kk.sum())
ksum_norm = round(float((kk / kk.sum()).sum()), 6)
m_norm = round(float(box_norm.mean()), 4)
m_unnorm = round(float(box_unnorm.mean()), 4)
d_blur_f2d = int(np.abs(box_norm.astype(int) - f2d_blur.astype(int)).max())
print("原图均值:", round(float(gray.mean()), 4))
print("blur(5,5) 均值:", m_norm, "| filter2D(均值核) 均值:", round(float(f2d_blur.mean()), 4),
      "| 两者最大差:", d_blur_f2d)
print("boxFilter 未归一化 均值:", m_unnorm, "| 唯一值数:", len(np.unique(box_unnorm)))
print("3x3 高斯样核 sum:", ksum, "→ 归一化后 sum:", ksum_norm)
print("解读: blur == filter2D(均值核)，逐像素最大差 0，两套写法完全等价。"
      "核的所有元素之和就是「直流增益」：sum=1 亮度不变；boxFilter 不归一化时 sum=25，"
      "结果被放大 25 倍直接饱和到 255（唯一值只剩 64 个）——这是新手最常踩的坑")

assert d_blur_f2d == 0 and m_norm == 116.1955
assert m_unnorm == 254.9622 and ksum == 16.0 and ksum_norm == 1.0
'''

S3_CODE = '''k1 = cv2.getGaussianKernel(5, 1.0)
k2 = cv2.getGaussianKernel(5, 2.0)
k5s = cv2.getGaussianKernel(5, 5.0)
k0 = cv2.getGaussianKernel(5, 0)
g2d = k1 @ k1.T
sum1, sum2 = round(float(k1.sum()), 6), round(float(g2d.sum()), 6)
d_sep = max_abs_diff(cv2.filter2D(f32, -1, g2d), cv2.sepFilter2D(f32, -1, k1, k1))
first1 = round(float(k1.ravel()[0]), 6)
first5 = round(float(k5s.ravel()[0]), 6)
auto0 = [round(float(v), 6) for v in k0.ravel()]
print("sigma=1 的 1D 核:", [round(float(v), 6) for v in k1.ravel()], "sum", sum1)
print("sigma=2 的 1D 核:", [round(float(v), 6) for v in k2.ravel()])
print("sigma=5 的 1D 核:", [round(float(v), 6) for v in k5s.ravel()])
print("外积核 sum:", sum2, "| 2D 滤波 vs sepFilter2D 最大差:", d_sep)
print("sigma=0（自动推算，即二项式系数）:", auto0)
print("解读: sigma 越大核越「平」（sigma=1 中心权重 0.4026，sigma=5 只有 0.2080）——"
      "平滑更强但更糊。1D 高斯的外积恰好等于 2D 高斯核，所以可以拆成横竖两趟做，"
      "这正是 sepFilter2D 快 15.5 倍的原理。sigma=0 时 OpenCV 按 ksize 反推，"
      "ksize=5 给出的是标准二项式系数 [0.0625, 0.25, 0.375, 0.25, 0.0625]")

assert sum1 == 1.0 and sum2 == 1.0 and d_sep <= 1e-4
assert first1 == 0.054489 and first5 == 0.192051
assert auto0 == [0.0625, 0.25, 0.375, 0.25, 0.0625]
'''

S4_CODE = '''k3e = np.ones((3, 3), np.float32) / 9.0
k5e = np.ones((5, 5), np.float32) / 25.0
base5 = cv2.filter2D(f32, -1, k5e, borderType=cv2.BORDER_REFLECT_101)
modes = {"REFLECT": cv2.BORDER_REFLECT, "REPLICATE": cv2.BORDER_REPLICATE,
         "CONSTANT": cv2.BORDER_CONSTANT}
border_stats = {}
for nm, bt in modes.items():
    out = cv2.filter2D(f32, -1, k5e, borderType=bt)
    d = np.abs(out - base5)
    border_stats[nm] = (round(float(d.max()), 4), int((d > 1e-6).sum()), bool((d[2:-2, 2:-2] < 1e-6).all()))
r1_reflect = cv2.filter2D(f32, -1, k3e, borderType=cv2.BORDER_REFLECT)
r1_replicate = cv2.filter2D(f32, -1, k3e, borderType=cv2.BORDER_REPLICATE)
r1_same = int((np.abs(r1_reflect - r1_replicate) > 1e-6).sum())
bor = {nm: cv2.copyMakeBorder(gray, 2, 2, 2, 2, bt)[:2, :4] for nm, bt in
       [("REFLECT_101", cv2.BORDER_REFLECT_101), ("REFLECT", cv2.BORDER_REFLECT),
        ("REPLICATE", cv2.BORDER_REPLICATE), ("CONSTANT", cv2.BORDER_CONSTANT),
        ("WRAP", cv2.BORDER_WRAP)]}
print("原图左上 2x4:", gray[:2, :4].tolist())
for nm, im in bor.items():
    print(f"   copyMakeBorder {nm:12s} 左上 2x4:", im.tolist())
for nm, (mx, cnt, inner) in border_stats.items():
    print(f"   {nm:10s} vs 默认: 最大差 {mx} | 差异像素 {cnt} | 内圈完全一致 {inner}")
print("半径 1 时 REFLECT vs REPLICATE 差异像素:", r1_same)
print("解读: 边界模式只影响最外 k 圈（内圈逐像素一致），但对边缘目标影响很大——"
      "CONSTANT 把边缘当 0，最大差能到 112.2。反直觉的是半径 1 时 REFLECT 与 REPLICATE "
      "完全等价（差异 0）：REFLECT 会把边缘像素本身也镜像一次，一步之内就是复制。"
      "另外 BORDER_WRAP / TRANSPARENT 在 filter2D 里不支持（会抛异常），"
      "要手工用 copyMakeBorder 铺边")

assert border_stats["REPLICATE"] == (16.88, 2143, True)
assert border_stats["CONSTANT"] == (112.2, 3568, True)
assert r1_same == 0
'''

S5_CODE = '''rs = np.random.RandomState(3)
m = rs.rand(*gray.shape)
sp = gray.copy()
sp[m < 0.02] = 0
sp[m > 0.98] = 255
gn = np.clip(gray + rs.normal(0, 20, gray.shape), 0, 255).astype(np.uint8)
filters = lambda src: {"中值5": cv2.medianBlur(src, 5),
                       "均值5": cv2.blur(src, (5, 5)),
                       "双边": cv2.bilateralFilter(src, 5, 50, 50)}
sp_gap = {k: gap_to_clean(v, gray) for k, v in filters(sp).items()}
gn_gap = {k: gap_to_clean(v, gray) for k, v in filters(gn).items()}
sp_std = {k: round(float(v.std()), 4) for k, v in filters(sp).items()}
gn_std = {k: round(float(v.std()), 4) for k, v in filters(gn).items()}
print("椒盐 2%: gap 原", gap_to_clean(sp, gray), "|", sp_gap, "| std 原", round(float(sp.std()), 4), sp_std)
print("高斯 σ=20: gap 原", gap_to_clean(gn, gray), "|", gn_gap, "| std 原", round(float(gn.std()), 4), gn_std)
print("解读: std 是「图有多平」，gap 是「离真值多远」，两把尺子会打架——"
      "椒盐图上均值滤波的 std 最低（41.1108 < 中值 43.7865），但 gap 反而最大（9.2347 > 5.8994），"
      "因为它把 0/255 抹成中间灰，看着平其实是糊。"
      "更反直觉的是：这个噪声密度下，「不做任何滤波」的 gap 只有 5.0841，比中值滤波还小——"
      "原图只被污染 2%，滤波却把整幅图都模糊了一遍，去噪收益抵不过模糊代价")

assert sp_gap == {"中值5": 5.8994, "均值5": 9.2347, "双边": 8.02}
assert gn_gap == {"中值5": 8.7138, "均值5": 8.5429, "双边": 7.5279}
assert sp_std["均值5"] == 41.1108 and sp_std["中值5"] == 43.7865
assert gap_to_clean(sp, gray) == 5.0841
'''

S6_CODE = '''sharp_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
lapl_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
sharped = cv2.filter2D(gray, -1, sharp_k)
flat = np.full((10, 10), 128, np.uint8)
flat_out = cv2.filter2D(flat, -1, lapl_k)
lap_f = cv2.Laplacian(f32, cv2.CV_32F)
blurred = cv2.GaussianBlur(gray, (0, 0), 3)
usm = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
s_sharp, s_lapl = float(sharp_k.sum()), float(lapl_k.sum())
flat_max = int(np.abs(flat_out.astype(int)).max())
sharp_std = round(float(sharped.std()), 4)
sharp_mean = round(float(sharped.mean()), 4)
usm_w = round(1.6 + (-0.6), 6)
usm_std = round(float(usm.std()), 4)
print("锐化核 sum:", s_sharp, "| 锐化后 std:", sharp_std, "（原", round(float(gray.std()), 4), "）",
      "| 均值:", sharp_mean, "（原", round(float(gray.mean()), 4), "）")
print("拉普拉斯核 sum:", s_lapl, "| 纯常量图经它的输出绝对最大值:", flat_max)
print("Laplacian(CV_32F) 均值:", round(float(lap_f.mean()), 6),
      "min/max:", round(float(lap_f.min()), 2), round(float(lap_f.max()), 2))
print("unsharp 权重和:", usm_w, "| 后 std:", usm_std, "（原", round(float(gray.std()), 4), "）")
print("解读: 1) 锐化核 sum=1，所以亮度守恒（均值 116.1954 → 116.5697 只差 0.37），"
      "只是把 std 从 45.59 拉到 60.71；2) 拉普拉斯核 sum=0，常量区域输出恒 0——"
      "这是「边缘检测」的数学本质：对平坦区域不响应；3) unsharp mask 的权重和也是 1，"
      "（1.6 − 0.6 = 1）所以亮度不漂，只是把高频放大")

assert s_sharp == 1.0 and s_lapl == 0.0 and flat_max == 0
assert sharp_std == 60.7078 and sharp_mean == 116.5697
assert usm_w == 1.0 and usm_std == 50.7266
'''

S7_CODE = '''edge_k = np.array([[1, 0, -1]], np.float32)
dx_u8 = cv2.filter2D(gray, -1, edge_k)
dx_f = cv2.filter2D(f32, -1, edge_k)
u8_min, u8_max = int(dx_u8.min()), int(dx_u8.max())
f_min, f_max = round(float(dx_f.min()), 2), round(float(dx_f.max()), 2)
n_neg = int((dx_f < 0).sum())
n_zero = int((dx_u8 == 0).sum())
abs_mean_f = round(float(np.abs(dx_f).mean()), 4)
mean_u8 = round(float(dx_u8.mean()), 4)
print("dx 核 {1,0,-1}: uint8 输出", u8_min, "~", u8_max, "| float32 输出", f_min, "~", f_max)
print("float 里负值像素:", n_neg, "| uint8 输出里等于 0 的:", n_zero)
print("abs(float).mean():", abs_mean_f, "vs uint8.mean():", mean_u8)
print("解读: 同一组卷积，uint8 把 −202 直接截成 0（负梯度全丢），输出均值只剩 uint8 的 5.4454；"
      "float32 的 |响应| 均值是 10.971，才是真实的梯度总量。"
      "所以方向性滤波（Sobel/Laplacian/自定义核）一律先转 float32，"
      "或者用 cv2.filter2D(gray, cv2.CV_32F, k) 让输出保持浮点")

assert u8_min == 0 and u8_max == 196 and f_min == -202.0 and f_max == 196.0
assert n_neg == 70003 and n_zero == 123276
assert abs_mean_f == 10.971 and mean_u8 == 5.4454
'''

S8_CODE = '''sizes = [3, 7, 15, 31]
blur_std = {}
blur_gap = {}
for ks in sizes:
    b = cv2.blur(f32, (ks, ks))
    blur_std[ks] = round(float(b.std()), 4)
for ks in sizes:
    print(f"   blur({ks:2d},{ks:2d}) std: {blur_std[ks]}")
d = round(blur_std[3] - blur_std[31], 4)
print("   std 从 3x3 到 31x31 降了", d, "| 核面积涨了", (31 * 31) / (3 * 3), "倍")
print("解读: 核大小与平滑强度不是线性关系——面积涨 106.8 倍，std 只降 7.37。"
      "因为均值滤波的方差按 1/核面积 递减，收益是开根号级的，"
      "所以「再糊一点」的边际代价很高、边际收益很低。选核大小要跟目标尺度匹配："
      "小目标用 3x3，去大块噪声才考虑 15 以上")

assert blur_std == {3: 43.4368, 7: 41.4624, 15: 39.0128, 31: 36.0681}
assert d == 7.3687
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch04 卷积与滤波

> 数据：`classic/home.jpg` → 灰度图 `(384, 512)` uint8，滤波素材。真值全部实跑（opencv 5.0.0）。
>
> 这一章把「滤波」从「调个 API」拉回到数学本质：**它到底在算什么、边界怎么办、
> 为什么换个核就变亮、为什么 uint8 会把边缘吃掉**。

**本章考点**

1. 手写 2D 相关（零填充 + 点积）与 `filter2D` 对账
2. **`filter2D` 做的是「相关」不是「卷积」**（核不翻转）
3. 核的**和 = 直流增益**：sum=1 保亮度、sum=25 饱和到 255
4. 高斯核：`sigma` 的影响、1D 外积 = 2D、`sigma=0` 自动推算
5. **可分离滤波** `sepFilter2D`（核元素 961 → 62）
6. **边界模式**五种补边对比 + `filter2D` 不支持 WRAP/TRANSPARENT
7. 两把尺子评估去噪：`std` vs **与真值的差**
8. 锐化核（sum=1）与拉普拉斯核（sum=0）
9. 输出类型：uint8 截断 vs float32 保负
10. 核大小与平滑强度的非线性关系

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| 相关 ≠ 卷积 | 非对称核差 352 | §4.1 |
| 核和 ≠ 1 | 未归一化直接饱和 | §4.2 |
| 边界补边 | CONSTANT 最大差 112.2 | §4.4 |
| std 会骗人 | 均值滤波 std 最低但 gap 最大 | §4.5 |
| uint8 吃负值 | 负梯度全变 0 | §4.7 |
| 核大小收益递减 | 面积涨 106.8 倍 std 只降 7.37 | §4.8 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 4.1 手写相关与「相关 vs 卷积」

卷积的定义要把核**翻转**再滑动，而 `cv2.filter2D` 是**不翻转**的（也就是说，它是相关）。
对称核看不出差别，非对称核一次就露馅：

| 做法 | 与 `filter2D` 的最大差 |
|---|---|
| 对称核（均值核）手写**相关** | 1.2e-05（浮点误差） |
| 非对称核 `[[1,2],[3,4]]` 手写**相关** | **0.0** |
| 非对称核手写**卷积**（核翻转） | **352.0** |

结论：**`filter2D` = 相关**。要真卷积，自己把核翻转（`k[::-1, ::-1]`）。
方向性核（Sobel、Gabor）必须想清楚自己用的是哪个口径。
"""
    ),
    code(S1_CODE),
    md(
        """
## 4.2 核的「和」= 直流增益

`cv2.blur` 与 `filter2D(均值核)` 逐像素最大差 **0**，是同一个东西的两种写法。
关键是核的元素之和：

| 核 | sum | 结果 |
|---|---|---|
| 均值核 5×5（已除 25） | 1 | 均值 116.195 → **116.1955**（亮度守恒） |
| `boxFilter(normalize=False)` | 25 | 均值 **254.9622**，唯一值只剩 64（饱和） |
| 高斯样核 `[[1,2,1],[2,4,2],[1,2,1]]` | 16 | 归一化后 sum = 1.0 |

**核和就是这张图会被乘上的倍数**。竞赛里写自定义核，第一件事是先算 `k.sum()`。
"""
    ),
    code(S2_CODE),
    md(
        """
## 4.3 高斯核与可分离性

`sigma` 决定核的「尖」还是「平」（ksize 固定为 5）：

| sigma | 中心权重 | 边缘权重 |
|---|---|---|
| 1.0 | **0.40262** | 0.054489 |
| 2.0 | 0.251379 | 0.152469 |
| 5.0 | **0.208046** | 0.192051 |

sigma 越大核越平 —— 平滑更强但更糊；sigma=5 时核已经几乎均匀了（跟均值核差不多）。

两个实用结论：

1. **1D 高斯的外积就是 2D 高斯核**，所以滤波能拆成横竖两趟；
   `filter2D(外积核)` 与 `sepFilter2D(横核, 纵核)` 最大差 7.6e-05。
   核元素从 31×31 = **961** 降到 31+31 = **62**（15.5 倍），这就是 `sepFilter2D` 的意义。
2. `sigma=0` 时 OpenCV 按 ksize 自动反推，ksize=5 给出的是标准二项式系数
   `[0.0625, 0.25, 0.375, 0.25, 0.0625]` —— 这也是「高斯 ≈ 多次均值」的离散体现。
"""
    ),
    code(S3_CODE),
    md(
        """
## 4.4 难点深挖：边界模式

`filter2D` 默认 `BORDER_REFLECT_101`，五种补边的实际效果（5×5 核）：

| 模式 | 与默认最大差 | 差异像素 | 内圈是否一致 |
|---|---|---|---|
| `REFLECT` | 9.96 | 2,133 | ✅ |
| `REPLICATE` | **16.88** | 2,143 | ✅ |
| `CONSTANT`（补 0） | **112.20** | 3,568 | ✅ |

两条反直觉的结论：

1. **差异只出现在最外 k 圈**（内圈逐像素一致）—— 但对「贴着画面边缘的目标」影响巨大。
2. **半径 1 时 `REFLECT` 与 `REPLICATE` 完全等价**（差异像素 0）：
   `REFLECT` 会把边缘像素本身也镜像一次（`fedcba|abc`），一步之内就是复制；
   只有 `REFLECT_101`（`gfedcb|abc`）才是「跳过边缘再镜像」。

另外 `BORDER_WRAP` / `BORDER_TRANSPARENT` **在 `filter2D` 里不支持**（会抛异常），
要手工 `copyMakeBorder` 铺边。
"""
    ),
    code(S4_CODE),
    md(
        """
## 4.5 难点深挖：两把尺子会打架

去噪到底好不好，取决于你拿什么量：

**椒盐 2%（盐 3,864 / 椒 4,033）**

| 处理 | `std` | 与干净图的平均差 |
|---|---|---|
| 不处理 | 51.5315 | **5.0841** |
| 中值 5×5 | 43.7865 | 5.8994 |
| 双边 | 47.7573 | 8.0200 |
| 均值 5×5 | **41.1108** | **9.2347** |

均值滤波的 `std` 最低（看着最「平」），但 gap 最大 —— 因为它把 0/255 抹成中间灰，
**平是平了，但糊了**。更值得记住的是：这个噪声密度下，**连「不处理」都比滤波更接近真值**
（5.0841 < 5.8994）—— 原图只被污染 2%，滤波却把整幅图都模糊了一遍。

**高斯噪声 σ=20**

| 处理 | `std` | 与干净图的平均差 |
|---|---|---|
| 不处理 | 49.7346 | 15.9185 |
| 中值 5×5 | 43.5553 | 8.7138 |
| 均值 5×5 | 42.5130 | 8.5429 |
| 双边 | 44.4988 | **7.5279** |

这里三把滤波都有明显收益（双边最好），因为高斯噪声是「处处都脏」，没有模糊代价能省。

> 结论：`std` 是「图有多平」，`gap` 是「离真值多远」。评估去噪**必须用后者**。
"""
    ),
    code(S5_CODE),
    md(
        """
## 4.6 锐化核与拉普拉斯核：看 sum

| 核 | sum | 效果 |
|---|---|---|
| 锐化 `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | **1** | std 45.5873 → **60.7078**，均值几乎不变（116.195 → 116.5697） |
| 拉普拉斯 `[[0,1,0],[1,-4,1],[0,1,0]]` | **0** | 纯常量图输出**恒为 0** |

- **sum=1 ⇒ 亮度守恒**：锐化只是把高频放大，整体明暗不变。
- **sum=0 ⇒ 对平坦区域不响应**：这正是「边缘检测」的数学本质。
- `unsharp mask` = `addWeighted(gray, 1.6, blur, -0.6, 0)`，两个权重之和也是 **1.0**，
  同样是「保亮度、放高频」的思路。

`cv2.Laplacian(f32, cv2.CV_32F)` 的均值只有 0.00233（≈0），min/max −402 / 399，
正负对称 —— 又一条可用来自检的自洽性。
"""
    ),
    code(S6_CODE),
    md(
        """
## 4.7 难点深挖：输出类型吃掉了一半信息

用方向性核 `{1, 0, -1}` 做水平差分，同一组卷积，两种输出类型：

| 输出类型 | min | max | 负值 | `mean` |
|---|---|---|---|---|
| `uint8`（默认） | **0** | 196 | 全部截成 0 | 5.4454 |
| `float32` | **−202** | 196 | 70,003 | `abs().mean()` = **10.971** |

`uint8` 把 −202 直接截成 0（负梯度全丢），输出均值只剩真实梯度总量的一半。
所以：

- 方向性滤波（Sobel / Laplacian / 自定义核）**一律先转 `float32`**，
- 或者写 `cv2.filter2D(gray, cv2.CV_32F, k)` 让输出保持浮点。

这也是「为什么我的边缘图一边有、一边没有」的经典原因。
"""
    ),
    code(S7_CODE),
    md(
        """
## 4.8 核大小与平滑强度：收益是开根号的

| 核 | std |
|---|---|
| 3×3 | 43.4368 |
| 7×7 | 41.4624 |
| 15×15 | 39.0128 |
| 31×31 | **36.0681** |

核面积从 9 涨到 961（**106.8 倍**），std 只降 **7.37**。
因为均值滤波的方差按 `1/核面积` 递减，std 是开根号级的 ——
**「再糊一点」的边际代价很高、边际收益很低。**

选核大小的原则是**跟目标尺度匹配**：小目标/保细节用 3×3，去大块噪声才考虑 15 以上。
"""
    ),
    code(S8_CODE),
    md(
        """
## 小结

1. 图像滤波 = 相关（`filter2D` 核不翻转）；真卷积要自己翻核（非对称核差 352）。
2. **核的 sum = 直流增益**：sum=1 保亮度，未归一化的 25 直接饱和到 255。
3. `blur` == `filter2D(均值核)`（最大差 0）。
4. 高斯核 sigma 越大越平；`sigma=0` 按 ksize 反推（ksize=5 → 二项式系数）。
5. **1D 外积 = 2D 高斯**，故可分离；31×31 的 961 个元素降到 62 个。
6. 边界模式只影响最外 k 圈；`REFLECT` 半径 1 时等价于 `REPLICATE`；`WRAP` 不被 `filter2D` 支持。
7. 评估去噪要用「与真值的差」，`std` 会把均值滤波吹成最优。
8. **低噪声密度下，不滤波可能比滤波更接近真值**（2% 椒盐：5.0841 < 5.8994）。
9. 核 sum=1 保亮度（锐化 / unsharp），sum=0 对平坦区无响应（拉普拉斯 / 边缘检测）。
10. 方向性滤波必须用 float32，uint8 会把负响应截成 0（梯度总量腰斩）。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch04 卷积与滤波（练习版）

> 按提示补全 `____`，跑通所有 assert。`gray`（(384,512) uint8）与 `f32`（float32 版）
> 已在 setup 里备好，`correlate_zero_pad` / `max_abs_diff` / `gap_to_clean` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；循环 / 分支写在挖空块之外（见任务 4、8）。
> 题目真值都来自实跑（opencv 5.0.0），照提示写就能对上。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md("## 任务 1：手写相关 vs `filter2D`\n\n再证明 `filter2D` 是相关不是卷积。"),
    code(E1_CODE),
    md("## 任务 2：核的 sum 决定亮度\n\n`blur` 对账 + 未归一化饱和。"),
    code(E2_CODE),
    md("## 任务 3：高斯核与可分离性\n\nsigma 影响 + 外积 + `sepFilter2D`。"),
    code(E3_CODE),
    md("## 任务 4：边界模式\n\n循环骨架已给好，块内写 3 行。"),
    code(E4_CODE),
    md("## 任务 5：两把尺子评估去噪\n\n`std` vs `gap_to_clean`。"),
    code(E5_CODE),
    md("## 任务 6：锐化核与拉普拉斯核\n\n看 sum 判断亮度行为。"),
    code(E6_CODE),
    md("## 任务 7：输出类型\n\nuint8 截断 vs float32 保负。"),
    code(E7_CODE),
    md("## 任务 8：核大小与平滑强度\n\n循环骨架已给好，块内写 2 行。"),
    code(E8_CODE),
    md(
        """
## 自查清单

- [ ] 能说出 `filter2D` 做的是相关还是卷积，以及怎么得到真卷积
- [ ] 会通过 `k.sum()` 判断一张滤波结果会不会变亮/变暗
- [ ] 知道 `sigma=0` 时高斯核长什么样
- [ ] 能解释 `sepFilter2D` 为什么快
- [ ] 知道 `REFLECT` 与 `REPLICATE` 在半径 1 时等价的原因
- [ ] 会用「与真值的差」而不是 `std` 评估去噪
- [ ] 能说出 sum=0 的核为什么能做边缘检测
- [ ] 知道方向性滤波必须用 float32 的原因
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
