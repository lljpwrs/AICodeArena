#!/usr/bin/env python3
"""cv01 图像读写与数组基础：imread/BGR/通道/resize/溢出/直方图/落盘往返。

数据：classic/camera.png（灰度存 PNG）、classic/home.jpg（真彩色）、classic/messi5.jpg。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv01_image_io.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch01_image_io"

IMPORTS = '''from pathlib import Path

import shutil
import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2
from PIL import Image

DATA = Path("data")
CL = DATA / "classic"
'''

SETUP = '''# 三张代表图
cam = cv2.imread(str(CL / "camera.png"))    # 存成 PNG 的灰度图（读进来仍是 3 通道）
home = cv2.imread(str(CL / "home.jpg"))     # 真彩色 JPEG（B-R 平均差 92.7，通道差异明显）
mess = cv2.imread(str(CL / "messi5.jpg"))   # JPEG 往返素材
print("cv2", cv2.__version__, "| camera", cam.shape, "| home", home.shape, "| messi5", mess.shape)
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
    """
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


def n_diff(a, b):
    """统计两个同形状数组中有多少个元素不相等（逐通道计数）。"""
    return int((np.asarray(a) != np.asarray(b)).sum())


tmp_dir = Path("_tmp_io")     # 落盘往返用，收尾会删掉
if not tmp_dir.exists():
    tmp_dir.mkdir()
print("脚手架就绪：probe / n_diff / tmp_dir")'''

# =========================================================================== #
# 练习（答案版内容，practice 由 builder 自动挖空）
# =========================================================================== #

E1_CODE = '''# @@todo(1) 元信息：shape / dtype / nbytes，再用灰度模式读同一张图
# @@hint cv2.imread(路径) 默认 IMREAD_COLOR → (H, W, 3)；cv2.IMREAD_GRAYSCALE → (H, W)
cam_shape = cam.shape
cam_dtype = str(cam.dtype)
cam_bytes = int(cam.nbytes)
gray_cam = cv2.imread(str(CL / "camera.png"), cv2.IMREAD_GRAYSCALE)
# @@end
print("camera:", cam_shape, cam_dtype, cam_bytes, "bytes | 灰度读:", gray_cam.shape)
print("解读: nbytes = H*W*C —— 512*512*3 = 786432，这是内存占用的最小估算")

assert cam_shape == (512, 512, 3)
assert cam_dtype == "uint8" and cam_bytes == 786432
assert gray_cam.shape == (512, 512), "灰度读少一个维度：H x W"
'''

E2_CODE = '''# @@todo(2) 读失败会怎样：cv2 静默返回 None，PIL 直接抛 FileNotFoundError
# @@hint cv2.imread 读不存在/非图片 → None；PIL 用脚手架 probe(Image.open, 路径) 接住异常
missing_cv = cv2.imread(str(CL / "nope.png"))
not_image = cv2.imread(str(DATA / "make_data.py"))
st_pil, err_pil = probe(Image.open, CL / "nope.png")
# @@end
print("cv2 读不存在的图:", missing_cv, "| cv2 读文本文件:", not_image)
print("PIL 读不存在的图 →", st_pil, err_pil)
print("解读: cv2 读失败不抛异常，只给 None。imread 之后必须判 None，"
      "否则后面 img.shape 报 AttributeError，报错位置离现场很远")

assert missing_cv is None and not_image is None
assert st_pil == "err" and err_pil == "FileNotFoundError"
'''

E3_CODE = '''# @@todo(3) 通道顺序对账：cv2 是 BGR，PIL 是 RGB，首像素必须互为倒序
# @@hint np.array(Image.open(路径)) 得到 RGB；home[..., ::-1] 把 BGR 翻成 RGB
pil_home = np.array(Image.open(CL / "home.jpg"))
pix_bgr = home[0, 0].tolist()
pix_rgb = pil_home[0, 0].tolist()
flip_ok = bool((home[..., ::-1] == pil_home).all())
max_diff = int(np.abs(home.astype(int) - pil_home[..., ::-1].astype(int)).max())
# @@end
print("cv2 首像素 BGR:", pix_bgr, "| PIL 首像素 RGB:", pix_rgb)
print("全图翻转后与 PIL 完全一致:", flip_ok, "| 最大差:", max_diff)
print("解读: 这条是 CV 第一坑——用 cv2 读图接 PIL/matplotlib 显示，不翻通道就会"
      "『人脸变阿凡达』。记法：cv2 存的是 BGR，显示库要 RGB")

assert pix_bgr == [164, 102, 31] and pix_rgb == [31, 102, 164]
assert flip_ok and max_diff == 0
'''

E4_CODE = '''# @@todo(4) 通道拆合：split 出 B/G/R 看均值（这张图偏蓝），merge 必须能还原
# @@hint b, g, r = cv2.split(img)；cv2.merge([b, g, r])
b_ch, g_ch, r_ch = cv2.split(home)
b_mean = round(float(b_ch.mean()), 4)
g_mean = round(float(g_ch.mean()), 4)
r_mean = round(float(r_ch.mean()), 4)
merge_ok = bool((cv2.merge([b_ch, g_ch, r_ch]) == home).all())
# @@end
print("通道均值 B/G/R:", b_mean, g_mean, r_mean, "| merge(split(x)) == x:", merge_ok)
print("解读: 三通道均值不相等才说明这是真彩图；camera.png 三通道完全相同"
      "（灰度存成 PNG），拿它做通道分析会看不出任何差异")

assert (b_mean, g_mean, r_mean) == (141.2399, 123.8721, 91.6084)
assert merge_ok
'''

E5_CODE = '''# @@todo(5) 灰度化两把尺子：cvtColor 加权公式 vs 三通道平均
# @@hint cvtColor(img, cv2.COLOR_BGR2GRAY)；官方权重 0.299R + 0.587G + 0.114B（R 在前！）
gray_cv = cv2.cvtColor(home, cv2.COLOR_BGR2GRAY)
gray_weighted = np.round(0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch).astype(np.uint8)
gray_avg = home.mean(axis=2)
d_weighted = int(np.abs(gray_weighted.astype(int) - gray_cv.astype(int)).max())
d_avg_max = round(float(np.abs(gray_avg - gray_cv).max()), 4)
d_avg_mean = round(float(np.abs(gray_avg - gray_cv).mean()), 4)
n_trunc_vs_cv = n_diff(gray_avg.astype(np.uint8), gray_cv)
n_trunc_vs_round = n_diff(gray_avg.astype(np.uint8), np.round(gray_avg).astype(np.uint8))
gray_mean = round(float(gray_cv.mean()), 3)
# @@end
print("手写加权 vs cvtColor 最大差:", d_weighted)
print("三通道平均 vs cvtColor: 最大差", d_avg_max, "| 平均差", d_avg_mean)
print("平均法直接 astype 截断后与 cvtColor 不同像素数:", n_trunc_vs_cv, "/", gray_cv.size)
print("解读: 加权公式等价于 cvtColor（差 0）；三通道『平均』是另一把尺子，"
      "最大差 20.33 —— 人眼对绿色最敏感，这就是权重 0.587 的来历")

assert d_weighted == 0, "手写加权公式与 cvtColor 完全等价"
assert (d_avg_max, d_avg_mean) == (20.3333, 9.8214)
assert n_trunc_vs_cv == 195079 and n_trunc_vs_round == 65393, "float→uint8 是截断不是四舍五入"
assert gray_mean == 116.195
'''

E6_CODE = '''# @@todo(6) 归一化到 [0,1] 再回写 uint8：会不会丢精度？
# @@hint img.astype(np.float32) / 255.0；回写 (f * 255).astype(np.uint8)
f32 = home.astype(np.float32) / 255.0
back_trunc = (f32 * 255).astype(np.uint8)
back_round = np.round(f32 * 255).astype(np.uint8)
n_back_trunc = n_diff(back_trunc, home)
n_back_round = n_diff(back_round, home)
f_min = round(float(f32.min()), 4)
f_max = round(float(f32.max()), 4)
# @@end
print(f32.dtype, "范围", f_min, "~", f_max, "| 占用", f32.nbytes, "bytes（uint8 的 4 倍）")
print("回写差异：截断", n_back_trunc, "| 四舍五入", n_back_round)
print("解读: n/255 再 *255 在 float32 下可逆，回写零损失；但换成『先算别的运算再回写』"
      "（如 gamma、均值），截断就会系统性压暗 —— 记住原则：回写前一律 round/clip")

assert (f_min, f_max) == (0.0, 1.0)
assert f32.nbytes == 2359296
assert n_back_trunc == 0 and n_back_round == 0, "纯归一化往返零损失"
'''

E7_CODE = '''# @@todo(7) resize 三插值：缩到 128 时 AREA / LINEAR / NEAREST 的差距
# @@hint cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)；fx/fy 走比例缩放
small_area = cv2.resize(home, (128, 128), interpolation=cv2.INTER_AREA)
small_linear = cv2.resize(home, (128, 128), interpolation=cv2.INTER_LINEAR)
small_nearest = cv2.resize(home, (128, 128), interpolation=cv2.INTER_NEAREST)
d_linear_area = round(float(np.abs(small_linear.astype(float) - small_area.astype(float)).mean()), 4)
d_nearest_area = round(float(np.abs(small_nearest.astype(float) - small_area.astype(float)).mean()), 4)
shape_half = cv2.resize(home, None, fx=0.5, fy=0.5).shape
shape_big = cv2.resize(home, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC).shape
# @@end
print("缩小 128 平均差：LINEAR-AREA", d_linear_area, "| NEAREST-AREA", d_nearest_area)
print("fx=0.5 →", shape_half, "| fx=1.5 (CUBIC) →", shape_big)
print("解读: 缩小用 INTER_AREA（像素面积平均，抗混叠），放大用 LINEAR/CUBIC；"
      "NEAREST 在缩小时直接丢像素（平均差 8.86，出现锯齿）")

assert small_area.shape == (128, 128, 3)
assert (d_linear_area, d_nearest_area) == (4.2628, 8.8559)
assert shape_half == (192, 256, 3) and shape_big == (576, 768, 3)
'''

E8_CODE = '''# @@todo(8) 像素加减的溢出：numpy 回绕 vs cv2 饱和，哪个是你想要的
# @@hint (img + 100).astype(np.uint8) 会回绕变暗；cv2.add(img, 100) 饱和在 255
wrap_add = (home + 100).astype(np.uint8)
sat_add = cv2.add(home, 100)
n_wrap_dark = int((wrap_add < home).sum())
n_sat_dark = int((sat_add < home).sum())
n_sat_255 = int((sat_add == 255).sum())
n_gt155 = int((home > 155).sum())
wrap_sub = (home - 100).astype(np.uint8)
n_sub_bright = int((wrap_sub > home).sum())
n_sub_0 = int((cv2.subtract(home, 100) == 0).sum())
n_lt100 = int((home < 100).sum())
# @@end
print("+100：回绕变暗的通道值", n_wrap_dark, "| 饱和写法变暗", n_sat_dark,
      "| 饱和到 255 的通道值", n_sat_255)
print("-100：回绕变亮的通道值", n_sub_bright, "| 饱和到 0 的通道值", n_sub_0)
print("解读: 回绕数 == 超过阈值（>155）的通道值数 =", n_gt155,
      "——uint8 加减法是模 256 运算，262 会变成 6。调亮度一律用 cv2.add/subtract")

assert n_wrap_dark == n_gt155 == 212145, "回绕数恰好等于越界像素数"
assert n_sat_dark == 0 and n_sat_255 == 215325
assert n_sub_bright == n_lt100 == 224627 and n_sub_0 == 228327
'''

E9_CODE = '''# @@todo(9) gamma 校正：把暗图提亮的非线性做法
# @@hint ((img / 255.0) ** gamma) —— gamma < 1 提亮，> 1 压暗；算完 clip 回 [0,255]
gamma_img = np.clip((home / 255.0) ** 0.5 * 255, 0, 255).astype(np.uint8)
g_mean_before = round(float(home.mean()), 4)
g_mean_after = round(float(gamma_img.mean()), 4)
n_brightened = int((gamma_img > home).sum())
n_darkened = int((gamma_img < home).sum())
# @@end
print("均值", g_mean_before, "→", g_mean_after, "| 提亮通道值", n_brightened, "| 压暗", n_darkened)
print("解读: gamma=0.5 让整体变亮；0 和 255 是幂运算的不动点，所以纯黑纯白不变。"
      "『均值直方图都对了』是调亮的标准验收")

assert (g_mean_before, g_mean_after) == (118.9068, 165.7185)
assert n_darkened == 0, "gamma<1 不会压暗任何像素"
'''

E10_CODE = '''# @@todo(10) ROI 切片与掩码统计：只关心感兴趣区域
# @@hint 切片 img[y1:y2, x1:x2] 是视图（改 ROI 会改原图）；cv2.mean(img, mask) 带掩码统计
roi = home[100:200, 200:300]
roi_mean = round(float(roi.mean()), 4)
full_mean = round(float(home.mean()), 4)
gray_home = cv2.cvtColor(home, cv2.COLOR_BGR2GRAY)
mask = np.zeros(gray_home.shape, np.uint8)
mask[100:200, 200:300] = 255
masked_mean = round(float(cv2.mean(gray_home, mask)[0]), 4)
hist = cv2.calcHist([gray_home], [0], None, [256], [0, 256]).ravel()
# @@end
print("ROI", roi.shape, "均值", roi_mean, "| 全图均值", full_mean, "| 掩码内灰度均值", masked_mean)
print("直方图 sum:", int(hist.sum()), "== 像素数", gray_home.size,
      "| 峰值 bin", int(hist.argmax()), "计数", int(hist.max()))
print("解读: ROI 是视图不是拷贝（roi[:] = 0 会连带改原图，要隔离用 .copy()）；"
      "直方图 256 个 bin 的计数总和恒等于像素数，这是自检公式")

assert roi.shape == (100, 100, 3)
assert roi_mean == 103.233 and full_mean == 118.9068
assert masked_mean == 93.2983
assert int(hist.sum()) == gray_home.size == 196608
assert int(hist.argmax()) == 111 and int(hist.max()) == 3908
'''

E11_CODE = '''# @@todo(11) 落盘往返：PNG 无损、JPEG 有损，q90 与 q50 各差多少
# @@hint cv2.imwrite(路径, img, [cv2.IMWRITE_JPEG_QUALITY, 90])；再 imread 回来对账
png_path = str(tmp_dir / "messi.png")
jpg90_path = str(tmp_dir / "messi_q90.jpg")
jpg50_path = str(tmp_dir / "messi_q50.jpg")
cv2.imwrite(png_path, mess)
cv2.imwrite(jpg90_path, mess, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
cv2.imwrite(jpg50_path, mess, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
back_png = cv2.imread(png_path)
back_jpg90 = cv2.imread(jpg90_path)
back_jpg50 = cv2.imread(jpg50_path)
n_png = n_diff(back_png, mess)
n_jpg90 = n_diff(back_jpg90, mess)
n_jpg50 = n_diff(back_jpg50, mess)
max_jpg90 = int(np.abs(back_jpg90.astype(int) - mess.astype(int)).max())
size_png = Path(png_path).stat().st_size
size_jpg90 = Path(jpg90_path).stat().st_size
size_jpg50 = Path(jpg50_path).stat().st_size
# @@end
print("体积 png", size_png, "| jpg q90", size_jpg90, "| jpg q50", size_jpg50)
print("不同通道值数 png", n_png, "| q90", n_jpg90, "| q50", n_jpg50, "（总", mess.size, "）")
print("解读: PNG 逐位无损（0 差异）；JPEG 有损且压缩比越高丢得越多——"
      "所以『标签/掩码图必须存 PNG』，而竞赛交付图片素材通常存 JPEG 省体积")

assert n_png == 0, "PNG 无损：往返 0 差异"
assert n_jpg90 == 424160 and n_jpg50 == 494055
assert max_jpg90 == 46
assert size_jpg90 == 53807 and size_png == 316930 and size_jpg50 == 19634

shutil.rmtree(tmp_dir, ignore_errors=True)   # 清理实验文件
print("已清理", tmp_dir)
'''

# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''cam_shape, cam_dtype, cam_bytes = cam.shape, str(cam.dtype), int(cam.nbytes)
gray_cam = cv2.imread(str(CL / "camera.png"), cv2.IMREAD_GRAYSCALE)
print("camera:", cam_shape, cam_dtype, cam_bytes, "bytes | 灰度读:", gray_cam.shape)
print("解读: 图像在内存里就是一个 3 维 ndarray（H, W, C），dtype 恒为 uint8（0~255）；"
      "nbytes = H*W*C 是最小内存估算，竞赛里上传大图前先算这个")

assert cam_shape == (512, 512, 3) and cam_bytes == 786432
'''

S2_CODE = '''missing_cv = cv2.imread(str(CL / "nope.png"))
not_image = cv2.imread(str(DATA / "make_data.py"))
st_pil, err_pil = probe(Image.open, CL / "nope.png")
print("cv2 读不存在的图:", missing_cv, "| cv2 读文本文件:", not_image)
print("PIL 读不存在的图 →", st_pil, err_pil)
print("解读: cv2 读失败静默返回 None，不抛异常——imread 之后必须 if img is None 兜底，"
      "否则 AttributeError 会在很远的地方才炸；PIL 走的是「抛异常」路线")

assert missing_cv is None and not_image is None
assert st_pil == "err"
'''

S3_CODE = '''pil_home = np.array(Image.open(CL / "home.jpg"))
print("cv2 首像素 BGR:", home[0, 0].tolist(), "| PIL 首像素 RGB:", pil_home[0, 0].tolist())
print("翻转后与 PIL 一致:", bool((home[..., ::-1] == pil_home).all()))
print("解读: OpenCV 的历史包袱是 BGR；PIL / matplotlib / 大多数模型库要 RGB。"
      "两个方向都要会写：cv2 读进来想显示 → img[:, :, ::-1]；"
      "RGB 喂 cv2 处理 → img[:, :, ::-1] 翻回去")

assert home[0, 0].tolist() == [164, 102, 31]
assert (home[..., ::-1] == pil_home).all()
'''

S4_CODE = '''b_ch, g_ch, r_ch = cv2.split(home)
print("通道均值 B/G/R:", round(float(b_ch.mean()), 4), round(float(g_ch.mean()), 4),
      round(float(r_ch.mean()), 4))
print("merge(split(x)) == x:", bool((cv2.merge([b_ch, g_ch, r_ch]) == home).all()))
print("解读: home.jpg 偏蓝（B 141 > G 124 > R 92）——三通道均值相等说明是灰度图存成的彩色 PNG，"
      "camera.png 就是这种：拿它演示通道差异会一无所获")

assert round(float(b_ch.mean()), 4) == 141.2399
assert (cv2.merge([b_ch, g_ch, r_ch]) == home).all()
'''

S5_CODE = '''gray_cv = cv2.cvtColor(home, cv2.COLOR_BGR2GRAY)
gray_weighted = np.round(0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch).astype(np.uint8)
gray_avg = home.mean(axis=2)
print("手写加权 vs cvtColor 最大差:", int(np.abs(gray_weighted.astype(int) - gray_cv.astype(int)).max()))
print("三通道平均 vs cvtColor: 最大差", round(float(np.abs(gray_avg - gray_cv).max()), 4),
      "| 平均差", round(float(np.abs(gray_avg - gray_cv).mean()), 4))
print("截断回写 vs 四舍五入 不同像素数:", n_diff(gray_avg.astype(np.uint8), np.round(gray_avg).astype(np.uint8)))
print("解读: 灰度 = 0.299R + 0.587G + 0.114B（人眼对绿最敏感）。"
      "三通道『平均』是另一把尺子，最大差 20.33；float 平均直接 astype 是截断，"
      "系统性偏暗 0.5 个灰阶——记住 round")

assert int(np.abs(gray_weighted.astype(int) - gray_cv.astype(int)).max()) == 0
assert round(float(np.abs(gray_avg - gray_cv).max()), 4) == 20.3333
assert n_diff(gray_avg.astype(np.uint8), np.round(gray_avg).astype(np.uint8)) == 65393
'''

S6_CODE = '''f32 = home.astype(np.float32) / 255.0
print("归一化:", f32.dtype, round(float(f32.min()), 4), "~", round(float(f32.max()), 4),
      "| 内存", f32.nbytes, "bytes")
print("回写差异（截断 / 四舍五入）:",
      n_diff((f32 * 255).astype(np.uint8), home), n_diff(np.round(f32 * 255).astype(np.uint8), home))
print("解读: 归一化是深度学习的输入规范（[0,1] 或标准化），float32 占 4 倍内存；"
      "纯 n/255*255 往返可逆，但任何中间运算之后都要 round + clip 再回写")

assert f32.nbytes == 2359296
assert n_diff((f32 * 255).astype(np.uint8), home) == 0
'''

S7_CODE = '''small_area = cv2.resize(home, (128, 128), interpolation=cv2.INTER_AREA)
small_linear = cv2.resize(home, (128, 128), interpolation=cv2.INTER_LINEAR)
small_nearest = cv2.resize(home, (128, 128), interpolation=cv2.INTER_NEAREST)
print("缩小 128 平均差：LINEAR-AREA", round(float(np.abs(small_linear.astype(float) - small_area.astype(float)).mean()), 4),
      "| NEAREST-AREA", round(float(np.abs(small_nearest.astype(float) - small_area.astype(float)).mean()), 4))
print("fx=0.5 →", cv2.resize(home, None, fx=0.5, fy=0.5).shape,
      "| fx=1.5 →", cv2.resize(home, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC).shape)
print("解读: cv2.resize 的 size 参数是 (宽, 高) —— 与 shape 的 (H, W) 顺序相反，这是高频低级错误。"
      "缩小用 AREA，放大用 LINEAR/CUBIC，放大别用 NEAREST（马赛克）")

assert small_area.shape == (128, 128, 3)
assert round(float(np.abs(small_nearest.astype(float) - small_area.astype(float)).mean()), 4) == 8.8559
'''

S8_CODE = '''print("+100 回绕变暗:", int(((home + 100).astype(np.uint8) < home).sum()),
      "| cv2.add 饱和变暗:", int((cv2.add(home, 100) < home).sum()),
      "| 饱和到 255:", int((cv2.add(home, 100) == 255).sum()),
      "| 原值 >155:", int((home > 155).sum()))
print("-100 回绕变亮:", int(((home - 100).astype(np.uint8) > home).sum()),
      "| cv2.subtract 到 0:", int((cv2.subtract(home, 100) == 0).sum()),
      "| 原值 <100:", int((home < 100).sum()))
print("解读: uint8 加减是模 256 运算（262 → 6），画面出现『黑点阵』；"
      "cv2.add/subtract 饱和在 [0,255]。调亮度、做残差图一律用 cv2 的运算")

assert int(((home + 100).astype(np.uint8) < home).sum()) == int((home > 155).sum())
assert int((cv2.subtract(home, 100) == 0).sum()) == 228327
'''

S9_CODE = '''gamma_img = np.clip((home / 255.0) ** 0.5 * 255, 0, 255).astype(np.uint8)
print("均值", round(float(home.mean()), 4), "→", round(float(gamma_img.mean()), 4),
      "| 提亮通道值", int((gamma_img > home).sum()), "| 压暗", int((gamma_img < home).sum()))
print("解读: gamma 校正 = 幂函数变换，gamma<1 提亮暗部（对比度重分配），"
      "0/255 是不动点。竞赛里『图像太暗』一类预处理题的标配; "
      "等价写法：查表 LUT —— lut = np.array([((i / 255.0) ** 0.5) * 255 for i in range(256)]).astype(np.uint8)")

assert round(float(gamma_img.mean()), 4) == 165.7185
assert int((gamma_img < home).sum()) == 0
'''

S10_CODE = '''roi = home[100:200, 200:300]
gray_home = cv2.cvtColor(home, cv2.COLOR_BGR2GRAY)
mask = np.zeros(gray_home.shape, np.uint8)
mask[100:200, 200:300] = 255
hist = cv2.calcHist([gray_home], [0], None, [256], [0, 256]).ravel()
print("ROI", roi.shape, "均值", round(float(roi.mean()), 4), "| 全图均值", round(float(home.mean()), 4))
print("掩码内灰度均值:", round(float(cv2.mean(gray_home, mask)[0]), 4))
print("直方图 sum:", int(hist.sum()), "== 像素数", gray_home.size, "| 峰值 bin", int(hist.argmax()))
print("解读: 切片是视图——roi[:] = 0 会连带改原图（隔离要 .copy()）。"
      "calcHist 的通道参数是列表 [0]；直方图计数总和 = 像素数是自检公式，"
      "『灰度<100 占比』直接 hist[:100].sum()/hist.sum() 就能算")

assert roi.shape == (100, 100, 3)
assert round(float(cv2.mean(gray_home, mask)[0]), 4) == 93.2983
assert int(hist.sum()) == 196608
'''

S11_CODE = '''png_path, jpg90_path, jpg50_path = (str(tmp_dir / n) for n in
                                      ["messi.png", "messi_q90.jpg", "messi_q50.jpg"])
cv2.imwrite(png_path, mess)
cv2.imwrite(jpg90_path, mess, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
cv2.imwrite(jpg50_path, mess, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
print("体积：png", Path(png_path).stat().st_size, "| jpg q90", Path(jpg90_path).stat().st_size,
      "| jpg q50", Path(jpg50_path).stat().st_size)
print("不同通道值数（总", mess.size, "）：png", n_diff(cv2.imread(png_path), mess),
      "| q90", n_diff(cv2.imread(jpg90_path), mess), "| q50", n_diff(cv2.imread(jpg50_path), mess))
print("解读: PNG 无损（0 差异）；JPEG 有损，q90 已丢 75% 的通道值（最大差 46）——"
      "掩码/标签图必须 PNG，照片素材用 JPEG 省体积。读回来的 dtype/shape 与写前一致，"
      "唯一区别是像素值")

assert n_diff(cv2.imread(png_path), mess) == 0
assert Path(jpg90_path).stat().st_size == 53807
shutil.rmtree(tmp_dir, ignore_errors=True)
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch01 图像读写与数组基础

> 数据：`classic/` 下的经典测试图——`camera.png`（灰度存 PNG）、`home.jpg`（真彩色）、
> `messi5.jpg`（JPEG 往返素材）。CV 的一切操作都建立在「图像 = ndarray」这个事实上，
> 本章把读、看、算、写四件事的真值和坑一次讲清。

**本章考点**

1. `imread` 元信息（shape / dtype / nbytes）与灰度模式读取
2. **读失败静默返回 `None`**（不抛异常）——必须判 None
3. **BGR vs RGB**：cv2 与 PIL 首像素互为倒序
4. 通道拆合 `split` / `merge` 与通道统计
5. 灰度化两把尺子：`cvtColor` 加权公式 vs 三通道平均
6. 归一化 float32 与回写的截断 / 四舍五入
7. `resize` 插值选型（size 是 (宽,高)，与 shape 相反）
8. **uint8 加减溢出**：numpy 回绕 vs `cv2.add` / `cv2.subtract` 饱和
9. gamma 校正与 LUT
10. ROI 视图 vs 拷贝、掩码统计、直方图自检公式
11. 落盘往返：PNG 无损、JPEG 有损

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| BGR / RGB | 不翻通道人脸变阿凡达 | §1.3 |
| imread 返回 None | 报错位置离现场很远 | §1.2 |
| 灰度图存成 PNG | 三通道相同，通道分析看不出差异 | §1.4 |
| uint8 回绕 | 262 → 6，画面出现黑点阵 | §1.8 |
| resize 参数顺序 | `(w, h)` 与 `shape=(H, W)` 相反 | §1.7 |
| float 回写截断 | 均值/gamma 后不 round 会系统性偏暗 | §1.5/§1.6 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 1.1 读取与元信息

图像 = `(H, W, C)` 的 `uint8` ndarray。`nbytes = H*W*C` 是最小内存估算；
`IMREAD_GRAYSCALE` 直接读成单通道 `(H, W)`，比读三通道再 `cvtColor` 少一次拷贝。
"""
    ),
    code(S1_CODE),
    md(
        """
## 1.2 难点深挖：`imread` 失败返回 `None`

OpenCV 是 C++ 血统的错误码风格：路径写错、文件不是图片、权限不足，**一律返回 `None`**，
不抛异常。所以生产代码的固定动作是 `if img is None: raise ...`。
PIL 相反——直接抛 `FileNotFoundError`，用脚手架 `probe()` 把它接成返回值才好写断言。
"""
    ),
    code(S2_CODE),
    md(
        """
## 1.3 难点深挖：BGR 与 RGB

OpenCV 读进来是 **BGR**；PIL / matplotlib / torchvision / 大多数预训练权重期望 **RGB**。
`home.jpg` 首像素 cv2 给 `[164, 102, 31]`、PIL 给 `[31, 102, 164]`——互为正逆序。
"""
    ),
    code(S3_CODE),
    md(
        """
## 1.4 通道拆合与统计

`split` 拆通道看统计量是「这图是什么货色」的第一手证据：三通道均值相等 ⇒ 灰度图存成的
彩色 PNG（`camera.png` 就是），做颜色类题目必须换图。
"""
    ),
    code(S4_CODE),
    md(
        """
## 1.5 难点深挖：灰度化两把尺子

官方的灰度公式是 **0.299R + 0.587G + 0.114B**（人眼对绿最敏感）。
手写加权与 `cvtColor` 完全等价（差 0）；三通道简单**平均**是另一把尺子，
最大差 **20.33**。另外 float 均值直接 `astype(np.uint8)` 是**截断**——
与四舍五入差 65393 个像素，画面整体偏暗半级。
"""
    ),
    code(S5_CODE),
    md(
        """
## 1.6 归一化与回写

深度学习输入规范是 `[0,1]`（`astype(np.float32)/255`）或标准化；代价是 4 倍内存。
纯归一化往返可逆（0 差异），但**任何中间运算之后回写，都要 `round` + `clip`**。
"""
    ),
    code(S6_CODE),
    md(
        """
## 1.7 难点深挖：resize 的插值选型

`cv2.resize(img, (w, h))` —— **参数顺序是 (宽, 高)**，与 `shape=(H, W)` 相反，高频低级错误。
缩小首选 `INTER_AREA`（面积平均、抗混叠），放大用 `LINEAR` / `CUBIC`；
`NEAREST` 缩小时直接丢像素（与 AREA 平均差 **8.86**，边缘出现锯齿）。
"""
    ),
    code(S7_CODE),
    md(
        """
## 1.8 难点深挖：uint8 溢出

`uint8` 加减是**模 256** 运算：`(img + 100)` 在 200 处变 44，画面出现黑点阵。
本图回绕的通道值数 **212145** 恰好等于「原值 >155 的通道值数」——溢出不是玄学，是可精确
对账的。做亮度调整、残差图一律用 `cv2.add` / `cv2.subtract`（饱和在 `[0,255]`）；
`convertScaleAbs` 是另一条常走的路。
"""
    ),
    code(S8_CODE),
    md(
        """
## 1.9 gamma 校正

`((img/255)**gamma)*255`：gamma < 1 提亮暗部，0/255 是幂函数不动点。
本图 gamma=0.5 把均值从 **118.91** 提到 **165.72**，且没有任何像素被压暗。
工程上等价写法是 256 项 LUT（一次建表、后续查表，快得多）。
"""
    ),
    code(S9_CODE),
    md(
        """
## 1.10 ROI、掩码与直方图

- ROI 切片 `img[y1:y2, x1:x2]` 是**视图**：`roi[:] = 0` 会连带改原图，隔离要 `.copy()`。
- `cv2.mean(img, mask)` 一行拿到掩码内均值；掩码是 `uint8` 的 0/255 图。
- `calcHist` 的通道参数必须写列表 `[0]`；**计数总和恒等于像素数**，是自检公式；
  想看「灰度 <100 的占比」，`hist[:100].sum() / hist.sum()` 直接算。
"""
    ),
    code(S10_CODE),
    md(
        """
## 1.11 落盘往返：PNG 无损、JPEG 有损

`imwrite` + `imread` 是交付前的必做验收：

| 格式 | 体积 | 与原始不同通道值数 | 最大差 |
|---|---|---|---|
| PNG | 316,930 | **0**（逐位无损） | 0 |
| JPEG q90 | 53,807 | 424,160 / 562,248 | 46 |
| JPEG q50 | 19,634 | 494,055 / 562,248 | 98 |

结论：**掩码/标签/中间结果存 PNG；照片素材存 JPEG**。
"""
    ),
    code(S11_CODE),
    md(
        """
## 小结

1. 图像 = `(H, W, C)` `uint8` ndarray；`nbytes = H*W*C`。
2. `imread` 失败返回 `None`（不抛异常）；PIL 抛 `FileNotFoundError`。
3. **cv2 是 BGR**：`img[..., ::-1]` 是 BGR↔RGB 的通用写法。
4. 灰度 = `0.299R + 0.587G + 0.114B`；三通道平均是另一把尺子（最大差 20.33）。
5. float 回写一律 `round` + `clip`；纯归一化往返无损。
6. `resize` 参数是 `(宽, 高)`；缩小用 `INTER_AREA`。
7. `uint8` 加减会回绕（模 256）；调亮度用 `cv2.add/subtract`。
8. gamma <1 提亮；LUT 是工程写法。
9. ROI 是视图；直方图计数总和 = 像素数。
10. PNG 无损、JPEG 有损；掩码图必须 PNG。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch01 图像读写与数组基础（练习版）

> 按提示补全 `____`，跑通所有 assert。`cam` / `home` / `mess` 已在 setup 里读好，
> `probe` / `n_diff` / `tmp_dir` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；循环 / 分支写在挖空块之外。
> 题目真值都来自实跑（opencv 5.0.0），照提示写就能对上。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md("## 任务 1：元信息与灰度读取\n\n`shape` / `dtype` / `nbytes` + `IMREAD_GRAYSCALE`。"),
    code(E1_CODE),
    md("## 任务 2：读失败怎么办\n\ncv2 返回 `None` vs PIL 抛异常。"),
    code(E2_CODE),
    md("## 任务 3：BGR 与 RGB 对账\n\n首像素互为倒序，全图翻转一致。"),
    code(E3_CODE),
    md("## 任务 4：通道拆合与统计\n\n`split` / `merge` + 三通道均值。"),
    code(E4_CODE),
    md("## 任务 5：灰度化两把尺子\n\n`cvtColor` vs 手写加权 vs 三通道平均。"),
    code(E5_CODE),
    md("## 任务 6：归一化与回写\n\nfloat32 归一化、截断与四舍五入。"),
    code(E6_CODE),
    md("## 任务 7：resize 三插值\n\nAREA / LINEAR / NEAREST + fx/fy。"),
    code(E7_CODE),
    md("## 任务 8：uint8 溢出\n\n回绕 vs 饱和，精确对账。"),
    code(E8_CODE),
    md("## 任务 9：gamma 校正\n\n非线性提亮与不动点。"),
    code(E9_CODE),
    md("## 任务 10：ROI、掩码与直方图\n\n视图语义 + 掩码统计 + 自检公式。"),
    code(E10_CODE),
    md("## 任务 11：落盘往返\n\nPNG 无损、JPEG q90/q50 有损。"),
    code(E11_CODE),
    md(
        """
## 自查清单

- [ ] 能说出 cv2 与 PIL 的通道顺序差异，并写出两种翻转写法
- [ ] 能解释 `imread` 返回 `None` 的后果与防御写法
- [ ] 能手写灰度加权公式（R/G/B 系数别记反）
- [ ] 知道 `resize` 的参数是 `(宽, 高)`
- [ ] 能解释 uint8 溢出为什么等于「越界像素数」
- [ ] 知道掩码图必须存 PNG
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
