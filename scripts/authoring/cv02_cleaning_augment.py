#!/usr/bin/env python3
"""cv02 图像清洗与数据增强：坏图体检 / 几何变换 / 亮度对比度 / 噪声 / 均衡 / bbox 同步。

数据：classic/home.jpg（亮度·噪声·均衡素材）、helmet/site_03.jpg（几何·bbox 同步素材）、
      helmet/labels.json（像素 xyxy 标注）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv02_cleaning_augment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch02_cleaning_augment"

IMPORTS = '''import json
import shutil
from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

DATA = Path("data")
CL = DATA / "classic"
HE = DATA / "helmet"
'''

SETUP = '''home = cv2.imread(str(CL / "home.jpg"))      # 真彩色，亮度/噪声/均衡素材
site3 = cv2.imread(str(HE / "site_03.jpg"))  # 工地小图，几何变换与 bbox 同步素材
labels = json.loads((HE / "labels.json").read_text())   # 教学标注（像素 xyxy）
print("home", home.shape, "| site_03", site3.shape,
      "| 标注图片数", len(labels["images"]), "| 类别", labels["classes"])
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。"""
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


tmp_dir = Path("_tmp_aug")    # 人造坏图落盘用，任务 1 收尾会删掉
if not tmp_dir.exists():
    tmp_dir.mkdir()
print("脚手架就绪：probe / tmp_dir")'''

# =========================================================================== #
# 练习（答案版内容，practice 由 builder 自动挖空）
# =========================================================================== #

E1_CODE = '''# @@todo(1) 坏图体检：造 4 张「有病」的图，看谁能被 imread 读出来
# @@hint 过小图用 np.zeros((8,8,3),np.uint8) 再赋值；常量图 np.full((64,64,3),200,np.uint8)；
# @@hint 假 jpg 用 Path(...).write_bytes(b"\\xff\\xd8\\xff\\xe0not-a-jpeg")；空文件写 b""
tiny_img = np.zeros((8, 8, 3), np.uint8)
tiny_img[:, :] = (10, 20, 30)
flat_img = np.full((64, 64, 3), 200, np.uint8)
tiny_p, flat_p = (str(tmp_dir / n) for n in ["tiny.png", "flat.png"])
corrupt_p, empty_p = (str(tmp_dir / n) for n in ["corrupt.jpg", "empty.png"])
cv2.imwrite(tiny_p, tiny_img)
cv2.imwrite(flat_p, flat_img)
Path(corrupt_p).write_bytes(b"\\xff\\xd8\\xff\\xe0not-a-jpeg")
Path(empty_p).write_bytes(b"")
bad_tiny, bad_flat = cv2.imread(tiny_p), cv2.imread(flat_p)
bad_corrupt, bad_empty = cv2.imread(corrupt_p), cv2.imread(empty_p)
flat_std = round(float(bad_flat.std()), 4)
tiny_unq = int(len(np.unique(bad_tiny)))
# @@end
print("tiny:", bad_tiny.shape, "std", round(float(bad_tiny.std()), 4), "唯一值", tiny_unq)
print("flat:", bad_flat.shape, "std", flat_std, "唯一值", int(len(np.unique(bad_flat))))
print("corrupt:", bad_corrupt, "| empty:", bad_empty)
print("解读: 坏图分三档——读不出（None）、太小（8×8 没法训练）、太单一（std=0 全常量）。"
      "清洗第一步永远是「逐张 imread + 判 None + 记 shape/std」，把可疑图挑出来再决定丢还是修")

assert bad_tiny.shape == (8, 8, 3) and tiny_unq == 3
assert flat_std == 0.0
assert bad_corrupt is None and bad_empty is None
shutil.rmtree(tmp_dir, ignore_errors=True)   # 体检完即清，不留垃圾
'''

E2_CODE = '''rows = []
files = sorted(p for p in CL.iterdir() if p.suffix.lower() in {".png", ".jpg"})
for p in files:
    # @@todo(2) 数据集体检：读每张图，记下 (名字, shape, 保留两位的 std)
    # @@hint 循环骨架已在挖空块之外；块内只写两行——cv2.imread(str(p)) 与 rows.append(...)
    img = cv2.imread(str(p))
    rows.append((p.name, img.shape, round(float(img.std()), 2)))
    # @@end
heights = [r[1][0] for r in rows]
widths = [r[1][1] for r in rows]
flattest = min(rows, key=lambda r: r[2])
tallest = max(rows, key=lambda r: r[1][0])
print("图数:", len(rows), "| 高", min(heights), "~", max(heights), "| 宽", min(widths), "~", max(widths))
print("最小 std:", flattest, "| 最高的图:", tallest[0], tallest[1])
print("解读: 同一批「经典图」尺寸从 172 到 660 差 3.8 倍、宽从 324 到 868 差 2.7 倍——"
      "进模型前必须统一到同一尺寸（后面 §2.3 的 resize）；std 最小的 moon.png 只有 13.33，"
      "是典型的低对比度图，做边缘/阈值类题目要特别当心")

assert len(rows) == 14
assert (min(heights), max(heights)) == (172, 660)
assert (min(widths), max(widths)) == (324, 868)
assert flattest[0] == "moon.png" and flattest[2] == 13.33
'''

E3_CODE = '''# @@todo(3) 三种 flip + rotate90 转四次：几何变换只挪像素，不改统计
# @@hint cv2.flip(img, 1) 左右 / cv2.flip(img, 0) 上下 / cv2.flip(img, -1) 同时翻；
# @@hint cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 顺时针 90°，连做 4 次应回到原图
hflip = cv2.flip(site3, 1)
vflip = cv2.flip(site3, 0)
both = cv2.flip(site3, -1)
r90 = cv2.rotate(site3, cv2.ROTATE_90_CLOCKWISE)
r180 = cv2.rotate(r90, cv2.ROTATE_90_CLOCKWISE)
r270 = cv2.rotate(r180, cv2.ROTATE_90_CLOCKWISE)
r360 = cv2.rotate(r270, cv2.ROTATE_90_CLOCKWISE)
mean_same = round(float(hflip.mean()), 4) == round(float(site3.mean()), 4)
# @@end
print("原", site3.shape, "→ r90", r90.shape, "→ r360", r360.shape)
print("hflip 首列 == 原末列:", bool((hflip[:, 0] == site3[:, -1]).all()))
print("flip(-1) == flip(1) 再 flip(0):", bool((both == cv2.flip(hflip, 0)).all()))
print("转 4 次回原图:", bool((r360 == site3).all()), "| 均值不变:", mean_same)
print("解读: 翻转/旋转是「像素搬家」，全部像素值都在，只是换了位置——所以 mean/std 完全不变，"
      "图分类任务可以放心用；但注意 rotate90 会把 H 和 W 互换（293×440 → 440×293），"
      "后续若按 shape 取参（如 bbox 归一化）必须先更新尺寸")

assert r90.shape == (440, 293, 3)
assert bool((r360 == site3).all()) and mean_same
assert bool((hflip[:, 0] == site3[:, -1]).all())
'''

E4_CODE = '''# @@todo(4) warpAffine 三件事：旋转补边、缩放、平移留黑
# @@hint cv2.getRotationMatrix2D((cx, cy), 角度, 缩放) 给 2x3 矩阵；warpAffine 第 3 个参数是输出尺寸 (宽, 高)
# @@hint 「全黑占比」= 三通道求和为 0 的像素比例，用来量黑边有多大
h0, w0 = site3.shape[:2]
M45 = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), 45, 1.0)
rot45 = cv2.warpAffine(site3, M45, (w0, h0))
black45 = round(float((rot45.sum(axis=2) == 0).mean()), 4)
cos_a, sin_a = abs(M45[0, 0]), abs(M45[0, 1])
nw, nh = int(h0 * sin_a + w0 * cos_a), int(h0 * cos_a + w0 * sin_a)
Mbig = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), 45, 1.0)
Mbig[0, 2] += nw / 2 - w0 / 2
Mbig[1, 2] += nh / 2 - h0 / 2
rot_big = cv2.warpAffine(site3, Mbig, (nw, nh))
black_big = round(float((rot_big.sum(axis=2) == 0).mean()), 4)
small = cv2.resize(site3, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
Mtr = np.float32([[1, 0, 20], [0, 1, 10]])
trans = cv2.warpAffine(site3, Mtr, (w0, h0))
black_tr = round(float((trans.sum(axis=2) == 0).mean()), 4)
# @@end
print("原尺寸内旋转 45°:", rot45.shape, "黑占比", black45)
print("扩边旋转 45°:", (nw, nh), "黑占比", black_big)
print("缩小一半:", small.shape, "| 平移(20,10) 黑占比:", black_tr)
print("解读: 旋转必然留黑角——扩边只是把图完整放进来，黑占比反而升到 0.5163"
      "（≈ 1 − 128920/268324，即旋转矩形的面积占比）。要彻底没黑边只能「先扩边再裁剪内接矩形」。"
      "平移同理：移多少像素，边缘就空多少。缩放不产生黑边，但它改尺寸")

assert rot45.shape == (293, 440, 3) and black45 == 0.2179
assert (nw, nh) == (518, 518) and black_big == 0.5163
assert small.shape == (146, 220, 3) and black_tr == 0.078
'''

E5_CODE = '''# @@todo(5) 亮度与对比度：convertScaleAbs(alpha, beta) 一行搞定
# @@hint alpha 是增益（改对比度）、beta 是偏置（改亮度）；饱和比例 = (原值 + beta > 255) 的通道占比
bright = cv2.convertScaleAbs(home, alpha=1.0, beta=50)
strong = cv2.convertScaleAbs(home, alpha=1.5, beta=0)
sat_ratio = round(float((home.astype(np.int16) + 50 > 255).mean()), 4)
mean0, mean_b, mean_s = (round(float(x.mean()), 4) for x in [home, bright, strong])
std_s = round(float(strong.std()), 4)
# @@end
print("原均值", mean0, "→ beta=+50:", mean_b, "→ alpha=1.5:", mean_s)
print("原 std", round(float(home.std()), 4), "→ alpha=1.5 后 std", std_s)
print("beta=+50 会饱和的通道占比:", sat_ratio)
print("解读: 加偏置简单但会把亮部顶到 255（本图 3.58% 的通道值被削平，细节永久丢失）；"
      "乘增益拉对比度更「安全」，std 从 58.99 拉到 79.85。"
      "注意 cv2.convertScaleAbs 内部做了饱和与取绝对值，比写 home*1.5 后再 astype 靠谱")

assert mean0 == 118.9068 and mean_b == 168.663 and mean_s == 171.1903
assert std_s == 79.854 and sat_ratio == 0.0358
'''

E6_CODE = '''# @@todo(6) gamma 校正：先归一化再取幂，必须 clip 才敢回 uint8
# @@hint gamma < 1 提亮；公式 np.clip(((img / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
g = 0.5
gamma_img = np.clip(((home / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
dark_cnt = int((gamma_img < home).sum())
bump = round(float(gamma_img.mean()) - float(home.mean()), 4)
# @@end
print("gamma=0.5 均值:", round(float(gamma_img.mean()), 4), "（原", round(float(home.mean()), 4), "）",
      "| 提升", bump)
print("被压暗的像素数:", dark_cnt, "| 0/255 是否不动点:",
      bool(gamma_img[home == 0].size == 0 or (gamma_img[home == 0] == 0).all()))
print("解读: gamma 是乘性变换，暗部提升多、亮部提升少，所以比线性加偏置「高级」。"
      "重放大小：0 → 0、255 → 255 是不动点，所以 0 变暗的像素一个都没有（信息不丢，只是重新分配）。"
      "竞赛里「图像太暗/过曝」类预处理题的标配答案就是 gamma")

assert round(float(gamma_img.mean()), 4) == 165.7185
assert dark_cnt == 0 and bump == 46.8117
'''

E7_CODE = '''# @@todo(7) 两类噪声：高斯（每个像素都动）vs 椒盐（只动 5% 但很猛），都要 seed 可复现
# @@hint rs = np.random.RandomState(1)；高斯 rs.normal(0,20,shape) 后 clip 回 uint8；
# @@hint 椒盐先 rs.rand(h,w) 再按 <0.025 置 255（盐）、>0.975 置 0（椒）
rs = np.random.RandomState(1)
gauss = np.clip(home + rs.normal(0, 20, home.shape), 0, 255).astype(np.uint8)
hh, ww = home.shape[:2]
mm = rs.rand(hh, ww)
sp = home.copy()
sp[mm < 0.025] = 255
sp[mm > 0.975] = 0
n_salt, n_pepper = int((mm < 0.025).sum()), int((mm > 0.975).sum())
sp_affect = int((sp != home).sum())
gauss_std = round(float(gauss.std()), 4)
gauss_mean = round(float(gauss.mean()), 4)
# @@end
print("高斯 sigma=20: 均值", gauss_mean, "std", gauss_std, "| 逐通道平均差",
      round(float(np.abs(gauss.astype(int) - home).mean()), 4))
print("椒盐: 盐", n_salt, "椒", n_pepper, "| 受影响通道值", sp_affect, "→ 3*(盐+椒) =", 3 * (n_salt + n_pepper))
print("解读: 高斯噪声动全部像素但每处只挪一点（均值几乎不变，std 从 58.99 抬到 61.50）；"
      "椒盐只动 5% 的像素，但一改成 0/255 就是「毁灭性」的。"
      "注意 sp_affect (29426) 比 3*(盐+椒) (29481) 少 55 —— 因为少数像素本来就是 0 或 255，改了等于没改")

assert gauss_mean == 118.8005 and gauss_std == 61.5029
assert (n_salt, n_pepper) == (4880, 4947) and sp_affect == 29426
assert sp_affect <= 3 * (n_salt + n_pepper)
'''

E8_CODE = '''# @@todo(8) 滤波降噪对账：高斯模糊治高斯噪声，中值滤波治椒盐
# @@hint 高斯 cv2.GaussianBlur(img, (5,5), 0)；中值 cv2.medianBlur(img, 5)，核必须是奇数
# @@hint 评价指标用「与原干净图的平均绝对差」，越小说明去噪越干净
blur_g = cv2.GaussianBlur(gauss, (5, 5), 0)
med_sp = cv2.medianBlur(sp, 5)
med_g = cv2.medianBlur(gauss, 5)
gap = lambda a: round(float(np.abs(a.astype(int) - home).mean()), 4)
gap_gg, gap_sm, gap_gm = gap(blur_g), gap(med_sp), gap(med_g)
blur_g_std = round(float(blur_g.std()), 4)
# @@end
print("高斯噪声 → 高斯滤波:", gap_gg, "| 高斯噪声 → 中值滤波:", gap_gm)
print("椒盐 噪声 → 中值滤波:", gap_sm)
print("高斯滤波后 std:", blur_g_std, "（噪声图 61.5029，干净图 58.9911）")
print("解读: 中值滤波对椒盐更有效（6.03 < 8.04 的高斯滤波结果），因为中值天然无视极端值；"
      "反过来拿中值治高斯噪声效果最差（8.71）。滤波必然同时磨掉细节——"
      "所以增强时「加噪」和「降噪」是两件事：加噪是为了让模型抗噪，降噪是为了让图好看")

assert gap_gg == 8.0414 and gap_gm == 8.7095 and gap_sm == 6.0301
assert blur_g_std == 56.6187
'''

E9_CODE = '''# @@todo(9) 直方图均衡：equalizeHist 全局拉 vs CLAHE 分块限幅
# @@hint cv2.equalizeHist(gray)；cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
gray = cv2.cvtColor(home, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
stds = [round(float(x.std()), 4) for x in [gray, eq, clahe]]
means = [round(float(x.mean()), 4) for x in [gray, eq, clahe]]
# @@end
print("std  : 原", stds[0], "| equalizeHist", stds[1], "| CLAHE", stds[2])
print("均值 : 原", means[0], "| equalizeHist", means[1], "| CLAHE", means[2])
print("equalizeHist 值域:", int(eq.min()), "~", int(eq.max()))
print("解读: equalizeHist 把累积分布强行拉平，std 从 45.59 冲到 73.32（对比度大增），"
      "但它对整张图用同一套映射，局部过曝/欠曝救不回来；CLAHE 分 8×8 块各自均衡再限幅插值，"
      "std 只到 52.97，肉眼更自然。竞赛做「低照度增强」优先 CLAHE")

assert stds == [45.5873, 73.3206, 52.9708]
assert (int(eq.min()), int(eq.max())) == (0, 255)
'''

E10_CODE = '''# @@todo(10) 增强必须同步改标注：hflip / vflip / 旋转外接框 / 裁剪越界
# @@hint hflip 后 x 坐标镜像：(x1,y1,x2,y2) → (W-x2, y1, W-x1, y2)；vflip 用 H-y
# @@hint 旋转后取「四角变换后的外接框」；裁剪后要把框 clip 回画布并丢掉空框
rec = labels["images"]["site_03.jpg"]
H0, W0 = rec["height"], rec["width"]
bx = np.array([o["bbox_xyxy"] for o in rec["objects"]], np.float64)
fb = np.stack([W0 - bx[:, 2], bx[:, 1], W0 - bx[:, 0], bx[:, 3]], axis=1)
vb = np.stack([bx[:, 0], H0 - bx[:, 3], bx[:, 2], H0 - bx[:, 1]], axis=1)
M = cv2.getRotationMatrix2D((W0 / 2, H0 / 2), 45, 1.0)
c4 = np.array([[bx[0, 0], bx[0, 1], 1], [bx[0, 2], bx[0, 1], 1],
               [bx[0, 2], bx[0, 3], 1], [bx[0, 0], bx[0, 3], 1]], np.float64).T
rc = (M @ c4).T
nbb = [rc[:, 0].min(), rc[:, 1].min(), rc[:, 0].max(), rc[:, 1].max()]
rot_ratio = round(((nbb[2] - nbb[0]) * (nbb[3] - nbb[1])) /
                  ((bx[0, 2] - bx[0, 0]) * (bx[0, 3] - bx[0, 1])), 4)
cx1, cx2 = np.clip(bx[:, 0], 50, 150), np.clip(bx[:, 2], 50, 150)
cy1, cy2 = np.clip(bx[:, 1], 100, 200), np.clip(bx[:, 3], 100, 200)
crop_ok = int(((cx2 > cx1) & (cy2 > cy1)).sum())
# @@end
gf = cv2.cvtColor(cv2.flip(site3, 1), cv2.COLOR_BGR2GRAY)
x1, y1, x2, y2 = bx[0].astype(int)
fx1, fy1, fx2, fy2 = fb[0].astype(int)
po = cv2.cvtColor(site3[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
pf = gf[fy1:fy2, fx1:fx2]
mirror_ok = bool((po[:, ::-1] == pf).all())
print("site_03", (H0, W0), "框数", len(bx), "| 首框", bx[0].tolist())
print("hflip 首框:", fb[0].tolist(), "| vflip 首框:", vb[0].tolist())
print("框内 patch", po.shape, "镜像后完全一致:", mirror_ok, "| 均值", round(float(po.mean()), 4))
print("旋转 45° 外接框:", [round(float(v), 2) for v in nbb], "| 面积膨胀比", rot_ratio)
print("裁剪 100:200,50:150 → 有效框", crop_ok, "个（clip 后），其余", len(bx) - crop_ok, "个被裁没了")
print("解读: 这是 CV 增强最容易埋雷的地方——图翻了标注没翻，模型学到的是「帽子在右边」。"
      "规则：hflip/vflip/缩放 都有一一对应的坐标公式（且可逆），"
      "而旋转会让轴对齐框面积膨胀 2.07 倍、裁剪会丢框，所以目标检测一般不乱旋转、"
      "裁剪后必须 clip 并丢弃无效框")

assert len(bx) == 6 and bx[0].tolist() == [47.0, 186.0, 66.0, 199.0]
assert fb[0].tolist() == [374.0, 186.0, 393.0, 199.0]
assert vb[0].tolist() == [47.0, 94.0, 66.0, 107.0]
assert crop_ok == 2 and rot_ratio == 2.0729 and mirror_ok and po.shape == (13, 19)
'''

# =========================================================================== #
# 讲解用代码块（与练习内容一一对应，去掉挖空标记）
# =========================================================================== #

S1_CODE = '''tiny_img = np.zeros((8, 8, 3), np.uint8)
tiny_img[:, :] = (10, 20, 30)
flat_img = np.full((64, 64, 3), 200, np.uint8)
tiny_p, flat_p = (str(tmp_dir / n) for n in ["tiny.png", "flat.png"])
corrupt_p, empty_p = (str(tmp_dir / n) for n in ["corrupt.jpg", "empty.png"])
cv2.imwrite(tiny_p, tiny_img)
cv2.imwrite(flat_p, flat_img)
Path(corrupt_p).write_bytes(b"\\xff\\xd8\\xff\\xe0not-a-jpeg")
Path(empty_p).write_bytes(b"")
bad_tiny, bad_flat = cv2.imread(tiny_p), cv2.imread(flat_p)
bad_corrupt, bad_empty = cv2.imread(corrupt_p), cv2.imread(empty_p)
flat_std = round(float(bad_flat.std()), 4)
tiny_unq = int(len(np.unique(bad_tiny)))
print("tiny:", bad_tiny.shape, "std", round(float(bad_tiny.std()), 4), "唯一值", tiny_unq)
print("flat:", bad_flat.shape, "std", flat_std, "唯一值", int(len(np.unique(bad_flat))))
print("corrupt:", bad_corrupt, "| empty:", bad_empty)
print("解读: 坏图分三档——读不出（None）、太小（8×8 没法训练）、太单一（std=0 全常量）。"
      "清洗第一步永远是「逐张 imread + 判 None + 记 shape/std」，把可疑图挑出来再决定丢还是修")

assert bad_tiny.shape == (8, 8, 3) and tiny_unq == 3
assert flat_std == 0.0
assert bad_corrupt is None and bad_empty is None
shutil.rmtree(tmp_dir, ignore_errors=True)   # 体检完即清，不留垃圾
'''

S2_CODE = '''rows = []
files = sorted(p for p in CL.iterdir() if p.suffix.lower() in {".png", ".jpg"})
for p in files:
    img = cv2.imread(str(p))
    rows.append((p.name, img.shape, round(float(img.std()), 2)))
heights = [r[1][0] for r in rows]
widths = [r[1][1] for r in rows]
flattest = min(rows, key=lambda r: r[2])
tallest = max(rows, key=lambda r: r[1][0])
print("图数:", len(rows), "| 高", min(heights), "~", max(heights), "| 宽", min(widths), "~", max(widths))
print("最小 std:", flattest, "| 最高的图:", tallest[0], tallest[1])
print("解读: 同一批「经典图」尺寸从 172 到 660 差 3.8 倍、宽从 324 到 868 差 2.7 倍——"
      "进模型前必须统一到同一尺寸（后面 §2.3 的 resize）；std 最小的 moon.png 只有 13.33，"
      "是典型的低对比度图，做边缘/阈值类题目要特别当心")

assert len(rows) == 14
assert (min(heights), max(heights)) == (172, 660)
assert (min(widths), max(widths)) == (324, 868)
assert flattest[0] == "moon.png" and flattest[2] == 13.33
'''

S3_CODE = '''hflip = cv2.flip(site3, 1)
vflip = cv2.flip(site3, 0)
both = cv2.flip(site3, -1)
r90 = cv2.rotate(site3, cv2.ROTATE_90_CLOCKWISE)
r180 = cv2.rotate(r90, cv2.ROTATE_90_CLOCKWISE)
r270 = cv2.rotate(r180, cv2.ROTATE_90_CLOCKWISE)
r360 = cv2.rotate(r270, cv2.ROTATE_90_CLOCKWISE)
mean_same = round(float(hflip.mean()), 4) == round(float(site3.mean()), 4)
print("原", site3.shape, "→ r90", r90.shape, "→ r360", r360.shape)
print("hflip 首列 == 原末列:", bool((hflip[:, 0] == site3[:, -1]).all()))
print("flip(-1) == flip(1) 再 flip(0):", bool((both == cv2.flip(hflip, 0)).all()))
print("转 4 次回原图:", bool((r360 == site3).all()), "| 均值不变:", mean_same)
print("解读: 翻转/旋转是「像素搬家」，全部像素值都在，只是换了位置——所以 mean/std 完全不变，"
      "图分类任务可以放心用；但注意 rotate90 会把 H 和 W 互换（293×440 → 440×293），"
      "后续若按 shape 取参（如 bbox 归一化）必须先更新尺寸")

assert r90.shape == (440, 293, 3)
assert bool((r360 == site3).all()) and mean_same
assert bool((hflip[:, 0] == site3[:, -1]).all())
'''

S4_CODE = '''h0, w0 = site3.shape[:2]
M45 = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), 45, 1.0)
rot45 = cv2.warpAffine(site3, M45, (w0, h0))
black45 = round(float((rot45.sum(axis=2) == 0).mean()), 4)
cos_a, sin_a = abs(M45[0, 0]), abs(M45[0, 1])
nw, nh = int(h0 * sin_a + w0 * cos_a), int(h0 * cos_a + w0 * sin_a)
Mbig = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), 45, 1.0)
Mbig[0, 2] += nw / 2 - w0 / 2
Mbig[1, 2] += nh / 2 - h0 / 2
rot_big = cv2.warpAffine(site3, Mbig, (nw, nh))
black_big = round(float((rot_big.sum(axis=2) == 0).mean()), 4)
small = cv2.resize(site3, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
Mtr = np.float32([[1, 0, 20], [0, 1, 10]])
trans = cv2.warpAffine(site3, Mtr, (w0, h0))
black_tr = round(float((trans.sum(axis=2) == 0).mean()), 4)
print("原尺寸内旋转 45°:", rot45.shape, "黑占比", black45)
print("扩边旋转 45°:", (nw, nh), "黑占比", black_big)
print("缩小一半:", small.shape, "| 平移(20,10) 黑占比:", black_tr)
print("解读: 旋转必然留黑角——扩边只是把图完整放进来，黑占比反而升到 0.5163"
      "（≈ 1 − 128920/268324，即旋转矩形的面积占比）。要彻底没黑边只能「先扩边再裁剪内接矩形」。"
      "平移同理：移多少像素，边缘就空多少。缩放不产生黑边，但它改尺寸")

assert rot45.shape == (293, 440, 3) and black45 == 0.2179
assert (nw, nh) == (518, 518) and black_big == 0.5163
assert small.shape == (146, 220, 3) and black_tr == 0.078
'''

S5_CODE = '''bright = cv2.convertScaleAbs(home, alpha=1.0, beta=50)
strong = cv2.convertScaleAbs(home, alpha=1.5, beta=0)
sat_ratio = round(float((home.astype(np.int16) + 50 > 255).mean()), 4)
mean0, mean_b, mean_s = (round(float(x.mean()), 4) for x in [home, bright, strong])
std_s = round(float(strong.std()), 4)
print("原均值", mean0, "→ beta=+50:", mean_b, "→ alpha=1.5:", mean_s)
print("原 std", round(float(home.std()), 4), "→ alpha=1.5 后 std", std_s)
print("beta=+50 会饱和的通道占比:", sat_ratio)
print("解读: 加偏置简单但会把亮部顶到 255（本图 3.58% 的通道值被削平，细节永久丢失）；"
      "乘增益拉对比度更「安全」，std 从 58.99 拉到 79.85。"
      "注意 cv2.convertScaleAbs 内部做了饱和与取绝对值，比写 home*1.5 后再 astype 靠谱")

assert mean0 == 118.9068 and mean_b == 168.663 and mean_s == 171.1903
assert std_s == 79.854 and sat_ratio == 0.0358
'''

S6_CODE = '''g = 0.5
gamma_img = np.clip(((home / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
dark_cnt = int((gamma_img < home).sum())
bump = round(float(gamma_img.mean()) - float(home.mean()), 4)
print("gamma=0.5 均值:", round(float(gamma_img.mean()), 4), "（原", round(float(home.mean()), 4), "）",
      "| 提升", bump)
print("被压暗的像素数:", dark_cnt, "| 0/255 是否不动点:",
      bool(gamma_img[home == 0].size == 0 or (gamma_img[home == 0] == 0).all()))
print("解读: gamma 是乘性变换，暗部提升多、亮部提升少，所以比线性加偏置「高级」。"
      "重放大小：0 → 0、255 → 255 是不动点，所以 0 变暗的像素一个都没有（信息不丢，只是重新分配）。"
      "竞赛里「图像太暗/过曝」类预处理题的标配答案就是 gamma")

assert round(float(gamma_img.mean()), 4) == 165.7185
assert dark_cnt == 0 and bump == 46.8117
'''

S7_CODE = '''rs = np.random.RandomState(1)
gauss = np.clip(home + rs.normal(0, 20, home.shape), 0, 255).astype(np.uint8)
hh, ww = home.shape[:2]
mm = rs.rand(hh, ww)
sp = home.copy()
sp[mm < 0.025] = 255
sp[mm > 0.975] = 0
n_salt, n_pepper = int((mm < 0.025).sum()), int((mm > 0.975).sum())
sp_affect = int((sp != home).sum())
gauss_std = round(float(gauss.std()), 4)
gauss_mean = round(float(gauss.mean()), 4)
print("高斯 sigma=20: 均值", gauss_mean, "std", gauss_std, "| 逐通道平均差",
      round(float(np.abs(gauss.astype(int) - home).mean()), 4))
print("椒盐: 盐", n_salt, "椒", n_pepper, "| 受影响通道值", sp_affect, "→ 3*(盐+椒) =", 3 * (n_salt + n_pepper))
print("解读: 高斯噪声动全部像素但每处只挪一点（均值几乎不变，std 从 58.99 抬到 61.50）；"
      "椒盐只动 5% 的像素，但一改成 0/255 就是「毁灭性」的。"
      "注意 sp_affect (29426) 比 3*(盐+椒) (29481) 少 55 —— 因为少数像素本来就是 0 或 255，改了等于没改")

assert gauss_mean == 118.8005 and gauss_std == 61.5029
assert (n_salt, n_pepper) == (4880, 4947) and sp_affect == 29426
assert sp_affect <= 3 * (n_salt + n_pepper)
'''

S8_CODE = '''blur_g = cv2.GaussianBlur(gauss, (5, 5), 0)
med_sp = cv2.medianBlur(sp, 5)
med_g = cv2.medianBlur(gauss, 5)
gap = lambda a: round(float(np.abs(a.astype(int) - home).mean()), 4)
gap_gg, gap_sm, gap_gm = gap(blur_g), gap(med_sp), gap(med_g)
blur_g_std = round(float(blur_g.std()), 4)
print("高斯噪声 → 高斯滤波:", gap_gg, "| 高斯噪声 → 中值滤波:", gap_gm)
print("椒盐 噪声 → 中值滤波:", gap_sm)
print("高斯滤波后 std:", blur_g_std, "（噪声图 61.5029，干净图 58.9911）")
print("解读: 中值滤波对椒盐更有效（6.03 < 8.04 的高斯滤波结果），因为中值天然无视极端值；"
      "反过来拿中值治高斯噪声效果最差（8.71）。滤波必然同时磨掉细节——"
      "所以增强时「加噪」和「降噪」是两件事：加噪是为了让模型抗噪，降噪是为了让图好看")

assert gap_gg == 8.0414 and gap_gm == 8.7095 and gap_sm == 6.0301
assert blur_g_std == 56.6187
'''

S9_CODE = '''gray = cv2.cvtColor(home, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
stds = [round(float(x.std()), 4) for x in [gray, eq, clahe]]
means = [round(float(x.mean()), 4) for x in [gray, eq, clahe]]
print("std  : 原", stds[0], "| equalizeHist", stds[1], "| CLAHE", stds[2])
print("均值 : 原", means[0], "| equalizeHist", means[1], "| CLAHE", means[2])
print("equalizeHist 值域:", int(eq.min()), "~", int(eq.max()))
print("解读: equalizeHist 把累积分布强行拉平，std 从 45.59 冲到 73.32（对比度大增），"
      "但它对整张图用同一套映射，局部过曝/欠曝救不回来；CLAHE 分 8×8 块各自均衡再限幅插值，"
      "std 只到 52.97，肉眼更自然。竞赛做「低照度增强」优先 CLAHE")

assert stds == [45.5873, 73.3206, 52.9708]
assert (int(eq.min()), int(eq.max())) == (0, 255)
'''

S10_CODE = '''rec = labels["images"]["site_03.jpg"]
H0, W0 = rec["height"], rec["width"]
bx = np.array([o["bbox_xyxy"] for o in rec["objects"]], np.float64)
fb = np.stack([W0 - bx[:, 2], bx[:, 1], W0 - bx[:, 0], bx[:, 3]], axis=1)
vb = np.stack([bx[:, 0], H0 - bx[:, 3], bx[:, 2], H0 - bx[:, 1]], axis=1)
M = cv2.getRotationMatrix2D((W0 / 2, H0 / 2), 45, 1.0)
c4 = np.array([[bx[0, 0], bx[0, 1], 1], [bx[0, 2], bx[0, 1], 1],
               [bx[0, 2], bx[0, 3], 1], [bx[0, 0], bx[0, 3], 1]], np.float64).T
rc = (M @ c4).T
nbb = [rc[:, 0].min(), rc[:, 1].min(), rc[:, 0].max(), rc[:, 1].max()]
rot_ratio = round(((nbb[2] - nbb[0]) * (nbb[3] - nbb[1])) /
                  ((bx[0, 2] - bx[0, 0]) * (bx[0, 3] - bx[0, 1])), 4)
cx1, cx2 = np.clip(bx[:, 0], 50, 150), np.clip(bx[:, 2], 50, 150)
cy1, cy2 = np.clip(bx[:, 1], 100, 200), np.clip(bx[:, 3], 100, 200)
crop_ok = int(((cx2 > cx1) & (cy2 > cy1)).sum())
gf = cv2.cvtColor(cv2.flip(site3, 1), cv2.COLOR_BGR2GRAY)
x1, y1, x2, y2 = bx[0].astype(int)
fx1, fy1, fx2, fy2 = fb[0].astype(int)
po = cv2.cvtColor(site3[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
pf = gf[fy1:fy2, fx1:fx2]
mirror_ok = bool((po[:, ::-1] == pf).all())
print("site_03", (H0, W0), "框数", len(bx), "| 首框", bx[0].tolist())
print("hflip 首框:", fb[0].tolist(), "| vflip 首框:", vb[0].tolist())
print("框内 patch", po.shape, "镜像后完全一致:", mirror_ok, "| 均值", round(float(po.mean()), 4))
print("旋转 45° 外接框:", [round(float(v), 2) for v in nbb], "| 面积膨胀比", rot_ratio)
print("裁剪 100:200,50:150 → 有效框", crop_ok, "个（clip 后），其余", len(bx) - crop_ok, "个被裁没了")
print("解读: 这是 CV 增强最容易埋雷的地方——图翻了标注没翻，模型学到的是「帽子在右边」。"
      "规则：hflip/vflip/缩放 都有一一对应的坐标公式（且可逆），"
      "而旋转会让轴对齐框面积膨胀 2.07 倍、裁剪会丢框，所以目标检测一般不乱旋转、"
      "裁剪后必须 clip 并丢弃无效框")

assert len(bx) == 6 and bx[0].tolist() == [47.0, 186.0, 66.0, 199.0]
assert fb[0].tolist() == [374.0, 186.0, 393.0, 199.0]
assert vb[0].tolist() == [47.0, 94.0, 66.0, 107.0]
assert crop_ok == 2 and rot_ratio == 2.0729 and mirror_ok and po.shape == (13, 19)
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch02 图像清洗与数据增强

> 数据：`classic/home.jpg`（亮度·噪声·均衡素材）、`helmet/site_03.jpg` + `helmet/labels.json`
> （几何变换与标注同步素材）。
>
> 竞赛里「数据准备及处理」这一档分，考的就是**把脏数据变成能训练的干净数据**，以及
> **增强时标注跟着一起变**。本章 10 个真值全部实跑（opencv 5.0.0）。

**本章考点**

1. 坏图体检三档：读不出 / 太小 / 太单一（std=0）
2. 数据集尺寸与对比度体检
3. 几何变换：`flip` 三模式、`rotate` 90° 与 H/W 互换
4. `warpAffine`：旋转黑角、扩边、平移留黑、缩放
5. 亮度与对比度：`convertScaleAbs(alpha, beta)` 与饱和代价
6. gamma 校正与不动点
7. 高斯噪声 vs 椒盐噪声（seed 可复现）
8. 滤波降噪对账：高斯 vs 中值
9. 直方图均衡：`equalizeHist` vs `CLAHE`
10. **增强与标注同步**：hflip / vflip / 缩放 / 旋转外接框 / 裁剪越界

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| 坏图静默通过 | `imread` 返回 `None` 或 8×8，训练时才炸 | §2.1 |
| 旋转黑角 | 扩边后黑占比反而升到 0.5163 | §2.4 |
| 亮度饱和 | 3.58% 通道被顶到 255，不可逆 | §2.5 |
| 椒盐 vs 高斯 | 去噪滤波器要配对，否则越滤越糊 | §2.8 |
| 均衡过头 | `equalizeHist` std 45.59→73.32 常过曝 | §2.9 |
| **标注不同步** | 图翻了框没翻，模型学反 | §2.10 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 2.1 坏图体检

清洗第一步永远是「逐张 `imread` + 判 `None` + 记 `shape`/`std`」。三类毛病：

| 类型 | 现象 | 本例 |
|---|---|---|
| 读不出 | `imread` 返回 `None` | 假 JPEG、0 字节 PNG |
| 太小 | shape 只有个位数 | 8×8 |
| 太单一 | `std == 0`、唯一值 1 | 64×64 纯色 |

后两类 `imread` 都会成功返回数组——**不会报错**，所以必须靠体检发现。
"""
    ),
    code(S1_CODE),
    md(
        """
## 2.2 数据集体检：尺寸与对比度

`classic/` 14 张图：高 172~660（3.8 倍）、宽 324~868（2.7 倍）。同批数据尺寸不齐，
训练前必须统一 resize。`std` 是「对比度」的廉价代理：`moon.png` 只有 13.33，
做边缘检测/阈值分割这类依赖梯度的任务时要当心（**低对比度 = 弱梯度 = 边缘检不出来**）。
"""
    ),
    code(S2_CODE),
    md(
        """
## 2.3 几何变换：flip 与 rotate

`cv2.flip(img, 1/0/-1)` = 左右 / 上下 / 同时翻。翻转与 90° 旋转都是**像素搬家**：
所有像素值都在，只换位置 ⇒ **mean/std 完全不变**（实测四种增强均值极差 0.0）。

唯一要注意的是 `rotate90` 把 H 和 W 互换：`293×440` → `440×293`。
"""
    ),
    code(S3_CODE),
    md(
        """
## 2.4 难点深挖：`warpAffine` 与黑边

`getRotationMatrix2D(中心, 角度, 缩放)` 只给 2×3 矩阵，`warpAffine` 第 3 个参数是
**输出尺寸 `(宽, 高)`**，且**默认不扩边**——转 45° 就有 21.79% 像素变黑。

想不丢内容就先算扩边矩形的尺寸再平移中心：

| 做法 | 输出尺寸 | 全黑占比 |
|---|---|---|
| 原地转 45° | `(293, 440)` | **0.2179** |
| 扩边后转 45° | `(518, 518)` | **0.5163** |

注意反直觉的一点：扩边后黑占比**更高**（0.5163 ≈ 1 − 128920/268324，即旋转矩形在其
外接框中的面积占比）。扩边解决的是「图被切掉」，不是「没有黑角」——
要彻底无黑边只能扩边后再裁内接矩形。
"""
    ),
    code(S4_CODE),
    md(
        """
## 2.5 亮度与对比度：`convertScaleAbs(alpha, beta)`

`alpha` 增益（对比度）、`beta` 偏置（亮度），一行完成 `|alpha*x + beta|` 并饱和到
`uint8`。代价是**亮部会被削平**：`beta=+50` 时有 3.58% 的通道值顶到 255，这部分
细节永久丢失（不是「被压缩」，是没了）。

`alpha=1.5` 则把 std 从 58.99 拉到 79.85——对比度是真的提上去了。
"""
    ),
    code(S5_CODE),
    md(
        """
## 2.6 gamma 校正

`(x/255)^γ * 255`，γ<1 提亮。相比线性加偏置，它的好处是**暗部提得多、亮部提得少**，
并且 `0 → 0`、`255 → 255` 是不动点——本图 0 个像素被压暗（信息只是重新分配，没丢）。

均值 118.9068 → 165.7185。实现上务必 `np.clip(..., 0, 255)` 再 `astype(np.uint8)`，
否则 float 转 uint8 的回绕会打出黑点阵。

工程写法是查表：`lut = np.array([((i / 255.0) ** 0.5) * 255 for i in range(256)]).astype(np.uint8)`。
"""
    ),
    code(S6_CODE),
    md(
        """
## 2.7 两类噪声

| 噪声 | 作用范围 | 幅度 | 效果 |
|---|---|---|---|
| 高斯 σ=20 | **全部**像素（58.99 万通道值） | 小 | 均值几乎不变，std 58.99 → 61.50 |
| 椒盐 5% | 只碰 9,827 个像素（≈2.94 万通道值） | 极大（→0/255） | 局部毁灭性 |

椒盐的「受影响通道值」29426 略小于 `3*(盐+椒) = 29481`——差 55，
因为少数像素本来就是 0 或 255，改了等于没改。**这类「差一点点」是最值得写进报告的自检线索。**

增强时**必须固定 seed**（`np.random.RandomState(1)`），否则同一次实验跑两遍得到两份增强集，
调参结论直接失效。
"""
    ),
    code(S7_CODE),
    md(
        """
## 2.8 难点深挖：滤波器要跟噪声配对

拿「与原干净图的平均绝对差」当尺子（越小越干净）：

| 噪声 → 滤波器 | 平均绝对差 |
|---|---|
| 椒盐 → 中值滤波 | **6.0301** |
| 高斯 → 高斯滤波 | **8.0414** |
| 高斯 → 中值滤波 | 8.7095（最差） |

原因：**中值天然无视极端值**，正好克椒盐；而高斯噪声是「每个像素都偏一点」，
中值取不到平滑信息反而更糊。结论：**加噪是为了让模型抗噪，降噪是为了让图好看，两件事别混。**
"""
    ),
    code(S8_CODE),
    md(
        """
## 2.9 直方图均衡：全局 vs 分块限幅

| 方法 | std | 均值 |
|---|---|---|
| 原灰度 | 45.5873 | 116.195 |
| `equalizeHist` | **73.3206** | 128.6276 |
| `CLAHE(2.0, 8×8)` | 52.9708 | 119.4462 |

`equalizeHist` 用同一套映射拉平整张图的累积分布——对比度冲得猛，但局部过曝/欠曝
救不回来；`CLAHE` 分 8×8 块各自均衡再限幅插值，结果更自然。低照度增强优先 CLAHE。
"""
    ),
    code(S9_CODE),
    md(
        """
## 2.10 难点深挖：增强与标注同步

这是 CV 增强最容易埋雷的地方——**图翻了，标注框没翻**，模型学到的是「安全帽在画面右边」。
`site_03.jpg`（293×440，6 个框）的实测：

| 变换 | 首框 `[47, 186, 66, 199]` 变成 | 坐标公式 |
|---|---|---|
| hflip | `[374, 186, 393, 199]` | `(W-x2, y1, W-x1, y2)` |
| vflip | `[47, 94, 66, 107]` | `(x1, H-y2, x2, H-y1)` |
| 缩放 0.5 | `[23.5, 93.0, 33.0, 99.5]` | 坐标同乘系数 |
| 旋转 45° | 外接框面积膨胀 **2.0729 倍** | 四角变换后取 min/max |

三条纪律：

1. **可逆变换**（flip / 缩放 / 平移）有精确公式，写完立刻做一次「变换两次回原」自检。
2. **旋转会膨胀轴对齐框**（2.07 倍），目标检测一般不旋转图像，分类/分割才放心转。
3. **裁剪后必须 `clip` 回画布并丢弃空框**：`[100:200, 50:150]` 后 6 个框只剩 2 个有效。

下面用「框内 patch 的镜像 == 翻转后图的对应 patch」做一次硬核对账——这是抓标注不同步
bug 的最直接证据。
"""
    ),
    code(S10_CODE),
    md(
        """
## 小结

1. 清洗 = 逐张体检（`None` / 太小 / `std=0`），坏图不会报错，只能主动发现。
2. `flip` / `rotate90` 是像素搬家，`mean`/`std` 不变；`rotate90` 会互换 H、W。
3. `warpAffine` 默认不扩边，旋转必留黑角；扩边只解决「内容被切」。
4. 亮度用 `convertScaleAbs(alpha, beta)`；加 `beta` 会削平亮部（本图 3.58%）。
5. gamma <1 提亮，0/255 是不动点；回写前必须 `clip`。
6. 高斯噪声动全部像素、椒盐只动 5% 但很猛；两者都要固定 seed。
7. 中值治椒盐、高斯治高斯，滤错反而更糊。
8. `CLAHE` 比 `equalizeHist` 温和，优先用在低照度增强。
9. **增强必须同步标注**：flip/缩放可逆有公式，旋转膨胀框，裁剪要 clip + 丢空框。
10. 本章 10 组真值全部来自实跑（opencv 5.0.0），照抄即对。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch02 图像清洗与数据增强（练习版）

> 按提示补全 `____`，跑通所有 assert。`home` / `site3` / `labels` 已在 setup 里读好，
> `probe` / `tmp_dir` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；循环 / 分支写在挖空块之外（见任务 2）。
> 题目真值都来自实跑（opencv 5.0.0），照提示写就能对上。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md("## 任务 1：坏图体检\n\n人造四张「有病」的图，看谁能被读出来。"),
    code(E1_CODE),
    md("## 任务 2：数据集体检\n\n循环骨架已给好，只在循环体内写 3 行。"),
    code(E2_CODE),
    md("## 任务 3：flip 与 rotate\n\n三种翻转 + 转四次回原图 + 均值不变。"),
    code(E3_CODE),
    md("## 任务 4：warpAffine\n\n旋转黑角、扩边、缩放、平移。"),
    code(E4_CODE),
    md("## 任务 5：亮度与对比度\n\n`convertScaleAbs(alpha, beta)` 与饱和代价。"),
    code(E5_CODE),
    md("## 任务 6：gamma 校正\n\n非线性提亮与不动点。"),
    code(E6_CODE),
    md("## 任务 7：两类噪声\n\n高斯 vs 椒盐，seed 固定可复现。"),
    code(E7_CODE),
    md("## 任务 8：滤波降噪对账\n\n什么噪声配什么滤波器。"),
    code(E8_CODE),
    md("## 任务 9：直方图均衡\n\n`equalizeHist` vs `CLAHE`。"),
    code(E9_CODE),
    md("## 任务 10：增强与标注同步\n\nhflip / vflip / 旋转外接框 / 裁剪越界 + patch 对账。"),
    code(E10_CODE),
    md(
        """
## 自查清单

- [ ] 能说出坏图的三种类型，以及为什么 `imread` 不会报错
- [ ] 能写出 hflip / vflip 的 bbox 坐标公式，并验证两次变换可逆
- [ ] 能解释「扩边后黑占比反而更高」的原因
- [ ] 知道加 `beta` 会不可逆地削平亮部
- [ ] 能说清 gamma <1 为什么是「提亮暗部」
- [ ] 能配对「椒盐 ↔ 中值、高斯 ↔ 高斯」
- [ ] 知道旋转为什么不适合目标检测增强
- [ ] 会写「框内 patch 镜像」这种标注同步的硬核自检
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
