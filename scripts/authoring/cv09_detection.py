#!/usr/bin/env python3
"""cv09 目标检测基础：手撕 IoU / 矢量 IoU 矩阵 / VOC-归一化-YOLO 互转 /
NMS（含「只与存活框比较」的坑）/ 框绘制与裁剪 / AP 指标。

数据：`helmet/` 的 `labels.json`（VOC xyxy + 类别名）与 `site_0X.txt`
（YOLO cxcywh 归一化）。两套标注一一对应，正好用来做格式互转对账。
真值全部实跑（opencv 5.0.0 / numpy 2.5.3）。

> 本章最值钱的三条：① NMS 只与**存活**框比较，与「所有已处理框」比较会过度抑制
> （实测 thr=0.5 时 [0, 2, 3] vs [0, 3]）；② IoU 对 1px 偏移的敏感度与目标尺寸成反比
> （20×20 掉到 0.904762，600×400 还有 0.996672）——小目标难就难在这里；
> ③ YOLO 归一化标注往返**有损**，txt 存 6 位小数会带来 5.125e-04 像素的偏差。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv09_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch09_detection"

IMPORTS = '''from pathlib import Path

import json

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

DATA = Path("data")
CL = DATA / "classic"
HL = DATA / "helmet"
IOU_THR = 0.5
'''

SETUP = '''lab = json.loads((HL / "labels.json").read_text(encoding="utf-8"))
site05_img = cv2.imread(str(HL / "site_05.jpg"))

print("类别表:", lab["classes"], "（class_id 就是它的下标）")
for nm, info in sorted(lab["images"].items()):
    names = [o["class_name"] for o in info["objects"]]
    print(f"   {nm:14s} {info['width']:4d}x{info['height']:4d}  对象 {len(info['objects']):2d}"
          f"  helmet={names.count('helmet')} head={names.count('head')}")
print("site_05 读入尺寸:", site05_img.shape, "（H, W, C）")
'''

SCAFFOLD = '''# 脚手架：常量与工具函数。其中 nms 的循环骨架已给好，缺口在 @@todo 里
ARR = np.array([[0, 0, 100, 100], [50, 0, 150, 100], [10, 10, 110, 110]], dtype=np.float64)

NBOX = np.array([(0, 0, 100, 100), (10, 10, 110, 110), (20, 20, 120, 120),
                 (200, 0, 300, 100), (205, 5, 305, 105)], dtype=np.float64)
NSCORE = [0.95, 0.90, 0.85, 0.80, 0.60]


def nms(boxes, scores, thr):
    """标准 NMS：按分数降序，逐个与「已存活框」比最大 IoU，不超过 thr 才留下。"""
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    keep = []
    for i in order:
        # @@todo 算第 i 个框与「已存活框」的最大 IoU（keep 为空时记 0.0）
        # @@hint 一条 max(...) 生成式，注意 default=0.0
        # @@hint 只跟 keep 里的框比 —— 不要跟已经被抑制掉的框比
        ov = max((iou(boxes[i], boxes[j]) for j in keep), default=0.0)
        # @@end
        if ov <= thr:
            keep.append(int(i))
    return keep


def draw_boxes(img, boxes, color, thickness=2):
    """把浮点框画到图上（返回新图）。"""
    canvas = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    return canvas


print("脚手架就绪：ARR / NBOX / NSCORE / nms / nms_all_prev / draw_boxes")
'''

# =========================================================================== #
# 练习（答案版内容，practice 由 builder 自动挖空）
# =========================================================================== #

E1_CODE = '''rec05 = lab["images"]["site_05.jpg"]
# @@todo 把 site_05 的 10 个框抽成 (10, 4) 的 float64 数组，并记下类别名
# @@hint 字段名是 "bbox_xyxy"；用列表推导 + np.array(..., dtype=np.float64)
site05 = np.array([o["bbox_xyxy"] for o in rec05["objects"]], dtype=np.float64)
cls05 = tuple(o["class_name"] for o in rec05["objects"])
# @@end
print("标注里的尺寸:", rec05["width"], "x", rec05["height"], "| cv2 读到的:", site05_img.shape)
print("框数组:", site05.shape, site05.dtype, "| 类别序列:", cls05)
print("前 3 个框:")
for b in site05[:3]:
    print("   ", tuple(int(v) for v in b))
areas = (site05[:, 2] - site05[:, 0]) * (site05[:, 3] - site05[:, 1])
print("面积:", tuple(float(v) for v in areas[:5]), "... 最小", float(areas.min()), "最大", float(areas.max()))
print("解读: 标注文件（json）与像素数组（numpy）是两种坐标系 —— json 里是 (x1, y1, x2, y2) 整数，"
      "numpy 里是 (N, 4) 的浮点表。**后续所有几何运算都在 numpy 上做**，"
      "并且统一用 float64，否则 IoU 的除法会掉精度")
assert site05.shape == (10, 4)
assert site05.dtype == np.float64
assert set(cls05) == {"helmet"}
assert float(areas.min()) == 850.0 and float(areas.max()) == 2484.0
'''

E2_CODE = '''def iou(a, b):
    """两个 xyxy 框的交并比。约定 (x1, y1, x2, y2)，x2 / y2 是右下角。"""
    # @@todo 手撕 IoU：交集左上取 max、右下取 min，宽高夹到 0，再除以并集
    # @@hint 交集宽 = max(0, min(ax2, bx2) - max(ax1, bx1))，高同理
    # @@hint 并集 = 两框面积之和 - 交集；并集为 0 时直接返回 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    value = inter / union if union > 0 else 0.0
    # @@end
    return value


A_BOX = (0, 0, 100, 100)
CASES = {
    "自身": (A_BOX, A_BOX),
    "半重叠(水平)": (A_BOX, (50, 0, 150, 100)),
    "半重叠(斜角)": (A_BOX, (50, 50, 150, 150)),
    "包含(B 在 A 内)": (A_BOX, (25, 25, 75, 75)),
    "边界接触": (A_BOX, (100, 0, 200, 100)),
    "完全分离": (A_BOX, (200, 200, 300, 300)),
    "细长(半宽)": (A_BOX, (0, 0, 50, 100)),
}
print("IoU 手撕（A 恒为 (0, 0, 100, 100)）")
for k, (p, q) in CASES.items():
    print(f"   {k:16s} {str(q):22s} -> {iou(p, q):.6f}")
print("解读: ① 「边界接触」是 0.000000 而不是「很小」—— 交集宽高必须夹到 0；"
      "② 「包含」时 IoU 退化成**面积比**（2500 / 10000 = 0.25），和大小关系一致但不等于 1；"
      "③ IoU 是**对称**的：iou(p, q) == iou(q, p)")
assert round(iou(A_BOX, A_BOX), 6) == 1.0
assert round(iou(A_BOX, (50, 0, 150, 100)), 6) == 0.333333
assert round(iou(A_BOX, (50, 50, 150, 150)), 6) == 0.142857
assert round(iou(A_BOX, (25, 25, 75, 75)), 6) == 0.25
assert round(iou(A_BOX, (100, 0, 200, 100)), 6) == 0.0
assert round(iou((25, 25, 75, 75), A_BOX), 6) == 0.25
assert round(iou(A_BOX, (0, 0, 50, 100)), 6) == 0.5
'''

E3_CODE = '''def iou_matrix(boxes_a, boxes_b):
    """向量化 IoU 矩阵，返回 (len(a), len(b))。"""
    # @@todo 一次算出 Na x Nb 个 IoU：a 补「行」轴、b 补「列」轴，交给广播
    # @@hint a[:, None, :] 配 b[None, :, :]；np.clip(x, 0, None) 顶替 max(0, x)
    # @@hint 面积用 a[..., 2] - a[..., 0] 这种「省略号索引」，保持最后一维是 4 个坐标
    a = np.asarray(boxes_a, dtype=np.float64)[:, None, :]
    b = np.asarray(boxes_b, dtype=np.float64)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    mat = inter / (area_a + area_b - inter)
    # @@end
    return mat


# @@todo 再用「双重循环」写一版逐对 IoU，和矢量版对账
# @@hint 外层包 np.array([... for p in ARR])，内层是 [iou(p, q) for q in ARR]
loop_mat = np.array([[iou(p, q) for q in ARR] for p in ARR])
# @@end
mat = iou_matrix(ARR, ARR)
print("逐对版:")
print(np.round(loop_mat, 6))
print("矢量版:")
print(np.round(mat, 6))
print("两版最大差:", float(np.abs(loop_mat - mat).max()))
print("对角线和:", float(np.diag(mat).sum()), "| 对称性 max|M - M.T|:", float(np.abs(mat - mat.T).max()))
print("解读: 广播版和双重循环版**结果完全一致**（差 0.0），但广播版把 O(N*M) 次 Python 调用"
      "压成 1 次数组运算 —— 检测后处理里常见 10^4 量级候选框，这个差别是「秒级」和「分钟级」。"
      "注意 `[..., 2]` 这种写法：最后一维才是坐标，前面两维是「行 / 列」")
assert float(np.abs(loop_mat - mat).max()) == 0.0
assert float(np.diag(mat).sum()) == 3.0
assert round(float(mat[0, 2]), 6) == 0.680672
assert float(np.abs(iou_matrix(site05, site05) - iou_matrix(site05, site05).T).max()) == 0.0
'''

E4_CODE = '''# @@todo 对 5 种尺寸的框，各算「整体右移 1px」后的 IoU，看谁跌得最快
# @@hint 一条列表推导，元素形如 (w, h, round(iou(base, shifted), 6))；base 是 [100, 100, 100+w, 100+h]
# @@hint 用「双倍括号」把整个推导写在同一个续行里，避免续行以 for 开头
sens = [(w, h, round(iou([100, 100, 100 + w, 100 + h], [101, 100, 101 + w, 100 + h]), 6)) for w, h in ((20, 20), (38, 40), (100, 100), (300, 300), (600, 400))]
# @@end
print("同一「右移 1px」在不同尺寸上的代价:")
for w, h, v in sens:
    print(f"   {w:3d}x{h:3d} -> IoU = {v:.6f}")
areas05 = (site05[:, 2] - site05[:, 0]) * (site05[:, 3] - site05[:, 1])
i_min, i_max = int(areas05.argmin()), int(areas05.argmax())
b_min, b_max = site05[i_min], site05[i_max]
got_min = round(iou(b_min, [b_min[0] + 1, b_min[1], b_min[2] + 1, b_min[3]]), 6)
got_max = round(iou(b_max, [b_max[0] + 1, b_max[1], b_max[2] + 1, b_max[3]]), 6)
print("真实数据里最小框", tuple(int(v) for v in b_min), "面积", float(areas05.min()), "-> 1px IoU =", got_min)
print("真实数据里最大框", tuple(int(v) for v in b_max), "面积", float(areas05.max()), "-> 1px IoU =", got_max)
print("解读: 同样错 1 个像素，20x20 的框 IoU 掉到 0.904762，而 600x400 只掉到 0.996672。"
      "**IoU 是「相对误差」尺度**，所以固定阈值天然对小目标苛刻 —— 这就是小目标检测要"
      "单独用一套判定标准（或用 0.5 : 0.95 多阈值平均）的根本原因")
assert sens[0] == (20, 20, 0.904762)
assert sens[-1] == (600, 400, 0.996672)
assert got_min == 0.942857 and got_max == 0.963636
assert abs((b_min[2] - b_min[0]) - 34) < 1e-9 and abs((b_max[2] - b_max[0]) - 54) < 1e-9
'''

E5_CODE = '''def xyxy_to_yolo(box, w, h):
    """VOC 像素 (x1, y1, x2, y2) -> YOLO 归一化 (cx, cy, bw, bh)。"""
    # @@todo 转成「中心点 + 宽高」，并**各自除以宽或高**归一化
    # @@hint cx 除以 w、cy 除以 h；bw 除以 w、bh 除以 h —— 四者分母不同，这是最常见的写错点
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    # @@end
    return (cx, cy, bw, bh)


def yolo_to_xyxy(t, w, h):
    """YOLO 归一化 (cx, cy, bw, bh) -> VOC 像素 (x1, y1, x2, y2)。"""
    # @@todo 反向：先还原中心点与宽高，再由「中心 - 半宽高」得左上角
    # @@hint 先乘回像素再减半宽；等价写法 (cx - bw / 2) * w
    cx, cy, bw, bh = t
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    # @@end
    return (x1, y1, x2, y2)


info05 = lab["images"]["site_05.jpg"]
W05, H05 = info05["width"], info05["height"]
print("第一个框三格式对照:")
print("   VOC  (x1, y1, x2, y2)   =", tuple(int(v) for v in site05[0]))
print("   COCO (x, y, w, h)       =", tuple(int(v) for v in (site05[0][0], site05[0][1],
                                                            site05[0][2] - site05[0][0], site05[0][3] - site05[0][1])))
print("   YOLO (cx, cy, bw, bh)   =", tuple(round(float(v), 6) for v in xyxy_to_yolo(site05[0], W05, H05)))
print("   同一行 .txt 原文        =", (HL / "site_05.txt").read_text().splitlines()[0])
print("解读: 三种格式只有「锚点」和「是否归一化」两个差异 —— VOC 用左上+右下且不归一化；"
      "COCO 用左上+宽高且不归一化；YOLO 用中心+宽高**且归一化**。"
      "**归一化的分母是宽 / 高各自的**，把 cx 除以 h 是最经典的错")
r0 = tuple(round(float(v), 6) for v in xyxy_to_yolo(site05[0], W05, H05))
assert r0 == (0.02303, 0.352181, 0.046061, 0.06462)
assert round(xyxy_to_yolo((0, 0, 825, 619), 825, 619)[2], 6) == 1.0
'''

E6_CODE = '''txt_check = {}
for name in ("site_01.jpg", "site_03.jpg", "site_05.jpg", "site_07.jpg"):
    info = lab["images"][name]
    w, h = info["width"], info["height"]
    rows = [ln.split() for ln in (HL / name.replace(".jpg", ".txt")).read_text().splitlines() if ln.strip()]
    got = [tuple(round(float(v), 6) for v in r[1:]) for r in rows]
    ids = [int(r[0]) for r in rows]
    # @@todo 用 xyxy_to_yolo 把同一张图的 json 标注转成同样格式（同样保留 6 位小数）
    # @@hint 一条列表推导套在 info["objects"] 上，取 "bbox_xyxy" 字段
    mine = [tuple(round(float(v), 6) for v in xyxy_to_yolo(o["bbox_xyxy"], w, h)) for o in info["objects"]]
    # @@end
    diff = max(abs(a - b) for A, B in zip(mine, got) for a, b in zip(A, B))
    txt_check[name] = (len(mine), len(got), tuple(ids), diff)
    print(f"   {name:14s} json {len(mine):2d} 框 / txt {len(got):2d} 框 | class_id = {tuple(ids)}")
    print(f"   {'':14s} 逐值最大差 = {diff:.2e}")
print("解读: 两套标注**逐值完全一致**（差 0.0），说明 txt 就是 json 转出来的。"
      "写数据加载器时可以直接二选一，但别忘了 class_id 与类别名的映射关系来自 json 里的 classes 数组")
assert txt_check["site_05.jpg"][3] == 0.0
assert txt_check["site_05.jpg"][2] == (1,) * 10
assert txt_check["site_07.jpg"][2] == (1, 1, 1, 1, 1, 0)
assert txt_check["site_01.jpg"][0] == 7 and txt_check["site_01.jpg"][2] == (0,) * 7
'''

E7_CODE = '''# @@todo 量化「往返损失」：把 txt 的归一化值转回像素，看和原始整数坐标差多少
# @@hint 两个 max(abs(...))：① back05 与 site05 比；② txt 的 6 位小数与完整精度的 raw05 比
# @@hint 用 yolo_to_xyxy(t, W05, H05) 还原；raw05 是未截断的完整精度归一化值
rows05 = [ln.split() for ln in (HL / "site_05.txt").read_text().splitlines() if ln.strip()]
yolo05 = [tuple(float(v) for v in r[1:]) for r in rows05]
raw05 = np.array([xyxy_to_yolo(o["bbox_xyxy"], W05, H05) for o in info05["objects"]])
back05 = np.array([yolo_to_xyxy(t, W05, H05) for t in yolo05])
round_err = float(np.abs(back05 - site05).max())
quant_err = float(np.abs(np.array(yolo05) - raw05).max())
# @@end
print("txt 存的是 6 位小数:", yolo05[0])
print("完整精度应为:       ", tuple(round(float(v), 10) for v in raw05[0]))
print("归一化空间的截断误差:", quant_err)
print("回到像素空间的往返误差:", round_err, "像素")
print("解读: 归一化坐标乘回像素尺寸会**放大**误差（这里 825 宽放大约 1000 倍）。"
      "7e-04 像素对画框完全无所谓，但**如果拿往返后的框去做严格相等断言就会挂** —— "
      "数据标注格式转换必须用容差比较，别用 ==")
assert abs(round_err - 0.0005125) < 1e-9
assert abs(quant_err - 4.545e-07) < 1e-9
assert float(np.abs(raw05 - np.array(yolo05)).max()) < 1e-6
'''

E8_CODE = '''M = iou_matrix(NBOX, NBOX)
print("=== 合成场景：5 个框其实只有 3 个目标 ===")
print("  几何布局: 框 0/1/2 是一簇（互相重叠），框 3/4 是另一簇（几乎重合），分数依次 0.95/0.90/0.85/0.80/0.60")
for i in range(5):
    print(f"   框{i} score={NSCORE[i]:.2f} 与其余 IoU =", tuple(round(float(v), 6) for v in M[i]))
print("  矩阵对称性 max|M - M.T| =", float(np.abs(M - M.T).max()))
print("解读: 框 0-1 IoU = 0.680672、框 1-2 也是 0.680672、框 0-2 只有 0.470588 —— "
      "**「两两重叠」不构成传递关系**，所以抑制顺序会影响最终结果")
assert float(np.abs(M - M.T).max()) == 0.0
assert round(float(M[0, 1]), 6) == 0.680672
assert round(float(M[0, 2]), 6) == 0.470588
assert round(float(M[3, 4]), 6) == 0.822323
'''

E9_CODE = '''# @@todo 四档阈值各跑一遍 NMS，记录保留下来的框下标与对应分数
# @@hint 一条字典推导：{thr: (nms(...), [NSCORE[i] for i in ...]) for thr in (0.3, 0.5, 0.7, 0.9)}
nms_scan = {thr: (nms(NBOX, NSCORE, thr), [NSCORE[i] for i in nms(NBOX, NSCORE, thr)]) for thr in (0.3, 0.5, 0.7, 0.9)}
# @@end
print("NMS 阈值扫描（5 个输入框 -> 真实目标 3 个）")
for thr in (0.3, 0.5, 0.7, 0.9):
    kept, sc = nms_scan[thr]
    print(f"   thr={thr}: 保留 {len(kept)} 个 {kept}  分数 {[round(v, 2) for v in sc]}")
print("解读: 三个必须说清的现象 ——")
print("  ① thr=0.3 只剩 [0, 3]：框 2 与框 0 的 IoU 是 0.470588 > 0.3，被抑制；")
print("  ② thr=0.5 变成了 [0, 2, 3]：框 2 与框 0 的 0.470588 <= 0.5，而框 1 已经"
      "**被抑制掉了、不再参与比较**，所以框 2 活下来了；")
print("  ③ thr=0.9 时 5 个全留（0.822323 也 < 0.9）—— 阈值越松，重复框越多")
assert nms_scan[0.3][0] == [0, 3]
assert nms_scan[0.5][0] == [0, 2, 3]
assert nms_scan[0.7][0] == [0, 1, 2, 3]
assert nms_scan[0.9][0] == [0, 1, 2, 3, 4]
assert [round(v, 6) for v in nms_scan[0.5][1]] == [0.95, 0.85, 0.8]
'''

E10_CODE = '''rng = np.random.default_rng(0)
jit_boxes, jit_scores = [], []
for b in site05:
    offs = rng.uniform(-3.0, 3.0, size=(3, 4))
    scores3 = rng.uniform(0.80, 0.99, size=3)
    for t in np.argsort(-scores3):
        jit_boxes.append(b + offs[t])
        jit_scores.append(round(float(scores3[t]), 6))
jit_boxes = np.array(jit_boxes, dtype=np.float64)
jm = iou_matrix(jit_boxes, site05)
# @@todo 统计 30 个抖动框「与最近 GT 的 IoU」，并跑四档阈值的 NMS
# @@hint 两个赋值：near = jm.max(axis=1)；保留数用 {thr: len(nms(...)) for thr in (...)}
near = jm.max(axis=1)
jitter_keep = {thr: len(nms(jit_boxes, jit_scores, thr)) for thr in (0.3, 0.5, 0.7, 0.9)}
# @@end
print("模拟「同一目标被反复检出」: 10 个 GT，每个生成 3 个 ±3px 抖动副本 =", len(jit_boxes), "个候选")
print("每个候选与最近 GT 的 IoU: min %.6f / mean %.6f / max %.6f" % (float(near.min()), float(near.mean()), float(near.max())))
print("NMS 保留数:", {k: v for k, v in jitter_keep.items()}, "（真值 10）")
kept50 = nms(jit_boxes, jit_scores, 0.5)
print("thr=0.5 留下的框落在哪个 GT 上:", tuple(int(v) for v in jm[kept50].argmax(axis=1)))
print("留下的分数:", tuple(round(jit_scores[i], 6) for i in kept50))
print("解读: 抖动只有 ±3px，所以候选与 GT 的 IoU 全在 0.785 以上 —— "
      "thr 取 0.3 / 0.5 / 0.7 都能把 30 个候选**精确压回 10 个**；"
      "thr=0.9 就压不住了（同一个目标的副本之间也可能只有 0.85 左右）。"
      "**NMS 阈值要跟「定位精度」一起定**，不是拍脑袋填 0.5")
assert len(jit_boxes) == 30 and len(jit_scores) == 30
assert round(float(near.min()), 6) == 0.785183
assert round(float(near.mean()), 6) == 0.859141
assert round(float(near.max()), 6) == 0.925049
assert jitter_keep == {0.3: 10, 0.5: 10, 0.7: 10, 0.9: 25}
assert len(set(int(v) for v in jm[kept50].argmax(axis=1))) == 10
'''

E11_PRE_CODE = '''def nms_all_prev(boxes, scores, thr):
    """错误示范：与「所有已处理框」（含已被抑制的）比较 —— 会过度抑制。"""
    order = list(np.argsort(-np.asarray(scores, dtype=np.float64)))
    keep = []
    for pos, i in enumerate(order):
        # @@todo 与 order[:pos]（所有更早处理过的框，不管死没死）比最大 IoU
        # @@hint 把 nms 里的 keep 换成 order[:pos] 即可，其余完全一样
        ov = max((iou(boxes[i], boxes[j]) for j in order[:pos]), default=0.0)
        # @@end
        if ov <= thr:
            keep.append(int(i))
    return keep


print("反面教材已就位：nms_all_prev")
'''

E11_CODE = '''# @@todo 把错误的「与所有已处理框比较」版本拉出来，四档阈值各跑一遍
# @@hint 一条字典推导，键是阈值，值是 nms_all_prev(NBOX, NSCORE, thr)
bad_scan = {thr: nms_all_prev(NBOX, NSCORE, thr) for thr in (0.3, 0.5, 0.7, 0.9)}
# @@end
print(f"{'thr':>5s} {'标准 NMS':>16s} {'与所有已处理框比较':>20s}")
for thr in (0.3, 0.5, 0.7, 0.9):
    print(f"{thr:>5.1f} {str(nms_scan[thr][0]):>16s} {str(bad_scan[thr]):>20s}")
print("解读: **两者只在 thr=0.5 时不同** —— 标准版是 [0, 2, 3]，错误版是 [0, 3]。差别来自："
      "框 1 已经被框 0 抑制掉了，标准 NMS 就不再拿它去衡量别人；错误版仍然拿框 1 去抑制框 2。"
      "结论: NMS 的「已抑制框退出比较」不是实现细节，而是**算法定义的一部分**，"
      "自己手写时写成 all_prev 那种会静默丢目标")
assert bad_scan[0.5] == [0, 3]
assert bad_scan[0.3] == [0, 3]
assert bad_scan[0.7] == [0, 1, 2, 3]
assert bad_scan[0.9] == nms_scan[0.9][0]
'''

E12_CODE = '''# @@todo 把 10 个框画成 2px 红框，统计真正被改动的像素数（与彩色图逐通道比较）
canvas = draw_boxes(site05_img, site05, (0, 0, 255), 2)
changed = int(np.any(canvas != site05_img, axis=2).sum())
# @@end
one_box = draw_boxes(site05_img, site05[:1], (0, 0, 255), 2)
mask_one = np.any(one_box != site05_img, axis=2)
ys, xs = np.where(mask_one)
print("画 10 个框改动像素:", changed, "| 占全图", round(changed / site05_img[:, :, 0].size, 6))
print("只画第一个框改动像素:", int(mask_one.sum()))
print("改动范围 bbox (x1, y1, x2, y2):", (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
print("原框坐标:", tuple(int(v) for v in site05[0]))
print("解读: ① `cv2.rectangle` 的**厚度以边线为中心向两侧扩**，所以实际改动范围比原框"
      "四周各多 1px —— 肉眼看不出来，但拿掩码做像素级评估时会算进去；"
      "② 颜色参数是 **BGR**，`(0, 0, 255)` 才是红框")
assert changed == 4769
assert int(mask_one.sum()) == 423
assert (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) == (0, 197, 39, 239)
'''

E13_CODE = '''# @@todo 把越界框裁到图像范围内（下界 0、上界分别是宽 / 高）
# @@hint 一条 np.clip(b, (0, 0, 0, 0), (w5, h5, w5, h5))，注意上界对 x 用宽、对 y 用高
h5, w5 = site05_img.shape[:2]
clip_pair = tuple(tuple(int(v) for v in np.clip(b, (0, 0, 0, 0), (w5, h5, w5, h5))) for b in ((-20, -30, 50, 60), (700, 500, 900, 700)))
# @@end
print("图像尺寸 (w, h) =", (w5, h5))
for b, c in zip(((-20, -30, 50, 60), (700, 500, 900, 700)), clip_pair):
    print(f"   {str(b):22s} -> {c}")
print("解读: 训练 / 推理前的框必须裁剪 —— ① 越界的框画不出来、切 ROI 会切错；"
      "② 越界框的**面积算大了**，会污染 IoU 与 NMS。注意裁剪后可能退化成零面积框，"
      "要顺手过滤掉（w <= 0 or h <= 0）")
assert clip_pair == ((0, 0, 50, 60), (700, 500, 825, 619))
assert tuple(int(v) for v in np.clip(site05[0], (0, 0, 0, 0), (w5, h5, w5, h5))) == (0, 198, 38, 238)
'''

E14_CODE = '''GT4 = site05[:4]
PRED = np.array([
    [0.0, 199.0, 38.0, 238.0],
    [400.0, 20.0, 460.0, 80.0],
    [58.0, 232.0, 104.0, 270.0],
    [178.0, 248.0, 228.0, 288.0],
    [600.0, 500.0, 660.0, 560.0],
    [2.0, 201.0, 40.0, 240.0],
], dtype=np.float64)
SCORES_DET = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
mm = iou_matrix(PRED, GT4)
hit = mm.max(axis=1) >= IOU_THR
order = np.argsort(-np.asarray(SCORES_DET, dtype=np.float64))
print("模拟检测结果: 4 个 GT / 6 个预测（2 个误检 + 1 个重复命中）")
print("预测 vs GT 的 IoU 矩阵:")
print(np.round(mm, 4))
print("每个预测的最大 IoU:", tuple(round(float(v), 6) for v in mm.max(axis=1)))
print("IoU >= 0.5 的命中标记:", tuple(bool(v) for v in hit))
used, is_tp = set(), []
for i in order:
    # @@todo 判断第 i 个预测算不算 TP：命中了、并且它对应的 GT 还没被占用
    # @@hint 三行：j = int(mm[i].argmax())；ok = bool(hit[i]) and j not in used；再 append
    j = int(mm[i].argmax())
    ok = bool(hit[i]) and j not in used
    is_tp.append(ok)
    # @@end
    if ok:
        used.add(j)
# @@todo 由 is_tp 累加出 precision 与 recall 序列
# @@hint tp_c = np.cumsum(is_tp)；分母：precision 用 tp_c + fp_c，recall 用 len(GT4)
tp_c = np.cumsum(is_tp).astype(np.float64)
fp_c = np.cumsum([not v for v in is_tp]).astype(np.float64)
prec = tp_c / (tp_c + fp_c)
rec = tp_c / len(GT4)
# @@end
print("按分数降序处理:", tuple(int(v) for v in order))
print("TP 判定:", tuple(bool(v) for v in is_tp), "| 最终 TP =", int(tp_c[-1]), "FP =", int(fp_c[-1]))
print("precision:", tuple(round(float(v), 6) for v in prec))
print("recall:   ", tuple(round(float(v), 6) for v in rec))
mrec = np.concatenate([[0.0], rec, [1.0]])
mpre = np.concatenate([[0.0], prec, [0.0]])
# @@todo 把 precision 曲线「单调化」—— 从右往左取累积最大值
# @@hint 一行：np.maximum.accumulate(mpre[::-1])[::-1]
mpre = np.maximum.accumulate(mpre[::-1])[::-1]
# @@end
ap_all = float(np.sum(np.diff(mrec) * mpre[1:]))
pts = np.linspace(0, 1, 11)
idx = np.searchsorted(mrec, pts, side="right") - 1
ap11 = float(mpre[idx].mean())
print("单调化后 precision:", tuple(round(float(v), 6) for v in mpre))
print("AP (all-point 积分):", round(ap_all, 6))
print("11 点采样 precision:", tuple(round(float(v), 6) for v in mpre[idx]))
print("AP (11 点平均):", round(ap11, 6))
print("解读: ① precision 曲线在「遇到误检」时会掉，而 recall 只增不减，所以**面积要用"
      "单调化后的曲线**算（否则同一个模型不同实现算出的 AP 会不一致）；"
      "② all-point（VOC2010+ / COCO 风格）只对真正变过 recall 的点积分，比 11 点采样更细，"
      "所以这里 0.625000 < 0.704545 —— 11 点采样在稀疏 PR 曲线上会**高估**")
assert round(ap_all, 6) == 0.625
assert round(ap11, 6) == 0.704545
assert int(tp_c[-1]) == 3 and int(fp_c[-1]) == 3
assert [round(float(v), 6) for v in prec] == [1.0, 0.5, 0.666667, 0.75, 0.6, 0.5]
assert [round(float(v), 6) for v in rec] == [0.25, 0.25, 0.5, 0.75, 0.75, 0.75]
assert round(float(mpre[idx][-1]), 6) == 0.0
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch09 目标检测基础

> 前八章一直在回答「这张图里有什么」，本章开始回答 **「东西在哪、有几个」**。
> 检测任务的全部工程复杂度，几乎都可以拆成三块：**框的表达**（IoU / 格式转换）、
> **框的筛选**（NMS）、**框的评价**（AP）。这三块在本章全部手撕。

## 本章路线

| 小节 | 内容 | 为什么要手写 |
|---|---|---|
| §9.1 | 数据集与三种标注格式 | VOC / COCO / YOLO 互转是数据工程的日常 |
| §9.2 | **手撕 IoU** | 后面所有判定（匹配 / NMS / 评估）的地基 |
| §9.3 | 向量化 IoU 矩阵 | 检测后处理是 10^4 量级候选框，循环会慢到不可用 |
| §9.4 | IoU 的尺度敏感性 | **小目标为什么难**的定量解释 |
| §9.5 | VOC ↔ YOLO 互转 + 对账 | 与真实 `.txt` 标注逐值比对 |
| §9.6 | **手撕 NMS** | 检测器一定会输出重复框，NMS 是标配后处理 |
| §9.7 | 抖动场景下的 NMS | 阈值与定位精度必须一起定 |
| §9.8 | 绘制与裁剪 | 画框的 BGR / 越界框，两个必踩的坑 |
| §9.9 | **AP 指标** | precision / recall / 单调化 / all-point vs 11 点 |

## 关于数据

`data/helmet/` 里有 **4 张带标注的现场图**，共 29 个目标（21 个 helmet + 8 个 head）。
标注有两套且互相一致：

- `labels.json`：VOC 风格 `bbox_xyxy` + `class_id` / `class_name`
- `site_0X.txt`：YOLO 风格 `class_id cx cy bw bh`（归一化）

两套标注的存在，刚好用来做**格式互转的逐值对账**。
"""
    ),
    md(
        """
## §9.0 环境与数据准备

先把标注读进来看看长什么样。注意 `class_id` 就是 `classes` 数组的下标。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## §9.1 把标注变成 numpy 数组

标注是 json、几何运算要 numpy —— 第一件事就是**统一到 (N, 4) 的 float64 表**。

用 `float64` 不是洁癖：IoU 里要做连除，`float32` 在 10^4 个框的矩阵上会攒出可见误差。
"""
    ),
    code(E1_CODE),
    md(
        """
## §9.2 手撕 IoU

IoU（Intersection over Union，交并比）是整个检测领域最基础的度量：

```
IoU = 交集面积 / 并集面积 = 交集 / (A + B - 交集)
```

三个必须记住的边界行为：

1. **交集宽高要夹到 0**：两个框只接触不重叠时，`min(x2) - max(x1)` 恰好是 0；
   如果写成 `abs(...)` 或者忘记 `max(0, ...)`，分离的框会算出正面积。
2. **并集要减掉交集**：`A + B` 是把交集算了两次。
3. **包含关系的 IoU 不等于 1**：大框套小框时，交集 = 小框，并集 = 大框，
   IoU 退化成面积比。
"""
    ),
    code(E2_CODE),
    md(
        """
## §9.3 向量化 IoU 矩阵

实际后处理里要算的是 **N 个预测 × M 个 GT 的整体 IoU 矩阵**。用双重循环写
在逻辑上没问题，但在 10^4 规模下会慢到不可用。

正确姿势是**利用广播**：把 a 的 shape 从 `(Na, 4)` 变成 `(Na, 1, 4)`，
b 从 `(Nb, 4)` 变成 `(1, Nb, 4)`，两者一运算就自动展开成 `(Na, Nb, 4)`，
再在最后一维上做 max / min 即可。

这里用 `a[..., 2]` 这种「省略号索引」而不是 `a[:, :, 2]`，好处是不用关心前面有几个轴。
"""
    ),
    code(E3_CODE),
    md(
        """
## §9.4 IoU 的尺度敏感性 —— 小目标为什么难

一个直觉上不容易发现、但对竞赛评分影响巨大的事实：**IoU 是相对度量**。
同样是「框错了 1 个像素」，小目标和大目标的 IoU 惩罚完全不同。

本节做完你就有了「小目标检测要单独定判定标准」的定量依据。
"""
    ),
    code(E4_CODE),
    md(
        """
## §9.5 三种标注格式与互转

| 格式 | 表达 | 归一化 | 锚点 |
|---|---|---|---|
| VOC | `x1 y1 x2 y2` | 否 | 左上 + 右下 |
| COCO | `x y w h` | 否 | 左上角 |
| YOLO | `cx cy w h` | **是** | 中心点 |

互转只有两件事要做：**换锚点**（角点 ↔ 中心）与**除不除尺寸**。

最容易写错的地方是归一化的**分母**：`cx` 除的是**宽**、`cy` 除的是**高**，
`bw` 除宽、`bh` 除高。四个量四个分母，写反了在方图上还看不出来，
一到 16:9 的图上就全错。
"""
    ),
    code(E5_CODE),
    md(
        """
### 与真实 `.txt` 逐值对账

`data/helmet/` 里同时存了 json 与 txt 两套标注 —— 这正好是一次免费的交叉验证：
把自己的转换结果与官方 `.txt` 逐值比，差应该恰好是 0。
"""
    ),
    code(E6_CODE),
    md(
        """
### 往返损失：归一化是有损的

`.txt` 里只存 6 位小数，转回像素坐标后**不会**回到原来的整数。

这不是数据质量事故（误差远小于 1 个像素），但它意味着：
**格式转换的验证必须用容差比较，不能用 `==`**。
"""
    ),
    code(E7_CODE),
    md(
        """
## §9.6 手撕 NMS

检测器（尤其是一阶段检测器）对同一个目标会输出**一大把重叠框**，
NMS（Non-Maximum Suppression，非极大值抑制）就是标准去重器，逻辑只有两条：

```
按分数从高到低排序
对每个框：如果它与「已保留框」的最大 IoU 超过阈值 -> 丢掉，否则保留
```

先看一个合成场景：5 个框，其实是 3 个目标。
"""
    ),
    code(E8_CODE),
    md(
        """
### 阈值扫描 —— 一个反直觉的结果

直觉上「阈值越低，留下来的框越少」。但真实实现里会看到
`thr=0.3 -> 2 个`、`thr=0.5 -> 3 个` 这种**不是单调**的跳变，原因在于
NMS 只与**存活框**比较：一旦某个框被抑制，它就退出比较，不再传递抑制。

自己手写时如果写成「与所有已处理框比较」，结果会静默不同。
"""
    ),
    code(E9_CODE),
    md(
        """
### 真实场景：同一目标被反复检出

合成场景是干净的。真实场景更常见的形态是「一个目标被检出 3~5 次，
每次框都差几个像素」。用 `labels.json` 的 10 个真实框加 ±3px 抖动来模拟。
"""
    ),
    code(E10_CODE),
    md(
        """
### 反面教材：与「所有已处理框」比较

把上一段提到的错误实现拉出来对照 —— 只在某个阈值下结果不同，
这种「大部分时候对、偶尔错」的 bug 最难查。
"""
    ),
    code(E11_PRE_CODE),
    code(E11_CODE),
    md(
        """
## §9.7 画框与裁剪

把框画出来是检测任务唯一的**目检手段** —— 数字再漂亮，图上框歪了就是错的。

这里有两个固定坑：

- 颜色是 **BGR**，`(0, 0, 255)` 才是红框；
- `cv2.rectangle` 的**厚度向两侧扩展**，2px 线宽时实际改动的像素范围
  会比原框四周各多 1px。
"""
    ),
    code(E12_CODE),
    md(
        """
### 越界框必须先裁剪

模型输出、坐标变换、resize 都可能把框推出画面。不裁剪会有三个后果：
画不出来、切 ROI 切错、**面积算大污染 IoU 与 NMS**。
"""
    ),
    code(E13_CODE),
    md(
        """
## §9.8 AP 指标

有了 IoU 就能判定「一个预测是不是真的命中了」：**IoU ≥ 0.5 算命中（TP）**。

但判定 TP 还要加一条：**一个 GT 只能被一个预测「认领」**。
否则同一个目标被检出 3 次会让 recall 虚高到「超过 100%」的荒谬程度。

有了 TP / FP，按分数降序就能画出 PR 曲线，曲线下面积就是 AP：

- **precision** = 到目前为止判对的比例（分母是已处理的预测数）
- **recall** = 到目前为止覆盖了多少 GT（分母是 GT 总数）
- **单调化**：AP 用的是「往右看，取最大 precision」的单调曲线
- **all-point vs 11 点**：前者对每个 recall 变化点积分，后者在 11 个等距点采样
"""
    ),
    code(E14_CODE),
    md(
        """
## 自查清单

- [ ] 能默写 IoU 的三行核心（交集 max/min、宽高夹 0、并集减交集）
- [ ] 知道「边界接触」的 IoU 是 0 而不是「很小」
- [ ] 知道包含关系下 IoU 退化成面积比
- [ ] 会用广播把 IoU 矩阵向量化，并说得清 `[..., 2]` 的含义
- [ ] 知道 IoU 对 1px 偏移的敏感度随目标变小而急剧上升
- [ ] 能说出 VOC / COCO / YOLO 三种格式的锚点与归一化差异
- [ ] 记得 YOLO 归一化的分母：cx 除宽、cy 除高
- [ ] 知道 YOLO 往返有损，格式转换验证要用容差
- [ ] 能默写 NMS 主循环（排序 + 与存活框比 + 阈值判定）
- [ ] **知道 NMS 只与存活框比较**，以及错误实现会过度抑制
- [ ] 知道 NMS 阈值与定位精度必须一起定（抖动 ±3px 时 0.3~0.7 都能压回 10 个）
- [ ] 记得画框颜色是 BGR、厚度会向两侧扩
- [ ] 知道越界框必须裁剪，以及不裁的三个后果
- [ ] 说得清「一个 GT 只能被认领一次」为什么是 recall 的硬约束
- [ ] 会算 precision / recall 序列，并知道 AP 要用单调化后的曲线
- [ ] 知道 all-point 与 11 点采样的差别，以及 11 点在稀疏曲线上会高估（0.704545 vs 0.625000）
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch09 目标检测基础（练习版）

> 按提示补全 `____`，跑通所有 assert。`lab` / `site05_img` 已在 setup 里备好；
> `ARR` / `NBOX` / `NSCORE` / `nms` / `nms_all_prev` / `draw_boxes` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；控制流写在挖空块之外
> （要写函数时 `def` 与 `return` 已经给好，你只填函数体，且**不许用 for**）。
> 题目真值都来自实跑（opencv 5.0.0）。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：把标注读成 numpy 表

`site_05` 的 10 个框，抽成 `(10, 4)` 的 float64 数组，同时记下类别名。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：手撕 IoU

填出 `iou` 函数体。注意交集宽高要夹到 0、并集要减掉交集。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：向量化 IoU 矩阵

用广播一次算出 `(Na, Nb)` 的 IoU 矩阵，再用双重循环写一版对账。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：IoU 的尺度敏感性

算「右移 1px」在五种尺寸上的不同代价，再在真实框上验证一次。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：VOC ↔ YOLO 互转

两个函数体都填上。**小心归一化的分母：cx 除以宽、cy 除以高**。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：与真实 `.txt` 逐值对账

把自己的转换结果与 `site_0X.txt` 比一遍，差值应该是 0。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：往返损失量化

把 txt 的 6 位小数归一化值转回像素，看与原始整数坐标差多少。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：看清 NMS 的输入

先只观察 5 框场景的 IoU 矩阵，理解「两两重叠不传递」。
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 9：阈值扫描

四档阈值各跑一遍 NMS。留意结果**不是单调的**。
"""
    ),
    code(E9_CODE),
    md(
        """
## 任务 10：真实抖动场景

10 个 GT × 3 个 ±3px 抖动副本 = 30 个候选，看 NMS 能不能压回 10 个。
（循环骨架已给，你只需要统计与跑 NMS。）
"""
    ),
    code(E10_CODE),
    md(
        """
## 任务 11：反面教材对照

先把错误实现 `nms_all_prev` 的函数体补上，再和标准版并排看差异。
"""
    ),
    code(E11_PRE_CODE),
    code(E11_CODE),
    md(
        """
## 任务 12：画框

画 2px 红框并统计改动像素。注意颜色是 BGR。
"""
    ),
    code(E12_CODE),
    md(
        """
## 任务 13：越界框裁剪

把两个越界框裁到画面内。
"""
    ),
    code(E13_CODE),
    md(
        """
## 任务 14：AP 指标

从 TP 判定一路算到 all-point AP 与 11 点 AP。
"""
    ),
    code(E14_CODE),
    md(
        """
## 自查清单

- [ ] 能默写 IoU 的三行核心
- [ ] 知道「边界接触」的 IoU 是 0
- [ ] 会用广播算 IoU 矩阵
- [ ] 知道 IoU 对小目标更苛刻
- [ ] 记得 YOLO 归一化的分母（cx 除宽、cy 除高）
- [ ] 知道 YOLO 往返有损
- [ ] 能默写 NMS 主循环
- [ ] **知道 NMS 只与存活框比较**
- [ ] 知道画框是 BGR、厚度向两侧扩
- [ ] 知道越界框必须裁剪
- [ ] 知道一个 GT 只能被一个预测认领
- [ ] 知道 AP 要用单调化曲线，以及 all-point 与 11 点的差别
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
