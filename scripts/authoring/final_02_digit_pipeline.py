#!/usr/bin/env python3
"""cv final_02 全流程（手写数字识别）：三分划分 / 手撕 HOG / 手写增强 ->
HOG+LinearSVC vs 小 CNN -> 低数据与数据量扫描 -> 错误分析 -> 报告。

数据：`mnist/mnist_train_sub.npz`（2000 张）+ `mnist/mnist_test_sub.npz`（500 张）。
本章把 2000 张**再切成 train 1600 / val 400**，测试集仍用那 500 张。
真值全部实跑（sklearn 1.9.1 + torch 2.14.0，CPU）。

> 本章的核心是一条**交叉曲线**：
> 训练样本只有 **100** 张时，HOG+LinearSVC **0.832** 碾压小 CNN **0.764**（差 0.068）；
> 样本涨到 **1600** 张时，CNN **0.960** 反超 SVM **0.946**。
> 再加上增强这条「免费的样本」：在 400 张的低数据档把样本增强到 ×4，
> CNN 从 0.890 涨到 **0.970**（+0.080），而 SVM 只从 0.896 涨到 0.918（+0.022）——
> **同一份增强，深度模型吃到的收益是经典方法的 3.6 倍。**
>
> 第二个核心是**错误分析**：20 个错分样本的平均墨迹像素只有 **135.5**
> （正确样本 153.3），平均置信度只有 **0.754**（正确 0.983）。
> 置信度 < 0.9 的 38 个样本里错分率高达 **0.421053** ——
> 这条信号可以直接用来做「拒识 → 转人工复核」。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/final_02_digit_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv" / "final"
NAME = "final_02_digit_pipeline"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import cv2

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import LinearSVC

DATA = Path("../data")
MN = DATA / "mnist"
SEED = 0
'''

SETUP = '''tr_npz = np.load(str(MN / "mnist_train_sub.npz"))
te_npz = np.load(str(MN / "mnist_test_sub.npz"))
Xall, yall = tr_npz["images"], tr_npz["labels"]
Xte_raw, yte = te_npz["images"], te_npz["labels"]

print("训练池", Xall.shape, Xall.dtype, "| 测试集", Xte_raw.shape, Xte_raw.dtype)
print("训练池标签分布", np.bincount(yall).tolist())
print("测试集标签分布", np.bincount(yte).tolist())
print("灰度范围", int(Xall.min()), "~", int(Xall.max()))
print("注意: 本章把 2000 张训练池**再切成 train 1600 / val 400**，测试集仍用那 500 张 —— "
      "所以本章数字与 ch07/ch11（用满 2000 张）**不可直接比大小**")
'''

SCAFFOLD = '''# 脚手架（全部给全，本章的缺口都在后面的函数定义与实验里）
CELL, BINS, BLOCK = 7, 9, 2
MEAN, STD = 0.1300110519, 0.3072552681


def near(a, b, tol=1e-6):
    """浮点容差比较。真值都实跑得到，用容差避免 == 在末位翻车。"""
    return abs(float(a) - float(b)) <= tol


def unit(v):
    """把向量归一化成单位长度（加 1e-6 防止除零）。"""
    return v / np.sqrt((v ** 2).sum() + 1e-6)


def hog_naive(img, cell=CELL, bins=BINS, block=BLOCK):
    """朴素三重循环版 HOG（边界用最近邻复制）。与 ch07 完全同口径 —— 慢，但每步看得见。"""
    g = img.astype(np.float64)
    h, w = g.shape
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    for i in range(h):
        for j in range(w):
            i0, i1 = max(i - 1, 0), min(i + 1, h - 1)
            j0, j1 = max(j - 1, 0), min(j + 1, w - 1)
            gx[i, j] = g[i, j1] - g[i, j0]
            gy[i, j] = g[i1, j] - g[i0, j]
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    bidx = np.minimum((ang * bins / 180.0).astype(np.int32), bins - 1)
    ny, nx = h // cell, w // cell
    cells = np.zeros((ny, nx, bins), np.float64)
    mm = mag[: ny * cell, : nx * cell]
    bb = bidx[: ny * cell, : nx * cell]
    for k in range(bins):
        cells[:, :, k] = (mm * (bb == k)).reshape(ny, cell, nx, cell).sum(axis=(1, 3))
    out = []
    for by in range(ny - block + 1):
        for bx in range(nx - block + 1):
            out.append(unit(cells[by : by + block, bx : bx + block, :].ravel()))
    return np.concatenate(out)


def hog_vec(img, cell=CELL, bins=BINS, block=BLOCK):
    """向量化版 HOG —— 输出必须与 hog_naive **逐元素相等**，只是梯度那步用切片代替了双循环。"""
    g = img.astype(np.float64)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gx[:, 0] = g[:, 1] - g[:, 0]
    gx[:, -1] = g[:, -1] - g[:, -2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    gy[0, :] = g[1, :] - g[0, :]
    gy[-1, :] = g[-1, :] - g[-2, :]
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    bidx = np.minimum((ang * bins / 180.0).astype(np.int32), bins - 1)
    ny, nx = g.shape[0] // cell, g.shape[1] // cell
    mm = mag[: ny * cell, : nx * cell].reshape(ny, cell, nx, cell)
    bb = bidx[: ny * cell, : nx * cell].reshape(ny, cell, nx, cell)
    cells = np.zeros((ny, nx, bins))
    for k in range(bins):
        cells[:, :, k] = (mm * (bb == k)).sum(axis=(1, 3))
    out = []
    for by in range(ny - block + 1):
        for bx in range(nx - block + 1):
            out.append(unit(cells[by : by + block, bx : bx + block, :].ravel()))
    return np.concatenate(out)


def hog_features(imgs):
    """对一批图逐张取 HOG 再堆叠成矩阵。"""
    return np.stack([hog_vec(im) for im in imgs])


class SmallCNN(nn.Module):
    """小 CNN（与 ch11 同结构）：Conv-ReLU-Pool x2 + 全连接，输出 (N, 10) logits。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


def to_tensor(imgs):
    """uint8 (N,28,28) -> float32 (N,1,28,28)，统计量沿用 ch11 的训练集口径。"""
    return ((torch.from_numpy(np.asarray(imgs)).float() / 255.0 - MEAN) / STD).unsqueeze(1)
'''

E1_CODE = '''perm = np.random.default_rng(SEED).permutation(len(Xall))
# @@todo 按 perm 把训练池切成 train 1600 / val 400（图像与标签都要切）
# @@hint 两行：Xtr, ytr = Xall[perm[:1600]], yall[perm[:1600]]
# @@hint 一行：Xva, yva = Xall[perm[1600:]], yall[perm[1600:]]
Xtr, ytr = Xall[perm[:1600]], yall[perm[:1600]]
Xva, yva = Xall[perm[1600:]], yall[perm[1600:]]
# @@end

print("三分: train %s / val %s / test %s" % (Xtr.shape, Xva.shape, Xte_raw.shape))
print("val 标签分布", np.bincount(yva).tolist())
print("划分指纹 perm[:5] =", perm[:5].tolist(), "（换 seed 结果就变，所以必须写进实验记录）")
print("Xtr[0] 非零像素 %d 标签 %d | Xva[0] 非零像素 %d 标签 %d"
      % (int((Xtr[0] > 0).sum()), int(ytr[0]), int((Xva[0] > 0).sum()), int(yva[0])))
print("解读: ① **为什么要三分？** train 用来学参数，val 用来选模型/挑轮数/调超参，"
      "test 只在最后看一次 —— 用 test 去调参就是**信息泄漏**，分数会虚高；"
      "② 划分必须固定 seed 并**记录指纹**（perm 的前几个数），否则实验不可复现；"
      "③ 本章数字与 ch07/ch11 不可直接比 —— 那两章用的是满 2000 张，本章只有 1600")
assert perm[:5].tolist() == [1946, 1236, 1380, 1949, 1633]
assert Xtr.shape == (1600, 28, 28) and Xva.shape == (400, 28, 28)
assert np.bincount(yva).tolist() == [38, 52, 30, 36, 43, 36, 36, 38, 40, 51]
assert int((Xtr[0] > 0).sum()) == 130 and int(ytr[0]) == 4
assert int((Xva[0] > 0).sum()) == 116 and int(yva[0]) == 8
'''

E2_CODE = '''# @@todo 用同一张图跑两版 HOG，算最大绝对差（应该为 0）
# @@hint 两行：d0 = float(np.abs(hog_naive(Xtr[0]) - hog_vec(Xtr[0])).max())
# @@hint 一行：d3 = float(np.abs(hog_naive(Xtr[3]) - hog_vec(Xtr[3])).max())
d0 = float(np.abs(hog_naive(Xtr[0]) - hog_vec(Xtr[0])).max())
d3 = float(np.abs(hog_naive(Xtr[3]) - hog_vec(Xtr[3])).max())
# @@end

v0 = hog_vec(Xtr[0])
blk = [round(float(np.linalg.norm(v0[i * 36 : (i + 1) * 36])), 6) for i in range(3)]
print("HOG 维度", v0.shape, "| 最大差（第 0 张 / 第 3 张）= %.8f / %.8f" % (d0, d3))
print("每块（2x2 cell x 9 bin = 36 维）的 L2 范数 前 3 个:", blk, "-> 归一化生效")
print("全零像素图的 HOG 范数: %.8f" % float(np.linalg.norm(hog_vec(np.zeros((28, 28), np.uint8)))))
print("解读: ① 两版**逐元素相等**（差 0.0）说明向量化只是换了写法，没换语义 —— "
      "这是重构时的验收标准；"
      "② 参数 CELL=7 / BINS=9 / BLOCK=2 在 28x28 上正好得到 4x4 个 cell、"
      "(4-2+1)²=9 个 block、**9x36 = 324 维**，与 ch07 的 `hog324` 完全一致；"
      "③ **全黑图的 HOG 是 324 个 0**（每块归一化时分子为 0）—— 线性 SVM 拿到这种"
      "零向量只会输出一个「某个类的偏置」，**没有任何信息**。这就是纯背景会被乱判的根源")
assert d0 == 0.0 and d3 == 0.0
assert v0.shape == (324,)
assert blk == [1.0, 1.0, 1.0]
assert float(np.linalg.norm(hog_vec(np.zeros((28, 28), np.uint8)))) == 0.0
'''

E3_CODE = '''# @@todo 三份 HOG 特征矩阵：train / val / test
# @@hint 三行：Htr = hog_features(Xtr)；Hva = hog_features(Xva)；Hte = hog_features(Xte_raw)
Htr = hog_features(Xtr)
Hva = hog_features(Xva)
Hte = hog_features(Xte_raw)
# @@end

print("Htr", Htr.shape, Htr.dtype, "| Hva", Hva.shape, "| Hte", Hte.shape)
print("Htr[0] 前 4 个值:", np.round(Htr[0][:4], 8).tolist())
print("解读: 特征矩阵一行 = 一张图。**同一套 HOG 必须用同样的参数分别作用在三个集合上** —— "
      "train 用 A 参数、test 用 B 参数，是最隐蔽的一类 bug（训练/推理不一致）")
assert Htr.shape == (1600, 324) and Hva.shape == (400, 324) and Hte.shape == (500, 324)
assert Htr.dtype == np.float64
'''

E4_CODE = '''# @@todo 训练 LinearSVC（C=1.0, max_iter=5000, dual="auto"），算 val 与 test 准确率
# @@hint 三行：svm = LinearSVC(C=1.0, max_iter=5000, dual="auto").fit(Htr, ytr)
# @@hint 一行：acc_svm_va = float(accuracy_score(yva, svm.predict(Hva)))
# @@hint 一行：acc_svm_te = float(accuracy_score(yte, svm.predict(Hte)))
svm = LinearSVC(C=1.0, max_iter=5000, dual="auto").fit(Htr, ytr)
acc_svm_va = float(accuracy_score(yva, svm.predict(Hva)))
acc_svm_te = float(accuracy_score(yte, svm.predict(Hte)))
# @@end

print("HOG + LinearSVC -> val %.6f | test %.6f" % (acc_svm_va, acc_svm_te))
print("解读: ① **val 比 test 低 0.0335** —— val 只有 400 张，抽样噪声大，"
      "400 张上 3% 的波动完全正常，别据此调参调过头；"
      "② ch07 用满 2000 张训到 0.952，本章 1600 张只有 0.946 —— 差 0.006 就是**少了 400 张样本**的代价；"
      "③ LinearSVC 不需要调 lr、不需要训练轮数、CPU 上几十毫秒跑完 —— "
      "**这是所有工程问题的第一根标杆**")
assert near(acc_svm_va, 0.9125)
assert near(acc_svm_te, 0.946)
'''

E5_CODE = '''pred_svm = svm.predict(Hte)
cm_svm = confusion_matrix(yte, pred_svm)
# @@todo 逐类召回 = 对角 / 每行和
# @@hint 一行：rec_svm = cm_svm.diagonal() / cm_svm.sum(axis=1)
rec_svm = cm_svm.diagonal() / cm_svm.sum(axis=1)
# @@end

print("SVM 测试错分 %d / 500" % int((pred_svm != yte).sum()))
print("对角（各类正确数）:", cm_svm.diagonal().tolist())
print("行和（各类真实数）:", cm_svm.sum(axis=1).tolist())
print("逐类召回:", np.round(rec_svm, 6).tolist())
print("最差类别 argmin = %d（%.6f）| 最好类 argmax = %d（%.6f）"
      % (int(np.argmin(rec_svm)), float(rec_svm.min()), int(np.argmax(rec_svm)), float(rec_svm.max())))
print("解读: 整体 0.946，但**数字 9 只有 0.789474** —— 38 个 9 里错了 8 个。"
      "这就是「整体准确率掩盖最差类」的活教材（ch11 §11.12 讲过同一件事）。"
      "**报告里只给一个总准确率，就是在隐藏问题**")
assert int((pred_svm != yte).sum()) == 27
assert cm_svm.diagonal().tolist() == [55, 53, 49, 37, 42, 38, 46, 57, 51, 45]
assert cm_svm.sum(axis=1).tolist() == [56, 54, 51, 39, 43, 40, 48, 58, 54, 57]
assert int(np.argmin(rec_svm)) == 9 and near(float(rec_svm.min()), 0.789474, 1e-6)
assert int(np.argmax(rec_svm)) == 7 and near(float(rec_svm.max()), 0.982759, 1e-6)
'''

E6_CODE = '''def aug_one(im, rng):
    """生成一张增强副本：随机旋转 ±12°、缩放 0.9~1.1、再平移 ±2 像素，边框填黑。"""
    # @@todo 抽 4 个随机参数：旋转角、缩放比、x 平移、y 平移
    # @@hint 两行：ang = float(rng.uniform(-12, 12))；sc = float(rng.uniform(0.9, 1.1))
    # @@hint 两行：dx = int(rng.integers(-2, 3))；dy = int(rng.integers(-2, 3))
    ang = float(rng.uniform(-12, 12))
    sc = float(rng.uniform(0.9, 1.1))
    dx = int(rng.integers(-2, 3))
    dy = int(rng.integers(-2, 3))
    # @@end
    M = cv2.getRotationMatrix2D((14.0, 14.0), ang, sc)
    M[0, 2] += dx
    M[1, 2] += dy
    # @@todo 用仿射矩阵把图像搬过去（双线性插值 + 黑边框）
    # @@hint 一行：return cv2.warpAffine(im, M, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)
    return cv2.warpAffine(im, M, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)
    # @@end


demo = aug_one(Xtr[0], np.random.default_rng(SEED))
print("aug_one(Xtr[0]) ->", demo.shape, demo.dtype,
      "| 非零像素 %d（原图 %d）" % (int((demo > 0).sum()), int((Xtr[0] > 0).sum())))
print("与原文逐像素最大差", int(np.abs(demo.astype(int) - Xtr[0].astype(int)).max()))
print("解读: ① 增强后**非零像素变多了** —— 因为 `warpAffine` 的双线性插值会在笔画边缘"
      "造出中间灰值，把原本是 0 的像素点亮；"
      "② 这不是 bug，但它提醒你：**增强改变的是像素分布**，不是「同一张图换个位置」那么简单；"
      "③ 旋转 ±12° 是刻意取的小角度 —— 手写数字的「6 / 9」「5 / 3」在大角度旋转下会彼此混淆，"
      "那不是增强，是**造错标签**")
assert demo.shape == (28, 28) and demo.dtype == np.uint8
assert not np.array_equal(demo, Xtr[0])
'''

E7_CODE = '''def augment(imgs, labels, times, seed=0):
    """把每张图展开成 times 份（第 0 份是原图，其余是增强副本）。返回 (图像数组, 标签数组)。

    **增强不改变标签** —— 这是整个增强范式的合法性前提。
    """
    rng = np.random.default_rng(seed)
    # @@todo 用双重列表推导：每张图先收原图，再补 times-1 个增强副本
    # @@hint 一行：out = [[im] + [aug_one(im, rng) for _ in range(times - 1)] for im in imgs]
    out = [[im] + [aug_one(im, rng) for _ in range(times - 1)] for im in imgs]
    # @@end
    # @@todo 把嵌套列表摊平成 (N*times, 28, 28) 图像数组 + 对应标签数组
    # @@hint 两行：Xa = np.stack([x for g in out for x in g])
    # @@hint 一行：ya = np.repeat(np.asarray(labels), times).astype(labels.dtype)
    Xa = np.stack([x for g in out for x in g])
    ya = np.repeat(np.asarray(labels), times).astype(labels.dtype)
    # @@end
    return Xa, ya
'''

E8_CODE = '''AX, AY = augment(Xtr[:400], ytr[:400], times=4, seed=SEED)
# @@todo 四件统计：标签集合是否不变 / 增强前后像素均值 / 标准差 / 非零像素占比
# @@hint 一行：same_label = sorted(set(AY.tolist())) == sorted(set(ytr[:400].tolist()))
# @@hint 一行：m0, m1 = float(Xtr[:400].mean()), float(AX.mean())
# @@hint 一行：s0, s1 = float(Xtr[:400].std()), float(AX.std())
# @@hint 一行：z0, z1 = float((Xtr[:400] > 0).mean()), float((AX > 0).mean())
same_label = sorted(set(AY.tolist())) == sorted(set(ytr[:400].tolist()))
m0, m1 = float(Xtr[:400].mean()), float(AX.mean())
s0, s1 = float(Xtr[:400].std()), float(AX.std())
z0, z1 = float((Xtr[:400] > 0).mean()), float((AX > 0).mean())
# @@end

print("augment(400 张, times=4) ->", AX.shape, AY.shape)
print("每类样本数", np.bincount(AY).tolist(), "| 标签集合不变?", same_label)
print("前 4 个标签", AY[:4].tolist(), "（= ytr[:4]，增强不重排）")
print("像素均值 %.6f -> %.6f | std %.6f -> %.6f | 非零占比 %.6f -> %.6f" % (m0, m1, s0, s1, z0, z1))
print("确定性: 同 seed 两次一致?", np.array_equal(AX, augment(Xtr[:400], ytr[:400], times=4, seed=SEED)[0]),
      "| times=1 是否就是原样?", np.array_equal(augment(Xtr[:400], ytr[:400], times=1, seed=SEED)[0], Xtr[:400]))
print("解读: ① **标签必须一一对应**：第 i 张图的所有副本标签都等于 ytr[i]，"
      "`np.repeat` 保证了这个顺序（第 0~3 个副本 = ytr[0]）；"
      "② 均值几乎不变（33.961703 -> 33.992993），但 **std 从 79.326340 掉到 75.869218** —— "
      "笔画被旋转/平移后分布更散，方差变小；"
      "③ **非零占比从 0.193798 涨到 0.238372**（插值灰边）；"
      "④ 增强必须是**确定性的**（传 seed），否则「同样配置跑两次数字不同」，实验没法做")
assert AX.shape == (1600, 28, 28) and AY.shape == (1600,)
assert same_label is True
assert np.bincount(AY).tolist() == [148, 176, 164, 128, 128, 152, 156, 192, 168, 188]
assert AY[:4].tolist() == [4, 4, 4, 4]
assert near(m0, 33.961703, 1e-6) and near(m1, 33.992993, 1e-6)
assert near(s0, 79.326340, 1e-6) and near(s1, 75.869218, 1e-6)
assert near(z0, 0.193798, 1e-6) and near(z1, 0.238372, 1e-6)
'''

E9_CODE = '''def train_cnn(imgs, labels, epochs=15, lr=2e-3, bs=64, seed=0):
    """训练 epochs 轮，返回模型。种子固定 + DataLoader 传 generator，保证可复现。"""
    torch.manual_seed(seed)
    net = SmallCNN()
    ds = TensorDataset(to_tensor(imgs), torch.from_numpy(np.asarray(labels)).long())
    dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        net.train()
        for xb, yb in dl:
            # @@todo 训练四步：清梯度 -> 前向 + 算 loss -> 反向传播 -> 更新参数
            # @@hint 四行：opt.zero_grad() / loss = crit(net(xb), yb) / loss.backward() / opt.step()
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            # @@end
    return net
'''

E10_CODE = '''def pred_cnn(net, imgs):
    """推理：eval + no_grad，返回预测类别数组。"""
    net.eval()
    with torch.no_grad():
        # @@todo 一行：前向 -> 在 dim=1 上 argmax -> 转成 numpy
        # @@hint 一行：return net(to_tensor(imgs)).argmax(1).numpy()
        return net(to_tensor(imgs)).argmax(1).numpy()
        # @@end
'''

E11_CODE = '''net = train_cnn(Xtr, ytr, epochs=15, seed=SEED)
# @@todo 三个集合的准确率：train / val / test
# @@hint 三行：pred_cnn_tr = pred_cnn(net, Xtr)；pred_cnn_va = pred_cnn(net, Xva)
# @@hint 一行：pred_cnn_te = pred_cnn(net, Xte_raw)
# @@hint 三行：acc_cnn_tr / acc_cnn_va / acc_cnn_te 都写成 float((预测 == 真值).mean())
pred_cnn_tr = pred_cnn(net, Xtr)
pred_cnn_va = pred_cnn(net, Xva)
pred_cnn_te = pred_cnn(net, Xte_raw)
acc_cnn_tr = float((pred_cnn_tr == ytr).mean())
acc_cnn_va = float((pred_cnn_va == yva).mean())
acc_cnn_te = float((pred_cnn_te == yte).mean())
# @@end

Xva_t = to_tensor(Xva)
Xte_t = to_tensor(Xte_raw)
print("小 CNN（15 轮）-> train %.6f | val %.6f | test %.6f" % (acc_cnn_tr, acc_cnn_va, acc_cnn_te))
print("Xva_t", tuple(Xva_t.shape), "| mean %.3e std %.6f" % (float(Xva_t.mean()), float(Xva_t.std())))
print("解读: ① **train 1.000000 / val 0.950000** —— gap 0.05，过拟合（1600 张训 9098 个参数）；"
      "② val 上 CNN 0.950000 > SVM 0.912500，test 上 CNN 0.960000 > SVM 0.946000，"
      "**两个集合上排序一致** —— 这才是「CNN 更好」的可靠证据；"
      "③ 如果 val 和 test 给出相反排序，说明差异落在噪声里，不该下结论")
assert near(acc_cnn_tr, 1.0)
assert near(acc_cnn_va, 0.95)
assert near(acc_cnn_te, 0.96)
assert tuple(Xva_t.shape) == (400, 1, 28, 28)
'''

E12_CODE = '''cm_cnn = confusion_matrix(yte, pred_cnn_te)
# @@todo CNN 的逐类召回（对角 / 行和）
# @@hint 一行：rec_cnn = cm_cnn.diagonal() / cm_cnn.sum(axis=1)
rec_cnn = cm_cnn.diagonal() / cm_cnn.sum(axis=1)
# @@end
# @@todo 两模型在测试集上的四种组合计数：都对 / 都错 / SVM错CNN对 / SVM对CNN错
# @@hint 两行：both_ok = int(((pred_svm == yte) & (pred_cnn_te == yte)).sum())
# @@hint 一行：both_bad = int(((pred_svm != yte) & (pred_cnn_te != yte)).sum())
# @@hint 两行：svm_bad_cnn_ok = int(((pred_svm != yte) & (pred_cnn_te == yte)).sum())
# @@hint 一行：svm_ok_cnn_bad = int(((pred_svm == yte) & (pred_cnn_te != yte)).sum())
both_ok = int(((pred_svm == yte) & (pred_cnn_te == yte)).sum())
both_bad = int(((pred_svm != yte) & (pred_cnn_te != yte)).sum())
svm_bad_cnn_ok = int(((pred_svm != yte) & (pred_cnn_te == yte)).sum())
svm_ok_cnn_bad = int(((pred_svm == yte) & (pred_cnn_te != yte)).sum())
# @@end

print("CNN 对角:", cm_cnn.diagonal().tolist())
print("CNN 逐类召回:", np.round(rec_cnn, 6).tolist())
print("CNN 最差类 argmin = %d（%.6f）| 最好类 argmax = %d（%.6f）"
      % (int(np.argmin(rec_cnn)), float(rec_cnn.min()), int(np.argmax(rec_cnn)), float(rec_cnn.max())))
print("两模型对照（500 张）: 都对 %d | 都错 %d | SVM错CNN对 %d | SVM对CNN错 %d"
      % (both_ok, both_bad, svm_bad_cnn_ok, svm_ok_cnn_bad))
print("净收益 = %d - %d = %d 张（准确率差 %.4f）"
      % (svm_bad_cnn_ok, svm_ok_cnn_bad, svm_bad_cnn_ok - svm_ok_cnn_bad,
         (svm_bad_cnn_ok - svm_ok_cnn_bad) / 500))
print("解读: ① **CNN 把数字 9 的召回从 0.789474 提到 0.894737（+0.105）**，"
      "而整体只涨 0.014 —— 与 ch11 §11.12 的结论一模一样的形状；"
      "② 两模型**都错 11 张**：这是「当前两个方法共同的天花板」，"
      "换更强的模型前先看看这 11 张长什么样；"
      "③ CNN 修好了 SVM 的 16 个错、又弄坏了 SVM 对的 9 个 -> 净 +7 张。"
      "**只看「准确率涨了 0.014」会漏掉这个结构**")
assert cm_cnn.diagonal().tolist() == [55, 53, 51, 38, 43, 39, 45, 55, 50, 51]
assert int(np.argmin(rec_cnn)) == 9 and near(float(rec_cnn.min()), 0.894737, 1e-6)
assert (both_ok, both_bad, svm_bad_cnn_ok, svm_ok_cnn_bad) == (464, 11, 16, 9)
assert svm_bad_cnn_ok - svm_ok_cnn_bad == 7
'''

E13_CODE = '''lite_rows = {}
for times in (1, 2, 4):
    I, L = augment(Xtr[:400], ytr[:400], times=times, seed=SEED)
    Hi = hog_features(I)
    s = LinearSVC(C=1.0, max_iter=5000, dual="auto").fit(Hi, L)
    # @@todo 算 SVM 测试准确率；训练一个 CNN 并算它的测试准确率；把 (样本数, 两 acc) 记进 lite_rows
    # @@hint 一行：a_svm = float(accuracy_score(yte, s.predict(Hte)))
    # @@hint 一行：n = train_cnn(I, L, epochs=15, seed=SEED)
    # @@hint 一行：a_cnn = float(accuracy_score(yte, pred_cnn(n, Xte_raw)))
    # @@hint 一行：lite_rows[times] = (len(I), round(a_svm, 6), round(a_cnn, 6))
    a_svm = float(accuracy_score(yte, s.predict(Hte)))
    n = train_cnn(I, L, epochs=15, seed=SEED)
    a_cnn = float(accuracy_score(yte, pred_cnn(n, Xte_raw)))
    lite_rows[times] = (len(I), round(a_svm, 6), round(a_cnn, 6))
    # @@end

print("低数据档：训练池只有 400 张，靠增强「造样本」")
print("  增强倍数 | 训练样本 | SVM test | CNN test")
for times in (1, 2, 4):
    n, a_s, a_c = lite_rows[times]
    print(f"     x{times}     |  {n:5d}   |  {a_s:.6f} | {a_c:.6f}")
print("SVM 增益 %.6f | CNN 增益 %.6f（相差 %.6f 倍）"
      % (lite_rows[4][1] - lite_rows[1][1], lite_rows[4][2] - lite_rows[1][2],
         (lite_rows[4][2] - lite_rows[1][2]) / (lite_rows[4][1] - lite_rows[1][1])))
print("解读: **同一份增强，两个模型的收益差了 3.6 倍。** "
      "① SVM 只涨 0.022（0.896000 -> 0.918000）—— 因为 HOG 本身就是「平移/光照不变量」的设计，"
      "增强喂进去的信息它**已经从特征里拿到了**；"
      "② CNN 涨 0.080（0.890000 -> 0.970000）—— 它没有内置任何不变性，"
      "「平移不变」是靠自己看数据学出来的，所以增强对它是**教学材料**；"
      "③ 结论：**增强的收益取决于「模型缺什么」**，不是「增强了就一定有收益」")
assert lite_rows[1] == (400, 0.896, 0.89)
assert lite_rows[2] == (800, 0.908, 0.912)
assert lite_rows[4] == (1600, 0.918, 0.97)
assert lite_rows[4][2] - lite_rows[1][2] > 3 * (lite_rows[4][1] - lite_rows[1][1])
'''

E14_CODE = '''sweep_rows = {}
for n_tr in (100, 200, 400, 800, 1600):
    Hi = hog_features(Xtr[:n_tr])
    s = LinearSVC(C=1.0, max_iter=5000, dual="auto").fit(Hi, ytr[:n_tr])
    # @@todo 同上一节：SVM 与 CNN 的测试准确率，记进 sweep_rows[n_tr]
    # @@hint 一行：a_svm = float(accuracy_score(yte, s.predict(Hte)))
    # @@hint 一行：n = train_cnn(Xtr[:n_tr], ytr[:n_tr], epochs=15, seed=SEED)
    # @@hint 一行：a_cnn = float(accuracy_score(yte, pred_cnn(n, Xte_raw)))
    # @@hint 一行：sweep_rows[n_tr] = (round(a_svm, 6), round(a_cnn, 6))
    a_svm = float(accuracy_score(yte, s.predict(Hte)))
    n = train_cnn(Xtr[:n_tr], ytr[:n_tr], epochs=15, seed=SEED)
    a_cnn = float(accuracy_score(yte, pred_cnn(n, Xte_raw)))
    sweep_rows[n_tr] = (round(a_svm, 6), round(a_cnn, 6))
    # @@end

print("数据量扫描（无增强）:")
print("  n_train |  SVM    |  CNN    | 谁赢")
for n_tr in (100, 200, 400, 800, 1600):
    a_s, a_c = sweep_rows[n_tr]
    print(f"   {n_tr:5d}  | {a_s:.6f} | {a_c:.6f} | {'CNN' if a_c > a_s else 'SVM'}")
cross = min((n for n in (100, 200, 400, 800, 1600) if sweep_rows[n][1] > sweep_rows[n][0]), default=None)
print("反超点: n_train =", cross, "（此之前 SVM 领先，之后 CNN 领先）")
print("解读: **这就是本章最值钱的一张表。** 100 张时 SVM 0.832000 领先 CNN 0.764000 达 0.068；"
      "200 张时 SVM 仍赢 0.030；400 张时 SVM 领先收窄到 0.006；"
      "800 张时 SVM 还赢（0.932000 vs 0.920000）；**到 1600 张 CNN 才反超（0.960000 vs 0.946000）**。"
      "① 注意 800 那一行：CNN 从 400 张的 0.890000 涨到 0.920000，"
      "但 SVM 涨得更猛（0.896000 -> 0.932000）—— **两条曲线的斜率在换**："
      "SVM 逼近它的天花板（0.932 -> 0.946，只再涨 0.014），CNN 还在爬（0.920 -> 0.960，再涨 0.040）；"
      "② **「深度模型更好」这句话没有主语 —— 它必须补上「在多少数据上」**")
assert sweep_rows[100] == (0.832, 0.764)
assert sweep_rows[200] == (0.88, 0.85)
assert sweep_rows[400] == (0.896, 0.89)
assert sweep_rows[800] == (0.932, 0.92)
assert sweep_rows[1600] == (0.946, 0.96)
assert cross == 1600
assert sweep_rows[1600][1] - sweep_rows[800][1] > sweep_rows[1600][0] - sweep_rows[800][0]
'''

E15_CODE = '''bad = np.where(pred_cnn_te != yte)[0]
# @@todo 算「墨迹像素」在全体 / 错分 / 正确样本上的均值
# @@hint 一行：ink = (Xte_raw > 0).sum(axis=(1, 2))
# @@hint 一行：ink_all = float(ink.mean())
# @@hint 一行：ink_bad = float(ink[bad].mean())
# @@hint 一行：ink_ok = float(ink[pred_cnn_te == yte].mean())
ink = (Xte_raw > 0).sum(axis=(1, 2))
ink_all = float(ink.mean())
ink_bad = float(ink[bad].mean())
ink_ok = float(ink[pred_cnn_te == yte].mean())
# @@end

print("错分样本 %d 个，索引前 12:" % len(bad), bad[:12].tolist())
print("错分的真值分布:", np.bincount(yte[bad], minlength=10).tolist())
print("错分的预测分布:", np.bincount(pred_cnn_te[bad], minlength=10).tolist())
print("墨迹像素均值: 全体 %.6f | 错分 %.6f | 正确 %.6f" % (ink_all, ink_bad, ink_ok))
print("解读: **错分样本的笔画明显更细/更少（135.5 vs 153.3）** —— "
      "「写得潦草、笔画断续」确实是模型的主要失败模式。"
      "这类样本**不是模型的错，是标注世界的长尾**；"
      "正确做法是补数据（把这类样本收集起来复训），而不是继续调网络结构")
assert len(bad) == 20
assert np.bincount(yte[bad], minlength=10).tolist() == [1, 1, 0, 1, 0, 1, 3, 3, 4, 6]
assert near(ink_all, 152.632, 1e-6) and near(ink_bad, 135.5, 1e-6)
assert near(ink_ok, 153.345833, 1e-6)
assert ink_bad < ink_ok
'''

E16_CODE = '''net.eval()
with torch.no_grad():
    prob = torch.softmax(net(Xte_t), dim=1).numpy()
conf = prob[np.arange(len(yte)), pred_cnn_te]
# @@todo 错分 / 正确样本的平均置信度，以及「置信度 < 0.9」子集的错分率
# @@hint 两行：conf_bad = float(conf[bad].mean())；conf_ok = float(conf[pred_cnn_te == yte].mean())
# @@hint 一行：lo = np.where(conf < 0.9)[0]
# @@hint 一行：lo_err = float(len(np.intersect1d(lo, bad)) / max(len(lo), 1))
conf_bad = float(conf[bad].mean())
conf_ok = float(conf[pred_cnn_te == yte].mean())
lo = np.where(conf < 0.9)[0]
lo_err = float(len(np.intersect1d(lo, bad)) / max(len(lo), 1))
# @@end

print("平均置信度: 错分 %.6f | 正确 %.6f" % (conf_bad, conf_ok))
print("置信度 < 0.9 的样本: %d 个（占 %.6f），其中错 %d 个 -> 子集错分率 %.6f"
      % (len(lo), len(lo) / len(yte), int((conf[bad] < 0.9).sum()), lo_err))
print("全体错分率 %.6f，对比子集错分率 %.6f -> 放大 %.2f 倍"
      % (len(bad) / len(yte), lo_err, lo_err / (len(bad) / len(yte))))
print("解读: ① 模型的**置信度和正确率是同步的**（0.983 vs 0.754），说明它「知道自己不确定」；"
      "② 用 `conf < 0.9` 卡出一个 38 张的子集，其中 **42.11% 是错的**（全体只有 4%）—— "
      "**放大 10.5 倍**；"
      "③ 工程上这就是**拒识 / 人工复核**的开关：把这 7.6% 的样本推给人看，"
      "剩下的自动通过，整体准确率立刻从 0.960 抬到接近 1.0。"
      "**「给一个准确率」不如「给一个准确率 + 一个可信度门槛」**")
assert near(conf_bad, 0.754318, 1e-6)
assert near(conf_ok, 0.983203, 1e-6)
assert len(lo) == 38 and int((conf[bad] < 0.9).sum()) == 16
assert near(lo_err, 0.421053, 1e-6)
assert lo_err > 10 * (len(bad) / len(yte))
'''

E17_CODE = '''# @@todo 把本章三条口径并排成一张表：(val, test) 二元组
# @@hint 三行：final_table = {"HOG+LinearSVC(1600)": (acc_svm_va, acc_svm_te),
# @@hint 一行：                "SmallCNN(1600)": (acc_cnn_va, acc_cnn_te),
# @@hint 一行：                "SmallCNN(400 + aug x4)": (None, lite_rows[4][2])}
final_table = {"HOG+LinearSVC(1600)": (acc_svm_va, acc_svm_te),
               "SmallCNN(1600)": (acc_cnn_va, acc_cnn_te),
               "SmallCNN(400 + aug x4)": (None, lite_rows[4][2])}
# @@end

print("=" * 66)
print("final_02 结论表")
print("-" * 66)
print("  %-26s %10s %10s" % ("方案", "val", "test"))
for k, (v, t) in final_table.items():
    print("  %-26s %10s %10.6f" % (k, "—" if v is None else "%.6f" % v, t))
print("-" * 66)
print("  参考(ch07/ch11，用满 2000 张): hog324+SVM test 0.952 | SmallCNN(15ep) test 0.960")
print("=" * 66)
print("交付结论：")
print("  1. 必须先建**可复现的三分划分**（记 seed 与 perm 指纹），test 只在最后看一次。")
print("  2. 特征工程路线：HOG(324 维) + LinearSVC，1600 张训到 test 0.946，几毫秒出结果。")
print("  3. 端到端路线：SmallCNN 15 轮 test 0.960，但 train 已 1.000（过拟合）。")
print("  4. **数据量决定谁赢**：100 张 SVM 领先 0.068 -> 800 张仍领先 -> 1600 张 CNN 才反超。")
print("  5. **增强=免费样本，但收益不对称**：低数据档 CNN +0.080 vs SVM +0.022（3.6 倍）。")
print("  6. 错误分析给行动项：错分样本笔画更细（墨迹 135.5 vs 153.3）-> 该补数据不是改结构。")
print("  7. 置信度 < 0.9 的子集错分率 42.11%（全体 4%）-> 直接可用作拒识门槛。")
assert len(final_table) == 3
assert near(final_table["SmallCNN(1600)"][1], 0.96)
assert near(final_table["SmallCNN(400 + aug x4)"][1], 0.97)
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# final_02 全流程：手写数字识别

> 03_cv 的第二个综合题。它把 ch02（增强）、ch07（特征 + 分类器）、ch11（深度学习）
> 三条线拧成一条**可交付的流水线**：
> **三分划分 → 手撕 HOG → 手写增强 → 特征/SVM 基线 → 小 CNN → 逐类指标 →
> 低数据实验 → 数据量扫描 → 错误分析 → 报告**。
>
> 它要回答的是一个**工程问题**（而不是「CNN 强不强」）：
> **手上只有这么多数据时，我该选哪条路？**

## 本章路线

| 小节 | 内容 | 关键结论 |
|---|---|---|
| §F2.1 | 环境与数据 | 1600 / 400 / 500 |
| §F2.1b | **三分划分** | 记 seed 与 perm 指纹 |
| §F2.2 | 手撕 HOG（naive vs 向量化对账） | 324 维，与 ch07 同口径 |
| §F2.3 | HOG 特征矩阵 | 全黑图的 HOG 是 324 个 0 |
| §F2.4 | HOG + LinearSVC 基线 | val 0.9125 / test **0.946** |
| §F2.5 | 逐类召回 | 数字 9 只有 **0.789474** |
| §F2.6 | **手写增强 `aug_one`** | 旋转 ±12° / 缩放 ±10% / 平移 ±2px |
| §F2.7 | **批量增强 `augment`** | 标签必须一一对应 |
| §F2.8 | 增强改变了什么 | std 79.33→75.87，非零 0.19→0.24 |
| §F2.9 | **训练四步（`train_cnn`）** | 与 ch11 完全一致 |
| §F2.10 | 推理 `pred_cnn` | eval + no_grad |
| §F2.11 | 小 CNN 三档准确率 | train 1.000 / val 0.950 / test **0.960** |
| §F2.12 | 两模型四组合计数 | CNN 修好 16、弄坏 9，净 +7 |
| §F2.13 | **低数据 + 增强** | CNN +0.080 vs SVM +0.022 |
| §F2.14 | **数据量扫描** | **100 张 SVM 赢 0.068；1600 张 CNN 才反超（斜率在换）** |
| §F2.15 | 错误分析（墨迹） | 错分样本笔画更细（135.5 vs 153.3） |
| §F2.16 | 错误分析（置信度） | 低置信子集错分率 **42.11%** |
| §F2.17 | 结论表 | 7 条交付结论 |

## 关于数据

`mnist/mnist_train_sub.npz`（2000 张）+ `mnist/mnist_test_sub.npz`（500 张）。
本章把 2000 张**再切成 train 1600 / val 400**，测试集仍用那 500 张。

> ⚠️ 所以本章数字**不能**和 ch07（hog324+SVM = 0.952）、ch11（CNN = 0.960）直接比大小 ——
> 那两章用的是满 2000 张。本章的意义在于**受控对比**：同一份 1600 张上，
> 两条路线、不同数据量、加不加增强，全部在同一条尺子上量。
"""
    ),
    md(
        """
## §F2.1 环境与数据准备
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## §F2.1b 三分划分

**test 只在最后看一次。** 用 test 调超参就是信息泄漏。
"""
    ),
    code(E1_CODE),
    md(
        """
## §F2.2 手撕 HOG：两版对账

向量化只是换写法，**输出必须逐元素相等** —— 这是重构的验收标准。
"""
    ),
    code(E2_CODE),
    md(
        """
## §F2.3 特征矩阵

同一套 HOG 参数必须分别作用在三个集合上。
"""
    ),
    code(E3_CODE),
    md(
        """
## §F2.4 特征工程基线：HOG + LinearSVC

**所有工程问题的第一根标杆。**
"""
    ),
    code(E4_CODE),
    md(
        """
## §F2.5 逐类召回：整体准确率会骗人

数字 9 是这一章的长尾。
"""
    ),
    code(E5_CODE),
    md(
        """
## §F2.6 手写增强：`aug_one`

旋转 ±12°、缩放 0.9~1.1、平移 ±2 像素。角度刻意取小 —— 大角度会把 6 转成 9。
"""
    ),
    code(E6_CODE),
    md(
        """
## §F2.7 批量增强：`augment`

难点不是「怎么增强一张」，而是**标签与顺序的对应**。
"""
    ),
    code(E7_CODE),
    md(
        """
## §F2.8 增强改变了什么？

不要只看「增强后图变了」。要看**统计量变了多少**。
"""
    ),
    code(E8_CODE),
    md(
        """
## §F2.9 训练四步（`train_cnn` 的循环体）

与 ch11 §11.9 完全一致的四行。
"""
    ),
    code(E9_CODE),
    md(
        """
## §F2.10 推理（`pred_cnn`）
"""
    ),
    code(E10_CODE),
    md(
        """
## §F2.11 小 CNN 在 1600 张上

train / val / test 三个数字一起看。
"""
    ),
    code(E11_CODE),
    md(
        """
## §F2.12 两模型对照：谁修好了谁

准确率差 0.014 这个数字**掩盖了结构**。拆开看四种组合。
"""
    ),
    code(E12_CODE),
    md(
        """
## §F2.13 低数据 + 增强：收益不对称

训练池只有 400 张，靠增强造样本。**这一节会给出一个反直觉的数字。**
"""
    ),
    code(E13_CODE),
    md(
        """
## §F2.14 数据量扫描：反超点在哪

**本章最值钱的一张表。**
"""
    ),
    code(E14_CODE),
    md(
        """
## §F2.15 错误分析（一）：错分样本长什么样
"""
    ),
    code(E15_CODE),
    md(
        """
## §F2.16 错误分析（二）：模型知道自己不确定吗
"""
    ),
    code(E16_CODE),
    md(
        """
## §F2.17 结论表
"""
    ),
    code(E17_CODE),
    md(
        """
## 自查清单

- [ ] 知道**三分划分**：train 学参数 / val 选模型 / test 只看一次
- [ ] 会记录划分的 **seed 与 perm 指纹**
- [ ] 知道重构时「向量化 vs 朴素版」要对账到 **差 0.0**
- [ ] 知道 HOG 参数（cell 7 / bin 9 / block 2）如何算出 **324 维**
- [ ] 知道**全黑图的 HOG 是零向量**，线性模型只能吐偏置
- [ ] **知道同一套特征参数必须作用在 train/val/test 三处**
- [ ] 会用 `confusion_matrix` 的对角/行和算**逐类召回**
- [ ] 知道整体准确率会掩盖最差类（9 只有 0.789474）
- [ ] 能写出手写增强（旋转 + 缩放 + 平移）并**保证标签对应**
- [ ] 知道**增强必须确定性**（传 seed）
- [ ] 知道增强会改变像素分布（std 79.33→75.87、非零占比 0.19→0.24）
- [ ] 知道增强角度不能太大（会把 6 转成 9，等于**造错标签**）
- [ ] **知道「深度模型更好」必须补上「在多少数据上」**
- [ ] **知道增强的收益取决于「模型缺什么」**（CNN +0.080 vs SVM +0.022）
- [ ] 会用「两模型四组合计数」看准确率差背后的结构
- [ ] 会用**墨迹像素**和**置信度**做错误分析
- [ ] 知道**低置信子集可直接当拒识门槛**（0.421053 vs 全体 0.04）

## 交付物

| 文件 | 内容 |
|---|---|
| `final_02_digit_pipeline.ipynb` | 讲解版（含结论与解读） |
| `final_02_digit_pipeline_practice.ipynb` | 练习版（挖空 + 提示） |
| `final_02_digit_pipeline_solution.ipynb` | 答案版（完整可跑） |
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# ===========================================================================

EXERCISE = [
    md(
        """
# final_02 全流程：手写数字识别（练习版）

> 按提示补全 `____`，跑通所有 assert。
> `Xall` / `yall` / `Xte_raw` / `yte` 以及 `near` / `unit` / `hog_naive` / `hog_vec` /
> `hog_features` / `SmallCNN` / `to_tensor` 脚手架可用；
> `aug_one` / `augment` / `train_cnn` / `pred_cnn` 需要你补全函数体。
>
> 注意：挖空块内每条语句保持**单行**；控制流写在挖空块之外
> （`augment` / `train_cnn` 的循环骨架都已给好，你只填里面的几行）。
> 题目真值都来自实跑（sklearn 1.9.1 / torch 2.14.0，CPU）。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：三分划分

按固定 perm 切出 train 1600 / val 400。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：HOG 两版对账

naive 与向量化版必须差 0.0。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：特征矩阵

三个集合各取一次 HOG。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：HOG + LinearSVC 基线
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：逐类召回
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：`aug_one`（增强一张图）
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：`augment`（批量增强）
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：增强前后统计量对比
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 9：`train_cnn` 的训练四步
"""
    ),
    code(E9_CODE),
    md(
        """
## 任务 10：`pred_cnn` 推理
"""
    ),
    code(E10_CODE),
    md(
        """
## 任务 11：小 CNN 的 train / val / test
"""
    ),
    code(E11_CODE),
    md(
        """
## 任务 12：两模型四组合计数
"""
    ),
    code(E12_CODE),
    md(
        """
## 任务 13：低数据 + 增强（400 张）
"""
    ),
    code(E13_CODE),
    md(
        """
## 任务 14：数据量扫描（找反超点）
"""
    ),
    code(E14_CODE),
    md(
        """
## 任务 15：错误分析（墨迹像素）
"""
    ),
    code(E15_CODE),
    md(
        """
## 任务 16：错误分析（置信度与拒识）
"""
    ),
    code(E16_CODE),
    md(
        """
## 任务 17：结论表
"""
    ),
    code(E17_CODE),
    md(
        """
## 自查清单

- [ ] 三分划分 + 记录 seed / perm 指纹
- [ ] 向量化重构要与朴素版对账到 0.0
- [ ] HOG 参数 → 324 维的算式
- [ ] 全黑图 HOG 是零向量
- [ ] 逐类召回（对角 / 行和）
- [ ] 手写增强 + 标签对应 + 确定性
- [ ] 增强会改变像素分布
- [ ] 「深度模型更好」要补「在多少数据上」
- [ ] 增强收益取决于模型缺什么
- [ ] 低置信子集可作拒识门槛
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
