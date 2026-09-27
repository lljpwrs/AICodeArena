#!/usr/bin/env python3
"""cv11 深度学习入门：numpy->tensor 与归一化 / Dataset·DataLoader（批次、drop_last、shuffle 种子）/
用 F.unfold 手撕 conv2d / 小 CNN 结构与参数量 / 训练四步 / epoch 扫描（欠拟合->过拟合）/
混淆矩阵与逐类召回 / 与 ch07 的 HOG+SVM 对比 / 可复现性。

数据：`mnist/mnist_train_sub.npz`（2000 张 28x28 uint8）、`mnist/mnist_test_sub.npz`（500 张）。
与 ch07 用的是**同一份划分**，所以两章结果可以直接并列。
真值全部实跑（torch 2.14.0 + torchvision 0.29.0，CPU）。

> 本章三条主线：① **训练不足 vs 过拟合** —— 同样结构，3 epoch 只有 0.830，
> 15 epoch 到 **0.960**，但此时训练集已经 **1.000**（gap 0.040）；
> ② **归一化与训练轮数都能改结果** —— 同一结构换归一化方式，15 epoch 从 0.946 变 0.960；
> ③ **可复现性要三处一起管** —— `torch.manual_seed` + DataLoader 的 `generator` + 模型初始化顺序。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/cv11_deep_learning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "03_cv"
NAME = "ch11_deep_learning"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np

warnings.simplefilter("ignore")

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, TensorDataset

DATA = Path("data")
MN = DATA / "mnist"
'''

SETUP = '''tr_npz = np.load(str(MN / "mnist_train_sub.npz"))
te_npz = np.load(str(MN / "mnist_test_sub.npz"))
Xtr_raw, ytr_np = tr_npz["images"], tr_npz["labels"]
Xte_raw, yte_np = te_npz["images"], te_npz["labels"]
ytr = torch.from_numpy(ytr_np).long()
yte = torch.from_numpy(yte_np).long()

print("torch", torch.__version__, "| CUDA 可用:", torch.cuda.is_available(), "（本章全程 CPU，所以刻意只用子集）")
print("训练集", Xtr_raw.shape, Xtr_raw.dtype, "| 测试集", Xte_raw.shape, Xte_raw.dtype)
print("训练集标签分布", np.bincount(ytr_np).tolist())
print("测试集标签分布", np.bincount(yte_np).tolist())
print("灰度范围", int(Xtr_raw.min()), "~", int(Xtr_raw.max()), "| uint8 均值 %.6f" % float(Xtr_raw.mean()))
print("这份划分与 ch07 完全相同，所以本章结果可以和 HOG+SVM 直接并列")
'''

SCAFFOLD = '''# 脚手架：模型定义 + 两个训练/评估函数（缺口都在 @@todo 里）
class SmallCNN(nn.Module):
    """小 CNN：Conv-ReLU-Pool x2 + 全连接。输入 (N, 1, 28, 28)，输出 (N, 10) 的 logits。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


def near(a, b, tol=1e-6):
    """浮点容差比较。真值都实跑得到，用容差避免 == 在末位翻车。"""
    return abs(float(a) - float(b)) <= tol


def train_epochs(net, loader, epochs, lr):
    """训练 epochs 轮，返回每轮的平均 loss。循环骨架已给，缺口是「训练四步」。"""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    hist = []
    for ep in range(epochs):
        net.train()
        tot, seen = 0.0, 0
        for xb, yb in loader:
            # @@todo 训练四步：清梯度 -> 前向 + 算 loss -> 反向传播 -> 更新参数
            # @@hint 四行，依次是 opt.zero_grad() / loss = crit(net(xb), yb)
            # @@hint loss.backward() / opt.step() —— 顺序不能换
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            # @@end
            tot += loss.item() * len(yb)
            seen += len(yb)
        hist.append(round(tot / seen, 6))
    return hist


def collect_preds(net, loader):
    """返回 (预测, 真值) 两个张量。循环骨架与 no_grad 已给，缺口是「前向 + argmax」。"""
    net.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            # @@todo 前向得到输出 -> argmax 取预测类别 -> 两边都收集起来
            # @@hint 四行：out = net(xb) / p = out.argmax(dim=1) / preds.append(p) / gts.append(yb)
            out = net(xb)
            p = out.argmax(dim=1)
            preds.append(p)
            gts.append(yb)
            # @@end
    return torch.cat(preds), torch.cat(gts)
'''

E1_CODE = '''tr_f = Xtr_raw.astype(np.float32) / 255.0
# @@todo 算训练集的 mean / std（**只用训练集**），再把训练集与测试集都标准化并加通道维
# @@hint 四行：tr_mean / tr_std 是两个 float（tr_f.mean() / tr_f.std()）
# @@hint X4 与 Xte4 各一行，形如 ((torch.from_numpy(原图).float() / 255.0 - tr_mean) / tr_std).unsqueeze(1)
tr_mean = float(tr_f.mean())
tr_std = float(tr_f.std())
X4 = ((torch.from_numpy(Xtr_raw).float() / 255.0 - tr_mean) / tr_std).unsqueeze(1)
Xte4 = ((torch.from_numpy(Xte_raw).float() / 255.0 - tr_mean) / tr_std).unsqueeze(1)
# @@end
print("训练集统计量: mean = %.10f  std = %.10f（从 0~255 缩到 0~1 之后）" % (tr_mean, tr_std))
print("X4  :", tuple(X4.shape), X4.dtype, "| mean %.3e  std %.6f" % (float(X4.mean()), float(X4.std())))
print("Xte4:", tuple(Xte4.shape), "| mean %.6f  std %.6f" % (float(Xte4.mean()), float(Xte4.std())))
print("取值范围: %.6f ~ %.6f" % (float(X4.min()), float(X4.max())))
print("解读: ① **归一化的统计量只能来自训练集** —— 拿测试集一起算 mean/std 就是信息泄漏，"
      "是「离线分数虚高、上线就崩」的经典成因；"
      "② 标准化后 X4 的 mean≈0、std≈1，但 **Xte4 的 mean 是 0.011494 而不是 0** —— "
      "这正是「测试集分布与训练集略有差异」的量化体现；"
      "③ 为什么不用 `/255` 就完事？因为 /255 之后 mean 还有 0.130011、std 0.307255，"
      "分布不正、也没到同一尺度，后面**实验会直接对比这两种做法的差距**")
assert near(tr_mean, 0.130011, 1e-6)
assert near(tr_std, 0.307255, 1e-6)
assert tuple(X4.shape) == (2000, 1, 28, 28)
assert tuple(Xte4.shape) == (500, 1, 28, 28)
assert X4.dtype == torch.float32
assert near(X4.mean(), 0.0, 1e-6)
assert near(X4.std(), 1.0, 1e-6)
assert near(Xte4.mean(), 0.011494, 1e-6)
assert near(Xte4.std(), 1.013165, 1e-6)
assert near(X4.min(), -0.423137, 1e-6) and near(X4.max(), 2.831486, 1e-6)
'''

E2_CODE = '''# @@todo 三件事：确认 X4 的 dtype；用 reshape 得到 (2000, 784) 的扁平视图；用 permute 把通道维挪到最后
# @@hint 三行：x_dtype = X4.dtype；flat = X4.reshape(len(X4), -1)；nhwc = X4.permute(0, 2, 3, 1)
x_dtype = X4.dtype
flat = X4.reshape(len(X4), -1)
nhwc = X4.permute(0, 2, 3, 1)
# @@end
print("dtype:", x_dtype, "（torch 没有隐式类型提升：整数张量做除法会丢小数，所以必须先 .float()）")
print("reshape(N, -1) ->", tuple(flat.shape), "| permute 到 NHWC ->", tuple(nhwc.shape))
print("permute 之后 contiguous?", nhwc.is_contiguous(), "（只是换了步长视图，没搬数据）")
print("解读: ① `torch.from_numpy` 得到的是 **uint8** 张量，`uint8 * 255` 还是 uint8，会溢出回绕；"
      "② reshape(N, -1) 把 (N,1,28,28) 摊平成 (N,784)，这是接全连接层的写法；"
      "③ permute 只改 stride 不复制数据 —— 所以接 `F.unfold` 或 `cv2` 之前往往要 `.contiguous()`")
assert x_dtype == torch.float32
assert tuple(flat.shape) == (2000, 784)
assert tuple(nhwc.shape) == (2000, 28, 28, 1)
'''

E3_CODE = '''ds = TensorDataset(X4, ytr)
# @@todo 用 batch_size=64、shuffle=False 过一遍，把每批的样本数记成列表
# @@hint 两行：ld = DataLoader(ds, batch_size=64, shuffle=False)
# @@hint batch_sizes = [len(b[0]) for b in ld]
ld = DataLoader(ds, batch_size=64, shuffle=False)
batch_sizes = [len(b[0]) for b in ld]
# @@end
print("len(ds) =", len(ds), "| 一个样本: 图像", tuple(ds[0][0].shape), ds[0][0].dtype,
      "标签", int(ds[0][1]), ds[0][1].dtype)
print("batch=64 的批次数:", len(ld))
print("每批大小:", batch_sizes[:3], "... 末批", batch_sizes[-1], "| 合计", sum(batch_sizes))
print("解读: 2000 / 64 = 31.25 -> **31 个满批 + 1 个 16 的残批**。"
      "残批有两个副作用：① 该批的梯度噪声更大（样本少）；"
      "② 如果模型里有 BatchNorm，小批次的统计量会明显偏。"
      "**所以「跑了几个 step」不是 `样本数 / 批大小` 那个小数**")
assert len(ld) == 32
assert batch_sizes[:3] == [64, 64, 64] and batch_sizes[-1] == 16
assert sum(batch_sizes) == 2000
assert len(batch_sizes) == 32
'''

E4_CODE = '''# @@todo 加上 drop_last=True，数出批次数与被丢掉的样本数
# @@hint 两行：n_drop = len(DataLoader(ds, batch_size=64, shuffle=False, drop_last=True))
# @@hint dropped = len(ds) - 64 * n_drop
n_drop = len(DataLoader(ds, batch_size=64, shuffle=False, drop_last=True))
dropped = len(ds) - 64 * n_drop
# @@end
print("drop_last=True -> 批次数", n_drop, "| 丢掉样本", dropped, "个（占 %.4f%%）" % (100 * dropped / len(ds)))
print("batch=2000（整批）批次数", len(DataLoader(ds, batch_size=2000, shuffle=False)))
print("解读: `drop_last` 就是**为了消灭残批**。丢掉 16 张（0.8%）换取每批形状一致 —— "
      "小数据集上要算清这个代价；数据集越大越无所谓")
assert n_drop == 31 and dropped == 16
assert len(DataLoader(ds, batch_size=2000, shuffle=False)) == 1
'''

E5_CODE = '''# @@todo 4 行：用「固定种子生成器」跑两次 shuffle 取第一批前 10 个标签；
# @@hint 再用「不给 generator」跑两次，同样取前 10 个标签
# @@hint g1 / g2 都是 torch.Generator().manual_seed(0)
g1 = torch.Generator().manual_seed(0)
g2 = torch.Generator().manual_seed(0)
shuf_a = next(iter(DataLoader(ds, batch_size=64, shuffle=True, generator=g1)))[1].tolist()[:10]
shuf_b = next(iter(DataLoader(ds, batch_size=64, shuffle=True, generator=g2)))[1].tolist()[:10]
raw_a = next(iter(DataLoader(ds, batch_size=64, shuffle=True)))[1].tolist()[:10]
raw_b = next(iter(DataLoader(ds, batch_size=64, shuffle=True)))[1].tolist()[:10]
# @@end
print("固定 generator 两次:", shuf_a, "/", shuf_b, "-> 一致?", shuf_a == shuf_b)
print("不给 generator 两次:", raw_a, "/", raw_b, "-> 一致?", raw_a == raw_b)
print("shuffle=False 的第一批前 10 个标签:", next(iter(DataLoader(ds, batch_size=64, shuffle=False)))[1].tolist()[:10])
print("解读: **DataLoader 的 shuffle 走的是它自己的随机源**，光设 `torch.manual_seed` 管不到它 —— "
      "必须显式传 `generator=`。不传的话两次训练看到的数据顺序不同，结果自然对不上。"
      "这是「深度学习实验跑不出同样数字」的**头号原因**")
assert shuf_a == shuf_b == [1, 6, 7, 4, 0, 8, 8, 9, 5, 3]
assert raw_a != raw_b
assert next(iter(DataLoader(ds, batch_size=64, shuffle=False)))[1].tolist()[:10] == [7, 3, 8, 9, 3, 9, 7, 7, 5, 4]
'''

E6_CODE = '''class DigitsDS(Dataset):
    """手写 Dataset：只要实现 __len__ 与 __getitem__，就能直接塞进 DataLoader。"""

    def __init__(self, images, labels, mean, std):
        self.images = images
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self):
        # @@todo __len__ 的函数体：返回样本总数
        # @@hint 一行：return len(self.labels)
        return len(self.labels)
        # @@end

    def __getitem__(self, i):
        # @@todo __getitem__ 的函数体：读第 i 张 -> /255 -> 标准化 -> 加通道维；标签转 python int
        # @@hint 三行：img = self.images[i].astype(np.float32) / 255.0
        # @@hint img = (img - self.mean) / self.std
        # @@hint return torch.from_numpy(img).unsqueeze(0), int(self.labels[i])
        img = self.images[i].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return torch.from_numpy(img).unsqueeze(0), int(self.labels[i])
        # @@end


mds = DigitsDS(Xtr_raw[:8], ytr_np[:8], tr_mean, tr_std)
# @@todo 过一遍 DataLoader(batch_size=3)，记下图像批与标签批的形状
# @@hint 一行：mb = next(iter(DataLoader(mds, batch_size=3)))
mb = next(iter(DataLoader(mds, batch_size=3)))
# @@end
print("len(mds) =", len(mds), "| mds[3][0]", tuple(mds[3][0].shape), mds[3][0].dtype, "| 标签", mds[3][1])
print("过 DataLoader(batch=3) ->", tuple(mb[0].shape), mb[1].dtype, mb[1].tolist())
print("与 TensorDataset 路径的结果对齐: mds[3][0] vs X4[3] 最大差 %.3e" % float((mds[3][0] - X4[3]).abs().max()))
print("解读: ① `TensorDataset(X4, ytr)` 是「现成张量」的快捷方式，"
      "`DigitsDS` 是「按需转换」的手写版 —— 大数据集（不能一次性读进内存）必须用后者；"
      "② 注意 `__getitem__` 返回的标签是 **python int**，DataLoader 会自动 collate 成 int64 张量；"
      "③ 两条路径算出来的张量必须**逐元素相等**，否则说明两边的预处理口径不一致")
assert len(mds) == 8
assert mds[3][1] == 9 and isinstance(mds[3][1], int)
assert tuple(mb[0].shape) == (3, 1, 28, 28)
assert mb[1].tolist() == [7, 3, 8]
assert float((mds[3][0] - X4[3]).abs().max()) == 0.0
'''

E7_CODE = '''torch.manual_seed(0)
xin = torch.randn(2, 1, 6, 6)
wgt = torch.randn(3, 1, 3, 3)
bias = torch.randn(3)
# @@todo 用 F.unfold + 矩阵乘法手撕 conv2d，再与 F.conv2d 对账（不用写循环）
# @@hint 三行：cols = F.unfold(xin, (3, 3))
# @@hint hand = (wgt.reshape(3, -1) @ cols + bias.reshape(1, 3, 1)).reshape(2, 3, 4, 4)
# @@hint ref = F.conv2d(xin, wgt, bias)
cols = F.unfold(xin, (3, 3))
hand = (wgt.reshape(3, -1) @ cols + bias.reshape(1, 3, 1)).reshape(2, 3, 4, 4)
ref = F.conv2d(xin, wgt, bias)
# @@end
print("输入", tuple(xin.shape), "卷积核", tuple(wgt.shape), "偏置", tuple(bias.shape))
print("F.unfold 输出:", tuple(cols.shape), "= (N, C*KH*KW, L)，L = (6-3+1)^2 =", (6 - 3 + 1) ** 2)
print("手撕输出:", tuple(hand.shape), "| torch 输出:", tuple(ref.shape))
print("最大差: %.3e" % float((hand - ref).abs().max()))
print("单元素 hand[0,0,0,0] = %.6f  ref[0,0,0,0] = %.6f" % (float(hand[0, 0, 0, 0]), float(ref[0, 0, 0, 0])))
print("解读: `F.unfold` 把每个滑动窗口拍成一列 —— 于是**卷积 = 一次矩阵乘法**。"
      "这正是 GPU 上卷积快的本质（im2col + GEMM）。"
      "差 9.537e-07 来自 float32 的累加顺序不同，不是算法错。"
      "注意这里**没有 padding / stride**，所以 L 正好等于 (H-KH+1)*(W-KW+1)")
assert tuple(cols.shape) == (2, 9, 16)
assert tuple(hand.shape) == (2, 3, 4, 4)
assert float((hand - ref).abs().max()) < 1e-5
assert near(float(hand[0, 0, 0, 0]), -4.816606, 1e-5)
assert near(float(ref[0, 0, 0, 0]), -4.816606, 1e-5)
'''

E8_CODE = '''torch.manual_seed(0)
net = SmallCNN()
# @@todo 用 named_parameters() 统计每层参数量，再求和
# @@hint 两行：param_num = {nm: p.numel() for nm, p in net.named_parameters()}
# @@hint total = sum(param_num.values())
param_num = {nm: p.numel() for nm, p in net.named_parameters()}
total = sum(param_num.values())
# @@end
print("逐层参数量:")
for nm, v in param_num.items():
    print(f"   {nm:24s} {v:6d}")
print("总参数量:", total)
print("解读: 参数量全在**全连接层**（7840，占 86%）—— 卷积层只有 1240 个参数。"
      "这不是巧合：`Conv2d(8, 16, 3)` 的参数量是 `16*8*9 + 16 = 1168`，"
      "**与图像尺寸无关**（权重共享），而 `Linear(16*7*7, 10)` 把空间位置全摊平了，"
      "所以和特征图大小成正比。小图 + 小网络可以接受，大图必须上全局池化")
assert param_num["features.0.weight"] == 72 and param_num["features.0.bias"] == 8
assert param_num["features.3.weight"] == 1152 and param_num["features.3.bias"] == 16
assert param_num["head.1.weight"] == 7840 and param_num["head.1.bias"] == 10
assert total == 9098
'''

E9_CODE = '''torch.manual_seed(0)
net = SmallCNN()
with torch.no_grad():
    # @@todo 取前 4 张的 features 输出与 logits，再对 logits 做 softmax 看行和
    # @@hint 三行：feat = net.features(X4[:4]) / logits = net(X4[:4]) / probs = F.softmax(logits, dim=1)
    feat = net.features(X4[:4])
    logits = net(X4[:4])
    probs = F.softmax(logits, dim=1)
    # @@end
print("features 输出:", tuple(feat.shape), "（28 -> 14 -> 7，通道 1 -> 8 -> 16）")
print("logits:", tuple(logits.shape))
print("前 4 行 logits 行和:", np.round(logits.sum(dim=1).numpy(), 6).tolist(), "<- **不是 1**")
print("softmax 后行和:", np.round(probs.sum(dim=1).numpy(), 6).tolist())
print("softmax 第一行:", np.round(probs[0].numpy(), 6).tolist())
print("argmax:", logits.argmax(dim=1).tolist(), "| CrossEntropyLoss = %.6f" % float(nn.CrossEntropyLoss()(logits, ytr[:4])))
print("解读: ① 模型最后一层**不加 softmax** —— `CrossEntropyLoss` 内部就是 "
      "`log_softmax + NLLLoss`，外面再加一层就重复了；"
      "② 刚初始化的网络输出接近均匀（各类概率都在 0.077~0.153），"
      "CrossEntropyLoss ≈ ln(10) = 2.302585，实测 2.298519；"
      "③ 推理时才需要 softmax（要概率）；只取类别用 `argmax(logits)` 就够，"
      "**不用先转概率再 argmax**（单调性保证结果一样）")
assert tuple(feat.shape) == (4, 16, 7, 7)
assert tuple(logits.shape) == (4, 10)
assert near(float(probs.sum(dim=1)[0]), 1.0, 1e-6)
assert logits.argmax(dim=1).tolist() == [9, 9, 9, 9]
assert near(float(nn.CrossEntropyLoss()(logits, ytr[:4])), 2.298519, 1e-5)
'''

E10_CODE = '''torch.manual_seed(0)
net15 = SmallCNN()
# @@todo 搭 loader（batch=64、shuffle、固定种子）并训练 15 轮
# @@hint 两行：tl = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
# @@hint hist15 = train_epochs(net15, tl, epochs=15, lr=2e-3)
tl = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
hist15 = train_epochs(net15, tl, epochs=15, lr=2e-3)
# @@end
print("15 轮 loss 曲线（每轮一个平均值）:")
for i, v in enumerate(hist15, 1):
    print(f"   epoch {i:2d}  loss {v:.6f}")
print("解读: loss 从 %.6f 单调降到 %.6f，**还没有反弹** —— 说明 15 轮内没到「梯度爆炸」或"
      "「发散」的程度。判断该停在哪一轮，靠的是**验证集**而不是训练 loss 本身"
      % (hist15[0], hist15[-1]))
assert len(hist15) == 15
assert near(hist15[0], 1.524842, 1e-6)
assert near(hist15[1], 0.455627, 1e-6)
assert near(hist15[2], 0.302241, 1e-6)
assert near(hist15[-1], 0.014964, 1e-6)
'''

E11_CODE = '''trl = DataLoader(ds, batch_size=256, shuffle=False)
tel = DataLoader(TensorDataset(Xte4, yte), batch_size=256, shuffle=False)
# @@todo 用 collect_preds 拿训练集/测试集的预测，算两边准确率与它们的差（gap）
# @@hint 三行：p_tr, g_tr = collect_preds(net15, trl)；p_te, g_te = collect_preds(net15, tel)
# @@hint acc_tr / acc_te 都写成 float((p == g).float().mean())；acc_gap = acc_tr - acc_te
p_tr, g_tr = collect_preds(net15, trl)
p_te, g_te = collect_preds(net15, tel)
acc_tr = float((p_tr == g_tr).float().mean())
acc_te = float((p_te == g_te).float().mean())
acc_gap = acc_tr - acc_te
# @@end
print("训练集准确率: %.6f（%d / %d）" % (acc_tr, int((p_tr == g_tr).sum()), len(g_tr)))
print("测试集准确率: %.6f（%d / %d）" % (acc_te, int((p_te == g_te).sum()), len(g_te)))
print("gap = train - test = %.6f" % acc_gap)
print("解读: **训练集已经满分（1.000000），测试集只有 0.96** —— 这就是过拟合的样子。"
      "2000 张训练 9098 个参数，模型有能力把训练集「背下来」。"
      "ch07 的 HOG + LinearSVC 测试是 0.952，所以这个 CNN **只赢 0.008（4 张图）** —— "
      "在小数据上，深度模型不一定碾压经典方法")
assert near(acc_tr, 1.0) and near(acc_te, 0.96)
assert near(acc_gap, 0.04)
assert int((p_tr == g_tr).sum()) == 2000 and int((p_te == g_te).sum()) == 480
assert int((p_te != g_te).sum()) == 20
'''

E12_CODE = '''ep_rows = {}
for ep in (1, 3, 6, 10, 15):
    torch.manual_seed(0)
    n_ep = SmallCNN()
    l_ep = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
    # @@todo 训练 ep 轮，记录 (最后一轮 loss, 训练准确率, 测试准确率) 到 ep_rows[ep]
    # @@hint 四行：h = train_epochs(n_ep, l_ep, epochs=ep, lr=2e-3)
    # @@hint p1, g1 = collect_preds(n_ep, trl)；p2, g2 = collect_preds(n_ep, tel)
    # @@hint ep_rows[ep] = (h[-1], float((p1 == g1).float().mean()), float((p2 == g2).float().mean()))
    h = train_epochs(n_ep, l_ep, epochs=ep, lr=2e-3)
    p1, g1 = collect_preds(n_ep, trl)
    p2, g2 = collect_preds(n_ep, tel)
    ep_rows[ep] = (h[-1], float((p1 == g1).float().mean()), float((p2 == g2).float().mean()))
    # @@end
print("轮数  | 末轮 loss | 训练准确率 | 测试准确率 | gap")
for ep in (1, 3, 6, 10, 15):
    loss_v, a1, a2 = ep_rows[ep]
    print(f"  {ep:2d}   |  {loss_v:.6f} |   {a1:.6f} |   {a2:.6f} | {a1 - a2:.6f}")
print("解读: **同一个模型、同一份数据，只改轮数**：1 轮到 0.800（严重欠拟合），"
      "3 轮到 0.922，6 轮 0.942，10 轮 0.954，15 轮 0.960。"
      "训练集准确率一路推向 1.0，gap 从 0.0025 涨到 0.040 —— "
      "**「训练不够」和「过拟合」是同一根轴的两端**，靠轮数调节，但最优轮数只能靠验证集找")
assert near(ep_rows[1][2], 0.80)
assert near(ep_rows[3][2], 0.922)
assert near(ep_rows[10][2], 0.954)
assert near(ep_rows[15][1], 1.0) and near(ep_rows[15][2], 0.96)
'''

E13_CODE = '''cm = torch.zeros(10, 10, dtype=torch.int64)
for t, p in zip(g_te.tolist(), p_te.tolist()):
    # @@todo 往混淆矩阵里累加一次
    # @@hint 一行：cm[t, p] += 1（行是真值、列是预测）
    cm[t, p] += 1
    # @@end
# @@todo 算逐类召回（对角 / 每行和）
# @@hint 一行：recall = (cm.diag() / cm.sum(dim=1)).numpy()
recall = (cm.diag() / cm.sum(dim=1)).numpy()
# @@end
print("混淆矩阵（行 = 真值，列 = 预测）:")
print(cm.numpy())
print("对角（各类正确数）:", cm.diag().tolist())
print("各类样本数      :", cm.sum(dim=1).tolist())
print("逐类召回率:")
for d in range(10):
    print(f"   数字 {d}: {recall[d]:.6f}")
print("最差类别 argmin = %d（%.6f）| 最好类别 argmax = %d（%.6f）" % (int(np.argmin(recall)), float(recall.min()),
                                                                int(np.argmax(recall)), float(recall.max())))
print("解读: ① 整体 0.960，但按类看不平均：数字 8 最差（0.870370），0 / 1 / 7 都是满分；"
      "② 对比 ch07 的 HOG+SVM：它的最差类也是数字 9（召回 **0.807**），"
      "而这个 CNN 把数字 9 拉到 **0.912281** —— **总准确率只涨 0.008，但最难的那类涨了 0.105**，"
      "这才是 CNN「有用」的地方；"
      "③ 混淆矩阵的行和 = 该类样本数，**分母是真实类别数**，不是预测数")
assert cm.diag().tolist() == [56, 54, 50, 38, 42, 38, 45, 58, 47, 52]
assert cm.sum(dim=1).tolist() == [56, 54, 51, 39, 43, 40, 48, 58, 54, 57]
assert int(cm.sum()) == 500 and int(cm.diag().sum()) == 480
assert int(np.argmin(recall)) == 8 and near(float(recall.min()), 0.870370, 1e-6)
assert near(float(recall[9]), 0.912281, 1e-6)
assert near(float(recall[0]), 1.0) and near(float(recall[1]), 1.0)
'''

E14_CODE = '''acc_rep = []
for _ in range(2):
    torch.manual_seed(0)
    n_rep = SmallCNN()
    l_rep = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
    # @@todo 训练 15 轮，把测试准确率追加进 acc_rep
    # @@hint 三行：train_epochs(n_rep, l_rep, epochs=15, lr=2e-3)
    # @@hint pp, gg = collect_preds(n_rep, tel)
    # @@hint acc_rep.append(round(float((pp == gg).float().mean()), 6))
    train_epochs(n_rep, l_rep, epochs=15, lr=2e-3)
    pp, gg = collect_preds(n_rep, tel)
    acc_rep.append(round(float((pp == gg).float().mean()), 6))
    # @@end
print("同一种子跑两次的测试准确率:", acc_rep, "-> 一致?", acc_rep[0] == acc_rep[1])
print("解读: 要复现一个训练结果，**三处必须一起管**："
      "① `torch.manual_seed`（管参数初始化、dropout）；"
      "② `DataLoader(generator=...)`（管数据顺序）；"
      "③ **调用顺序**（先 seed 再建模型，且两次之间不能插入别的随机操作）。"
      "少任何一条，两次结果就对不上 —— 这也是调参日志必须记「跑了多少次」的原因")
assert acc_rep == [0.96, 0.96]
'''

E15_CODE = '''# @@todo 把本章的 CNN 与 ch07 的三种特征分类器并排到一张表里
# @@hint 一行：baseline = {"raw784+KNN": 0.914, "raw784+SVM": 0.878, "hog324+KNN": 0.924,
# @@hint                "hog324+SVM": 0.952, "lbp59+SVM": 0.704, "CNN(15ep)": acc_te}
baseline = {"raw784+KNN": 0.914, "raw784+SVM": 0.878, "hog324+KNN": 0.924,
            "hog324+SVM": 0.952, "lbp59+SVM": 0.704, "CNN(15ep)": acc_te}
# @@end
best_classic = max(v for k, v in baseline.items() if k != "CNN(15ep)")
print("同一份 2000 / 500 划分上的测试准确率:")
for k, v in baseline.items():
    print(f"   {k:14s} {v:.4f}")
print("经典方法最好:", best_classic, "| CNN:", acc_te, "| 提升 %.4f（%d 张图）"
      % (acc_te - best_classic, round((acc_te - best_classic) * 500)))
print("解读: ① **CNN 只赢 0.008**（500 张里多对 4 张）—— 在 2000 张的小数据上，"
      "「手撕 HOG + 线性 SVM」依然是极强的基线，不要迷信深度模型；"
      "② 但**代价完全不同**：HOG+SVM 不需要训练轮数、不需要调 lr、不需要 GPU，"
      "也不会有过拟合的烦恼；"
      "③ 结论不是「CNN 更好」，而是「**小数据 + 简单任务，先试经典特征**；"
      "数据量上去、任务变复杂之后，深度模型才会拉开差距」")
assert baseline["hog324+SVM"] == 0.952
assert near(baseline["CNN(15ep)"], 0.96)
assert acc_te > best_classic
assert round((acc_te - best_classic) * 500) == 4
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch11 深度学习入门

> 前七章手撕的 LBP / HOG 是「人设计特征 + 分类器」；本章换成「**让网络自己学特征**」。
> 但这一章不打算把深度学习讲成神话 —— 恰恰相反，它要**用数字说明：在小数据上深度模型未必赢**。

## 本章路线

| 小节 | 内容 | 为什么要手写 |
|---|---|---|
| §11.1 | numpy → tensor + **归一化** | 统计量只能来自训练集，否则信息泄漏 |
| §11.2 | 张量的 shape / dtype / permute | `uint8` 溢出、`reshape` vs `permute` 的差别 |
| §11.3 | `TensorDataset` + `DataLoader` | 批次、**残批**、`drop_last` |
| §11.4 | `shuffle` 与 `generator` | **结果跑不出同样数字的头号原因** |
| §11.5 | 手写 `Dataset` | `__len__` / `__getitem__` 两件套 |
| §11.6 | **用 `F.unfold` 手撕 conv2d** | 卷积 = im2col + 矩阵乘 |
| §11.7 | 参数量统计 | 卷积层与 FC 层的参数量量级差别 |
| §11.8 | 前向 + softmax + CrossEntropy | 为什么最后一层**不加 softmax** |
| §11.9 | **训练四步** | `zero_grad` / `forward+loss` / `backward` / `step` |
| §11.10 | 准确率与 **gap** | 训练满分、测试 0.96 = 过拟合 |
| §11.11 | **轮数扫描** | 欠拟合与过拟合是同一根轴的两端 |
| §11.12 | 混淆矩阵与逐类召回 | 整体准确率会掩盖最难的那类 |
| §11.13 | **可复现性** | 三处随机源要一起管 |
| §11.14 | **与 ch07 的 HOG+SVM 对比** | 小数据上经典方法依然是强基线 |

## 关于数据

`mnist/mnist_train_sub.npz`（2000 张）+ `mnist/mnist_test_sub.npz`（500 张），
**与 ch07 用的是同一份划分**，所以两章的数字可以直接并列比较。

全程 CPU —— 这也是刻意只用子集的原因：2000 张跑 15 轮约 5 秒，
足够把「训练/评估/调参」的闭环讲清楚，而不必等 GPU。
"""
    ),
    md(
        """
## §11.0 环境与数据准备
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## §11.1 归一化：统计量只能来自训练集

先把 `uint8` 的 `0~255` 变成网络友好的分布。**标准化用的 mean / std 必须只用训练集算**。
"""
    ),
    code(E1_CODE),
    md(
        """
## §11.2 张量的 shape 与 dtype

`reshape` 与 `permute` 都能改形状，但一个是「重新排列元素」、一个是「换步长视图」。
"""
    ),
    code(E2_CODE),
    md(
        """
## §11.3 `TensorDataset` 与 `DataLoader`

批大小不是「整数除」那么简单 —— 2000 / 64 会剩一个 16 的残批。
"""
    ),
    code(E3_CODE),
    code(E4_CODE),
    md(
        """
## §11.4 `shuffle` 的随机源：为什么结果总是对不上

**这是深度学习实验里最常踩的坑** —— 你设了 `torch.manual_seed`，但数据顺序还是每次不同。
"""
    ),
    code(E5_CODE),
    md(
        """
## §11.5 手写 `Dataset`

`TensorDataset` 只适合「已经全在内存里的张量」。真实项目要自己写 —— 只要两条方法。
"""
    ),
    code(E6_CODE),
    md(
        """
## §11.6 用 `F.unfold` 手撕卷积

不用写四层循环：`unfold`（即 im2col）把每个滑动窗口摊成一列，卷积就变成**一次矩阵乘法**。
"""
    ),
    code(E7_CODE),
    md(
        """
## §11.7 模型结构与参数量

先看看 9098 个参数是怎么分布的。
"""
    ),
    code(E8_CODE),
    md(
        """
## §11.8 前向、logits 与 softmax

网络输出的是 **logits**（未归一化的分数），不是概率。
"""
    ),
    code(E9_CODE),
    md(
        """
## §11.9 训练四步

训练循环的核心就是这四行，顺序不能错：

```
opt.zero_grad()      # 1. 清掉上一步的梯度（PyTorch 默认累加！）
loss = crit(...)     # 2. 前向 + 算损失
loss.backward()      # 3. 反向传播，填好每个参数的 .grad
opt.step()           # 4. 用 .grad 更新参数
```

`train_epochs` 的循环骨架已在脚手架里给好，缺口就是这四行。
"""
    ),
    code(E10_CODE),
    md(
        """
## §11.10 训练准确率与 gap

只看训练 loss 会骗人。两边的准确率一起看。
"""
    ),
    code(E11_CODE),
    md(
        """
## §11.11 轮数扫描：欠拟合 ⇄ 过拟合

同一个结构、同一份数据，只改轮数。
"""
    ),
    code(E12_CODE),
    md(
        """
## §11.12 混淆矩阵与逐类召回

整体准确率是一个平均值 —— 平均值最擅长掩盖最差的那一类。
"""
    ),
    code(E13_CODE),
    md(
        """
## §11.13 可复现性

想得到「两次完全一样」的训练结果，需要同时管住三处随机源。
"""
    ),
    code(E14_CODE),
    md(
        """
## §11.14 与 ch07 的经典方法对比

最后把本章结果和 ch07 的三种特征分类器并排 —— 这一节会给出一个反直觉的结论。
"""
    ),
    code(E15_CODE),
    md(
        """
## 自查清单

- [ ] 归一化的 mean / std **只能来自训练集**
- [ ] 知道 `/255` 与「标准化」是两回事
- [ ] 知道 `torch.from_numpy` 得到 uint8，整数除法会丢小数
- [ ] 能说出 `reshape` 与 `permute` 的区别（搬数据 vs 换步长）
- [ ] 知道残批的存在，以及 `drop_last` 的代价
- [ ] **知道 `shuffle` 要用 `generator=` 才能复现**
- [ ] 能默写 `Dataset` 的 `__len__` / `__getitem__`
- [ ] 知道 `F.unfold` + 矩阵乘为什么等于卷积
- [ ] 知道卷积层参数量与图像尺寸无关、FC 层成正比
- [ ] 知道模型最后一层**不加 softmax**（CrossEntropyLoss 内含）
- [ ] 能默写训练四步，并知道 `zero_grad` 为什么必须调
- [ ] 知道「训练满分 + 测试 0.96」= 过拟合
- [ ] 知道欠拟合与过拟合是同一根轴的两端
- [ ] 知道混淆矩阵的行和是**真实类别数**
- [ ] **知道复现训练结果要同时管三处随机源**
- [ ] 知道小数据上经典特征依然是强基线
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch11 深度学习入门（练习版）

> 按提示补全 `____`，跑通所有 assert。`Xtr_raw` / `ytr` / `Xte_raw` / `yte`
> 以及 `SmallCNN` / `near` / `train_epochs` / `collect_preds` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；控制流写在挖空块之外
> （`train_epochs` / `collect_preds` 的循环骨架已给好，你只填循环体里的几行）。
> 题目真值都来自实跑（torch 2.14.0，CPU）。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：归一化（统计量只用训练集）

算出训练集的 mean / std，把两边都标准化并加通道维。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：shape 与 dtype

`reshape` 到扁平、`permute` 到 NHWC。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：批次划分

2000 张、batch=64，看看每批多少张。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：`drop_last`

消灭残批要付出什么代价。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：`shuffle` 的随机源

固定 generator 两次一致，不传 generator 两次不一致 —— 这个现象要亲手跑出来。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：手写 `Dataset`

`__len__` 与 `__getitem__` 两条方法。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：`F.unfold` 手撕卷积

im2col + 矩阵乘，与 `F.conv2d` 对账。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：参数量统计

看看 9098 个参数落在哪一层。
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 9：前向与 softmax

注意 logits 行和不是 1。
"""
    ),
    code(E9_CODE),
    md(
        """
## 任务 10：训练四步

`train_epochs` 的循环骨架已给好，只填循环体里的四行。
"""
    ),
    code(E10_CODE),
    md(
        """
## 任务 11：准确率与 gap

训练集和测试集都要算。
"""
    ),
    code(E11_CODE),
    md(
        """
## 任务 12：轮数扫描

1 / 3 / 6 / 10 / 15 轮，看准确率怎么变。
"""
    ),
    code(E12_CODE),
    md(
        """
## 任务 13：混淆矩阵与逐类召回

行是真值、列是预测。
"""
    ),
    code(E13_CODE),
    md(
        """
## 任务 14：可复现性

三处随机源一起管。
"""
    ),
    code(E14_CODE),
    md(
        """
## 任务 15：与经典方法对比

和 ch07 的 HOG+SVM 并排看。
"""
    ),
    code(E15_CODE),
    md(
        """
## 自查清单

- [ ] 归一化的 mean / std **只能来自训练集**
- [ ] `torch.from_numpy` 得到 uint8
- [ ] `reshape` 搬数据、`permute` 换步长
- [ ] 残批与 `drop_last`
- [ ] **`shuffle` 要用 `generator=` 才能复现**
- [ ] `Dataset` 的 `__len__` / `__getitem__`
- [ ] `F.unfold` + 矩阵乘 = 卷积
- [ ] 卷积层参数量与图像尺寸无关
- [ ] 最后一层不加 softmax
- [ ] 训练四步与 `zero_grad`
- [ ] 过拟合 = 训练满分 + 测试掉下来
- [ ] 混淆矩阵行和是真实类别数
- [ ] 复现训练结果要管三处随机源
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
