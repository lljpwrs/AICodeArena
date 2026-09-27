#!/usr/bin/env python3
"""ch07 —— 训练与评估：MSE/MAE/RMSE/MAPE / 零点陷阱 / 早停·裁剪·调度 / 多步误差累积 / 残差诊断

本章主线（六条）：

1. **四个指标的公式与陷阱**：MSE / MAE / RMSE / MAPE 手写 + 与 sklearn 逐位对账；
   `RMSE ≥ MAE` 恒成立，`RMSE/MAE` 的比值反映**误差分布的形状**
2. **MAPE 的零点陷阱**：分母是真值，真值接近 0 时百分比爆炸（实测 6.69% → 2481.95%）；
   修法 `sMAPE` / `WAPE`
3. **训练循环模板**：train / valid 双循环 + **早停（patience）** + **梯度裁剪** + **StepLR 调度**，
   用一个**真能触发早停**的设置把 `best_epoch` / `stopped_epoch` 打印出来
4. **多步误差累积**：递归（自己喂自己）vs 直接多输出，MAE 随步长 1..24 的曲线，
   并解释「递归误差方差为什么随时间放大」
5. **预测曲线可视化**：整段抽样（h=1 连续曲线）+ 单日放大（同一起点的 24 步）
6. **残差诊断**：残差 ACF（手写 + 与 pandas `autocorr` 对账）/ 按小时分组的箱线统计 /
   系统性偏差的 t 检验，并区分「统计显著」与「实际重要」

数据：`data/load_curve.csv` → A 台区（居民型）8760 h，`clean_series` 后取 `.values`
口径：`window = 72`、`horizon = 24`、起点从 `START = 168` 开始（与 ch08 对齐）、按时间 8:2 切分
生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ts07_train_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "05_timeseries"
NAME = "ch07_train_eval"

# =========================================================================== #
# 1. 公共导入 / 数据加载 / 脚手架（两版都给，不挖空）                          #
# =========================================================================== #

IMPORTS = '''from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("data")          # notebook 的 cwd = coding/05_timeseries
CSV = DATA / "load_curve.csv"

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

raw = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["时间戳"])
print("原始行数", len(raw), "| 列", list(raw.columns))
print("台区", sorted(raw["台区编号"].unique()))
print("torch", torch.__version__, "| CUDA 可用？", torch.cuda.is_available(), "（数据小，CPU 足够）")
'''

SETUP = '''# 去重 + 排序（顺序不能反，理由见 ch01 §1.1）
df = raw.drop_duplicates().sort_values("时间戳").reset_index(drop=True)
frame = df[df["台区编号"] == "STATION_A_01"].set_index("时间戳").sort_index()


def clean_series(s):
    """按**小时分组**的中位数做 3 倍阈值判异常，再线性插值（ch01 §1.3.2 的结论）。"""
    med = s.groupby(s.index.hour).transform("median")
    return s.where(s.between(0, 3 * med)).interpolate()


s = clean_series(frame["负荷值"].asfreq("h"))
ser = s.values                      # 本章统一用 ndarray，方便做窗口切片

print("A 台区 %d 个整点 | 原始缺失 %d" % (len(s), int(frame["负荷值"].asfreq("h").isna().sum())))
print("清洗后 min %.2f | max %.2f | mean %.6f" % (ser.min(), ser.max(), ser.mean()))
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

SCAFFOLD = '''# 脚手架：下面的常量与工具函数两版都有，你只填 @@todo 块里的内容
WINDOW, HORIZON = 72, 24
# 起点取 168 而不是 WINDOW：与 ch08 对齐（ch08 要用 lag168，历史至少要 168 小时）
START = 168


class MLP(nn.Module):
    """两层 ReLU 的全连接网络。

    输入 = 展平的 72 小时窗口；输出维度由 out 决定：
    递归单步模型用 out=1，直接多步模型用 out=24（见 §7.3）。
    """

    def __init__(self, inp, hid, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out),
        )

    def forward(self, x):
        return self.net(x)


def mae_by_step(y_true, y_pred):
    """返回长度 HORIZON 的逐步 MAE：第 h 个元素 = 「预测 h+1 步」的平均绝对误差。"""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)), axis=0)


def mae_all(y_true, y_pred):
    """24 步合起来的单个 MAE（后面所有对比表的排序依据）。"""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
'''

# =========================================================================== #
# 2. 任务代码块                                                              #
# =========================================================================== #

T1_CODE = '''# @@todo 构造全部「预测起点」——从 START 起，一直到「后面还留得下 24 步」为止
# @@hint np.arange(START, len(ser) - HORIZON + 1)
origins = np.arange(START, len(ser) - HORIZON + 1)
# @@end

# @@todo 把每个起点切成 (72,) 的输入窗口 + (24,) 的目标序列
# @@hint X 取 ser[t - WINDOW:t]**不含 t** —— t 是第一个待预测点，不能进特征
# @@hint Y 取 ser[t:t + HORIZON] —— 第 0 列是 h=1，第 23 列是 h=24
X = np.stack([ser[t - WINDOW:t] for t in origins]).astype("float32")
Y = np.stack([ser[t:t + HORIZON] for t in origins]).astype("float32")
# @@end

# @@todo 按时间 8:2 切分：前 80% 训练、后 20% 测试（时序**绝不 shuffle**）
cut = int(len(origins) * 0.8)
# @@end

X_train, Y_train = X[:cut], Y[:cut]
X_test, Y_test = X[cut:], Y[cut:]
t_test = frame.index[origins[cut:]]

# 朴素基线：季节性朴素（用昨天同一小时），§7.1 拿它当「被打分的预测」
PN = np.stack([ser[t + np.arange(1, HORIZON + 1) - 24] for t in origins[cut:]])

print("样本 %d（起点 %d ~ %d）| 切分点 %d | 训练 %d / 测试 %d"
      % (len(origins), origins[0], origins[-1], cut, len(X_train), len(X_test)))
print("训练段目标均值 %.6f | 测试段目标均值 %.4f" % (Y_train.mean(), Y_test.mean()))
print("测试段时段 %s ~ %s" % (t_test[0], t_test[-1]))
print("测试段真值 min %.4f max %.4f | 季节朴素预测 min %.4f max %.4f"
      % (Y_test.min(), Y_test.max(), PN.min(), PN.max()))
print("窗口最后一个点 X_train[0, -1] = %.4f（= ser[%d]，是**已知**的）" % (X_train[0, -1], START - 1))
print("第一个待预测点 Y_train[0, 0]   = %.4f（= ser[%d]，窗口里没有它）" % (Y_train[0, 0], START))

# ---- 验收 ----
assert len(origins) == 8569 and cut == 6855
assert X_train.shape == (6855, WINDOW) and Y_test.shape == (1714, HORIZON)
assert abs(float(Y_train.mean()) - 452.674164) < 1e-5
assert abs(float(Y_test.mean()) - 483.2289) < 1e-4
assert str(t_test[0]) == "2025-10-20 15:00:00" and str(t_test[-1]) == "2025-12-31 00:00:00"
assert X_train[0, -1] == np.float32(ser[START - 1]) and Y_train[0, 0] == np.float32(ser[START])
assert Y_train[0, 23] == np.float32(ser[START + HORIZON - 1]), "第 23 列就是 h=24"
'''

T2_CODE = '''def mse(y_true, y_pred):
    # @@todo 均方误差 = 残差平方的均值
    # @@hint np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    # @@end


def mae(y_true, y_pred):
    # @@todo 平均绝对误差 = 残差绝对值的均值
    # @@hint np.mean(np.abs(...))
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    # @@end


def rmse(y_true, y_pred):
    # @@todo 均方根误差 = sqrt(MSE)。注意**不是** MAE 开方
    # @@hint 直接复用上面写好的 mse()：np.sqrt(mse(...))
    return float(np.sqrt(mse(y_true, y_pred)))
    # @@end


def mape(y_true, y_pred):
    # @@todo 平均绝对百分比误差——分母是**真值**（这正是它的死穴，见 §7.1.1）
    # @@hint np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    # @@end


MSE, MAE, RMSE, MAPE = mse(Y_test, PN), mae(Y_test, PN), rmse(Y_test, PN), mape(Y_test, PN)
print("MSE  %12.6f" % MSE)
print("MAE  %12.6f" % MAE)
print("RMSE %12.6f" % RMSE)
print("MAPE %12.6f %%" % MAPE)
print("RMSE / MAE = %.6f   ← 恒 ≥ 1，这个比值在说误差分布的形状" % (RMSE / MAE))
print("sklearn 对账：MAE %.6f | MSE %.6f" % (mean_absolute_error(Y_test, PN),
                                             mean_squared_error(Y_test, PN)))
print("手写 - sklearn：MAE %.3e | MSE %.3e"
      % (MAE - mean_absolute_error(Y_test, PN), MSE - mean_squared_error(Y_test, PN)))

# ---- 验收 ----
assert abs(MSE - 1802.744577) < 1e-5
assert abs(MAE - 32.998794) < 1e-5
assert abs(RMSE - 42.458740) < 1e-5
assert abs(MAPE - 6.693991) < 1e-5
assert RMSE >= MAE, "RMSE ≥ MAE 恒成立（Jensen 不等式）"
assert abs(RMSE / MAE - 1.286676) < 1e-5
assert abs(MAE - mean_absolute_error(Y_test, PN)) < 1e-9
assert abs(MSE - mean_squared_error(Y_test, PN)) < 1e-9
'''

T3_CODE = '''# @@todo 人为构造一段「接近 0」的序列：把测试目标的**前 6 步**整体乘 0.01
# @@hint Y_trap = Y_test.copy()；然后只改前 6 列
Y_trap = Y_test.copy()
Y_trap[:, :6] = Y_trap[:, :6] * 0.01
# @@end


def smape(y_true, y_pred):
    # @@todo 对称 MAPE：分母从「真值」换成 (|真值| + |预测|)，于是分母不会归零
    # @@hint np.mean(2 * np.abs(y - yp) / (np.abs(y) + np.abs(yp))) * 100
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100)
    # @@end


def wape(y_true, y_pred):
    # @@todo 加权绝对百分比误差（100 − 它就是「准确率」）：把「逐点相对误差的均值」
    #       换成「绝对误差之和 / 真值之和」—— 分母换成总量，就不会归零
    # @@hint np.sum(np.abs(y - yp)) / np.sum(np.abs(y)) * 100
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)
    # @@end


print("缩放后真值 min %.6f（原 min %.4f）—— 分母塌了" % (Y_trap.min(), Y_test.min()))
print("MAE   %12.6f（原 %.6f，只涨 %.2f 倍）" % (mae(Y_trap, PN), MAE, mae(Y_trap, PN) / MAE))
print("RMSE  %12.6f（原 %.6f，涨 %.2f 倍）" % (rmse(Y_trap, PN), RMSE, rmse(Y_trap, PN) / RMSE))
print("MAPE  %12.6f（原 %.6f，涨 %.1f 倍）← 爆炸" % (mape(Y_trap, PN), MAPE, mape(Y_trap, PN) / MAPE))
print("sMAPE %12.6f（原 %.6f，涨 %.1f 倍）" % (smape(Y_trap, PN), smape(Y_test, PN),
                                              smape(Y_trap, PN) / smape(Y_test, PN)))
print("WAPE  %12.6f（原 %.6f，涨 %.1f 倍）" % (wape(Y_trap, PN), wape(Y_test, PN),
                                              wape(Y_trap, PN) / wape(Y_test, PN)))

print()
print("单点极端例：真值 0.05，预测 5.0 —— 绝对误差只有 %.2f kW" % abs(5.0 - 0.05))
print("  MAPE  %10.4f %%   ← 一个点就把整份报告毁掉" % mape(np.array([0.05]), np.array([5.0])))
print("  sMAPE %10.4f %%   ← 有上界（≤ 200）" % smape(np.array([0.05]), np.array([5.0])))

# ---- 验收 ----
assert abs(mae(Y_trap, PN) - 143.890302) < 1e-5
assert abs(rmse(Y_trap, PN) - 243.244359) < 1e-5
assert abs(mape(Y_trap, PN) - 2481.953943) < 1e-5
assert abs(smape(Y_trap, PN) - 54.016410) < 1e-5
assert abs(wape(Y_trap, PN) - 39.555869) < 1e-5
assert abs(smape(Y_test, PN) - 6.682859) < 1e-5
assert abs(wape(Y_test, PN) - 6.828811) < 1e-5
assert mape(Y_trap, PN) / MAPE > 300, "MAPE 涨了 300 倍以上，sMAPE/WAPE 只涨个位数倍"
assert smape(Y_trap, PN) / smape(Y_test, PN) < 10 and wape(Y_trap, PN) / wape(Y_test, PN) < 10
assert abs(mape(np.array([0.05]), np.array([5.0])) - 9900.0) < 1e-6
assert abs(smape(np.array([0.05]), np.array([5.0])) - 196.0396) < 1e-3
'''

T4_CODE = '''# 标准化参数只能来自训练段（ch01 §1.4.1 的纪律，零成本）
MU, SD = float(X_train.mean()), float(X_train.std())
MUY, SDY = float(Y_train.mean()), float(Y_train.std())

Xtr = torch.tensor((X_train - MU) / SD)
Ytr = torch.tensor((Y_train - MUY) / SDY)
Xte = torch.tensor((X_test - MU) / SD)

# 训练段内部再切一段「时间靠后」的验证集，专门给早停看（还是按时间，不 shuffle）
vcut = int(len(Xtr) * 0.85)
X_fit, Y_fit = Xtr[:vcut], Ytr[:vcut]
X_val, Y_val = Xtr[vcut:], Ytr[vcut:]

SUB, PATIENCE, MAX_EPOCH, CLIP, STEP, GAMMA = 2500, 5, 40, 1.0, 8, 0.5
print("训练子集 %d | 拟合段 %d / 验证段 %d | patience %d | 最大轮数 %d | 裁剪阈值 %.1f"
      % (SUB, len(X_fit), len(X_val), PATIENCE, MAX_EPOCH, CLIP))

with torch.random.fork_rng():
    torch.manual_seed(42)
    es_model = MLP(WINDOW, 128, 1)
    opt = torch.optim.Adam(es_model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP, gamma=GAMMA)
    tr_hist, va_hist, gn_hist, lr_hist = [], [], [], []
    best_val, best_epoch, wait, stopped_epoch = float("inf"), -1, 0, MAX_EPOCH
    for ep in range(1, MAX_EPOCH + 1):
        es_model.train()
        opt.zero_grad()
        # @@todo 在训练**子集**上算一个 full-batch 的 MSE 损失（只学 h=1 那一列）
        # @@hint 子集切片是 X_fit[:SUB] / Y_fit[:SUB]；目标列用 [:, 0:1] 保持二维
        loss = ((es_model(X_fit[:SUB]) - Y_fit[:SUB, 0:1]) ** 2).mean()
        # @@end
        # @@todo 反向传播，并用 clip_grad_norm_ 做梯度裁剪；把**裁剪前**的范数存进 gnorm
        # @@hint loss.backward()，再 torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        # @@hint clip_grad_norm_ 的**返回值就是裁剪前的总范数**，用 float(...) 转出来
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(es_model.parameters(), CLIP))
        # @@end
        opt.step()
        sched.step()
        es_model.eval()
        with torch.no_grad():
            # @@todo 在验证段上算同样的 MSE —— 它决定早停，不参与梯度
            # @@hint 别忘了 .eval() + torch.no_grad()（上面两层已经写好了）
            val_loss = float(((es_model(X_val) - Y_val[:, 0:1]) ** 2).mean())
            # @@end
        tr_hist.append(float(loss.detach()))
        va_hist.append(val_loss)
        gn_hist.append(gnorm)
        lr_hist.append(opt.param_groups[0]["lr"])
        # 早停计数：验证损失不再创新低就 wait+1，连续 PATIENCE 轮就停
        if val_loss < best_val - 1e-9:
            best_val, best_epoch, wait = val_loss, ep, 0
        else:
            wait = wait + 1
            if wait >= PATIENCE:
                stopped_epoch = ep
                break

print("实际跑了 %d 轮 | 最好的一轮 best_epoch = %d（val %.6f）| 停在第 %d 轮"
      % (len(tr_hist), best_epoch, best_val, stopped_epoch))
print("训练损失 %.4f → %.4f（还在降）| 验证损失 %.4f → %.4f（先降后升）"
      % (tr_hist[0], tr_hist[-1], va_hist[0], va_hist[-1]))
print("梯度范数 %.3f → %.3f | 最大 %.3f（裁剪阈值 %.1f）| 被裁过的轮数 %d"
      % (gn_hist[0], gn_hist[-1], max(gn_hist), CLIP, sum(1 for g in gn_hist if g > CLIP)))
print("学习率 %.6f → %.6f（StepLR：每 %d 轮 ×%.1f）"
      % (lr_hist[0], lr_hist[-1], STEP, GAMMA))

# ---- 验收 ----
assert stopped_epoch < MAX_EPOCH, "早停必须真的触发，否则这个演示没有意义"
assert stopped_epoch == best_epoch + PATIENCE, "停下来的那一轮 = 最好的一轮 + patience"
assert 3 <= best_epoch <= 25 and len(tr_hist) == stopped_epoch
assert va_hist[best_epoch - 1] == min(va_hist), "best_epoch 就是验证损失最小那一轮"
assert tr_hist[-1] < tr_hist[0], "训练损失还在降 —— 典型的过拟合形状"
assert max(gn_hist) > CLIP, "至少有一轮的梯度范数超过阈值，裁剪才是必要的"
assert abs(lr_hist[0] - 2e-3) < 1e-18
assert abs(lr_hist[-1] - 2e-3 * GAMMA ** (stopped_epoch // STEP)) < 1e-18
'''

T5_CODE = '''eps = np.arange(1, len(tr_hist) + 1)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
axes[0].plot(eps, tr_hist, marker="o", ms=3, label="训练损失（子集）")
axes[0].plot(eps, va_hist, marker="s", ms=3, label="验证损失")
axes[0].axvline(best_epoch, color="seagreen", ls="--", lw=1.2, label="best_epoch=%d" % best_epoch)
axes[0].axvline(stopped_epoch, color="crimson", ls=":", lw=1.2, label="stopped_epoch=%d" % stopped_epoch)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("MSE（标准化后）")
axes[0].set_title("早停：验证损失先降后升")
axes[0].legend(fontsize=8)

axes[1].plot(eps, gn_hist, marker="o", ms=3, color="tab:orange", label="梯度范数（裁剪前）")
axes[1].axhline(CLIP, color="crimson", ls="--", lw=1.2, label="clip = %.1f" % CLIP)
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("‖grad‖")
axes[1].set_title("梯度裁剪")
axes[1].legend(fontsize=8)

# @@todo 把学习率随 epoch 的变化画到第三张子图上（纵轴用对数刻度才看得出阶梯）
# @@hint axes[2].plot(eps, lr_hist, ...)；axes[2].set_yscale("log")
axes[2].plot(eps, lr_hist, marker="o", ms=3, color="tab:green")
axes[2].set_yscale("log")
axes[2].set_xlabel("epoch")
axes[2].set_ylabel("学习率")
axes[2].set_title("StepLR：每 %d 轮减半" % STEP)
# @@end

plt.tight_layout()
print("三个量都能从历史列表里还原：loss %d / grad %d / lr %d 个点"
      % (len(tr_hist), len(gn_hist), len(lr_hist)))

# ---- 验收 ----
assert len(eps) == stopped_epoch
assert int(np.argmin(va_hist)) + 1 == best_epoch
assert lr_hist[-1] <= lr_hist[0] and len(set(lr_hist)) <= 3
'''

T6_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(7)
    rec_model = MLP(WINDOW, 128, 1)
    rec_opt = torch.optim.Adam(rec_model.parameters(), lr=1e-2)
    for _ in range(40):
        rec_opt.zero_grad()
        # @@todo 递归模型的每一轮：目标只有 h=1 那一列
        # @@hint ((rec_model(Xtr) - Ytr[:, 0:1]) ** 2).mean().backward()
        ((rec_model(Xtr) - Ytr[:, 0:1]) ** 2).mean().backward()
        # @@end
        rec_opt.step()
    rec_model.eval()

    buf = ((X_test - MU) / SD).copy()
    rec_pred = np.zeros_like(Y_test)
    with torch.no_grad():
        for h in range(HORIZON):
            # @@todo 用递归模型预测下一步：输出要**反标准化**回 kW，再写进第 h 列
            # @@hint 输入是 buf；输出 * SDY + MUY 才是 kW
            p = rec_model(torch.tensor(buf)).numpy().ravel() * SDY + MUY
            rec_pred[:, h] = p
            # @@end
            # @@todo 把这一步的预测「重新标准化」后接到窗口末尾，同时丢掉最老的一格
            # @@hint np.concatenate([buf[:, 1:], ((p - MUY) / SD)[:, None]], axis=1)
            buf = np.concatenate([buf[:, 1:], ((p - MUY) / SD)[:, None]], axis=1)
            # @@end

print("递归模型 24 步 MAE %.6f（h=1 %.4f → h=24 %.4f）"
      % (mae_all(Y_test, rec_pred), mae_by_step(Y_test, rec_pred)[0], mae_by_step(Y_test, rec_pred)[23]))
print("递归预测的均值 %.4f / std %.4f | 真值均值 %.4f / std %.4f"
      % (rec_pred.mean(), rec_pred.std(), Y_test.mean(), Y_test.std()))

with torch.random.fork_rng():
    torch.manual_seed(7)
    dir_model = MLP(WINDOW, 128, HORIZON)
    dir_opt = torch.optim.Adam(dir_model.parameters(), lr=1e-2)
    for _ in range(40):
        dir_opt.zero_grad()
        # @@todo 直接多输出模型的每一轮：一次吐出 24 个值
        # @@hint ((dir_model(Xtr) - Ytr) ** 2).mean().backward()
        ((dir_model(Xtr) - Ytr) ** 2).mean().backward()
        # @@end
        dir_opt.step()
    dir_model.eval()
    with torch.no_grad():
        # @@todo 直接模型推理：一次前向就得到 24 步，反标准化回 kW
        # @@hint dir_model(Xte).numpy() * SDY + MUY
        dir_pred = dir_model(Xte).numpy() * SDY + MUY
        # @@end

print("直接多输出 24 步 MAE %.6f（h=1 %.4f → h=24 %.4f）"
      % (mae_all(Y_test, dir_pred), mae_by_step(Y_test, dir_pred)[0], mae_by_step(Y_test, dir_pred)[23]))
print("两个模型的 MAE 都远好于季节朴素基线 %.6f" % mae_all(Y_test, PN))

# ---- 验收 ----
assert rec_pred.shape == dir_pred.shape == Y_test.shape == (1714, HORIZON)
assert 10 < mae_all(Y_test, rec_pred) < 25, "递归模型应在「远好于基线」的量级，但不该是死值"
assert 10 < mae_all(Y_test, dir_pred) < 25
assert mae_all(Y_test, rec_pred) < mae_all(Y_test, PN)
assert mae_all(Y_test, dir_pred) < mae_all(Y_test, PN)
assert abs(mae_all(Y_test, PN) - 32.998794) < 1e-5, "基线是确定性的，可以卡死值"
assert rec_pred.mean() > 0 and rec_pred.std() > 0
'''

T7_CODE = '''# @@todo 把「递归 / 直接 / 季节朴素」三条曲线的逐步 MAE 都算出来（长度 24）
# @@hint 脚手架里的 mae_by_step(y_true, y_pred) 直接可用
step_rec = mae_by_step(Y_test, rec_pred)
step_dir = mae_by_step(Y_test, dir_pred)
step_pn = mae_by_step(Y_test, PN)
# @@end

print("h      递归      直接      季节朴素")
for h in range(HORIZON):
    print("%2d  %8.4f  %8.4f  %8.4f" % (h + 1, step_rec[h], step_dir[h], step_pn[h]))
print("整体 %8.4f  %8.4f  %8.4f" % (step_rec.mean(), step_dir.mean(), step_pn.mean()))

err_rec = np.abs(Y_test - rec_pred)
print()
print("递归误差的**标准差**（不是 MAE）：h1 %.4f → h8 %.4f → h24 %.4f（涨 %.2f 倍）"
      % (err_rec[:, 0].std(), err_rec[:, 7].std(), err_rec[:, 23].std(),
         err_rec[:, 23].std() / err_rec[:, 0].std()))

# @@todo 拟合「误差标准差 ~ 步长的幂律」：std(h) ≈ a · h^b，b 就是放大速度的指数
# @@hint np.polyfit(np.log(1..24), np.log(逐步误差标准差), 1) 拿斜率
h_grid = np.arange(1, HORIZON + 1)
std_by_step = err_rec.std(axis=0)
b_exp = float(np.polyfit(np.log(h_grid), np.log(std_by_step), 1)[0])
# @@end

print("幂律拟合 std(h) ∝ h^%.4f  ← 若是「独立误差线性累加」b 会接近 0.5" % b_exp)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
axes[0].plot(h_grid, step_rec, marker="o", ms=3, label="递归（自己喂自己）")
axes[0].plot(h_grid, step_dir, marker="s", ms=3, label="直接多输出")
axes[0].plot(h_grid, step_pn, marker="^", ms=3, label="季节朴素 t-24")
axes[0].set_xlabel("预测步长 h（小时）")
axes[0].set_ylabel("MAE (kW)")
axes[0].set_title("多步误差累积：递归曲线单调上翘")
axes[0].legend(fontsize=9)

axes[1].plot(h_grid, std_by_step, marker="o", ms=3, color="crimson", label="递归误差 std")
axes[1].plot(h_grid, std_by_step[0] * h_grid ** b_exp, ls="--", lw=1.2,
             label="幂律拟合 h^%.3f" % b_exp)
axes[1].set_xlabel("预测步长 h（小时）")
axes[1].set_ylabel("误差标准差 (kW)")
axes[1].set_title("方差随时间放大")
axes[1].legend(fontsize=9)
plt.tight_layout()

# ---- 验收 ----
assert step_rec.shape == step_dir.shape == step_pn.shape == (HORIZON,)
assert abs(step_pn[0] - 33.0215) < 1e-4 and abs(step_pn[23] - 33.1928) < 1e-4
assert step_rec[-1] > step_rec[0] * 1.15, "递归误差随步长明显上翘"
assert step_dir[-1] > step_dir[0], "直接多输出也会涨，但通常更平"
assert err_rec[:, 23].std() > err_rec[:, 0].std() * 1.15, "误差方差随时间放大"
assert 0.0 < b_exp < 0.6, "放大速度介于「完全独立」（0.5）与「几乎不涨」（0）之间"
'''

T8_CODE = '''# @@todo 整段抽样：测试段每 4 个起点取一个（1714 → 429 个点，画得动）
# @@hint np.arange(0, len(Y_test), 4)
show = np.arange(0, len(Y_test), 4)
# @@end

k = 369                      # 单日放大的起点（测试段内第 370 个窗口）
day_idx = np.arange(k, k + HORIZON)
print("整段抽样 %d 个点 | 单日放大起点 %s（h=1 对应 %s）"
      % (len(show), t_test[k], t_test[k]))

fig, axes = plt.subplots(2, 1, figsize=(13, 7.2),
                         gridspec_kw={"height_ratios": [1.35, 1]})
axes[0].plot(t_test[show], Y_test[show, 0], lw=1.0, label="真实（h=1 的时刻值）")
axes[0].plot(t_test[show], rec_pred[show, 0], lw=1.0, label="递归模型 1 步预测")
axes[0].set_ylabel("负荷 kW")
axes[0].set_title("整段抽样：测试段 1714 个起点里每 4 个取 1 个")
axes[0].legend(fontsize=9)

# @@todo 在第二张子图上画「同一起点的 24 步」四条曲线：真实 / 递归 / 直接 / 季节朴素
# @@hint 横轴用 np.arange(1, HORIZON + 1)，因为目标只有 24 步，没有时间戳
axes[1].plot(np.arange(1, HORIZON + 1), Y_test[k], marker="o", ms=4, color="black", label="真实")
axes[1].plot(np.arange(1, HORIZON + 1), rec_pred[k], marker="s", ms=3, label="递归")
axes[1].plot(np.arange(1, HORIZON + 1), dir_pred[k], marker="^", ms=3, label="直接多输出")
axes[1].plot(np.arange(1, HORIZON + 1), PN[k], marker="x", ms=3, label="季节朴素 t-24")
# @@end

axes[1].set_xlabel("预测步长 h（小时）")
axes[1].set_ylabel("负荷 kW")
axes[1].set_title("单日放大：起点 %s 的 24 步预测" % t_test[k])
axes[1].legend(fontsize=9)
plt.tight_layout()

# ---- 验收 ----
assert len(show) == 429
assert abs(float(Y_test[k, 0]) - float(ser[origins[cut] + k])) < 1e-3, "窗口目标与原始序列对齐"
assert str(t_test[k]) == "2025-11-05 00:00:00"
assert Y_test[k].shape == rec_pred[k].shape == dir_pred[k].shape == PN[k].shape == (HORIZON,)
'''

T9_CODE = '''res1 = rec_pred[:, 0] - Y_test[:, 0]
res24 = rec_pred[:, 23] - Y_test[:, 23]


def acf(x, lag):
    # @@todo 自相关：先减去均值，再算 sum(x[:-lag] * x[lag:]) / sum(x * x)
    # @@hint 先做 x = np.asarray(x, dtype=float) - np.mean(x)
    # @@hint 注意 lag 必须 ≥ 1：x[:-0] 是空数组，lag=0 会直接 Broadcasting 报错
    x = np.asarray(x, dtype=float) - np.mean(x)
    return float(np.sum(x[:-lag] * x[lag:]) / np.sum(x * x))
    # @@end


LAGS = (1, 2, 3, 24, 168)
# @@todo 算出 h=1 / h=24 残差在 LAGS 上的自相关
# @@hint [round(acf(res1, k), 4) for k in LAGS]
acf_r1 = [round(acf(res1, k), 4) for k in LAGS]
acf_r24 = [round(acf(res24, k), 4) for k in LAGS]
# @@end

naive1 = np.array([ser[t - 24] for t in origins[cut:]]) - Y_test[:, 0]

print("lag      " + "  ".join("lag%-6d" % k for k in LAGS))
print("h=1  残差 " + "  ".join("%+.4f " % v for v in acf_r1))
print("h=24 残差 " + "  ".join("%+.4f " % v for v in acf_r24))
print("朴素 残差 " + "  ".join("%+.4f " % acf(naive1, k) for k in LAGS))
print()
print("手写 acf vs pandas .autocorr（同一份残差，两种口径）：")
print("  lag=1  手写 %+.6f | pandas %+.6f | 差 %+.6f"
      % (acf(res1, 1), pd.Series(res1).autocorr(1), acf(res1, 1) - pd.Series(res1).autocorr(1)))
print("  lag=24 手写 %+.6f | pandas %+.6f | 差 %+.6f"
      % (acf(res1, 24), pd.Series(res1).autocorr(24),
         acf(res1, 24) - pd.Series(res1).autocorr(24)))
print("  （差别来自分母：手写用全序列方差，pandas 用两个错位子序列各自的 std）")

# ---- 验收 ----
assert len(acf_r1) == len(acf_r24) == 5
assert -0.3 < acf_r1[0] < 0.4, "h=1 残差的自相关很弱 —— 模型把日周期吃掉了"
assert acf_r24[0] > acf_r1[0] + 0.2, "多步残差仍有强自相关：误差是「成串」的"
assert abs(acf(naive1, 1) - 0.3804) < 1e-4, "朴素残差是确定性的，可以卡死值"
assert abs(acf(naive1, 24) - (-0.3149)) < 1e-4
assert abs(acf(res1, 1) - pd.Series(res1).autocorr(1)) < 0.05
try:
    acf(res1, 0)
    raise AssertionError("lag=0 应该报错")
except ValueError:
    print("lag=0 → ValueError（x[:-0] 是空数组）—— 手写 ACF 的边界坑")
'''

T10_CODE = '''# @@todo 把 h=1 残差按「预测时刻的小时」分组，算每组的均值与标准差
# @@hint pd.DataFrame({"hour": t_test.hour, "res": res1}).groupby("hour")["res"]
res_frame = pd.DataFrame({"hour": t_test.hour, "res": res1})
by_hour = res_frame.groupby("hour")["res"]
hour_mean, hour_std = by_hour.mean(), by_hour.std()
# @@end

# @@todo 系统性偏差的 t 统计量：t = 均值 / 标准误，标准误 = std(ddof=1) / sqrt(n)
# @@hint res1.std(ddof=1) / np.sqrt(len(res1))
t1 = float(res1.mean() / (res1.std(ddof=1) / np.sqrt(len(res1))))
t24 = float(res24.mean() / (res24.std(ddof=1) / np.sqrt(len(res24))))
# @@end

for nm, r, t in [("h=1 ", res1, t1), ("h=24", res24, t24)]:
    print("%s 残差均值 %+8.4f kW（占负荷均值 %+.4f%%）| std %7.4f | t = %+.4f"
          % (nm, r.mean(), r.mean() / Y_test.mean() * 100, r.std(), t))
print("临界值：n=%d 时 |t| > 1.96 就算「在 5%% 水平上显著」" % len(res1))
print()
print("按小时的残差均值：最低 %+.4f kW（%d 点）| 最高 %+.4f kW（%d 点）| 极差 %.4f"
      % (hour_mean.min(), int(hour_mean.idxmin()), hour_mean.max(), int(hour_mean.idxmax()),
         hour_mean.max() - hour_mean.min()))
print("按小时的残差标准差：最稳 %d 点 %.4f | 最不稳 %d 点 %.4f"
      % (int(hour_std.idxmin()), hour_std.min(), int(hour_std.idxmax()), hour_std.max()))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
axes[0].boxplot([res_frame.loc[res_frame["hour"] == h, "res"].values for h in range(24)],
                tick_labels=list(range(24)), showfliers=False)
axes[0].axhline(0, color="black", lw=1)
axes[0].set_xlabel("预测时刻（小时）")
axes[0].set_ylabel("残差 kW")
axes[0].set_title("残差按小时分组：晚间波动更大")
axes[1].bar(hour_mean.index, hour_mean.values, color="tab:blue")
axes[1].axhline(0, color="black", lw=1)
axes[1].set_xlabel("预测时刻（小时）")
axes[1].set_ylabel("残差均值 kW")
axes[1].set_title("系统性偏差：午后高估、清晨低估")
plt.tight_layout()

# ---- 验收 ----
assert len(hour_mean) == 24 and len(hour_std) == 24
assert int(by_hour.size().sum()) == len(res1) == 1714
assert hour_std.max() > hour_std.min() * 1.5, "晚间（18~21 点）的残差波动明显更大"
assert abs(t1) < 6 and abs(t24) < 20
assert abs(res1.mean()) / Y_test.mean() < 0.02, "偏差占负荷不到 2% —— 统计显著但实际不重要"
assert abs(t24) > abs(t1), "步长越长，同一份样本量下的偏差越容易被检出"
'''

LESSON = [
    md(
        """
# ch07 训练与评估（讲解版）

> **本节考点**：模型训练（**技能操作 40%**）+ 模型性能评估（**10%**）+ 数据可视化。
> 前面几章都在回答「怎么造特征、怎么搭模型」，本章回答的是**怎么知道模型好不好**——
> 而这件事的翻车率，比调模型高得多。

| 竞赛评分点 | 分值含义 | 本章覆盖 |
|---|---|---|
| 模型训练 | **技能操作 40%** | 训练/验证双循环、早停、梯度裁剪、lr 调度 |
| 模型性能评估 | **10%** | MSE / MAE / RMSE / MAPE、多步误差累积、残差诊断 |
| 数据可视化 | 独立计分 | 预测曲线（整段 + 单日放大）、残差箱线图 |
| 新能源功率 / 负荷预测 | 行业赛题 | 评估口径直接决定「谁的模型更好」 |

**与 `ch05` 的分工**：ch05 讲**怎么产生**多步预测（递归 / 直接多输出 / Seq2Seq / teacher forcing）；
本章讲**怎么评估**它 —— 同一条递归预测，在 ch05 里是「能跑通」，在本章里要回答
「误差随步长怎么涨、为什么涨、涨得快不快」。

## 学习目标

1. 手写 MSE / MAE / RMSE / MAPE，说清每一个的**量纲**与**对离群点的敏感性**，并与 sklearn 逐位对账
2. 说清 **MAPE 的零点陷阱**为什么致命，会用 `sMAPE` / `WAPE` 兜底
3. 会写一个**带早停 + 梯度裁剪 + lr 调度**的训练循环，并解释 `best_epoch` 与 `stopped_epoch` 的关系
4. 会画「MAE vs 步长」曲线，并解释**递归误差的方差为什么随时间放大**
5. 会用**残差 ACF / 按小时分组 / t 检验**三件套诊断模型，并区分「统计显著」与「实际重要」

## 本章统一口径

```
数据   data/load_curve.csv → 去重排序 → A 台区（居民型）→ clean_series → 8760 个整点
窗口   window = 72（过去 72 小时）   步长 horizon = 24（预测未来 24 小时）
起点   从 168 开始（与 ch08 对齐，ch08 要用 lag168）
切分   前 80% 训练（6855 个起点） / 后 20% 测试（1714 个起点），**绝不 shuffle**
```

## API 速查表

| 方法 / 属性 | 关键参数 | 返回 | 一句话 |
|---|---|---|---|
| `np.mean((y - yp) ** 2)` | — | `float` | MSE；量纲是 kW²，不直观 |
| `np.mean(np.abs(y - yp))` | — | `float` | MAE；量纲是 kW，最直观 |
| `np.sqrt(mse)` | — | `float` | RMSE；**不是** MAE 开方 |
| `np.mean(np.abs((y - yp) / y))` | — | `float` | MAPE；分母是真值，**接近 0 会爆** |
| `np.sum(abs(y-yp)) / np.sum(abs(y))` | — | `float` | WAPE；分母换成总量，不会爆 |
| `torch.nn.utils.clip_grad_norm_(params, c)` | `max_norm` | `Tensor` | **返回裁剪前的总范数**，别丢掉 |
| `torch.optim.lr_scheduler.StepLR(opt, s, g)` | `step_size` / `gamma` | — | 每 `s` 轮把 lr 乘 `g`，**在 `opt.step()` 之后调** |
| `Series.autocorr(lag)` | `lag` | `float` | 自相关；口径与手写版略有差别（分母不同） |
| `DataFrame.groupby(col)["x"].std()` | — | `Series` | 分组统计，残差按小时分组的核心 |
| `mean / (std / sqrt(n))` | — | `float` | t 统计量；检验**系统性偏差**是否显著 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md(
        """
## 7.1 四个指标：公式、量纲、以及它们各自怕什么

先把「被打分的对象」定下来。本章统一用**季节朴素（seasonal naive，t − 24）**当基准预测：

```python
PN = np.stack([ser[t + np.arange(1, HORIZON + 1) - 24] for t in origins[cut:]])
```

它的意思是「**明天这个点 = 昨天这个点**」。别小看它 —— 后面 ch08 会看到，它比
「用最近一个值外推」（MAE 60.52）强得多，因为日周期被它整段对齐了。

四个指标的公式与量纲：

| 指标 | 公式 | 量纲 | 对离群点 |
|---|---|---|---|
| MSE | $\\frac{1}{n}\\sum (y_i-\\hat y_i)^2$ | kW² | **极敏感**（平方放大） |
| MAE | $\\frac{1}{n}\\sum \\lvert y_i-\\hat y_i\\rvert$ | kW | 不敏感（线性） |
| RMSE | $\\sqrt{\\text{MSE}}$ | kW | 敏感 |
| MAPE | $\\frac{1}{n}\\sum \\lvert\\frac{y_i-\\hat y_i}{y_i}\\rvert\\times100\\%$ | %（无量纲） | **看分母**（见 §7.1.1） |

`RMSE ≥ MAE` **恒成立**（由 Jensen 不等式，平方是凸函数）。更有用的一条是**看比值**：

| `RMSE/MAE` | 说明误差分布 | 典型场景 |
|---|---|---|
| ≈ 1.00 | 误差**几乎等幅**（全是同一个大小） | 系统被量化误差主导 |
| 1.25 ~ 1.30 | 误差近似**高斯** | 本数据：**1.286676** |
| ≫ 1.4 | 少数**大离群点**把 RMSE 抬起来 | 有尖峰 / 未清洗的异常值 |

> **所以报告里 MAE 和 RMSE 要一起给**：MAE 说「典型误差多大」，`RMSE/MAE` 说
> 「有没有少数特别糟的点」。只给一个，等于只说了一半。
>
> 顺手记一个诊断经验：**如果 `RMSE/MAE` 突然变大，先回去查异常值，不要先调模型。**
"""
    ),
    code(T1_CODE),
    code(T2_CODE),
    md(
        """
### 7.1.1 难点深挖：MAPE 的零点陷阱 —— 一个 0.05 的点能毁掉整份报告

**为什么难**：MAPE 的分母是**真值** $y_i$。而电力负荷、光伏出力、风速这些量的真值
**天然会接近 0**（夜间光伏、检修时段、轻载台区）。分母一小，单点误差就爆炸。

先做一个受控实验：把测试目标的（本来在 352 kW 以上的）**前 6 步整体乘 0.01**，
让它们掉到 3.53 kW 附近。**注意只动了真值、没动预测**：

```
被评分的点总数 = 1714 个起点 × 24 步 = 41136 个
被改动的点     = 1714 个起点 × 6 步  = 10284 个（25%）
改动后真值最小值 = 3.525300 kW（原来最小的也有 352.5300）
```

**实测结果（同一份预测，只改真值）：**

| 指标 | 原尺度 | 前 6 步 ×0.01 | 涨了几倍 |
|---|---|---|---|
| MAE | 32.998794 | 143.890302 | **4.36×** |
| RMSE | 42.458740 | 243.244359 | **5.73×** |
| **MAPE** | 6.693991 % | **2481.953943 %** | **370.8×** ← 膨胀到无意义 |
| sMAPE | 6.682859 % | 54.016410 % | 8.08× |
| WAPE | 6.828811 % | 39.555869 % | 5.79× |

极端单点：真值 `0.05`、预测 `5.0`，**绝对误差只有 4.95 kW**，
但 MAPE = **9900%**，sMAPE = 196.04%。

**两种修法**：

```python
sMAPE = mean(2·|y − ŷ| / (|y| + |ŷ|)) × 100     # 分母加对称项 → 上界 200%
WAPE  = sum(|y − ŷ|) / sum(|y|)      × 100       # 分母换成总量 → 分母不会归零
```

**判定规则**：

| 场景 | 该用哪个 |
|---|---|
| 真值远离 0，且业务真的关心**相对误差** | MAPE 可用，但**必须**同时报 MAE |
| 真值可能接近 0（光伏、风速、轻载） | **禁用 MAPE**；用 WAPE 或 sMAPE |
| 多个量级差异很大的序列要合并算一个分 | 用 WAPE（天然加权） |

> **一句话记住**：MAPE 惩罚的是「相对误差」，而「相对」这件事在分母趋零时没有定义。
> 光伏功率预测题里，夜间真值全是 0，**用 MAPE 会直接算出 inf** —— 这不是模型的问题。
"""
    ),
    code(T3_CODE),
    md(
        """
### 7.1.2 难点深挖：`RMSE/MAE` 是「误差分布形状」的指纹

三个指标都是**同一个误差向量的不同汇总量**，所以它们的信息并不重复：

```
误差 = [3, 3, 3, 3]        → MAE 3, RMSE 3      → 比值 1.000
误差 = [1, 2, 3, 4, 5]     → MAE 3, RMSE 3.32   → 比值 1.105
误差 = [0.5, 0.5, 0.5, 11.5]→ MAE 3.25, RMSE 5.77 → 比值 1.776
```

第三组和第一组**平均一样痛**，但显然是两种完全不同的失败模式：
一组是「均匀地差一点」，另一组是「绝大部分很准，偶尔崩一次」。

**本数据实测**：

| 量 | 值 |
|---|---|
| MAE | 32.998794 kW |
| RMSE | 42.458740 kW |
| RMSE / MAE | **1.286676** |

1.2867 落在「近似高斯」的区间里 —— 说明季节性朴素的误差**没有量级失控的离群点**，
是「每天都差一点」而不是「偶尔崩一次」。

**判定规则**：

> 报告里 `MAE` 与 `RMSE` 必须成对出现。看到 `RMSE/MAE > 1.4`，
> 第一反应是**回数据里找那 1% 的烂点**（ch01 的尖峰 / 哨兵值 / 未清洗异常），
> 第二反应才是改模型。
"""
    ),
    md(
        """
## 7.2 训练循环模板：双循环 + 早停 + 梯度裁剪 + lr 调度

这一节给一个**可以直接抄走的模板**。四个零件缺一不可：

| 零件 | 代码 | 解决什么问题 |
|---|---|---|
| train / valid **双循环** | `model.train()` / `model.eval()` | 验证段能提前发现过拟合 |
| **早停** | `patience` 计数 + `best_epoch` | 不用猜「该训多少轮」 |
| **梯度裁剪** | `clip_grad_norm_(params, CLIP)` | 时序数据有尖峰，梯度会突然炸 |
| **lr 调度** | `StepLR(opt, step_size, gamma)` | 后期用小 lr 收敛到更细的解 |

**四个坑点**：

1. **验证段必须按时间取**（我们取训练段末 15%），不能随机抽 —— 理由和 ch01 §1.4.1 一样
2. **`clip_grad_norm_` 在 `backward()` 之后、`step()` 之前**，顺序反了就没效果
3. **`sched.step()` 在 `opt.step()` 之后**（PyTorch 2.x 起，先调 scheduler 会警告）
4. **`best_epoch` 与 `stopped_epoch` 是两个不同的数**：前者是「验证损失最低的那一轮」，
   后者是「忍不下去的那一轮」，两者相差**正好 `patience`**（见 §7.2.1）

**为什么用 full-batch（不加 DataLoader）**：本章 6855 个样本、72 维输入，
全批一次前向只要毫秒级；不 shuffle 也就意味着**结果与内核重启次数无关**，
每个 `assert` 都能对齐。真实项目样本上百万时才必须上 `DataLoader`（ch06 讲）。

为了让**早停真的触发**，这里刻意做了两件事：只用前 2500 个训练样本（欠采样 → 快速过拟合）、
把 lr 放到 2e-3（前期梯度大 → 裁剪有戏）。
"""
    ),
    code(T4_CODE),
    md(
        """
### 7.2.1 难点深挖：早停到底「停」在哪 —— `best_epoch` vs `stopped_epoch`

初学者最常写错的早停是这样的：

```python
if val_loss < best:              # ← 只有这一半
    best = val_loss
    torch.save(model.state_dict(), "best.pt")
```

它**只保存了最优权重，却没让训练停下来** —— 训练继续跑，`best.pt` 停在最好的那轮，
但你不知道「最好」是哪一轮，也不知道该不该继续跑更久。

正确的写法要维护**一个计数器**：

```python
if val_loss < best_val - 1e-9:      # 1e-9 是容差，避免浮点抖动白刷新
    best_val, best_epoch, wait = val_loss, ep, 0
else:
    wait = wait + 1
    if wait >= PATIENCE:            # 连续 PATIENCE 轮不创新低 → 停
        stopped_epoch = ep
        break
```

**`best_epoch` 与 `stopped_epoch` 的关系是恒等式**：

```
stopped_epoch = best_epoch + PATIENCE
```

**实测（`patience = 5`）：`best_epoch = 14`、`stopped_epoch = 19`、`14 + 5 = 19` ✓**

**为什么是恒等式**：`wait` 只在「不刷新」时累加，所以从最好那一轮开始，
必须连续 `PATIENCE` 轮都不刷新才会触发；期间一旦刷新，`wait` 归零、`best_epoch` 一起前移。

**读到这条曲线要能说出三件事**（见下一步的图）：

| 现象 | 含义 |
|---|---|
| 训练损失**持续下降**、验证损失**先降后升** | 典型的过拟合 |
| 验证损失在 `best_epoch` 之后**抬头的幅度很小** | 数据规整、噪声不大 → 早停主要省时间，不是救命 |
| 梯度范数**前几轮远大于阈值**，后面自然降到阈值内 | 裁剪的作用期在**训练早期**（冷启动阶段） |

> **判定规则**：`patience` 怎么选？经验值是 `epochs` 的 10%~20%，且**至少 3**。
> 太小会被验证损失的抖动骗停；太大就失去省时间的意义。想更稳就配 `ReduceLROnPlateau`
> —— 它会在验证损失停滞时**先降 lr**，往往能再挤出几轮。
"""
    ),
    code(T5_CODE),
    md(
        """
## 7.3 多步误差累积：递归 vs 直接多输出

预测未来 24 小时有两条路线，它们的评估结果**完全不同**：

| 路线 | 训练目标 | 推理方式 | 一次前向 |
|---|---|---|---|
| **递归**（recursive） | 只学 h=1 | 把预测**接回窗口末尾**，再预测下一步 | 要跑 24 次 |
| **直接多输出**（direct） | 一次学 h=1~24 全部 24 个值 | 一次前向吐 24 个数 | 只要 1 次 |

```python
# 递归：自己喂自己
p = rec_model(buf)                 # 预测下一步
buf = concat(buf[:, 1:], p)        # 丢掉最老的一格，把预测接上去
```

```python
# 直接：一次吐出 24 个
pred24 = dir_model(window)         # shape (B, 24)
```

**先看结论**（本章实测，两个模型都是 128 隐层、40 轮 Adam lr=1e-2）：

| 模型 | 24 步 MAE | h=1 MAE | h=24 MAE | 误差 std（h1 → h24） |
|---|---|---|---|---|
| 递归 | 17.266384 | **14.952325** | 19.115711 | 11.5343 → **15.0208** |
| 直接多输出 | **17.035627** | 15.176231 | **18.490346** | — |
| 季节朴素 t−24 | 32.998794 | 33.0215 | 33.1928 | — |

两个模型都**把基线砍掉了近一半**（33.0 → 17.0 / 17.3），但它们的**误差形状完全不同**：

- **递归在 h=1 上更强**（14.95 vs 15.18）—— 它只有一个目标，全部容量都用在「下一小时」上
- **直接多输出在 h=24 上更强**（18.49 vs 19.12），整体 MAE 也略好（17.036 vs 17.266）

这说明「递归一定更好」或「直接一定更好」都是**错觉**，两边的强弱取决于任务：

> 递归把「长程能力」外包给了「自己上一时刻的输出」——单步很准，但一旦某一步偏了，
> 误差会滚进窗口（见 §7.3.1）；直接多输出没有滚雪球效应，但 24 个头共享一个隐层，
> 近端那几列会被难学的远端「拖累」。**本章数据上后者净收益更大。**
"""
    ),
    code(T6_CODE),
    md(
        """
### 7.3.1 难点深挖：递归误差为什么会随时间放大

**为什么难**：递归的每一小时都用「上一小时的预测值」当输入。于是第 h 步的输入里
已经掺进了前面 h−1 步的全部误差。误差不是「加」进去的，是**沿着窗口滚进去**的。

形式化一点。设单步误差为 $e_t$（零均值、方差 $\\sigma^2$），输入窗口长 $W$。
第 h 步的输入里有 $\\min(h-1, W)$ 个位置是「预测值」而不是真值，
所以第 h 步的误差大致是

$$\\hat y^{(h)} - y^{(h)} \\;\\approx\\; \\underbrace{e_h}_{\\text{本步新误差}} \\;+\\; \\sum_{k=1}^{\\min(h-1,W)} \\alpha_k \\, e_{h-k}$$

其中 $\\alpha_k$ 是模型对「输入的 k 步前那个值」的敏感度。

- 如果各步误差**相互独立**，方差是**线性累加**的：$\\mathrm{Var} \\propto h$，即 **std ∝ h^0.5**
- 但模型对输入的敏感度 $\\alpha_k$ **不是 1** —— 窗口里 72 个数，一个位置变化 1% 只挪动输出一点点，
  所以实际指数会**小于 0.5**

**实测**（递归模型，h=1..24 的残差标准差做幂律拟合）：

```
std(h) ∝ h^0.0861      ← 远小于 0.5
h1 std 11.5343 → h8 13.6496 → h24 15.0208（涨 1.30 倍）
```

**再对比「直接多输出」的曲线形状**：

| 曲线特征 | 含义 |
|---|---|
| 递归的 MAE 从 h=1 到 h=24 **单调上翘**，中间没有回落 | 误差沿窗口持续注入 |
| 直接多输出的曲线**锯齿状、不平滑** | 24 个独立头各自学习，h 之间没有耦合 |
| 朴素基线的曲线**几乎水平** | 它根本没有「累积」这一说：t−24 与步长无关 |

**判定规则**：

> 递归误差放大的**速度**由「模型对输入扰动的敏感度」决定，不由步长单独决定。
> 想压住它，三条路：① 直接多输出（牺牲 h=1 的精度换 h=24 的稳定）；
> ② 用 `scheduled sampling` 在训练时按概率喂预测值（ch05 teacher forcing 的延伸）；
> ③ 把「误差反馈」显式建模 —— 这是 Seq2Seq + attention 的动机。
>
> **反过来说**：如果 `std(h)` 从 h=1 到 h=24 只涨了 30%（像本例），
> 说明单步模型已经很稳 —— 这时**没必要**为了压累积误差去上复杂的 Seq2Seq。
"""
    ),
    code(T7_CODE),
    md(
        """
## 7.4 预测曲线可视化：整段抽样 + 单日放大

评估指标是**压缩过的信息**，一定会丢东西。两张图能把它补回来：

**图一：整段抽样。** 测试段 1714 个起点全画会糊成一团，**每 4 个取 1 个**（429 点），
画「真实值」与「递归模型 h=1 预测」。看三件事：

1. **相位对不对** —— 峰谷是不是踩在同一个小时上（错位一天 = 灾难，但 MAE 可能只涨一点）
2. **幅度对不对** —— 有没有系统性压低峰值（回归模型常见的「向均值收缩」）
3. **周末 / 工作日** —— 居民型台区周末更忙，看曲线有没有跟上

**图二：单日放大。** 挑一个起点，把它**同一批 24 步**的四条曲线画在一起：
真实 / 递归 / 直接多输出 / 季节朴素。这张图回答的是「**模型错在哪几步上**」——
是整天平移（偏差）还是峰谷错位（相位），还是只在高负荷段偏。

> **可视化纪律**：任何「模型 A 比模型 B 好 2%」的结论，都要有这两张图撑着。
> 见过太多「MAE 一样、曲线完全不同」的情况 —— 一个是相位准、一个是幅度准，
> 选哪个取决于业务（要不要用来做调峰决策）。
"""
    ),
    code(T8_CODE),
    md(
        """
## 7.5 残差诊断：ACF / 按小时分组 / 显著性

指标只回答「误差多大」，残差诊断回答「**误差还剩什么结构**」—— 剩下的结构就是
下一轮迭代的方向。三件套：

| 诊断 | 看什么 | 什么算「有问题」 |
|---|---|---|
| **残差 ACF** | 残差还有没有自相关 | lag ≥ 2 后仍显著非 0 → 还有结构没学到 |
| **按小时分组** | 残差均值 / 标准差随小时的形状 | 某些小时长期偏 → 有系统性偏差或漏了变量 |
| **t 检验** | 残差均值是否显著非 0 | \\|t\\| > 1.96 且**偏差量级**不可忽略 |

**手写 ACF 与 pandas `autocorr` 的区别**（对账时会看到）：

```python
手写： sum(x[:-lag] * x[lag:]) / sum(x * x)        # 分母 = 全序列方差
pandas： Series(x).autocorr(lag)                    # 分母 = 两个错位子序列各自的 std
```

lag 很小时两者几乎一样；lag 大到接近序列长度时会明显分开 —— **报告里写清用的是哪一种**。

**本数据实测（递归模型的残差）**：

| lag | 1 | 2 | 3 | 24 | 168 |
|---|---|---|---|---|---|
| h=1 残差 | +0.1543 | +0.1111 | +0.0218 | −0.0011 | +0.1646 |
| h=24 残差 | +0.5439 | +0.4796 | +0.3996 | −0.0617 | +0.4398 |
| 季节朴素残差 | +0.3804 | +0.3541 | +0.2840 | −0.3149 | +0.3421 |

**两个结论**：

1. **h=1 残差基本白了**（lag 2、3 就掉到 0.11 / 0.02），比朴素残差（lag1 = 0.3804）干净得多
   —— 模型确实把「相位」学到了
2. **h=24 残差还很脏**（lag 1~3 都在 0.4~0.54，形状与朴素残差几乎一样）
   —— 24 小时外的预测，本质上退化回了「t−24 那个朴素解」

> **一句话**：**h=1 的残差 ACF 用来判断「模型有没有学到日内结构」；
> h=24 的残差 ACF 用来判断「模型在远期是不是已经放弃治疗」。**
"""
    ),
    code(T9_CODE),
    md(
        """
### 7.5.1 难点深挖：统计显著 ≠ 实际重要（n 一大，什么都能显著）

**为什么难**：t 检验的公式是 $t = \\bar e / (s / \\sqrt n)$。`||` 里唯一的自由度是 $n$。
我们手里有 **1714 个**测试点、**41136 个**残差 —— 哪怕偏差只有 1 kW，
只要残差标准差别太大，t 值也会「显著」。

**实测**：

| 残差 | 均值 | 占负荷均值 | std | t |
|---|---|---|---|---|
| h=1 | −1.0629 kW | −0.22% | 18.8542 | **−2.3333** |
| h=24 | −2.1429 kW | −0.44% | 24.2166 | **−3.6624** |

两个 t 的绝对值都超过了 1.96 的临界值，**「统计上显著」**。但偏差只有 **−0.22% / −0.44%** ——
对负荷预测来说，**完全不用管**（传感器自身的漂移都比这个数大）。

再看按小时分组的形状：

```
按小时残差均值：最低 −6.5399 kW（4 点）| 最高 +2.9676 kW（10 点）| 极差 9.5075
按小时残差 std ：最稳 13.7081（13 点）| 最不稳 26.7899（20 点）| 后者是前者的 1.95 倍
```

**这是两条完全不同的信息**：

- **残差均值随小时的形状**（午后高估、凌晨低估，极差 9.5 kW）→ 系统性偏差，
  说明还漏了变量（比如温度、节假日），**下一轮迭代该加特征**
- **残差标准差随小时的形状**（20 点的波动是 13 点的 1.95 倍）→ **异方差**，
  说明晚间高峰难预测；这决定了「置信区间不能做成等宽的」

**判定规则**：

> 报残差诊断时，**必须同时给「量级」和「p 值」**。判断顺序是：
>
> 1. 先看**偏差占目标的比例**（< 1% 就基本可以忽略）
> 2. 再看**按小时 / 按天分组的形状**（有形状 = 有可用信息）
> 3. 最后才看 t 值（它只能告诉你「这点偏差不是随机噪声」，不能告诉你「这点偏差要紧」）
"""
    ),
    code(T10_CODE),
    md(
        """
## 易错点清单

1. **`MAPE` 的分母是真值**，真值接近 0 时爆炸。本数据只改 25% 的点，MAPE 就从 6.69% 涨到 2481.95%（**370.8 倍**）。
2. **光伏 / 风速 / 轻载场景禁用 MAPE**，用 `WAPE`（分母换成总量）或 `sMAPE`（分母加对称项，上界 200%）。
3. **`RMSE` 不是 `MAE` 开方**，是 `MSE` 开方。`MAE` 开方得到的是一个没有名字的数。
4. **`RMSE ≥ MAE` 恒成立**，看到 `RMSE < MAE` 一定是代码写错了。
5. **`RMSE/MAE` 要一起报**：1.286676 说明误差近似高斯；> 1.4 先查离群点，别急着调模型。
6. **验证段必须按时间切**（本例取训练段末 15%），随机切会让早停看到未来。
7. **`clip_grad_norm_` 必须在 `backward()` 之后、`opt.step()` 之前**；它的**返回值**是裁剪前范数，别丢。
8. **`sched.step()` 在 `opt.step()` 之后**调（PyTorch 2.x 的纪律）。
9. **`stopped_epoch = best_epoch + patience`** 是恒等式（实测 14 + 5 = 19）；两者混用是最常见 bug。
10. **早停只保存权重是错的**，还要 `break` 掉循环 —— 否则你永远不知道「最好」出现在第几轮。
11. **递归预测的输入必须重新标准化**：预测值是 kW，接回窗口前要 `(p - MUY) / SD`，忘了就是量纲灾难。
12. **递归误差的方差放大指数远小于 0.5**（实测 h^0.0861），因为模型对单点输入的敏感度 $\\alpha_k \\ll 1$。
13. **直接多输出与递归各有胜负**：整体 17.035627 vs 17.266384（直接赢），但 h=1 是 15.1762 vs **14.9523**（递归赢）—— 别背结论，要实测。
14. **`autocorr` 与手写 ACF 分母不同**，lag 小的时候几乎一样，lag 大时必须说明口径。
15. **`h=24` 残差的自相关还很高（+0.5439）**，说明远期预测退化成朴素解 —— 这是结构问题，不是调参问题。
16. **统计显著不代表重要**：−0.22% 的偏差也能有 t = −2.33，报告里必须同时给量级。
17. **残差按小时分组的「均值形状」和「标准差形状」是两件事**：前者是偏差，后者是异方差。
18. **所有涉及随机性的指标（torch 训出来的 MAE）都不能卡死值**，要用区间断言；只有朴素基线/指标函数才可以。

## 本章小结

**一句话**：评估不是「算一个数」，而是**用四个数 + 三张图 + 一个残差诊断**回答
「模型学到了什么、还差什么」。

**可背的四个判定规则**：

| 场景 | 规则 |
|---|---|
| MAPE 能不能用 | 真值可能接近 0 → **禁用**，换 WAPE / sMAPE |
| 误差有没有离群点 | `RMSE/MAE`：≈1.29 高斯；> 1.4 先查异常值 |
| 训练该停在哪 | `stopped_epoch = best_epoch + patience`；训练损失还在降、验证损失抬头 = 停 |
| 递归还值不值得用 | 看 `std(h)` 的放大指数：接近 0.1 → 很稳，没必要上 Seq2Seq |

**训练循环模板（可以背下来直接抄）**：

```python
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)
best_val, best_epoch, wait, stopped_epoch = float("inf"), -1, 0, MAX_EPOCH
for ep in range(1, MAX_EPOCH + 1):
    model.train(); opt.zero_grad()
    loss = ((model(X_fit) - Y_fit) ** 2).mean()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 裁剪 + 记录
    opt.step(); sched.step()                                          # 顺序不能反
    model.eval()
    with torch.no_grad():
        val_loss = float(((model(X_val) - Y_val) ** 2).mean())
    if val_loss < best_val - 1e-9:
        best_val, best_epoch, wait = val_loss, ep, 0
    else:
        wait += 1
        if wait >= PATIENCE:
            stopped_epoch = ep
            break                                                     # 别忘了 break
```

**下一步**：本章的模型都很朴素（MLP）。ch08 会给出**真正有资格当基线的对手** ——
朴素基线家族、特征工程 + LightGBM、以及 TCN / ProbSparse Transformer，
并把它们和本章的 24 步 MAE 放到同一张表里排序。
"""
    ),
]

EXERCISE = [
    md(
        """
# ch07 训练与评估（练习版）

> 补全所有 `____`，让每个代码块末尾的 `assert` 全绿。
> 数据：`data/load_curve.csv` → A 台区 8760 h；窗口 72、步长 24、起点从 168 开始、按时间 8:2 切分。
>
> **每题都跟了 `# 提示：`**，照着提示能独立做完。
> 如果 assert 的数字对不上，**先回头看上一步的输出**，别急着改 assert —— 那些数字都是实跑出来的。
> 注意：涉及 torch 训练的断言都是**区间**（不是死值），因为训练结果依赖随机数种子；
> 只有**朴素基线**和**指标函数本身**是确定性的，那几个数才能卡死。

## 任务清单

| 任务 | 主题 | 挖空 | 对应讲解小节 |
|---|---|---|---|
| 1 | 构造 72→24 样本 + 8:2 时间切分 | 3 | §7.1 |
| 2 | 手写 MSE / MAE / RMSE / MAPE + 对账 | 4 | §7.1 |
| 3 | MAPE 零点陷阱 + sMAPE / WAPE | 3 | §7.1.1 |
| 4 | 早停 + 梯度裁剪 + StepLR 训练循环 | 3 | §7.2 |
| 5 | 训练曲线可视化（loss / grad / lr） | 1 | §7.2.1 |
| 6 | 递归 vs 直接多输出 | 4 | §7.3 |
| 7 | 逐步 MAE 曲线 + 方差放大 | 2 | §7.3.1 |
| 8 | 预测曲线可视化（整段 + 单日） | 2 | §7.4 |
| 9 | 残差 ACF（手写 + 对账） | 2 | §7.5 |
| 10 | 按小时残差 + 显著性检验 | 2 | §7.5.1 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md("## 任务 1：构造 72 步窗口与 8:2 时间切分（§7.1）"),
    code(T1_CODE),
    md("## 任务 2：手写四个指标，并与 sklearn 对账（§7.1）"),
    code(T2_CODE),
    md("## 任务 3：MAPE 的零点陷阱与两种修法（§7.1.1）"),
    code(T3_CODE),
    md("## 任务 4：训练循环模板 —— 早停 / 梯度裁剪 / lr 调度（§7.2）"),
    code(T4_CODE),
    md("## 任务 5：把训练过程画出来（§7.2.1）"),
    code(T5_CODE),
    md("## 任务 6：递归预测 vs 直接多输出（§7.3）"),
    code(T6_CODE),
    md("## 任务 7：多步误差累积曲线（§7.3.1）"),
    code(T7_CODE),
    md("## 任务 8：预测曲线可视化（§7.4）"),
    code(T8_CODE),
    md("## 任务 9：残差自相关（§7.5）"),
    code(T9_CODE),
    md("## 任务 10：按小时分组的残差与显著性（§7.5.1）"),
    code(T10_CODE),
    md(
        """
## 自查清单（能不看笔记答出来才算过）

- [ ] 10 个代码块的 assert 全部通过
- [ ] `MAPE` 的分母是什么？为什么它会「爆」？本数据实测涨了多少倍？
- [ ] `sMAPE` 和 `WAPE` 分别怎么修分母？各自的上界是多少？
- [ ] `RMSE ≥ MAE` 为什么恒成立？`RMSE/MAE = 1.2867` 说明误差是什么形状？
- [ ] 训练循环里 `clip_grad_norm_` 放在哪两步之间？它的返回值是什么？
- [ ] `sched.step()` 和 `opt.step()` 谁先谁后？
- [ ] `best_epoch` 与 `stopped_epoch` 的恒等式是什么？为什么成立？
- [ ] 递归预测把预测值接回窗口前，必须做哪一步变换？
- [ ] 递归误差方差的放大指数实测是多少？为什么远小于 0.5？
- [ ] 直接多输出一定比递归好吗？给出本章的实测数字。
- [ ] 手写 ACF 与 `Series.autocorr` 的分母差在哪？
- [ ] h=1 残差与 h=24 残差的 ACF 差多少？这说明什么？
- [ ] 「统计显著」和「实际重要」在本章分别对应哪两个数字？

## 关键真值对照（做完再对）

| 量 | 值 |
|---|---|
| 样本数 / 切分点 / 训练 / 测试 | 8569 / 6855 / 6855 / 1714 |
| 训练段目标均值 / 测试段目标均值 | 452.674164 / 483.2289 |
| 测试段时段 | 2025-10-20 15:00 ~ 2025-12-31 00:00 |
| MSE / MAE / RMSE | 1802.744577 / 32.998794 / 42.458740 |
| MAPE / sMAPE / WAPE（季节朴素） | 6.693991 / 6.682859 / 6.828811 |
| `RMSE / MAE` | 1.286676 |
| 零点陷阱后 MAE / RMSE | 143.890302 / 243.244359 |
| 零点陷阱后 MAPE / sMAPE / WAPE | 2481.953943 / 54.016410 / 39.555870 |
| MAPE 放大倍数 | 370.8× |
| 单点（0.05 vs 5.0）MAPE / sMAPE | 9900.0000% / 196.0396% |
| 早停：best_epoch / stopped_epoch / patience | 14 / 19 / 5 |
| 训练损失首末（最后一轮 0.1084） | 1.1775 → 0.1084 |
| 验证损失首末（最低 0.104366） | 0.6464 → 0.1099 |
| 梯度范数最大值（阈值 1.0）/ 被裁轮数 | 3.485 / 7 |
| 学习率 首 / 末 | 0.002 → 0.0005 |
| 递归 24 步 MAE（h1 / h24） | 17.266384（14.952325 / 19.115711） |
| 直接多输出 24 步 MAE（h1 / h24） | 17.035627（15.176231 / 18.490346） |
| 季节朴素 24 步 MAE（h1 / h24） | 32.998794（33.0215 / 33.1928） |
| 递归误差 std：h1 / h8 / h24 | 11.5343 / 13.6496 / 15.0208 |
| 方差放大幂律指数 | h^0.0861 |
| 朴素残差 ACF(1) / ACF(24) | +0.3804 / −0.3149 |
| h=1 残差 ACF(1/2/3/24/168) | +0.1543 / +0.1111 / +0.0218 / −0.0011 / +0.1646 |
| h=24 残差 ACF(1/2/3/24/168) | +0.5439 / +0.4796 / +0.3996 / −0.0617 / +0.4398 |
| h=1 残差 均值 / 占比 / t | −1.0629 kW / −0.22% / −2.3333 |
| h=24 残差 均值 / 占比 / t | −2.1429 kW / −0.44% / −3.6624 |
| 按小时残差均值 极值（小时） | −6.5399（4 点）/ +2.9676（10 点） |
| 按小时残差 std 极值（小时） | 13.7081（13 点）/ 26.7899（20 点） |
"""
    ),
]

if __name__ == "__main__":
    report([build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)])
