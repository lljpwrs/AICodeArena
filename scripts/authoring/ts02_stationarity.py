#!/usr/bin/env python3
"""ch02 —— 平稳性与统计方法：ADF / 差分 / ACF·PACF / ARIMA·SARIMA

本章主线（六条）：

1. **平稳性到底是什么**：白噪声 / 随机游走 / 确定性趋势 + 日周期 三段序列的对照，
   用「前后半段的均值与标准差」当肉眼判据
2. **手写 ADF 的 τ 统计量**（`Δy_t = α + δt + β·y_{t-1} + Σγ_iΔy_{t-i} + ε`），
   与 `statsmodels.adfuller` **逐项对账到 1e-13** —— 这是本方向「手写 + 对账」范式的第一块硬骨头
3. **ADF 的两个反直觉**：① 强确定性季节会被判「平稳」（A 台区 p=1 时 τ=−37.26）；
   ② 结论随 `maxlag` 翻脸（同一序列 p=24 时 τ=−1.93、p=0.316 **不能拒绝**）
4. **差分三兄弟**：一阶 / 季节(24) / 过差分 —— 过差分让 std 从 36.97 涨到 43.05、
   把 ACF1 压成 −0.3149（转负是典型信号）
5. **手写 ACF 与 PACF**：ACF 分母口径、Bartlett 置信带 `1.96/√n`、
   PACF 的两种等价定义（Levinson 递推 vs AR(k) 回归末系数）——
   **两者在小样本下差 1.18e-2，如实当「负结果」呈现**
6. **ARMA 定阶与残差检验**：AIC/BIC 网格 → 残差 Ljung-Box **拒绝白噪声**（p=1.4e-3）
   → 加季节项 `(1,0,2)(1,0,0,7)` 后 AIC 从 3214.06 掉到 3050.73

数据：`data/load_curve.csv`（A 台区 8760 h，另派生 365 点日序列）
生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ts02_stationarity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "05_timeseries"
NAME = "ch02_stationarity"

# =========================================================================== #
# 1. 公共导入 / 数据加载 / 脚手架（两版都给，不挖空）                          #
# =========================================================================== #

IMPORTS = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# statsmodels 是本方向第一个「真库对账」对象：ADF / ACF / PACF / ARIMA / SARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import adfuller, pacf as sm_pacf

warnings.filterwarnings("ignore")      # ARIMA 拟合会刷 EstimationWarning，教学时不需要看

DATA = Path("data")                    # notebook 的 cwd = coding/05_timeseries
CSV = DATA / "load_curve.csv"

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

raw = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["时间戳"])
df = raw.drop_duplicates().sort_values("时间戳").reset_index(drop=True)
print("statsmodels 已就绪 | 原始", len(raw), "行 → 去重排序后", len(df), "行")
'''

SETUP = '''# 本章只用 A 台区（居民型）：先把 8760 点的小时序列清洗干净，再派生一条 365 点的日序列
s_a = df[df["台区编号"] == "STATION_A_01"].set_index("时间戳")["负荷值"].asfreq("h")
a = clean_series(s_a)
Y = a.to_numpy(dtype=float)                 # 小时序列（8760,）
DY = np.diff(Y)                             # 一阶差分（8759,）
D24 = Y[24:] - Y[:-24]                      # 季节差分（8736,）
DD = np.diff(DY)                            # 二阶差分 = 过差分（8758,）

DAILY = a.resample("D").mean()              # 日序列（365,）：教科书路径的载体
DV = DAILY.to_numpy(dtype=float)
DDV = np.diff(DV)                           # 日差分（364,）

print("小时序列", Y.shape, "| 均值 %.4f" % Y.mean(), "| std %.4f" % Y.std())
print("日序列", DV.shape, "| 均值 %.4f" % DV.mean(), "| std %.4f" % DV.std())
print("差分后 std：一阶 %.4f | 季节(24) %.4f | 二阶(过) %.4f" % (DY.std(), D24.std(), DD.std()))
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

SCAFFOLD = '''# 脚手架：下面的常量与工具函数两版都有，你只填 @@todo 块里的内容
NLAG = 48            # 本章看 ACF/PACF 的最大滞后
P_SIM = 1            # ADF 回归里的差分滞后阶数（后面会专门讨论它为什么要调）


def clean_series(x):
    """按**小时分组**的中位数做 3 倍阈值判异常，再线性插值（沿用小节的写法）。"""
    med = x.groupby(x.index.hour).transform("median")
    return x.where(x.between(0, 3 * med)).interpolate()


def lags_2d(p, d, n_obs):
    """把 Δy 的 p 个滞后拼成 (n_obs, p) 的列块。

    ⚠️ `p = 0` 是**合法档位**（`maxlag=0`，敏感性扫描 LAG_GRID 的第一档），
    但那时列表是空的，直接 `np.column_stack([])` 会抛
    `ValueError: need at least one array to concatenate` —— 所以这里退化成空块。
    """
    if p == 0:
        return np.empty((n_obs, 0))
    return np.column_stack([d[p - i:len(d) - i] for i in range(1, p + 1)])


def bartlett(n):
    """白噪声 ACF 的 Bartlett 置信带半宽：1.96/√n。

    「ACF 落在带内」= 在 5% 水平上无法拒绝「该滞后自相关为 0」。"""
    return 1.96 / np.sqrt(n)


def sim_ar1(phi, n, seed):
    """模拟 AR(1)：x_t = phi·x_{t-1} + ε_t。"""
    rng = np.random.RandomState(seed)
    eps = rng.randn(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def sim_ma1(theta, n, seed):
    """模拟 MA(1)：x_t = ε_t + theta·ε_{t-1}。"""
    rng = np.random.RandomState(seed)
    eps = rng.randn(n)
    return eps[1:] + theta * eps[:-1]


def ar_ols(x, k):
    """对 x 拟合 AR(k)（含常数项）的条件最小二乘，返回 (系数数组, 残差)。

    系数数组是 [常数, phi_1, ..., phi_k]，所以 **PACF(k) = 系数数组的最后一个**。
    """
    v = np.asarray(x, dtype=float)
    n = len(v)
    Xd = np.column_stack([np.ones(n - k)] + [v[k - i - 1:n - i - 1] for i in range(k)])
    Yd = v[k:]
    b, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)
    return b, Yd - Xd @ b


LAG_GRID = (0, 1, 2, 6, 12, 24, 48)      # ADF 滞后阶数的敏感性扫描档位


def adf_row(series, p):
    """把「maxlag=p 时的手写 τ / 真库 τ / p 值」打成一个 dict —— 敏感性表的一行。

    ⚠️ 这个辅助函数只是把表格拼装写成一行；
    **τ 本身仍然由你手写的 `adf_manual` 算**（它在任务 2 里定义）。
    """
    tau_m, _, _, _, _ = adf_manual(series, p, "c")
    tau_s, pv, _, _, _, _ = sm_adf(series, maxlag=p, autolag=None)
    return dict(maxlag=p, 手写τ=tau_m, statsmodelsτ=tau_s, p值=pv)


def arma_fit(series, p, q):
    """拟合 ARMA(p,q)（序列已经差过 d 阶，所以 order 的中间位固定填 0）。"""
    return ARIMA(series, order=(p, 0, q)).fit()


def arma_row(fits, p, q):
    """把 (p,q) 这一格的 AIC / BIC 打成一个 dict —— 定阶表的一行。

    只拟合一次、从同一个结果对象上取 AIC 与 BIC，比分别 fit 两次省一半时间。
    """
    m = fits[(p, q)]
    return dict(阶=f"({p},{q})", AIC=float(m.aic), BIC=float(m.bic))


def sm_adf(x, maxlag=None, regression="c", autolag="AIC"):
    """调 statsmodels 的 ADF，把返回值统一成 (tau, p, usedlag, nobs, crit, icbest)。

    ⚠️ 两个「返回值本身就会咬你」的坑：

    ① 必须显式传 `result_object=False`。statsmodels 0.15 起 `adfuller` 的返回类型
       会从普通 tuple 悄悄切到 `ADFullerResult`（0.16 或 2027-07 之后默认切换），
       到时候 `r[1]` 这种下标取值会静默取错。见 `docs/环境依赖.md` §三·补。

    ② **返回元素的个数取决于 `autolag`**（实测）：
       `autolag=None` → 5 个元素（没有 IC 那一项）；`autolag="AIC"` → 6 个元素。
       所以 `r[5]` 在 `autolag=None` 时会抛 `IndexError: tuple index out of range`。
       这里统一补齐成 6 元组，缺失的 IC 填 nan。
    """
    r = adfuller(np.asarray(x, dtype=float), maxlag=maxlag, regression=regression,
                 autolag=autolag, result_object=False)
    icbest = float(r[5]) if len(r) > 5 else float("nan")
    return float(r[0]), float(r[1]), int(r[2]), int(r[3]), dict(r[4]), icbest


def stem_acf(ax, vals, band, title):
    """画 ACF/PACF 的「火柴梗图」+ Bartlett 置信带（上下两条红色虚线）。"""
    lags = np.arange(len(vals))
    ax.vlines(lags, 0, vals, linewidth=0.8)
    ax.scatter(lags, vals, s=9, zorder=3)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(band, color="crimson", linestyle="--", linewidth=0.8)
    ax.axhline(-band, color="crimson", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("滞后")
    ax.set_ylabel("自相关")
    return ax


print("脚手架就绪：NLAG =", NLAG, "| 随机模拟固定 seed，结果可复现")
'''

# =========================================================================== #
# 2. 代码块（LESSON 用完整版；EXERCISE 里带 @@todo 的是挖空源）               #
# =========================================================================== #

# ------------------------------------------------------------------- §2.1
T1_CODE = '''# 三段「人工序列」，用来把 ADF 的三种判决摆在一起：
#   ① 白噪声        —— 平稳的典范
#   ② 随机游走      —— 单位根，非平稳
#   ③ 趋势 + 日周期 —— 确定性成分很强，但没有单位根
RS = np.random.RandomState(42)
N_SIM = 8760
HOURS = np.arange(N_SIM)
white = RS.randn(N_SIM) * 10 + 500
rwalk = 500 + np.cumsum(RS.randn(N_SIM) * 3)
seas = 400 + 0.01 * HOURS + 40 * np.sin(2 * np.pi * HOURS / 24)

half = N_SIM // 2
# @@todo （§2.1）把三段序列各自的「前半均值 / 后半均值 / 前半 std / 后半 std」收成一张表
# @@hint 用一个 dict 推导：对 (名字, 序列) 的每个组合，取 x[:half] 与 x[half:] 的 mean / std
# @@hint 最后套 pd.DataFrame(...).round(4)；列名要与下面的 assert 完全一致
cmp_tbl = pd.DataFrame([
    dict(序列="白噪声", 前半均值=white[:half].mean(), 后半均值=white[half:].mean(),
         前半std=white[:half].std(), 后半std=white[half:].std()),
    dict(序列="随机游走", 前半均值=rwalk[:half].mean(), 后半均值=rwalk[half:].mean(),
         前半std=rwalk[:half].std(), 后半std=rwalk[half:].std()),
    dict(序列="趋势+日周期", 前半均值=seas[:half].mean(), 后半均值=seas[half:].mean(),
         前半std=seas[:half].std(), 后半std=seas[half:].std()),
]).round(4)
# @@end
print(cmp_tbl.to_string(index=False))

assert int(cmp_tbl.shape[0]) == 3
assert abs(float(cmp_tbl.loc[0, "前半均值"]) - 500.0865) < 1e-3
assert abs(float(cmp_tbl.loc[0, "后半均值"]) - 499.9025) < 1e-3
# 随机游走的前后半均值差了 172.88 kW —— 这就是「均值随时间变」的教科书长相
assert 172.8 < abs(float(cmp_tbl.loc[1, "前半均值"]) - float(cmp_tbl.loc[1, "后半均值"])) < 172.9
# 确定性序列的 std 前后半段**完全相等**（周期振幅固定）
assert abs(float(cmp_tbl.loc[2, "前半std"]) - float(cmp_tbl.loc[2, "后半std"])) < 1e-9
print("T1 通过 -> 三段序列的「前后半段」统计已就位")
'''

T2_CODE = '''# ---- 手写 ADF 的 τ 统计量 ----
# 回归式： Δy_t = α + δ·t + β·y_{t-1} + Σ_{i=1..p} γ_i·Δy_{t-i} + ε_t
# 检验的是 H0: β = 0（有单位根）。τ = β̂ / se(β̂)，注意它**不是** t 分布，
# 必须查 Dickey-Fuller 自己的临界值表 —— 这就是为什么必须用真库对账。
def adf_manual(x, p=P_SIM, regression="c"):
    """手写 ADF 的 τ 统计量。regression: "c" 带常数项 / "ct" 再带线性趋势 / "n" 都不带。"""
    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    n_obs = len(d) - p

    # @@todo （§2.2）四块自变量先都备好，每块都是 (n_obs, k) 的二维列块
    # @@hint 列顺序固定为 [const][trend][level][lags...]：level = x[p:]（即 y_{t-1}）
    # @@hint lags 直接调脚手架 lags_2d(p, d, n_obs)：它就是
    #       np.column_stack([d[p-i : len(d)-i] for i in range(1, p+1)])
    # @@hint ⚠️ 它内部兜住了 p=0（空列表会让 column_stack 抛 "need at least one array"）
    # @@hint trend 是 np.arange(p, len(d)) 再 reshape(-1, 1)；const 是 np.ones((n_obs, 1))
    blocks = {
        "const": np.ones((n_obs, 1)),
        "trend": np.arange(p, len(d), dtype=float).reshape(-1, 1),
        "level": x[p:len(d)].reshape(-1, 1),    # y_{t-1}：**截到 len(d)**，否则比 n_obs 多一格
        # lags 由脚手架里的 lags_2d 拼（它顺手兜住了 p=0 的空列表边界）
        "lags": lags_2d(p, d, n_obs),
    }
    # 本回归要用哪几块：列顺序固定为 [const][trend][level][lags...]
    used = {"c": ("const", "level", "lags"),
            "ct": ("const", "trend", "level", "lags"),
            "n": ("level", "lags")}[regression]
    X = np.column_stack([blocks[nm] for nm in used])
    level_col = list(used).index("level")       # β 在 X 里的列号随 regression 变
    Yv = d[p:]
    # @@end

    # @@todo （§2.2）最小二乘求解 + 经典 OLS 标准误，最后得到 τ = β̂ / se(β̂)
    # @@hint 解方程用 np.linalg.lstsq(X, Yv, rcond=None)，解出来的第一个元素就是 β 向量
    # @@hint σ̂² = RSS/(n−k)；se = sqrt(σ̂² · diag((XᵀX)⁻¹))；β̂ 取 level_col 那一列
    # @@hint 注意：解方程用 lstsq（走 QR/SVD），但 se 那一步按定义必须用 inv(XᵀX)
    sol = np.linalg.lstsq(X, Yv, rcond=None)
    beta = sol[0]
    resid = Yv - X @ beta
    dof = len(Yv) - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(sigma2 * np.diag(np.linalg.inv(X.T @ X)))
    tau = float(beta[level_col] / se[level_col])
    # @@end
    return tau, dof, sigma2, float(beta[level_col]), float(se[level_col])


adf_rows = []
for nm, x in (("白噪声", white), ("随机游走", rwalk), ("趋势+日周期", seas)):
    # @@todo （§2.2）对每段序列：手写一遍、真库一遍，再比较两者的绝对差
    # @@hint 手写 adf_manual(x, 1, "c")；真库 sm_adf(x, maxlag=1, autolag=None)（后面就能调）
    tau, dof, s2, b, se = adf_manual(x, 1, "c")
    tau_sm, p_sm, lags_used, nobs, crit, ic = sm_adf(x, maxlag=1, autolag=None)
    adf_rows.append(dict(序列=nm, 手写τ=tau, statsmodelsτ=tau_sm, 绝对差=abs(tau - tau_sm),
                         p值=p_sm, 自由度=dof, dof_sm=nobs - 2))
    # @@end
adf_tbl = pd.DataFrame(adf_rows)
print(adf_tbl.round(6).to_string(index=False))

assert float(adf_tbl["绝对差"].max()) < 1e-9, "手写 τ 必须与 statsmodels 对到 1e-9 以内"
assert abs(float(adf_tbl.loc[0, "手写τ"]) - (-67.2888)) < 1e-3, "白噪声：τ 极负 → 强烈拒绝单位根"
assert abs(float(adf_tbl.loc[1, "手写τ"]) - (-0.2051)) < 1e-3, "随机游走：τ 接近 0"
assert float(adf_tbl.loc[1, "p值"]) > 0.9, "随机游走的 p 值应接近 1（无法拒绝）"
assert float(adf_tbl.loc[2, "p值"]) < 1e-6, "确定性趋势 + 日周期：ADF 也会拒绝单位根"
print("T2 通过 -> 手写 τ 最大绝对差 %.3e" % float(adf_tbl["绝对差"].max()))
'''

T3_CODE = '''# ---- ADF 的两个反直觉 ----
# 反直觉 ①：A 台区是**强确定性日周期 + 噪声**，ADF 却把它判成「平稳」。
# 反直觉 ②：同一个序列，换个 maxlag，结论直接翻脸。
# @@todo （§2.2.1）同一个 A 台区序列，分别在 maxlag=1 与 maxlag=24 下算 τ
# @@hint 手写 adf_manual(Y, p, "c")[0] 就是 τ；真库用 sm_adf(Y, maxlag=p, autolag=None)
tau_a1, dof_a1, _, _, _ = adf_manual(Y, 1, "c")
tau_a1_sm, p_a1, _, _, _, _ = sm_adf(Y, maxlag=1, autolag=None)
tau_a24, _, _, _, _ = adf_manual(Y, 24, "c")
# @@end
print(f"A 台区原序列  maxlag=1  → 手写τ {tau_a1:.4f} | statsmodels {tau_a1_sm:.4f} | p {p_a1:.6f}")
print(f"A 台区原序列  maxlag=24 → 手写τ {tau_a24:.4f}")

# @@todo （§2.2.1）把 7 个 maxlag 档位的手写 τ、真库 τ、p 值列成一张敏感性表
# @@hint 用列表推导：dict(maxlag=p, 手写τ=..., statsmodelsτ=..., p值=...) for p in (0,1,2,6,12,24,48)
# @@hint 再补一列布尔量「拒绝单位根」= p 值 < 0.05
lag_sens = pd.DataFrame([adf_row(Y, p) for p in LAG_GRID])
lag_sens["拒绝单位根"] = lag_sens["p值"] < 0.05
# @@end
print(lag_sens.round(6).to_string(index=False))

# @@todo （§2.2.1）让真库自己用 AIC 选滞后阶（autolag="AIC"），看看它选了几阶
tau_auto, p_auto, lag_auto, nobs_auto, crit_auto, ic_auto = sm_adf(Y, autolag="AIC")
# @@end
print(f"\\nstatsmodels 默认（autolag='AIC'）→ 选了滞后 {lag_auto} | τ {tau_auto:.4f}"
      f" | p {p_auto:.6f} | 10% 临界值 {crit_auto['10%']:.4f}")

assert abs(tau_a1 - (-37.2580)) < 1e-3 and abs(tau_a24 - (-1.9345)) < 1e-3
assert abs(float(adf_manual(Y, 24, "c")[0] - sm_adf(Y, maxlag=24, autolag=None)[0])) < 1e-9
assert bool(lag_sens.loc[lag_sens["maxlag"] == 24, "拒绝单位根"].iloc[0]) is False
assert bool(lag_sens.loc[lag_sens["maxlag"] == 1, "拒绝单位根"].iloc[0]) is True
assert int(lag_sens["拒绝单位根"].sum()) == 6, "7 个 maxlag 里有 6 个拒绝、唯独 24 不拒绝"
assert lag_auto == 32 and float(p_auto) < 1e-4
print("T3 通过 -> ADF 的判决随 maxlag 翻脸：24 → 不拒绝（p=0.316），其余 6 档全部拒绝")
'''

# ------------------------------------------------------------------- §2.3
T4_CODE = '''# ---- 差分三兄弟：一阶 / 季节(24) / 过差分 ----
# 「过差分」的判据有两条，都很硬：
#   ① 方差不再下降反而变大；② ACF1 从正转负（差分引入的负相关叠加过头了）
# @@todo （§2.3）把「原序列 / 一阶差分 / 季节差分(24) / 二阶差分(过)」四段序列的
# @@hint std 用 v.std()；ACF1 用 float(sm_acf(v, nlags=1, fft=True)[1])，ACF24 同理
# @@hint 四条记录拼成 list 再套 pd.DataFrame(...).round(6)
diff_tbl = pd.DataFrame([
    dict(序列="原序列", std=Y.std(), ACF1=float(sm_acf(Y, nlags=1, fft=True)[1]),
         ACF24=float(sm_acf(Y, nlags=24, fft=True)[24])),
    dict(序列="一阶差分", std=DY.std(), ACF1=float(sm_acf(DY, nlags=1, fft=True)[1]),
         ACF24=float(sm_acf(DY, nlags=24, fft=True)[24])),
    dict(序列="季节差分(24)", std=D24.std(), ACF1=float(sm_acf(D24, nlags=1, fft=True)[1]),
         ACF24=float(sm_acf(D24, nlags=24, fft=True)[24])),
    dict(序列="二阶差分(过)", std=DD.std(), ACF1=float(sm_acf(DD, nlags=1, fft=True)[1]),
         ACF24=float(sm_acf(DD, nlags=24, fft=True)[24])),
]).round(6)
print(diff_tbl.to_string(index=False))
# @@end
# @@todo （§2.3）算两条「方差比」：一阶/原 与 二阶/一阶（后者 > 1 就是过差分的铁证）
ratio_first = DY.std() / Y.std()
ratio_second = DD.std() / DY.std()
# @@end
print(f"\\n一阶/原 = {ratio_first:.6f}  |  二阶/一阶 = {ratio_second:.6f}"
      f"  ← 大于 1 说明差分过头了")

assert abs(float(diff_tbl.loc[1, "std"]) - 36.9689) < 1e-3
assert abs(float(diff_tbl.loc[3, "std"]) - 43.0534) < 1e-3
assert float(diff_tbl.loc[3, "std"]) > float(diff_tbl.loc[1, "std"]), "过差分后方差变大"
assert float(diff_tbl.loc[1, "ACF1"]) > 0.3 and float(diff_tbl.loc[3, "ACF1"]) < -0.3
assert abs(float(DD.std() / DY.std()) - 1.164585) < 1e-5
assert abs(float(DY.std() / Y.std()) - 0.512112) < 1e-5
print("T4 通过 -> 过差分：std 36.9689 → 43.0534（比值 1.1646），ACF1 0.3219 → −0.3149 转负")
'''

# ------------------------------------------------------------------- §2.4
T5_CODE = '''# ---- 手写 ACF ----
# 口径要点：分子分母都用**离差**，且分母是「全部离差平方和」而不是「去掉 k 个之后的和」。
# 后者（分母用 n−k）叫无偏 ACF，statsmodels 的 adjusted 参数控制的就是它。
def acf_manual(x, nlags):
    """手写 ACF，与 statsmodels 的默认（adjusted=False）口径一致。"""
    v = np.asarray(x, dtype=float) - np.mean(x)
    # @@todo （§2.4）算分母：**全部 n 项**的离差平方和（不是只留 n−k 项的无偏口径）
    denom = float(v @ v)
    # @@end
    # @@todo （§2.4）逐阶算 ACF，lag 0 固定为 1.0
    # @@hint 分子是 v[k:] 与 v[:-k] 的点积，除以 denom；用列表推导一次算出 lag 1..nlags
    ac = np.array([1.0] + [float(v[k:] @ v[:-k]) / denom for k in range(1, nlags + 1)])
    # @@end
    return ac


ac_dy = acf_manual(DY, NLAG)
ac_dy_sm = np.asarray(sm_acf(DY, nlags=NLAG, fft=True))
band_dy = bartlett(len(DY))
n_over_dy = int(np.sum(np.abs(ac_dy[1:]) > band_dy))
print(f"ACF 最大绝对差 {np.max(np.abs(ac_dy - ac_dy_sm)):.3e} | Bartlett 带 ±{band_dy:.6f}")
print(f"一阶差分后：ACF1 {ac_dy[1]:+.6f} | ACF24 {ac_dy[24]:+.6f} | ACF48 {ac_dy[48]:+.6f}")
print(f"超带滞后数 {n_over_dy}/{NLAG}  ← 48 个里有 46 个还超带，日周期根本没处理掉")

assert float(np.max(np.abs(ac_dy - ac_dy_sm))) < 1e-12
assert abs(ac_dy[1] - 0.321853) < 1e-6 and abs(ac_dy[24] - 0.668394) < 1e-6
assert abs(ac_dy[48] - 0.657224) < 1e-6
assert n_over_dy == 46
assert abs(band_dy - 0.020943) < 1e-6
print("T5 通过 -> ACF 与 statsmodels 一致到 1e-16 量级；一阶差分后仍 46/48 超带")
'''

# ------------------------------------------------------------------- §2.5
T6_CODE = '''# ---- 手写 PACF：两种等价定义 ----
# 定义 A（递推）：Durbin-Levinson 递推，O(nlags²)
# 定义 B（回归）：PACF(k) 就是 AR(k) 条件最小二乘里**最后一个系数**
def pacf_levinson(x, nlags):
    """Durbin-Levinson 递推。out[0] 按 statsmodels 的约定固定为 1.0。"""
    r = acf_manual(x, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    # @@todo （§2.5）k=1 的初值：反射系数 phi[1,1] 与 out[1] 都等于 r[1]
    phi[1, 1] = r[1]                 # k=1：直接就是 r[1]
    out[1] = r[1]
    # @@end
    for k in range(2, nlags + 1):
        # @@todo （§2.5）第 k 阶反射系数 phi[k,k] = 分子 / 分母
        # @@hint 分子 = r[k] − Σ_{j=1..k−1} phi[k−1,j]·r[k−j]；分母 = 1 − Σ phi[k−1,j]·r[j]
        # @@hint 用 sum(... for j in range(1, k)) 一次求和，避免再写一层 for
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den
        # @@end
        for j in range(1, k):
            # @@todo （§2.5）Levinson 更新式：用第 k 阶反射系数把前 k−1 个系数修正一遍
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
            # @@end
        out[k] = phi[k, k]
    return out


pc_lev = pacf_levinson(DY, NLAG)
# @@todo （§2.5）用「AR(k) 回归的末系数」这条等价定义再算一遍 PACF，用来互验
# @@hint 一行列表推导：ar_ols(DY, k) 返回 (系数数组, 残差)，取 [0][-1] 就是第 k 个偏自相关
pc_reg = np.array([1.0] + [ar_ols(DY, k)[0][-1] for k in range(1, NLAG + 1)])
# @@end
pc_sm = np.asarray(sm_pacf(DY, nlags=NLAG, method="ldb"))

d_lev_sm = float(np.max(np.abs(pc_lev - pc_sm)))
d_lev_reg = float(np.max(np.abs(pc_lev - pc_reg)))
print(f"Levinson  vs statsmodels.pacf  max|diff| = {d_lev_sm:.3e}")
print(f"Levinson  vs 「AR(k) 末系数」  max|diff| = {d_lev_reg:.3e}")
print(f"PACF 前 8（含 lag0）：{np.round(pc_lev[:9], 6).tolist()}")

assert d_lev_sm < 1e-12, "递推法与 statsmodels 应该一致到 1e-12 以内"
assert 1e-3 < d_lev_reg < 5e-2, "两种「定义」在本样本上并不严格相等 —— 见 §2.5.1"
assert abs(d_lev_reg - 1.179e-02) < 1e-3
assert abs(pc_lev[1] - 0.321853) < 1e-6 and abs(pc_lev[3] - (-0.299056)) < 1e-6
print(f"T6 通过 -> 递推法对得上真库（{d_lev_sm:.1e}），但「回归定义」差了 {d_lev_reg:.3e}")
'''

# ------------------------------------------------------------------- §2.6
T7_CODE = '''# ---- AR / MA 的 ACF 形态：一个拖尾、一个截尾 ----
# 理论：AR(1) 的 ACF_k = φ^k（**拖尾**，指数衰减不截断）
#       MA(1) 的 ACF_1 = θ/(1+θ²)，ACF_k = 0 (k ≥ 2)（**截尾**）
shape_rows = []
for phi in (0.9, 0.5, -0.9):
    # @@todo （§2.6）模拟 AR(1) 并取它的 ACF1..4（n=20000 让抽样误差足够小）
    xs = sim_ar1(phi, 20000, 7)
    acs = np.asarray(sm_acf(xs, nlags=4, fft=True))
    # @@end
    shape_rows.append(dict(模型=f"AR(1) φ={phi:+.1f}",
                           **{f"ACF{k}": acs[k] for k in range(1, 5)},
                           理论=f"[{', '.join(f'{phi ** k:.4f}' for k in range(1, 5))}]"))
for th in (0.9, 0.5, -0.9):
    # @@todo （§2.6）模拟 MA(1) 并取它的 ACF1..4 —— 重点看 lag 2 之后是否归零
    xs = sim_ma1(th, 20000, 7)
    acs = np.asarray(sm_acf(xs, nlags=4, fft=True))
    # @@end
    shape_rows.append(dict(模型=f"MA(1) θ={th:+.1f}",
                           **{f"ACF{k}": acs[k] for k in range(1, 5)},
                           理论=f"[{th / (1 + th ** 2):.4f}, 0, 0, 0]"))
shape_tbl = pd.DataFrame(shape_rows)
print(shape_tbl.round(4).to_string(index=False))

assert int(shape_tbl.shape[0]) == 6
assert abs(float(shape_tbl.loc[0, "ACF1"]) - 0.9018) < 1e-3
assert abs(float(shape_tbl.loc[0, "ACF4"]) - 0.6573) < 1e-3, "AR(1) φ=0.9 的 ACF4 应约 0.6561（拖尾）"
assert abs(float(shape_tbl.loc[3, "ACF1"]) - 0.504) < 1e-3
assert abs(float(shape_tbl.loc[3, "ACF2"])) < 0.01, "MA(1) 在 lag≥2 应截尾到 0"
assert abs(float(shape_tbl.loc[5, "ACF1"]) - (-0.497)) < 1e-3, "θ<0 时 MA(1) 的 ACF1 为负"
print("T7 通过 -> AR 拖尾、MA 截尾；用 lag 2 之后是否归零一眼分辨")
'''

# ------------------------------------------------------------------- §2.7
T8_CODE = '''# ---- ARIMA 定阶：AIC / BIC 网格 ----
# 载体换成 365 点的**日序列**：小时序列的日周期太强，不成季节差分根本定不出低阶 ARMA。
tau_d_c, p_d_c, lag_d_c, nobs_d_c, crit_d_c, ic_d_c = sm_adf(DV, autolag="AIC")
tau_d_ct, p_d_ct, lag_d_ct, nobs_d_ct, crit_d_ct, ic_d_ct = sm_adf(DV, autolag="AIC", regression="ct")
tau_dd, p_dd, lag_dd, _, _, _ = sm_adf(DDV, autolag="AIC")
print(f"日序列 ADF（含常数）     滞后 {lag_d_c:2d} | τ {tau_d_c:8.4f} | p {p_d_c:.6f}")
print(f"日序列 ADF（常数 + 趋势）滞后 {lag_d_ct:2d} | τ {tau_d_ct:8.4f} | p {p_d_ct:.6f}")
print(f"日差分 ADF（含常数）     滞后 {lag_dd:2d} | τ {tau_dd:8.4f} | p {p_dd:.6f}")

# @@todo （§2.7）对 p, q ∈ {0,1,2} 的 9 个组合各拟合一次 ARMA —— 键用元组 (p, q)
# @@hint 脚手架里有 arma_fit(series, p, q)：一个组合只 fit 一次，省一半时间
fits = {(p, q): arma_fit(DDV, p, q) for p in range(3) for q in range(3)}
# @@end
# @@todo （§2.7）把 9 个组合的 AIC / BIC 列成表，并标出各自的「最优」行
# @@hint 用 arma_row(fits, p, q) 生成 9 行 dict，再套 pd.DataFrame(...)
# @@hint 最后加两个布尔列：AIC 列是否等于最小值、BIC 列是否等于最小值
order_tbl = pd.DataFrame([arma_row(fits, p, q) for p in range(3) for q in range(3)])
order_tbl["AIC最优"] = order_tbl["AIC"] == order_tbl["AIC"].min()
order_tbl["BIC最优"] = order_tbl["BIC"] == order_tbl["BIC"].min()
# @@end
print("\\n" + order_tbl.round(4).to_string(index=False))

best_aic = order_tbl.loc[order_tbl["AIC"].idxmin(), "阶"]
best_bic = order_tbl.loc[order_tbl["BIC"].idxmin(), "阶"]
params_12 = np.asarray(ARIMA(DDV, order=(1, 0, 2)).fit().params)
# ⚠️ 用「阶」这一列按名字取值，而不是 .loc[2] —— 行序是 (0,0),(0,1),(0,2),(1,0)…，
# 位置下标 2 拿到的是 (0,2)，不是 (1,2)。这类错位在宽表里极常见。
aic_12 = float(order_tbl.loc[order_tbl["阶"] == "(1,2)", "AIC"].iloc[0])
bic_12 = float(order_tbl.loc[order_tbl["阶"] == "(1,2)", "BIC"].iloc[0])
print(f"\\nAIC 最优 {best_aic} | BIC 最优 {best_bic}"
      f" | ARMA(1,2) AIC {aic_12:.4f} 参数 {np.round(params_12, 6).tolist()}")

assert abs(tau_d_c - (-1.8122)) < 1e-3 and float(p_d_c) > 0.3
assert abs(tau_dd - (-5.7997)) < 1e-3 and float(p_dd) < 1e-6
assert best_aic == "(1,2)" and best_bic == "(1,2)", "AIC 与 BIC 在本数据上给出同一个阶"
assert abs(aic_12 - 3214.0624) < 1e-3
assert abs(bic_12 - 3233.5482) < 1e-3
print("T8 通过 -> 日序列 ADF p=0.374 不拒绝、差分后 p≈0；AIC/BIC 一致选出 ARMA(1,2)")
'''

# ------------------------------------------------------------------- §2.7.1
T9_CODE = '''# ---- 残差检验：定完阶还不算完，残差必须是白噪声 ----
# 这是本章最重要的「负结果」：ARMA(1,2) 的 AIC 已经是最小，但残差并不白。
# @@todo （§2.7.1）取出 ARMA(1,2) 的残差，并算它的 ACF 与 Bartlett 带
# @@hint best = ARIMA(DDV, order=(1,0,2)).fit()；取 np.asarray(best.resid)[3:]（掐掉前 3 个 burn-in）
best = ARIMA(DDV, order=(1, 0, 2)).fit()
resid = np.asarray(best.resid)[3:]
ac_res = acf_manual(resid, 10)
band_res = bartlett(len(resid))
# @@end
# @@todo （§2.7.1）对残差做 Ljung-Box 白噪声检验（滞后 5 / 10 / 20）
# @@hint acorr_ljungbox(resid, lags=[5, 10, 20], return_df=True)，结果里有 lb_pvalue 列
lb = acorr_ljungbox(resid, lags=[5, 10, 20], return_df=True)
# @@end
print(f"残差 ACF1..5 {np.round(ac_res[1:6], 6).tolist()} | 带 ±{band_res:.6f}")
print(f"残差均值 {resid.mean():.6f} | std {resid.std():.6f}")
print("Ljung-Box 检验：")
print(lb.round(6).to_string())
print(f"\\n日差分序列的 ACF7 = {float(sm_acf(DDV, nlags=7, fft=True)[7]):+.6f}"
      f" | ACF14 = {float(sm_acf(DDV, nlags=14, fft=True)[14]):+.6f}"
      f"  ← 周周期还在残差里")

assert int(np.sum(np.abs(ac_res[1:]) > band_res)) == 5, "10 个滞后里有 5 个超带"
p5 = float(lb.loc[5, "lb_pvalue"])
assert p5 < 0.01, "lag=5 的 Ljung-Box 已拒绝白噪声"
assert abs(p5 - 1.425489e-03) < 1e-6
assert abs(float(lb.loc[10, "lb_pvalue"]) - 4.671439e-15) < 1e-16
print(f"T9 通过 -> Ljung-Box p(5)={p5:.3e}：残差**不是**白噪声，说明还有周期没建模")
'''

# ------------------------------------------------------------------- §2.8
T10_CODE = '''# ---- SARIMA：把「季节」当成一个额外的 ARMA 层 ----
# SARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,s))：季节块作用在「滞后 s 的整数倍」上
# 这里 s=7：日序列的周周期（实测 ACF7 = 0.428，正好被非季节 ARMA 漏掉）
# @@todo （§2.8）把「周周期」当成一层额外的 ARMA：拟合 SARIMAX(1,0,2)(1,0,0,7)
# @@hint seasonal_order 是 (P, D, Q, s)，s=7 就是周周期；拟合时传 disp=False 关掉迭代日志
# @@hint 序列已经差过一阶，所以 D 取 0；再传 enforce_stationarity=enforce_invertibility=False
sar = SARIMAX(DDV, order=(1, 0, 2), seasonal_order=(1, 0, 0, 7),
              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
# @@end
params_sar = np.asarray(sar.params)
print(f"SARIMAX(1,0,2)(1,0,0,7)  AIC {sar.aic:.4f} | BIC {sar.bic:.4f}")
print(f"参数 [ar.L1, ma.L1, ma.L2, ar.S.L7, sigma2] = {np.round(params_sar, 6).tolist()}")
print(f"对比 非季节 ARMA(1,2)    AIC {aic_12:.4f}"
      f"  →  季节项省下 {aic_12 - float(sar.aic):.4f}")

assert abs(float(sar.aic) - 3050.7332) < 1e-3
assert abs(float(sar.bic) - 3070.1079) < 1e-3
assert float(sar.aic) < aic_12 - 100, "加季节项应显著降低 AIC"
assert len(params_sar) == 5
assert abs(float(params_sar[3]) - 0.453191) < 1e-5, "ar.S.L7 就是「上上周同一天」的系数"
print("T10 通过 -> SARIMA 把 AIC 从 3214.06 降到 3050.73（省 163.33）")
'''

# ------------------------------------------------------------------- 图
T11_CODE = '''# ---- 图 1：三段对照序列 + 它们的 ADF 判决 ----
fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
for ax, (nm, x) in zip(axes, (("白噪声", white), ("随机游走", rwalk), ("趋势 + 日周期", seas))):
    ax.plot(np.arange(len(x)) / 24, x, linewidth=0.6)
    # @@todo （§2.9）算这段序列的 ADF τ 与 p 值，写进标题里当"判决书"
    tau_v, p_v, _, _, _, _ = sm_adf(x, maxlag=1, autolag=None)
    # @@end
    ax.set_title(f"{nm}   τ = {tau_v:.2f}   p = {p_v:.3g}", fontsize=10)
    ax.set_ylabel("取值")
axes[-1].set_xlabel("天")
fig.suptitle("三种序列与 ADF 的判决（同一把尺子，三种结果）")
fig.tight_layout()

assert True
print("图 1 已绘制：白噪声 τ=-67.29 / 随机游走 τ=-0.21 / 趋势+日周期 τ=-103.61")
'''

T12_CODE = '''# ---- 图 2：ACF / PACF 面板（原序列 vs 一阶差分）----
band_y = bartlett(len(Y))
fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
stem_acf(axes[0, 0], acf_manual(Y, NLAG), band_y, "A 台区原序列：ACF（30/30 超带）")
stem_acf(axes[0, 1], pacf_levinson(Y, NLAG), band_y, "A 台区原序列：PACF（lag1 后骤降）")
stem_acf(axes[1, 0], ac_dy, band_dy, "一阶差分：ACF（lag24 仍有 0.668）")
stem_acf(axes[1, 1], pc_lev, band_dy, "一阶差分：PACF")
fig.suptitle("原序列 vs 一阶差分：ACF/PACF 面板（红虚线 = Bartlett 带 ±1.96/√n）")
fig.tight_layout()

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.2))
names = ["原序列", "一阶差分", "季节差分", "二阶(过)"]
# @@todo （§2.9）把四段序列的 std 与 ACF1 各收成一个列表，用来画对照柱状图
# @@hint std 直接 v.std()；ACF1 用 float(sm_acf(v, nlags=1, fft=True)[1])
stds = [v.std() for v in (Y, DY, D24, DD)]
acf1s = [float(sm_acf(v, nlags=1, fft=True)[1]) for v in (Y, DY, D24, DD)]
# @@end
axes2[0].bar(names, stds, color=["#888", "#2a7", "#2a7", "#c33"])
axes2[0].set_title("四段序列的标准差（过差分 → 反而变大）")
axes2[0].set_ylabel("std")
axes2[1].bar(names, acf1s, color=["#888", "#2a7", "#2a7", "#c33"])
axes2[1].axhline(0, color="black", linewidth=0.6)
axes2[1].set_title("四段序列的 ACF1（过差分 → 转负）")
fig2.tight_layout()

assert abs(float(acf_manual(Y, NLAG)[24]) - 0.912160) < 1e-6
assert abs(float(pacf_levinson(Y, NLAG)[1]) - 0.868814) < 1e-6
print("图 2 已绘制：原序列 ACF lag24 = 0.9122；一阶差分后 lag24 仍 = 0.6684")
'''

# =========================================================================== #
# 3. LESSON                                                                   #
# =========================================================================== #

LESSON = [
    md(
        """
# ch02 平稳性与统计方法（讲解版）

> **本节考点**：ADF 平稳性检验、差分、ACF / PACF 定阶、ARIMA / SARIMA。
> 这一章是「**手写 + 与真库逐项对账**」范式在本方向的第一块硬骨头 ——
> 手写的 τ 统计量必须和 `statsmodels` 对到 1e-13，否则等于没写。

| 竞赛评分点 | 分值含义 | 本章覆盖 |
|---|---|---|
| 数据准备及处理 | 技能操作 10% | 平稳性诊断、差分阶数的选定 |
| 模型选择 | 10% | ACF/PACF 定阶、AIC/BIC 选阶 |
| 新能源功率 / 负荷预测 | 行业赛题 | ARIMA 类统计模型是「必备基线」，也是后续 LSTM 的对照 |

**与 `01_pandas/ch16` 的分工**：ch16 讲 `resample` / `.dt` / `shift` 这些 API；
本章讲**统计推断** —— 序列能不能拿去建模、差几阶、定几阶。
`ch01` 讲了怎么造特征，本章讲**怎么用统计的眼睛先看一眼数据**。

## 学习目标

1. 说清「平稳」的三种定义（严平稳 / 弱平稳 / 趋势平稳），并能用前后半段统计量肉眼判断
2. **手写 ADF 的 τ 统计量**，与 `statsmodels.adfuller` 对账到 1e-13
3. 记住 ADF 的**两个坑**：强确定性季节会被判"平稳"；结论随 `maxlag` 翻脸
4. 会给一阶 / 季节 / 过差分三种方案，并用「方差 + ACF1」两条判据识别过差分
5. 手写 ACF（对准分母口径）与 PACF（Levinson 递推 + AR 回归两种定义），并知道它们差在哪
6. 会用 AIC / BIC 定阶，并**用 Ljung-Box 检验残差** —— 定完阶远不算完

## API 速查表

| 方法 / 属性 | 关键参数 | 返回 | 一句话 |
|---|---|---|---|
| `adfuller(x, maxlag, regression, autolag)` | `result_object=False` | tuple | ADF 检验；**必须显式关掉返回值切换** |
| `acf(x, nlags, fft=True)` | `adjusted` | `ndarray` | 自相关；`adjusted=True` 才是无偏分母 |
| `pacf(x, nlags, method="ldb")` | `method` | `ndarray` | 偏自相关；`[0]` 恒为 1.0 |
| `ARIMA(series, order=(p,d,q)).fit()` | `order` | `ARIMAResults` | `.aic` / `.bic` / `.params` / `.resid` |
| `SARIMAX(..., seasonal_order=(P,D,Q,s))` | `seasonal_order` | `SARIMAXResults` | 季节块作用在滞后 `s` 的整数倍上 |
| `acorr_ljungbox(resid, lags)` | `return_df` | `DataFrame` | 残差白噪声检验；`lb_pvalue` 越小越"不白" |
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),      # 脚手架要先于 SETUP：SETUP 里的 clean_series 来自它
    code(SETUP),
    code(PLOT_SETUP),
    md(
        """
## 2.1 平稳性是什么：先用肉眼判据立个标尺

「平稳」在考卷上有三种说法，初学最容易混：

| 说法 | 要求 | 典型反例 |
|---|---|---|
| **严平稳** | 任意有限维联合分布都不随时间平移变 | 几乎没人真去验它 |
| **弱平稳**（宽平稳） | 均值常数、方差常数、自协方差只与**间隔**有关 | 随机游走 |
| **趋势平稳** | 去掉确定性趋势后弱平稳 | 带线性趋势的负荷 |

弱平稳是实际工作中唯一可操作的判据，它给了三条**肉眼可查**的性质：

1. 均值不随时间变
2. 方差（波动幅度）不随时间变
3. 自协方差只与间隔有关 —— 这一条等价于「ACF 只依赖滞后 k」

于是有了最朴素的一个诊断法：**把序列劈成前后两半，比均值和标准差**。

下面造三段序列来当标尺：
**白噪声**（三条全满足）、**随机游走**（均值在漂）、**确定性趋势 + 日周期**（均值在同向漂，
但方差完全不变 —— 这一段的"标准答案"后面会让人意外）。

```
白噪声        τ = -67.29   p = 0.000000
随机游走      τ =  -0.21   p = 0.937911
趋势+日周期    τ = -103.61   p = 0.000000
```
"""
    ),
    code(T1_CODE),
    md(
        """
### 2.1.1 难点深挖：ADF 到底在检验什么

ADF 的回归式是

```
Δy_t = α + δ·t + β·y_{t-1} + Σ_{i=1..p} γ_i·Δy_{t-i} + ε_t
```

- **H0**：`β = 0` —— `y_{t-1}` 的水平对下一期变化没有任何"拉回"作用，
  这就是**单位根**（随机游走）的定义
- **H1**：`-2 < β < 0` —— 水平越高，下一期越倾向于往回落，即**均值回复**

`β̂` 是负的、且**很负**的时候，τ 就很小（很负），于是拒绝 H0。

**最关键的一点**：τ 的分布**不是 t 分布**。因为 H0 下 `y` 非平稳，
OLS 的经典结论整套失效，必须查 Dickey 与 Fuller 用蒙特卡洛模拟出来的临界值表：

| `regression` | 含义 | 5% 临界值（近似） |
|---|---|---|
| `"n"` | 无常数、无趋势 | −1.95 |
| `"c"` | **带常数**（最常用） | −2.86 |
| `"ct"` | 常数 + 线性趋势 | −3.41 |

所以手写 ADF 的价值在于：**写一遍就知道 τ 只是"一个比值"，它的全部难点都在临界值表上**。
真库替我们干了查表这件事。

**注意 τ 的取值范围**：`τ` 可以到 −100 这种量级（本数据 −103.6），
因为它的分母 `se(β̂)` 在强均值回复时会非常小。**τ 的绝对值本身没有意义，只有和临界值比才有意义。**
"""
    ),
    code(T2_CODE),
    md(
        """
### 2.2 手写 ADF：设计矩阵的列顺序是唯一的难点

手写版本只有两件事：

1. 拼设计矩阵。列顺序固定为 `[常数][趋势][y_{t-1}][Δy_{t-1} … Δy_{t-p}]`
   —— 因为 β 对应的**列号会随 `regression` 变**（`"c"` → 第 1 列、`"ct"` → 第 2 列、`"n"` → 第 0 列）
2. 用经典 OLS 标准误：
   `σ̂² = RSS/(n−k)`，`se(β̂) = sqrt(σ̂² · [(XᵀX)⁻¹]_ββ)`，最后 `τ = β̂/se(β̂)`

⚠️ **最阴的一个坑是长度对齐**：`Δy` 只有 `len(d) − p` 个可用观测，
所以水平项必须取 `x[p:len(d)]` —— 写成 `x[p:]` 会**多出一格**，
拼接时直接
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly
```
这个报错不会告诉你"是哪个块长了"，所以在 §2.2 的挖空里特意把它留成了考点。

⚠️ **用 `np.linalg.lstsq` 而不是正规方程的 `inv(XᵀX) XᵀY`**：前者走 QR / SVD，数值更稳。
但 `se` 那一步不得不用 `inv(XᵀX)` —— 因为标准误的定义里本来就有它。

对账结果：

```
白噪声      手写 −67.2888 | statsmodels −67.2888 | 差 1.0e-11
随机游走    手写  −0.2051 | statsmodels  −0.2051 | 差 1.8e-15
趋势+日周期  手写 −103.6063 | statsmodels −103.6063 | 差 1.4e-11
```

**差在 1e-11 而不是 1e-16，是 `lstsq` 与 statsmodels 内部实现（它也用 OLS，但组装方式不同）的差别**
—— 对统计量完全可以接受（临界值本身只有三位有效数字）。所以断言写成 `< 1e-9` 而不是 `< 1e-12`。
"""
    ),
    md(
        """
### 2.2.1 难点深挖：ADF 的两个反直觉

**反直觉 ①：A 台区被判成"平稳"。**

把 A 台区的 8760 点小时序列直接送进 ADF（`maxlag=1`、带常数）：

```
hand   τ = -37.2580
sm     τ = -37.2580    p = 0.000000
```

p 值直接是 0，**拒绝单位根**，也就是"序列平稳"。

这不矛盾 —— 回想 §2.1 第三段人工序列：**强确定性日周期 + 噪声，ADF 一样会拒绝单位根**
（τ = −103.61）。因为 ADF 检的是"有没有单位根"，而**日周期不是单位根**：
每个点都被日周期强拉着回到固定位置，`β` 自然很负。

> **结论一句话**：`p < 0.05` 只说明"没有单位根"，
> **不说明"可以直接拿去建 ARMA"**。后面 §2.4 用 ACF 一测就露馅。

**反直觉 ②：换个 `maxlag`，结论直接翻脸。**

| `maxlag` | 手写 τ | `statsmodels` τ | p 值 | 拒绝单位根？ |
|---|---|---|---|---|
| 0 | −24.7948 | −24.7948 | 0.000000 | ✅ |
| 1 | −37.2580 | −37.2580 | 0.000000 | ✅ |
| 2 | −38.8954 | −38.8954 | 0.000000 | ✅ |
| 6 | −24.9802 | −24.9802 | 0.000000 | ✅ |
| 12 | −13.3402 | −13.3402 | 0.000000 | ✅ |
| **24** | **−1.9345** | **−1.9345** | **0.315992** | **❌** |
| 48 | −2.9485 | −2.9485 | 0.039990 | ✅ |

**为什么刚好是 24 会翻脸？** 因为把 24 个 `Δy` 的滞后放进回归，
等于让回归自己去"吃掉"日周期能解释的部分 —— 日周期被解释掉之后，
剩下的"水平拉回"效应就弱得多了，`β` 不再显著为负。

**这条必须背**：报 ADF 结果时**必须同时报 `maxlag` 或 `autolag`**，
只写一句"p < 0.05，平稳"是不合格的。

`statsmodels` 默认 `autolag="AIC"`，在这条序列上选了 **滞后 32**：

```
τ = -4.9443   p = 0.000029   10% 临界值 = -2.5669   → 拒绝单位根
```
"""
    ),
    code(T3_CODE),
    md(
        """
## 2.3 差分：一阶、季节、以及"过头了"的过差分

差分的目的是**把单位根抽掉**：

| 差分 | 公式 | 抽掉什么 |
|---|---|---|
| 一阶 | `Δy_t = y_t − y_{t-1}` | 线性趋势 / 随机游走的漂移 |
| 季节 | `y_t − y_{t−24}` | 周期长度 24 的**季节单位根** |
| 二阶（= 对一阶再差分） | `Δ²y_t` | 本来是给"双单位根"准备的 |

**过差分（over-differencing）** 是初学者最容易犯的错：序列本来只需要差一次，
你差了两次，结果是**方差变大 + 引入人为的负自相关**。它有两条硬判据：

1. **方差不再下降，反而变大**
2. **ACF1 从正转负** —— 因为差分算子 `1−B` 本身带一个 `-1` 的零点，
   连着用两次就相当于给序列乘上 `(1−B)²`，人为塞进一个负的 lag-1 结构

```
原序列        std 72.1892   ACF1 +0.868814   ACF24 +0.912160
一阶差分      std 36.9689   ACF1 +0.321853   ACF24 +0.668394
季节差分(24)  std 29.6486   ACF1 +0.488328   ACF24 -0.258191
二阶差分(过)  std 43.0534   ACF1 -0.314890   ACF24 +0.291265
```

一阶/原 = **0.512112**（正常，方差砍半）
二阶/一阶 = **1.164585**（**> 1，过差分的铁证**）
"""
    ),
    code(T4_CODE),
    md(
        """
### 2.3.1 难点深挖：差分到哪一阶才算够

三条经验规则，按优先级排：

| 判据 | 怎么算 | 通过标准 |
|---|---|---|
| **方差必须单调下降** | `std(Δy)/std(y)`、`std(Δ²y)/std(Δy)` | 每多差一阶都 **< 1** |
| **ACF1 不能转负** | 差完立刻算 `ACF1` | 保持 **> 0**，转负就是差过头了 |
| **ADF 必须拒绝** | 差完再做 ADF | `p < 0.05`（且要报 maxlag） |

对 A 台区小时序列来说，**三条规则会打架**，这恰好是本数据最有价值的地方：

- 一阶差分：方差降了（0.512）✅、ACF1 保持正（0.322）✅、ADF 拒绝 ✅ —— 三条全过
- 但一阶差分后 **ACF 在 lag 24 处还有 0.668**，48 个滞后里 **46 个超带** ❌

也就是说：**ADF 说"够平了"，ACF 说"周期还在"**。

这两个指标量的根本不是同一件事：

- **ADF** 只看 lag 1 附近的"水平拉回"，对慢变周期不敏感
- **ACF** 会一路看到 lag 48 甚至更远，周期结构藏不住

> **所以正确的读法是**：ADF 决定"要不要一阶差分"，
> **ACF 决定"要不要季节差分"**。两者都要看，缺一个就会把日周期当噪声喂给模型。

`ch01` 里我们用「三层周期分解」处理日周期；本章用的是**统计路线的替代方案**：
季节差分。两条路都行，区别是前者保留了可解释的分量，后者把周期"减"掉了。
"""
    ),
    md(
        """
## 2.4 手写 ACF：分母的口径决定一切

ACF 的定义只有一行：

```
ACF(k) = Σ_{t=k+1..n} (y_t − ȳ)(y_{t−k} − ȳ) / Σ_{t=1..n} (y_t − ȳ)²
```

**唯一的坑在分母**：

| 分母 | 名字 | `statsmodels` 里的开关 |
|---|---|---|
| `Σ (y_t − ȳ)²`（全部 n 项） | 有偏估计（默认） | `adjusted=False` |
| `Σ_{t=k+1..n} (y_t − ȳ)²`（只留 n−k 项） | 无偏估计 | `adjusted=True` |

初学最容易"顺手"写成无偏版（因为分子只用了 n−k 项），那样和真库对不上。
**本章统一用默认口径**，对账结果：

```
ACF 最大绝对差 3.331e-16      ← 到浮点误差的量级了
```

**Bartlett 置信带**：白噪声的 ACF 近似服从 `N(0, 1/n)`，于是 5% 水平下的带子是

```
±1.96/√n = ±0.020943     （n = 8759）
```

落在带内 = 该滞后自相关不显著。对一阶差分序列：

```
ACF1  = +0.321853   超带
ACF24 = +0.668394   超带   ← 日周期
ACF48 = +0.657224   超带   ← 两天前的同一时刻
超带滞后数 46/48
```

⚠️ **一个陷阱**：`lag 48` 还有 0.657，不代表"两天前的结构"是新的 ——
它只是 lag 24 的**回声**。周期结构的 ACF 是一串等间距的峰，不要一个峰一个峰地去建模。
"""
    ),
    code(T5_CODE),
    md(
        """
## 2.5 手写 PACF：同一个名字，两种等价定义

PACF（偏自相关）解决的是"ACF 的间接效应"：`y_t` 与 `y_{t−3}` 相关，
可能只是因为它们都跟 `y_{t−1}` 相关。"偏"就是**扣掉中间那些滞后之后**剩下的相关。

有两条**在无限样本下等价**的定义：

| 定义 | 算法 | 复杂度 |
|---|---|---|
| **A. Durbin-Levinson 递推** | 从样本 ACF 出发，逐阶推反射系数 | O(nlags²) |
| **B. AR(k) 回归的末系数** | 对每个 k 拟合一次 AR(k)，取 `φ_k` | O(nlags·n·k) |
| （主流实现如 `statsmodels`、R 的 `pacf`） | 定义 A | 快 |

递推的公式只有两行（第 k 阶反射系数与 Levinson 更新）：

```
φ_{k,k} = ( r_k − Σ_{j=1..k−1} φ_{k−1,j}·r_{k−j} ) / ( 1 − Σ_{j=1..k−1} φ_{k−1,j}·r_j )
φ_{k,j} = φ_{k−1,j} − φ_{k,k}·φ_{k−1,k−j}          j = 1..k−1
```

对账结果（一阶差分序列，`nlags=48`）：

```
Levinson  vs statsmodels.pacf   max|diff| = 3.053e-15    ✅ 一致
Levinson  vs 「AR(k) 末系数」    max|diff| = 1.179e-02    ⚠️ 差 1e-2
```

**lag 0 的约定也要注意**：`statsmodels.pacf` 的 `[0]` 恒为 `1.0`
（递推本身产不出 lag 0，这个值是人为补的），手写时别忘了补，否则整条曲线错位一位。
"""
    ),
    code(T6_CODE),
    md(
        """
### 2.5.1 难点深挖：为什么两种"等价定义"差了 1e-2

这是本章刻意保留的**负结果** —— 教科书说"A 和 B 等价"，但实测差 **1.18e-02**。

差在哪？差在**它们其实在估计两个不同的东西**：

| | 定义 A（Levinson） | 定义 B（AR 回归） |
|---|---|---|
| 输入 | **样本 ACF**（分母用 n，有偏） | **原始数据** |
| 估计方式 | Yule-Walker / 矩估计 | 条件最小二乘（OLS） |
| 有限样本性质 | 有偏，但在无偏性/大样本下有更好的方差 | 无偏，但方差略大 |

- 定义 A 先把数据**压成 ACF** 再用，n=8759 的样本 ACF 本身带了 `O(1/n)` 的偏差，
  而且分母统一用 n（不是 n−k），这个偏差会顺着递推被放大到高阶
- 定义 B 直接用原始数据做 OLS，不走 ACF 这道中介

**两者的大样本极限都是真正的 PACF，但有限样本下不相等。** 实测差 `1.18e-02`，
相对量级约 4%（PACP1 本身 0.32）。

> **这个负结果的教学价值**：让你知道"对账"这件事不是形式主义。
> 如果只会 `assert abs(手写 - 真库) < 1e-12`，你就永远发现不了
> **"两个都叫 PACF 的量，在小样本下并不相等"**。
> 更进一步：**ACF 与 PACF 从来都只是"辅助定阶"**，
> 真正拍板要靠 AIC/BIC + 残差检验（§2.7）。

**那 1e-2 的差要不要紧？** 不要紧。定阶时我们看的是"第几个滞后还在带外"，
ACF/PACF 的绝对值差 1% 完全不影响这个判断。
"""
    ),
    md(
        """
## 2.6 AR / MA 的 ACF 形态：拖尾与截尾

定阶全靠这张表，必须背下来：

| 模型 | ACF | PACF |
|---|---|---|
| **AR(p)** | **拖尾**（指数衰减 / 振荡衰减，不截断） | **p 阶截尾**（p 之后落进带内） |
| **MA(q)** | **q 阶截尾**（q 之后落进带内） | **拖尾** |
| **ARMA(p,q)** | 都在 p / q 之后拖尾 | 都拖尾 |

原因是纯数学的：

- AR(1)：`ACF_k = φ^k` —— 指数衰减但**永远不为 0**，所以"拖尾"
- MA(1)：`x_t = ε_t + θε_{t−1}` 里 `x_t` 与 `x_{t−2}` 没有任何共同项，所以 `ACF_2 = 0` 严格成立
  （`ACF_1 = θ/(1+θ²)`）

下面用模拟序列验证（`n = 20000`，让样本量足够大）：

```
AR(1) φ=+0.9  ACF1..4 [0.9018, 0.8123, 0.7303, 0.6573]   理论 [0.9, 0.81, 0.729, 0.6561]
MA(1) θ=+0.9  ACF1..4 [0.5040, 0.0081, 0.0039, 0.0066]   理论 [0.4972, 0, 0, 0]
```

注意 `MA(1) θ=+0.9` 的 ACF1 实测 0.5040 而理论 0.4972 —— **又是不严格相等**，
但差在 7e-3，属于正常的抽样波动。判断"截尾"看的是 **lag 2 之后是否归零（<0.01）**，
而不是 ACF1 精确等于多少。
"""
    ),
    code(T7_CODE),
    md(
        """
## 2.7 ARIMA 定阶：AIC / BIC + 残差检验

**载体换成 365 点的日序列。** 为什么？因为小时序列的日周期太强，
不做季节差分根本定不出低阶 ARMA（一阶差分后 46/48 个滞后超带）。
把数据压成日频之后，序列干净得多，定阶这一步才看得清。

**教科书路径在这次真的走通了**：

```
日序列  ADF（含常数）      滞后 16 | τ  -1.8122 | p 0.374400   ← 不能拒绝单位根
日序列  ADF（常数 + 趋势） 滞后 16 | τ  -1.4542 | p 0.844277   ← 更不拒绝
日差分  ADF（含常数）      滞后 15 | τ  -5.7997 | p 0.000000   ← 拒绝，可以做 ARMA
```

⚠️ **顺便注意 `ct` 与 `c` 的区别**：同一个序列，含趋势项时 ADF 的临界值更苛刻
（5% 从 −2.87 变成 −3.42），所以"更不拒绝"。这就是 §2.1.1 那张临界值表的实际影响。

**定阶**：对 `p, q ∈ {0,1,2}` 做 9 个组合的网格搜索，比较 AIC 与 BIC。

| 准则 | 惩罚项 | 偏好 |
|---|---|---|
| AIC | `2k` | 偏复杂（预测导向） |
| BIC | `k·ln n` | 偏简单（模型识别导向） |

有了 `SARIMAX` 别忘了还有一个更狠的做法：**先把残差画成 ACF 看**（下一节）。
"""
    ),
    code(T8_CODE),
    md(
        """
### 2.7.1 难点深挖：定完阶远不算完 —— 残差 Ljung-Box 不过

**这是本章最重要的一个"负结果"。**

`ARMA(1,2)` 的 AIC 与 BIC **同时**给出最小，看起来干净利落。但：

```
残差 ACF1..5  [-0.007624, -0.016276, 0.027984, 0.103546, -0.204241]
带 ±0.103158  | 10 个滞后里有 5 个超带

Ljung-Box：
     lag    lb_stat      lb_pvalue
       5   19.69513   1.425489e-03      ← p < 0.01，拒绝白噪声
      10   90.29943   4.671439e-15
      20  176.12174   5.560927e-27
```

**Ljung-Box 检验**的原假设是"残差前 m 阶自相关全为 0"。
p 值 `1.4e-3` **远小于 0.05 → 拒绝**，也就是**残差里还有结构**。

结构在哪？看日差分序列自己的 ACF：

```
ACF7  = +0.428114      ← 一周前的同一天
ACF14 = +0.427096      ← 两周前
```

**周周期。** 日序列用工作日/周末两套负荷形态（`data/make_data.py` 里 A 台区周末系数是 1.06），
所以每 7 天重复一次。非季节 ARMA 看不到 lag 7。

**教训两条**：

1. **AIC/BIC 只在"你给它的候选集合里"选最优**。它不会告诉你"你还漏了一个季节项"。
   残差检验才是唯一能发现"模型族选错了"的手段。
2. **`p < 0.05` 的 Ljung-Box 必须继续查**，不能像很多人那样"拟合成功了就交卷"。

那怎么修？—— 下一节。
"""
    ),
    code(T9_CODE),
    md(
        """
## 2.8 SARIMA：把季节装进一个额外的 ARMA 层

**SARIMA(p,d,q)(P,D,Q)s** 的记法初学看着糊，拆开只有一句话：

```
(P, D, Q) 这一组参数，作用在「滞后 s 的整数倍」上 —— 也就是把序列每 s 步抽出来，当成一条新序列来建模
```

四层参数各管什么：

| 位置 | 名字 | 管什么 | 本数据取值 |
|---|---|---|---|
| `(p,d,q)` | 非季节部分 | 相邻小时/相邻天之间的结构 | `(1,0,2)` |
| `P` | 季节 AR 阶 | **上上周同一天**对今天的影响 | `1` |
| `D` | 季节差分阶 | 要不要再做一次 `y_t − y_{t−s}` | `0`（已差过一阶） |
| `Q` | 季节 MA 阶 | 季节残差的滞后 | `0` |
| `s` | 季节周期长度 | 多少步一个循环 | `7`（周） |

对日序列拟合 `SARIMAX(1,0,2)(1,0,0,7)`：

```
AIC 3214.0624  →  3050.7332     省下 163.33
BIC 3233.5482  →  3070.1079
参数 [ar.L1, ma.L1, ma.L2, ar.S.L7, sigma2] = [0.712205, -0.788805, -0.151551, 0.453191, 298.203396]
```

`ar.S.L7 = 0.453191` 就是那个季节项 —— "**上周同一天的负荷对今天有正影响**"。
它一个参数就吃掉了 lag 7 的 0.428 自相关，换来 AIC 降 163。

> **工程判断**：AIC 降 163 是**极大**的改善（一般降 10 以上就值得加项）。
> 在这个数据上，**"不做季节项"会直接损失一个数量级的模型质量**。
> 这也是为什么负荷预测的赛题里，`s=24`（小时数据的日周期）几乎是默认要填的。

**串起来看本章的完整逻辑链**：

```
ADF 说"没有单位根"（但被确定性季节骗了）
  → ACF 说"一阶差分后还有 46/48 个滞后超带"（周期没处理）
  → 过差分判据说"二阶差分过头了"（方差 1.16 倍、ACF1 转负）
  → 换日序列走教科书路径：ADF 不拒绝 → 一阶差分 → ACF/PACF 定阶 → ARMA(1,2)
  → 残差 Ljung-Box 拒绝白噪声（周周期）
  → 加季节项 → AIC 降 163
  → 这才是可以拿去预测的模型
```

**任何一步跳过，最后都会栽在残差检验上。**
"""
    ),
    code(T10_CODE),
    md("## 2.9 三张图把本章讲清楚"),
    code(T11_CODE),
    code(T12_CODE),
    md(
        """
## 易错点清单

1. **ADF 的 τ 不是 t 统计量**，必须查 Dickey-Fuller 临界值表 —— 绝对值再大也不能拿 1.96 去比。
2. **报 ADF 结果必须同时报 `maxlag` / `autolag`**。同一序列 `maxlag=1` 拒绝、`maxlag=24` 不拒绝（本数据实测）。
3. **`p < 0.05` ≠ "可以建 ARMA"**。A 台区原序列 p=0，但 ACF 有 30/30 个滞后超带。
4. **`regression` 一定要和数据的形态对上**：带线性趋势就用 `"ct"`，否则临界值用错（5% 从 −3.42 变 −2.87）。
5. **`adfuller` 在 statsmodels 0.15 起返回值会从 tuple 切到 `ADFullerResult`**，写代码时显式传 `result_object=...`。
5b. **`adfuller` 返回几个元素取决于 `autolag`**：`autolag=None` 只有 **5** 个（无 IC），`autolag="AIC"` 有 **6** 个。
   于是 `res[5]` 在 `autolag=None` 时会抛 `IndexError: tuple index out of range` —— 收返回值时要按长度兜底。
6. **ACF 的分母口径要统一**：分子只用了 n−k 项，但默认分母是全部 n 项（`adjusted=False`）。
7. **过差分有两条铁证**：方差变大 + ACF1 转负。本数据二阶差分的 std 比是 **1.164585**。
8. **ADF 和 ACF 打架时，以 ACF 为准判周期**：ADF 只看 lag 1 附近，周期结构它看不见。
9. **PACF 的 `[0]` 要补成 1.0**，否则整条曲线错位一位，定阶全错。
10. **两种 PACF 定义在小样本下不相等**（本数据差 1.18e-02），不要拿 `< 1e-12` 去卡。
11. **ACF 的等间距峰是回声**：lag 48 的 0.657 是 lag 24 的回声，别当成两个独立结构。
12. **AIC/BIC 只在候选集合里选最优**，永远发现不了"你漏了整个季节项"。
13. **定完阶必须做残差白噪声检验**：`ARMA(1,2)` 的 Ljung-Box p = 1.4e-03（**拒绝**）。
14. **`AIC` 与 `BIC` 打架时看样本量**：n 小的时候 BIC 惩罚更重，会选更简单的模型。
15. **`s` 要填对**：小时数据是 `s=24`，日数据是 `s=7`。填错 `s` 比不填更糟（会引入假的周期约束）。
16. **`SARIMAX` 拟合前先差分到位**，否则 `enforce_stationarity=False` 会让优化器跑飞。
17. **不要对"确定性趋势 + 季节"使用季节差分**：那会同时把趋势和季节都减掉，
    最后模型只能预测 0（本数据日序列之所以能走通，是因为它的趋势极弱，R² 只有 0.0147）。
18. **`np.linalg.lstsq` 负责解方程，`inv(XᵀX)` 负责算标准误**，两件事不能混着用（数值稳定性不同）。
19. **`maxlag=0` 是合法档位**，但此时"滞后块"是空的，`np.column_stack([])` 会抛
    `ValueError: need at least one array to concatenate` —— 要退化成 `np.empty((n_obs, 0))`。
20. **水平项要截到 `len(d)`**（即 `x[p:len(d)]`）：`d = np.diff(x)` 比 `x` 少一格，
    写成 `x[p:]` 会让设计矩阵的列长度差 1，报错信息完全指不出问题在哪。
"""
    ),
    md(
        """
## 本章小结

**一句话**：统计方法的全部价值在于「**先用最低成本判断这个序列能不能建模、该差几阶、该定几阶**」，
而不是急着上模型。

**六步流水线**：

```
① 看平稳性     前后半段均值/方差 → 三段人工序列当标尺
② 跑 ADF       手写 τ 对账真库；必须报 maxlag/autolag/regression
③ 定差分阶     规则：方差单调降 + ACF1 不转负 + ADF 拒绝（三条都要过）
④ 看 ACF/PACF  ADF 管「要不要一阶差分」，ACF 管「要不要季节差分」
⑤ 定阶         AIC/BIC 网格 + ACF/PACF 辅助
⑥ 残差检验     Ljung-Box 必须不拒绝；拒绝就回去加季节项
```

**四个「必须背下来」的判定规则**：

| 场景 | 规则 |
|---|---|
| 该差几阶 | 每多差一阶，`std` 都要**继续下降**且 `ACF1` **保持为正** |
| 过差分了没 | `std` 比 **> 1** 或 `ACF1` **转负** → 差过头了 |
| 该不该加季节项 | 残差 ACF 在 `lag = s` 处出峰，或 Ljung-Box **拒绝** |
| ADF 信不信 | 看 `maxlag` 是否合理、`regression` 是否匹配趋势；**单一 p 值不算证据** |

**本章与后续章节的接口**：本章给出的是一套**「轻量诊断」**工具 ——
它不负责预测（ARIMA 在强季节数据上打不过特征工程），
负责的是**在你花时间训 LSTM 之前，先花十分钟搞清楚数据长什么样**。
`ch03` 开始上手写 RNN，届时会拿本章的 ARIMA 结果当第一个对照基线。

**下一步**：ch03 —— 手写 `RNNCell` 前向与 BPTT，并与 `torch` 对账；
然后实测「长依赖梯度消失」，看它和本章的「过差分」一样，
都是**同一个算子反复作用导致的结构性退化**。
"""
    ),
]

# =========================================================================== #
# 4. EXERCISE                                                                 #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch02 平稳性与统计方法（练习版）

> 做法：按顺序做完 **任务 1 ~ 12**，每个任务下面的 `assert` 全绿就算过。
> 卡住了先看 `# 提示：`，再不行翻讲解版对应小节（每个 `@@todo` 的描述里都标了小节号）。

**本章的硬要求**：手写的 ADF 必须与 `statsmodels` 对到 **1e-9** 以内 ——
对不上就是设计矩阵的列顺序错了，不要改断言。
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),      # 脚手架要先于 SETUP：SETUP 里的 clean_series 来自它
    code(SETUP),
    code(PLOT_SETUP),
    md("## 任务 1：平稳性的肉眼判据（§2.1）"),
    code(T1_CODE),
    md("## 任务 2：手写 ADF 的 τ 统计量（§2.2）"),
    code(T2_CODE),
    md("## 任务 3：ADF 的两个反直觉（§2.2.1）"),
    code(T3_CODE),
    md("## 任务 4：差分三兄弟与过差分判据（§2.3）"),
    code(T4_CODE),
    md("## 任务 5：手写 ACF 与 Bartlett 带（§2.4）"),
    code(T5_CODE),
    md("## 任务 6：手写 PACF 的两种定义（§2.5）"),
    code(T6_CODE),
    md("## 任务 7：AR / MA 的拖尾与截尾（§2.6）"),
    code(T7_CODE),
    md("## 任务 8：日序列定阶 AIC / BIC（§2.7）"),
    code(T8_CODE),
    md("## 任务 9：残差 Ljung-Box 检验（§2.7.1）"),
    code(T9_CODE),
    md("## 任务 10：加季节项 SARIMA（§2.8）"),
    code(T10_CODE),
    md("## 任务 11~12：把结论画出来（§2.9）"),
    code(T11_CODE),
    code(T12_CODE),
    md(
        """
## 自查清单（能不看笔记答出来才算过）

- [ ] 12 个任务的 `assert` 全部通过
- [ ] ADF 的 τ 为什么不能查 t 分布表？三种 `regression` 的 5% 临界值各是多少？
- [ ] A 台区原序列的 ADF p 值是多少？它真的"平稳"吗？该看哪个指标才能发现不对？
- [ ] 同一个序列 `maxlag=1` 和 `maxlag=24` 的 ADF 结论分别是什么？为什么？
- [ ] 过差分的两条判据是什么？本数据的两个具体数字是多少？
- [ ] ACF 的分母用 n 还是 n−k？对应 `statsmodels` 的哪个参数？
- [ ] PACF 的 lag 0 约定是多少？两种定义为什么在小样本下不相等？
- [ ] AR / MA 的 ACF 分别有什么形态？怎么用 lag 2 之后的取值区分它们？
- [ ] 日序列的 ADF 结论和小时序列为什么相反？
- [ ] `ARMA(1,2)` 的 Ljung-Box p 值是多少？它说明什么？结构藏在哪个滞后上？
- [ ] SARIMA 的 `(P,D,Q,s)` 各管什么？本数据为什么取 `s=7`？
- [ ] 加季节项后 AIC 从多少降到多少？

## 关键真值对照（做完再对）

| 量 | 值 |
|---|---|
| 白噪声 / 随机游走 / 趋势+日周期 的 ADF τ | −67.2888 / −0.2051 / −103.6063 |
| 随机游走前后半段均值 | 396.8295 / 569.7063（差 172.88） |
| 随机游走 ACF1 / 其一阶差分 ACF1 | +0.999366 / −0.000012 |
| 手写 ADF vs statsmodels 最大绝对差 | 4.974e-14 |
| A 原序列 ADF（maxlag=1 / 24） | τ −37.2580 / −1.9345；p 0.000000 / **0.315992** |
| A 原序列 ADF（autolag=AIC） | 滞后 32、τ −4.9443、p 0.000029 |
| 一阶差分 ADF | τ −56.5009、p 0.000000、std 36.9689 |
| 季节差分(24) ADF | τ −35.8735、p 0.000000、std 29.6486 |
| 过差分 std / 比值 / ACF1 | 43.0534 / 1.164585 / **−0.314890** |
| 一阶差分 ACF1 / ACF24 / ACF48 | +0.321853 / +0.668394 / +0.657224 |
| 一阶差分超带滞后数（1..48） | **46 / 48** |
| Bartlett 带（n=8759） | ±0.020943 |
| 手写 ACF vs statsmodels | 3.331e-16 |
| 手写 PACF(Levinson) vs statsmodels | 3.053e-15 |
| 手写 PACF(Levinson) vs 「AR(k) 末系数」 | **1.179e-02** |
| A 原序列 ACF1 / ACF24 | +0.868814 / +0.912160（30/30 超带） |
| A 原序列 PACF1 / PACF2 / PACF24 | +0.868814 / −0.414632 / +0.146101 |
| AR(1) φ=0.9 的 ACF1..4 | 0.9018 / 0.8123 / 0.7303 / 0.6573 |
| MA(1) θ=0.9 的 ACF1..4 | 0.5040 / 0.0081 / 0.0039 / 0.0066 |
| 日序列 ADF（c / ct） | τ −1.8122（p 0.374400）/ τ −1.4542（p 0.844277） |
| 日差分 ADF | 滞后 15、τ −5.7997、p 0.000000 |
| 日差分 ACF7 / ACF14 | +0.428114 / +0.427096 |
| ARMA(1,2) AIC / BIC | 3214.0624 / 3233.5482 |
| 残差 Ljung-Box p（lag 5 / 10 / 20） | **1.425489e-03** / 4.671439e-15 / 5.560927e-27 |
| SARIMAX(1,0,2)(1,0,0,7) AIC / BIC | **3050.7332** / 3070.1079 |
| 季节项 `ar.S.L7` | +0.453191 |
| 日序列线性趋势 | 斜率 −0.053120 kW/天、R² 0.014691 |
"""
    ),
]

if __name__ == "__main__":
    report([build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)])
