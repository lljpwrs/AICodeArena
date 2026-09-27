"""ch18 —— 可视化：`df.plot` + matplotlib 组合

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch18_visualization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch18_visualization"

HEADER = '''import io
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)

# 中文显示：字体列表按平台命中率从高到低排；负号必须显式修复
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA = Path("data")

dev = pd.read_csv(DATA / "device_defects.csv")
load = pd.read_csv(DATA / "load_curve.csv")
load["时间戳"] = pd.to_datetime(load["时间戳"])

st = load[load["台区编号"] == "STATION_A_01"].sort_values("时间戳").set_index("时间戳")
daily = st["负荷值"].resample("D").mean()

print("pandas", pd.__version__, "| backend:", pd.options.plotting.backend)
print("dev", dev.shape, "| daily", daily.shape)'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def close_all():
    """每次画完关掉图窗，避免 40+ 张图把 notebook 撑爆。"""
    plt.close("all")


def png_bytes(fig):
    """把图渲染成 PNG 字节流（不用写文件即可验证 savefig 真的产出了图）。"""
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight")
    return buf.getvalue()


print("中文字体列表已配置；负号修复 axes.unicode_minus = False")'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch18 可视化：`df.plot` + matplotlib 组合

> 方向：数据预处理 ｜ 竞赛对应：**数据可视化是独立计分项（每项 2 分）**——
> 图出得来只是底线，标题 / 轴标签 / 中文不方框 / 图存得下来才是满分动作。

可视化坑的共性是**不报错**：

1. **中文字体找不到只出方框**——字体列表 + `axes.unicode_minus` 双配置
2. **直方图被离群值撑扁**——9999 一根柱把其他柱压成一条线
3. **`ax=` 传错图就画错地方**——subplots 布局与传入 Axes 必须对上
4. **饼图文字 / 图例顺序是码位序**——与 value_counts 顺序对账
5. **headless 环境不显示图**——`savefig` 才是可交付物

> 本章断言基于 matplotlib 的**对象模型**（`ax.patches` / `ax.lines` /
> `ax.collections`），图不用肉眼看也能对账。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据可视化 | 每项 2 分 | bar / hist / line / scatter / box / pie，标题+轴标签齐全 |
| （隐性）交付能力 | 图能进报告 | `savefig` / `dpi` / `bbox_inches` |

**学习目标**：任何一列/两列数据，30 秒出一张"标题、轴标签、中文正常、
可存盘"的图。
"""
    ),
    md(
        """
## 二、API 速查表

| 方法 / 参数 | 说明 |
|---|---|
| `df.plot(kind=...)` | 返回 **Axes**；kind: bar / barh / hist / line / scatter / box / pie |
| `ax=` | 把图画进已有 Axes（subplots 布局的关键） |
| `plt.subplots(nrows, ncols, figsize)` | 返回 `(Figure, axes数组)` |
| `ax.set_title / set_xlabel / set_ylabel` | 补齐三件套 |
| `ax.patches / ax.lines / ax.collections` | 柱 / 线 / 点的对象池，**对账用** |
| `fig.savefig(目标, dpi, bbox_inches="tight")` | 交付物；目标可以是文件或字节流 |
| `plt.close("all")` | 释放图窗，notebook 防爆内存 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 bar：类别分布 + 柱高对账

`value_counts().plot(kind="bar")` 是类别列的标配。柱高必须与
`value_counts()` 一一对应——用 `ax.patches` 直接对账。
"""
    ),
    code(
        """vc = dev["设备类型"].value_counts()
ax = vc.plot(kind="bar", title="设备类型分布")
ax.set_xlabel("设备类型")
ax.set_ylabel("缺陷条数")

print("返回类型:", type(ax).__name__, "| 柱数:", len(ax.patches))
print("x 标签:", [t.get_text() for t in ax.get_xticklabels()])
print("柱高:", [int(p.get_height()) for p in ax.patches])
print("与 value_counts 一致:",
      [int(p.get_height()) for p in ax.patches] == vc.tolist())

assert type(ax).__name__ == "Axes"
assert len(ax.patches) == 5
assert [int(p.get_height()) for p in ax.patches] == [59, 58, 57, 55, 41]
assert ax.get_title() == "设备类型分布"
close_all()"""
    ),
    md(
        """
### 3.2 难点深挖：直方图被离群值撑扁——`range` 圈出主战场

**为什么难**：`负荷值` 有 9999 的哨兵值。直接 `plot(kind="hist", bins=20)`
时，x 轴拉到 10000，**99% 的数据挤在最左边一根柱里**——图"能看"但啥也
看不出来，且**不报任何错**。

**正误对照**（`range=(0, 1000)` 截掉长尾）：

| 写法 | 结果 |
|---|---|
| `hist(bins=20)` 全量 | 主分布挤成一柱，无法判读 |
| `hist(bins=20, range=(0, 1000))` | 20 柱总频数 **238** = 270 − 21 NaN − 7 个 >1000 − 4 个负值 |

**判定规则**：**画 hist 前先看 max（`s.describe()`），有离群值就 `range=`
截主分布，并在报告里注明"截断范围"；总频数 = 非 NaN 数 − range 外数，
必须能对上账。**
"""
    ),
    code(
        """ax = dev["负荷值"].plot(kind="hist", bins=20, range=(0, 1000), edgecolor="white")
ax.set_title("负荷值分布（截断 0~1000）")
ax.set_xlabel("负荷值")

total = sum(p.get_height() for p in ax.patches)
n_above = int((dev["负荷值"] > 1000).sum())
n_below = int((dev["负荷值"] < 0).sum())
print("总频数:", total, "| NaN:", int(dev['负荷值'].isna().sum()),
      "| 上限外:", n_above, "| 负值:", n_below,
      "| 对账:", total == len(dev) - int(dev['负荷值'].isna().sum()) - n_above - n_below)

assert len(ax.patches) == 20
assert total == 238
assert n_above == 7 and n_below == 4
close_all()"""
    ),
    md(
        """
## 3.3 line：时序曲线 + `ax=` 复用

时序线图用 `resample` 聚合后再画（衔接 ch16）。`plot(ax=...)` 可以把图画进
指定 Axes——先 `plt.subplots` 定布局，再逐个往里填。
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(10, 3))
daily.plot(kind="line", ax=ax)
ax.set_title("STATION_A_01 日均负荷（2026 Q1）")
ax.set_ylabel("负荷值")

print("线数:", len(ax.lines), "| 数据点:", len(ax.lines[0].get_xdata()),
      "| y NaN:", int(np.isnan(ax.lines[0].get_ydata()).sum()))

assert len(ax.lines) == 1
assert len(ax.lines[0].get_xdata()) == 90
assert int(np.isnan(ax.lines[0].get_ydata()).sum()) == 0
close_all()"""
    ),
    md(
        """
## 3.4 scatter：两列关系

`DataFrame.plot(kind="scatter", x=..., y=...)` ——**点数 = 总行数**：
NaN 坐标照样进 offsets，只是画布上不渲染。想数"可见点"要自己过滤。
"""
    ),
    code(
        """ax = dev.plot(kind="scatter", x="温度", y="负荷值", alpha=0.5,
              title="温度 × 负荷值")
n_all = len(ax.collections[0].get_offsets())
n_visible = int(dev[["温度", "负荷值"]].notna().all(axis=1).sum())
print("offsets 行数:", n_all, "| 真正可见的点:", n_visible,
      "（", int(dev["负荷值"].isna().sum()), "个 NaN 坐标不渲染）")

assert len(ax.collections) == 1
assert n_all == 270, "scatter 不丢行：NaN 坐标也在 offsets 里"
assert n_visible == 228, "270 − 21 − 23 + 重叠缺失 2"
close_all()"""
    ),
    md(
        """
## 3.5 boxplot：三列并排

箱线图一眼看出中位数 / 四分位 / 离群点。多列 box 的 `ax.lines` 有 21 根
（每箱 6 根：中位、箱体 2、须 2、帽 2、飞点 1 —— 3 箱共 21）。
"""
    ),
    code(
        """ax = dev[["负荷值", "温度", "湿度"]].plot(kind="box")
ax.set_title("三列数值分布对比")

print("lines:", len(ax.lines), "| x 标签数:", len(ax.get_xticklabels()))

assert len(ax.lines) == 21
assert len(ax.get_xticklabels()) == 3
close_all()"""
    ),
    md(
        """
### 3.6 难点深挖：`subplots` + `ax=`——两张图必须各就各位

**为什么难**：`plt.subplots(1, 2)` 返回的 `axes` 是**数组**（一维时是
长度 2 的 array）。不用 `ax=axes[i]` 而直接连画两张 `plot()`，两张图会
**叠在同一个默认 Axes 上**——不报错，图废了。

**正误对照**：

| 写法 | 结果 |
|---|---|
| 连续两次 `df.plot(...)`（不传 ax） | 两图叠加在一张默认图上 ❌ |
| `df.plot(kind=..., ax=axes[0])` / `ax=axes[1]` | 各画各的 ✅ |

**判定规则**：**多子图流程 = `plt.subplots` 先布好局 → 每张图显式传
`ax=` → 每张补齐 `set_title`。** 饼图记得 `ylabel=""` 清掉残留轴标签。
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

dev["缺陷等级"].value_counts().plot(kind="bar", ax=axes[0], title="缺陷等级")
axes[0].set_xlabel("")

dev["处理状态"].value_counts().plot(kind="pie", ax=axes[1], title="处理状态",
                                    ylabel="")

print("axes 形状:", np.array(axes).shape)
print("左图 title:", axes[0].get_title(), "| 柱数:", len(axes[0].patches))
print("右图 title:", axes[1].get_title(), "| 文本块:", len(axes[1].texts),
      "| 扇区数:", len(axes[1].patches))

assert np.array(axes).shape == (2,)
assert axes[0].get_title() == "缺陷等级" and len(axes[0].patches) == 3
assert axes[1].get_title() == "处理状态" and len(axes[1].patches) == 8
close_all()"""
    ),
    md(
        """
### 3.7 难点深挖：`savefig` 才是交付物——不写文件也能验证

**为什么难**：竞赛提交物是**报告里的图**，不是 notebook 里的内嵌输出。
`fig.savefig(路径, dpi=..., bbox_inches="tight")` 是标准动作；在无法写盘的
环境（服务器 / 自动判题），可以存到**内存字节流**验证"图真的渲染出来了"。

**判定规则**：**报告图一律 savefig；dpi ≥ 110、`bbox_inches="tight"` 防裁切；
验证图形产物用字节长度（PNG 有几十 KB，空图只有几百字节）。**
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(10, 3))
daily.plot(kind="line", ax=ax)
ax.set_title("日均负荷")

png = png_bytes(fig)
print("PNG 字节数:", len(png), "| PNG 魔数:", png[:4])

assert png[:4] == b"\\x89PNG", "确实渲染成了 PNG"
assert len(png) > 20_000, "有内容的图不会只有几百字节"
close_all()"""
    ),
    md(
        """
## 3.8 barh 与图例：多列分组条形图

`groupby` 产出的多列 DataFrame 直接 `barh`，两组柱 + 图例自动生成——
图例文字与列名必须一致。
"""
    ),
    code(
        """ax = dev.groupby("设备类型")[["负荷值", "温度"]].mean().plot(kind="barh")
ax.set_title("各设备类型：负荷 vs 温度")

print("柱组数:", len(ax.containers), "| 图例:", [t.get_text() for t in
      ax.get_legend().get_texts()])
print("y 标签:", [t.get_text() for t in ax.get_yticklabels()])

assert len(ax.containers) == 2
assert [t.get_text() for t in ax.get_legend().get_texts()] == ["负荷值", "温度"]
assert len(ax.get_yticklabels()) == 5
close_all()"""
    ),
    md(
        """
---

## 四、易错点清单

1. 中文方框 = 字体列表里没有系统装了的字体；负号变成方块 = 忘了 `axes.unicode_minus`。
2. hist 默认全量范围——9999 压扁主分布；`range=` 截断后对账要同时扣
   NaN、上限外、**负值**三类。
3. scatter 静默丢任一轴为 NaN 的行，点数要按非缺失行数对账。
4. 多子图不传 `ax=` 会全部叠在默认图上，且不报错。
5. `plt.subplots` 的 `axes` 在 1×N 时是一维数组，取元素用 `axes[i]`。
6. 饼图扇区数 = 类别数（含脏值！本数据 处理状态 8 类，清洗前别急着画饼）。
7. `plot` 返回 Axes，链式 `set_title` 写法在 `ax=` 模式下容易漏。
8. 交付靠 `savefig`（dpi ≥ 110 + `bbox_inches="tight"`），内嵌输出不算数。
9. 画完 `plt.close("all")`，否则 40 张图把 notebook 撑爆。

## 五、本章小结

- 统一流程：`subplots` 布局 → `plot(ax=)` 填图 → `set_title/label` 三件套 →
  `savefig` 交付 → `close` 释放。
- 对账三件套：柱高看 `ax.patches`、线看 `ax.lines`、点看 `ax.collections`。
- 离群值先截断再画 hist；脏类别先清洗再画饼。

### 复盘提问

1. 中文字体配置的两行 rcParams 分别解决什么问题？
2. hist 的总频数怎么对账？238 是怎么算出来的？
3. scatter 的点数为什么是 249 而不是 270？
4. 两张图叠在一起的原因和修法？
5. 无法写盘的环境怎么验证 savefig 产出了真图？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch18 练习：可视化

与讲解版逐 Cell 对应，`assert` 验收保留。卡住回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 数据可视化 | bar / hist / line / scatter / box / pie 六种图 + 交付 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：bar 柱高对账（对应讲解 §3.1）"),
    code(
        """# @@todo(1) 缺陷等级的 value_counts 画 bar（标题「缺陷等级分布」），
#            柱高对账存入 heights
ax = dev["缺陷等级"].value_counts().plot(kind="bar", title="缺陷等级分布")
heights = [int(p.get_height()) for p in ax.patches]
# @@end

print("柱数:", len(ax.patches), "| 柱高:", heights)
assert len(ax.patches) == 3
assert heights == [164, 75, 31]
assert ax.get_title() == "缺陷等级分布"
close_all()"""
    ),
    md("## 题 2：hist 截断与对账（对应讲解 §3.2）"),
    code(
        """# @@todo(2) 负荷值画 hist：bins=20、range=(0, 1000)；总频数存入 total
ax = dev["负荷值"].plot(kind="hist", bins=20, range=(0, 1000), edgecolor="white")
total = int(sum(p.get_height() for p in ax.patches))
# @@end

print("总频数:", total)
assert len(ax.patches) == 20
assert total == 238
close_all()"""
    ),
    md("## 题 3：line + ax= 复用（对应讲解 §3.3）"),
    code(
        """# @@todo(3) 先 plt.subplots(figsize=(10,3))，再把 daily 用 line 画进该 Axes；
#            标题「日均负荷」；数据点数存入 n_pts
fig, ax = plt.subplots(figsize=(10, 3))
daily.plot(kind="line", ax=ax)
ax.set_title("日均负荷")
n_pts = len(ax.lines[0].get_xdata())
# @@end

print("数据点:", n_pts)
assert n_pts == 90
assert len(ax.lines[0].get_xdata()) == 90
close_all()"""
    ),
    md("## 题 4：scatter 点数对账（对应讲解 §3.4）"),
    code(
        """# @@todo(4) 温度 × 湿度 散点图；点数存入 n_pts2
# @@hint dev.plot(kind="scatter", x="温度", y="湿度")
ax = dev.plot(kind="scatter", x="温度", y="湿度", alpha=0.5)
n_pts2 = len(ax.collections[0].get_offsets())
# @@end

print("点数:", n_pts2, "| 温度 NaN:", int(dev["温度"].isna().sum()))
assert n_pts2 == 270, "scatter 不丢行：NaN 坐标也在 offsets 里"
close_all()"""
    ),
    md("## 题 5：boxplot（对应讲解 §3.5）"),
    code(
        """# @@todo(5) 负荷值 / 温度 两列并排 box；lines 数存入 n_lines
ax = dev[["负荷值", "温度"]].plot(kind="box")
n_lines = len(ax.lines)
# @@end

print("lines:", n_lines, "| 标签数:", len(ax.get_xticklabels()))
assert n_lines == 14, "每箱 7 根 × 2 列"
assert len(ax.get_xticklabels()) == 2
close_all()"""
    ),
    md("## 题 6：subplots 布局（对应讲解 §3.6）"),
    code(
        """# @@todo(6) 1 行 2 列 subplots：左图画 设备类型 bar（标题「设备类型」）、
#            右图画 缺陷等级 pie（ylabel=\"\"）；右图扇区数存入 n_wedges
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
dev["设备类型"].value_counts().plot(kind="bar", ax=axes[0], title="设备类型")
dev["缺陷等级"].value_counts().plot(kind="pie", ax=axes[1], ylabel="")
n_wedges = len(axes[1].patches)
# @@end

print("左图柱数:", len(axes[0].patches), "| 右图扇区:", n_wedges)
assert len(axes[0].patches) == 5
assert axes[0].get_title() == "设备类型"
assert n_wedges == 3
close_all()"""
    ),
    md("## 题 7（综合）：savefig 到内存（对应讲解 §3.7）"),
    code(
        """# @@todo(7) 画 daily 折线图（ax 用 plt.subplots 造），用 png_bytes 渲染成
#            PNG 字节流存入 png；是 PNG 且 >20000 字节写入断言
fig, ax = plt.subplots(figsize=(10, 3))
daily.plot(kind="line", ax=ax)
ax.set_title("日均负荷")
png = png_bytes(fig)
# @@end

print("PNG 字节:", len(png), "| 魔数:", png[:4])
assert png[:4] == b"\\x89PNG"
assert len(png) > 20_000
close_all()"""
    ),
    md(
        """
---

## 综合自查

1. 题 1 柱高 [164, 75, 31] 和哪张真值表对上？
2. 题 2 的 238 怎么算出来的？
3. 题 4 的点数为什么是 249？
4. 题 6 如果不传 `ax=` 会发生什么？

全部答得上来，01_pandas 的常规章节全部完成（final 综合题另行安排）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
