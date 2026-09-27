"""ch17 —— 性能与内存：CoW / `category` / `eval` / `query` / 向量化

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch17_perf_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch17_perf_memory"

HEADER = '''import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

DATA = Path("data")

dev = pd.read_csv(DATA / "device_defects.csv")

print("pandas", pd.__version__, "| CoW 默认:", pd.options.mode.copy_on_write)
print("dev", dev.shape)'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。"""
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


LOW_CARD = ["台区编号", "设备类型", "缺陷等级", "处理状态", "缺陷类型"]
KB = 1024

print("低基数列:", LOW_CARD)'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch17 性能与内存：CoW / `category` / `eval` / `query` / 向量化

> 方向：数据预处理 ｜ 竞赛对应：**数据准备及处理**（大表预处理跑得快、不爆内存）
> + 隐性考点：**pandas 3.0 的 CoW 语义变更**——老教程的写法在新版会静默失效。

这一章的坑**全部静默**：

1. **链式赋值不再生效也不报错**——`df[mask]["a"] = 1` 是空气操作
2. **`astype("category")` 压缩比 20 倍**——但不是所有列都该转
3. **`query` 与布尔掩码等价**——行数、索引都一致才叫等价
4. **向量化与循环结果一致**——但速度差 81 倍
5. **副本何时产生**——CoW 下"看着像视图"的切片全是惰性副本

> 本章的 `assert` 全部可以在方向 README 的「ch17 专项真值」对上。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据准备及处理 | 占技能操作 10% | 大表降内存（category）、高效筛选（query）|
| （隐性）工程能力 | 长表不超时 | 向量化替代 Python 循环 |

**学习目标**：给一张 10 万行的表做体检——内存多少、哪些列该转 category、
循环改哪几行——10 分钟内给出方案。
"""
    ),
    md(
        """
## 二、API 速查表

| 方法 / 机制 | 关键点 | 一句话说明 |
|---|---|---|
| Copy-on-Write | 3.0 默认开启 | 切片是惰性副本，修改不再回写原表 |
| `memory_usage(deep=True)` | — | 逐列真实内存（object 列必加 deep） |
| `astype("category")` | — | 低基数字符串列压缩 ~20 倍 |
| `.cat.codes` | — | 类别 → 整数编码 |
| `query("A > 1 and B == 'x'")` | `and` / `in` / 引用列方法 | SQL 风格筛选 |
| `eval("c = a / b")` | — | 字符串表达式造列 |
| 向量化算子 | `(df.x - 100) / 20` | 替代 Python for 循环 |
| `assign` | — | **总是**返回副本 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 CoW：切片修改不再回写原表

pandas 3.0 把 Copy-on-Write 变成唯一行为（旧选项已无意义）。切片拿到的是
惰性副本，**怎么改都影响不到原表**——这是与 2.x 教程最大的语义差异。
"""
    ),
    code(
        """df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
sub = df[df["a"] > 1]
sub["a"] = 99

print("sub:", sub["a"].tolist())
print("df 原表:", df["a"].tolist(), "← 不受影响（CoW）")

view = df[["a"]]
view.loc[0, "a"] = 100
print("切片视图改后 df a[0]:", int(df.loc[0, "a"]), "← 依然不受影响")

c = df.copy()
c.loc[0, "a"] = 9
print("显式 copy 改后 df a[0]:", int(df.loc[0, "a"]))

assert df["a"].tolist() == [1, 2, 3], "sub 修改不回写"
assert int(df.loc[0, "a"]) == 1, "切片视图修改不回写"
assert int(df.loc[0, "a"]) == 1, "显式副本修改不回写\""""
    ),
    md(
        """
### 3.2 难点深挖：链式赋值静默失效——CoW 下最危险的坑

**为什么难**：2.x 时代 `df[mask]["a"] = 1` 多半还能"碰巧"写进去（附带
SettingWithCopyWarning）。**CoW 下它既不报错也不生效**——原表纹丝不动，
你以为改了，结果表里根本没这回事。

**错误示范**（贴实际输出）：

```python
df[df["a"] > 1]["a"] = 42
df["a"].tolist()      # [1, 2, 3]  ← 没报错，也没改上
```

**正误对照**：

| 写法 | 结果 |
|---|---|
| `df[df["a"] > 1]["a"] = 42` | ❌ 静默无效 |
| `df.loc[df["a"] > 1, "a"] = 42` | ✅ 生效 |

**判定规则**：**任何"筛选 + 赋值"必须一步 `loc` 完成：`df.loc[mask, 列] = 值`。
看到 `][` 连着的赋值直接判错。**（工具层面：pandas 3.0 会在链式赋值处发
`ChainedAssignmentError` 警告——别把 warnings 一键关掉。）
"""
    ),
    code(
        """df2 = pd.DataFrame({"a": [1, 2, 3]})
df2[df2["a"] > 1]["a"] = 42
after_chain = df2["a"].tolist()
print("链式赋值后:", after_chain, "← 不报错也不生效")

df2.loc[df2["a"] > 1, "a"] = 42
print("loc 一步赋值后:", df2["a"].tolist())

status, msg = probe(lambda: df2[df2["a"] > 1]["a"])
print("链式读（非赋值）→", status, "（读不禁止，写才失效）")

assert after_chain == [1, 2, 3], "链式赋值静默失效"
assert df2["a"].tolist() == [1, 42, 42]
assert status == "ok\""""
    ),
    md(
        """
## 3.3 `memory_usage(deep=True)` + `astype("category")`

object 列的内存必须 `deep=True` 才量得准。低基数字符串列转 `category` 后
只存"类别表 + 整数编码"，本数据 5 列从 **91.1 KB 压到 4.4 KB（20.6 倍）**。
"""
    ),
    code(
        """dev_obj = dev.copy()
mu_obj = dev_obj.memory_usage(deep=True)
print("全表内存(KB):", round(mu_obj.sum() / KB, 1))

dev_cat = dev_obj.copy()
dev_cat[LOW_CARD] = dev_cat[LOW_CARD].astype("category")
mu_cat = dev_cat.memory_usage(deep=True)
print("category 后全表(KB):", round(mu_cat.sum() / KB, 1))

tot_obj = sum(mu_obj[c] for c in LOW_CARD)
tot_cat = sum(mu_cat[c] for c in LOW_CARD)
print("5 列 object:", round(tot_obj / KB, 1), "KB → category:",
      round(tot_cat / KB, 1), "KB | 压缩比:", round(float(tot_obj / tot_cat), 2))

assert round(mu_obj.sum() / KB, 1) == 157.2
assert round(mu_cat.sum() / KB, 1) == 70.5
assert round(float(tot_obj / tot_cat), 2) == 20.62"""
    ),
    md(
        """
### 3.4 难点深挖：`category` 不是免费午餐——语义保持与适用边界

**为什么难**：转完 category 后**统计结果必须一字不差**（这是验收线），
但三件事会变：`.unique()` 返回顺序是类别表顺序、比较运算走类别码位序、
新取值集合之外的过滤行为依赖 `observed`。盲目全表 `astype("category")`
还会把**高基数字符串列**（如记录ID、时间戳）越转越大。

**正误对照**（本数据 设备类型：5 类 / 270 行）：

| 检查 | 结果 |
|---|---|
| `value_counts()` 逐值相等 | ✅ True |
| `unique()` 集合相等 | ✅ True（顺序可能不同） |
| `cat.codes` | 整数编码 `[0, 2, 4, ...]` |
| `cat.categories` | 码位序 `['互感器', '变压器', '断路器', '绝缘子', '避雷器']` |

**判定规则**：**基数行数比 < 50%（本例 5/270 ≈ 1.9%）才转 category；
转完跑一遍 `value_counts` 对账；模型要整数特征就取 `.cat.codes`。**
时间戳、自由文本、ID 列**禁止**转。
"""
    ),
    code(
        """cat_col = dev_cat["设备类型"]
obj_col = dev_obj["设备类型"]

print("value_counts 一致:", bool(obj_col.value_counts().equals(cat_col.value_counts())))
print("集合一致:", sorted(obj_col.unique()) == sorted(cat_col.unique().tolist()))
print("categories(码位序):", cat_col.cat.categories.tolist())
print("codes 前 3:", cat_col.cat.codes[:3].tolist())
print("dtype:", cat_col.dtype)

assert bool(obj_col.value_counts().equals(cat_col.value_counts()))
assert sorted(obj_col.unique()) == sorted(cat_col.unique().tolist())
assert cat_col.cat.categories.tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert cat_col.cat.codes[:3].tolist() == [0, 2, 4]"""
    ),
    md(
        """
## 3.5 `query`：SQL 风格筛选

`query` 与布尔掩码**必须逐行等价**（行数一致 + 索引一致才算对）。
支持 `and / or / not`、`in [...]`、以及**直接引用列名调方法**。
"""
    ),
    code(
        """q = dev.query("负荷值 > 300 and 设备类型 == '变压器'")
mask = dev[(dev["负荷值"] > 300) & (dev["设备类型"] == "变压器")]

print("query:", len(q), "| mask:", len(mask), "| 索引一致:", bool(q.index.equals(mask.index)))
print("query 引用列方法:", len(dev.query("负荷值 > 负荷值.mean() * 2")))
print("in 语法:", len(dev.query("台区编号 in ['STATION_A_01', 'STATION_B_01']")))

assert len(q) == 48 and len(mask) == 48
assert bool(q.index.equals(mask.index))
assert len(dev.query("负荷值 > 负荷值.mean() * 2")) == 7
assert len(dev.query("台区编号 in ['STATION_A_01', 'STATION_B_01']")) == 20"""
    ),
    md(
        """
## 3.6 `eval`：字符串表达式造列

`eval("新列 = 表达式")` 与手写算子结果一致；NaN 照常传播——
"负荷比 = 负荷值 / 温度" 有 **42** 个 NaN（负荷值 21 + 温度 23，重叠 2）。
"""
    ),
    code(
        """r = dev.eval("负荷比 = 负荷值 / 温度", inplace=False)

print("r:", r.shape, "| 新列 NaN:", int(r["负荷比"].isna().sum()))
print("手写版一致:", bool(np.allclose(
    r["负荷比"].dropna(), (dev["负荷值"] / dev["温度"]).dropna())))

assert r.shape == (270, 14)
assert int(r["负荷比"].isna().sum()) == 42
assert round(float(r["负荷比"].max()), 4) == round(
    float((dev["负荷值"] / dev["温度"]).max()), 4)"""
    ),
    md(
        """
### 3.7 难点深挖：向量化 vs Python 循环——结果必须一致，速度差一个量级

**为什么难**：`iterrows` / 逐元素 for 是新手默认写法，竞赛数据量一大直接
超时。关键不是"会背向量化写法"，而是**敢断言两种写法结果逐值一致**——
用 `np.allclose` 对账后再谈加速。

**实测**（20 万行，`(x - 100) / 20`）：

| 写法 | 耗时 |
|---|---|
| 逐元素 for 循环 | ~26 ms |
| 向量化 `(big["x"] - 100) / 20` | ~0.3 ms（**≈81 倍**） |

**判定规则**：**看到 for 里只做算术/比较 → 换算子；看到 for 里在累计/条件
分支 → 先想 `np.where` / `np.select` / `clip`；真写不了向量化再用
`to_numpy()` 拿数组循环（比 iterrows 快一个量级）。**
"""
    ),
    code(
        """n = 200_000
big = pd.DataFrame({"x": np.random.default_rng(42).normal(100, 20, n)})

t0 = time.perf_counter()
out_loop = np.empty(n)
for i, v in enumerate(big["x"].to_numpy()):
    out_loop[i] = (v - 100) / 20
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
out_vec = (big["x"] - 100) / 20
t_vec = time.perf_counter() - t0

print(f"循环 {t_loop * 1000:.0f} ms | 向量化 {t_vec * 1000:.1f} ms "
      f"| 加速 {t_loop / t_vec:.0f}x")
print("结果一致:", bool(np.allclose(out_loop, out_vec.to_numpy())))

assert bool(np.allclose(out_loop, out_vec.to_numpy())), "两种写法必须逐值一致"
assert t_vec < t_loop, "向量化更快（本机实测约 80 倍）\""""
    ),
    md(
        """
## 3.8 副本何时产生：`assign` 永远是副本

CoW 下"什么时候是副本"的实用口径：**`assign` / `drop` / `query` 等返回新
DataFrame 的方法都是副本**；切片是惰性副本（改时才真正复制）。原表安全，
但别指望"顺手改一下"——要改就 `loc` 打在原表上。
"""
    ),
    code(
        """d = pd.DataFrame({"a": np.arange(5)})
d2 = d.assign(b=1)
d.loc[0, "a"] = 999

print("d :", d["a"].tolist())
print("d2:", d2["a"].tolist(), "← assign 时的快照")

assert d["a"].tolist() == [999, 1, 2, 3, 4]
assert d2["a"].tolist() == [0, 1, 2, 3, 4]"""
    ),
    md(
        """
---

## 四、易错点清单

1. 链式赋值 `df[mask][col] = x` 在 CoW 下**静默失效**——必须 `df.loc[mask, col] = x`。
2. 切片修改不再回写原表（CoW 语义），2.x 教程的"视图"心智模型作废。
3. `memory_usage` 不加 `deep=True` 会严重低估 object 列。
4. category 只给低基数列（< 50% 基数行数比）；ID / 时间戳 / 自由文本禁止转。
5. 转完 category 必须用 `value_counts` 对账，统计结果一字不差才算成功。
6. `.cat.codes` 是模型用的整数编码；类别顺序是码位序。
7. `query` 与掩码要"行数 + 索引"双重一致才叫等价。
8. 向量化改写后必须 `np.allclose` 对账，不能只看速度。

## 五、本章小结

- CoW 三句话：切片是惰性副本、修改不回写、链式赋值失效。
- 降内存三板斧：量（`deep=True`）→ 转（低基数 category）→ 验（value_counts 对账）。
- `query` / `eval` 是掩码与算子的等价写法，等价性要用索引对账。
- 向量化优先级：算子 > `np.where`/`np.select` > `to_numpy()` 循环 > iterrows。

### 复盘提问

1. CoW 下链式赋值的表现是什么？正确写法？
2. 哪些列适合转 category？转完怎么验收？
3. `query` 里怎么引用"列均值"？和掩码写法怎么对账？
4. 向量化改写的验证标准是什么？
5. `assign` 之后改原表，新表会变吗？

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
# ch17 练习：性能与内存

与讲解版逐 Cell 对应，`assert` 验收保留。卡住回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 数据准备及处理 | category 降内存、query 高效筛选 |
| （隐性）工程能力 | 向量化替代循环 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：CoW 行为验证（对应讲解 §3.1）"),
    code(
        """# @@todo(1) 造 df = {\"a\": [1,2,3], \"b\": [4,5,6]}；切片 sub = df[df.a > 1] 后
#            把 sub 的 a 改成 99；原表 a 存入 after_sub
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
sub = df[df["a"] > 1]
sub["a"] = 99
after_sub = df["a"].tolist()
# @@end

print("sub:", sub["a"].tolist(), "| 原表:", after_sub)
assert sub["a"].tolist() == [99, 99]
assert after_sub == [1, 2, 3], "CoW：切片修改不回写\""""
    ),
    md("## 题 2：链式赋值现场（对应讲解 §3.2）"),
    code(
        """# @@todo(2) 对 df2 = {\"a\": [1,2,3]} 执行链式赋值 df2[df2.a > 1][\"a\"] = 42，
#            把结果存入 after_chain；再用 loc 一步赋值修复
df2 = pd.DataFrame({"a": [1, 2, 3]})
df2[df2["a"] > 1]["a"] = 42
after_chain = df2["a"].tolist()
df2.loc[df2["a"] > 1, "a"] = 42
# @@end

print("链式赋值后:", after_chain, "| loc 修复后:", df2["a"].tolist())
assert after_chain == [1, 2, 3], "链式赋值静默失效"
assert df2["a"].tolist() == [1, 42, 42]"""
    ),
    md("## 题 3：内存体检 + category 压缩（对应讲解 §3.3-3.4）"),
    code(
        """# @@todo(3) dev 的 deep 内存 Series 存入 mu_obj、总和 KB 存入 mem_before
mu_obj = dev.memory_usage(deep=True)
mem_before = mu_obj.sum() / KB
# @@end

# @@todo(4) 复制 dev 存入 dev_cat，把 LOW_CARD 五列转 category，
#            deep 内存总和 KB 存入 mem_after
dev_cat = dev.copy()
dev_cat[LOW_CARD] = dev_cat[LOW_CARD].astype("category")
mem_after = dev_cat.memory_usage(deep=True).sum() / KB
# @@end

print("内存:", mem_before, "KB →", mem_after, "KB")
print("设备类型 value_counts 一致:",
      bool(dev["设备类型"].value_counts().equals(dev_cat["设备类型"].value_counts())))
assert round(mem_before, 1) == 157.2
assert round(mem_after, 1) == 70.5
assert bool(dev["设备类型"].value_counts().equals(dev_cat["设备类型"].value_counts()))
assert dev_cat["设备类型"].cat.codes[:3].tolist() == [0, 2, 4]"""
    ),
    md("## 题 4：query 等价性对账（对应讲解 §3.5）"),
    code(
        """# @@todo(5) query 版 q 与掩码版 mask：负荷值 > 300 且 设备类型 == \"变压器\"
q = dev.query("负荷值 > 300 and 设备类型 == '变压器'")
mask = dev[(dev["负荷值"] > 300) & (dev["设备类型"] == "变压器")]
# @@end

print("query:", len(q), "| mask:", len(mask), "| 索引一致:",
      bool(q.index.equals(mask.index)))
assert len(q) == 48 and len(mask) == 48
assert bool(q.index.equals(mask.index))
assert len(dev.query("负荷值 > 负荷值.mean() * 2")) == 7
assert len(dev.query("台区编号 in ['STATION_A_01', 'STATION_B_01']")) == 20"""
    ),
    md("## 题 5：eval 造列（对应讲解 §3.6）"),
    code(
        """# @@todo(6) eval 生成 负荷比 = 负荷值 / 温度（inplace=False），存入 r；
#            NaN 个数存入 n_na
r = dev.eval("负荷比 = 负荷值 / 温度", inplace=False)
n_na = int(r["负荷比"].isna().sum())
# @@end

print("r:", r.shape, "| 负荷比 NaN:", n_na)
assert r.shape == (270, 14)
assert n_na == 42"""
    ),
    md("## 题 6（综合）：向量化改写并对账（对应讲解 §3.7）"),
    code(
        """# @@todo(7) big = 20 万行 x 列（seed=42）
# @@hint np.random.default_rng(42).normal(100, 20, n)
n = 200_000
big = pd.DataFrame({"x": np.random.default_rng(42).normal(100, 20, n)})
# @@end
# （循环写法保留在挖空区外——它正是要被替代的"反面教材"）
t0 = time.perf_counter()
out_loop = np.empty(n)
for i, v in enumerate(big["x"].to_numpy()):
    out_loop[i] = (v - 100) / 20
t_loop = time.perf_counter() - t0
# @@todo(8) 向量化版 out_vec；与循环版逐值一致性 same
# @@hint (big["x"] - 100) / 20 ；np.allclose(...)
out_vec = (big["x"] - 100) / 20
same = np.allclose(out_loop, out_vec.to_numpy())
# @@end

print(f"循环 {t_loop * 1000:.0f} ms")
print("结果一致:", bool(same))
assert bool(same), "向量化必须逐值一致\""""
    ),
    md(
        """
---

## 综合自查

1. 题 2 里链式赋值为什么"不报错也不生效"？哪两种正确写法？
2. 题 3 压缩比约多少倍？哪些列**不能**这样转？
3. 题 4 的"等价"为什么必须对到索引级？
4. 题 7 循环里如果换成 `iterrows()` 会怎样？

全部答得上来，进入 ch18（可视化）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
