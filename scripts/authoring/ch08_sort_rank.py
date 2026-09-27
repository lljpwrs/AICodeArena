"""ch08 —— 排序、排名与采样：`sort_values` / `rank` / `nlargest` / `sample`

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch08_sort_rank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch08_sort_rank"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

DATA = Path("data")

dev = pd.read_csv(DATA / "device_defects.csv")
load = pd.read_csv(DATA / "load_curve.csv")

KEY = "台区编号"
VAL = "负荷值"
CAT = "设备类型"

print("pandas", pd.__version__)
print("dev", dev.shape, "| load", load.shape)
print("负荷值缺失:", int(dev[VAL].isna().sum()))'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
    """
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


def stable_check(frame, by, kind):
    """稳定排序判据：同一取值内部，原始「序号」必须严格递增。

    不稳定的排序只保证「值有序」，并列行之间的先后是**未定义的**。
    """
    ordered = frame.sort_values(by, kind=kind)
    pos = ordered["序号"].to_numpy()
    keys = ordered[by].to_numpy()
    same = keys[1:] == keys[:-1]
    return bool(np.all(pos[1:][same] > pos[:-1][same]))


def topn_by_sort(frame, by, n):
    """分组 TopN 的推荐写法：整体排序 + head，比 groupby.apply 快且结果确定。"""
    return (
        frame.sort_values(by, ascending=False, kind="stable")
        .groupby(KEY, sort=False)
        .head(n)
    )


# 演示稳定性用的「大量并列」数据：2 万行、只有 5 个取值
_rng = np.random.default_rng(0)
BIG = pd.DataFrame({"分组": _rng.integers(0, 5, 20_000)})
BIG["序号"] = np.arange(len(BIG))

# 演示 rank 的 method 差异用的小样本（有 2 组并列）
vals = pd.Series([10, 20, 20, 30, 40, 40, 40, 50])
METHODS = ["average", "min", "max", "first", "dense"]

print("BIG", BIG.shape, "| 组数", BIG["分组"].nunique(), "| 平均每组",
      len(BIG) // BIG["分组"].nunique(), "行")'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch08 排序、排名与采样：`sort_values` / `rank` / `nlargest` / `sample`

> 方向：数据预处理 ｜ 竞赛对应：**数据探索**（找极值 / 看分布）+ **数据清洗**（组内名次标记异常）
> + **数据准备及处理**（TopN 报表、可复现抽样）。

本章的六个陷阱，前四个都**不会报错**，只会给你一个看起来正常但其实不对的结果：

1. **`sort_values` 默认 `kind="quicksort"` 是不稳定排序** —— 并列行的先后每次都可能不同
2. **排序不会重置索引** —— 之后 `df.loc[0]` 和 `df.iloc[0]` 指向的是两个完全不同的行
3. **`rank` 默认 `method="average"`** —— 想要"第 1 名 / 第 2 名"的名次，默认给的是 `1.0 / 2.5`
4. **`nlargest(k)` 与"排序后取前 k 行"在并列时结果不同** —— 一个受 `keep` 控制，一个要多少给多少
5. **`sample(frac=0.001)` 会静默返回空表** —— 因为行数是 `round(n * frac)`
6. **`sample(weights=...)` 有两道硬门槛** —— 权重不能为负；不放回时还要求 `n * max(权重) <= 1`

> 本章的 `assert` 全部可以在方向 README 的「ch08 专项真值」对上。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据探索 | 每项 2 分 | `sort_values` 找极值、`value_counts` 看分布、`nlargest` 看 TopN |
| 数据清洗 | 每项 2 分 | 组内 `rank` 标记"本台区第几名"、异常值定位 |
| 数据准备及处理 | 占技能操作 10% | 排序 + 采样 + `reset_index` 交付可复现的子集 |
| 结果交付 | 提交格式错 = 白做 | 索引复位、TopN 表结构、抽样必须固定 `random_state` |

**为什么单独成章**：这是"看起来最简单、扣分最隐蔽"的一章。

- 排序的坑在 **并列**——数据里只要有两行取值相同，默认排序给的结果就不保证可复现，
  而竞赛是**按提交结果对答案**的，一次跑一个样就等于没做；
- 采样的坑在 **`random_state`**——不写死种子，你的结果和别人对不上，也无法重跑；
- 排名的坑在 **默认 `average`**——它把并列的平均掉，产出 `2.5` 这种"半个名次"，
  而业务上的名次是整数。

这一章的操作量很小，但**每一处都要求你写对参数**，属于典型的"一分都丢不起"的章节。
"""
    ),
    md(
        """
## 二、API 速查表

### 2.1 六个入口的分工

| 入口 | 作用对象 | 返回 | 一句话说明 |
|---|---|---|---|
| `df.sort_values(by, ascending, kind, na_position, ignore_index)` | **值** | 新对象（默认） | 按列的值排；索引**不重置** |
| `df.sort_index(axis, ascending, level)` | **标签** | 新对象 | 按行/列**标签**排；常在 `groupby` 之后用 |
| `s.rank(method, ascending, pct, na_option)` | 值 | 名次 Series | 输出**名次**而不是有序数据 |
| `df.nlargest(n, columns, keep)` / `nsmallest` | 值 | 前 n 行 | 比 `sort_values().head(n)` 快，且**跳过 NaN** |
| `df.sample(n, frac, replace, weights, random_state)` | **行** | 随机子集 | 抽样；不写 `random_state` 就不可复现 |
| `s.value_counts(sort, ascending, normalize, dropna, bins)` | 值 | 计数 Series | 频次统计；**默认丢 NaN、默认降序** |

> **`rank` 与 `sort_values` 的区别**：`sort_values` 给你的是**排好序的原始数据**，
> `rank` 给你的是**每一行排第几**（还是原来的行序）。要"名次"就用 `rank`，
> 要"排名单"就用 `sort_values`。混用是本节的第一个高频错误。

### 2.2 关键参数

| 参数 | 取值 | 默认 | 什么时候必须改 |
|---|---|---|---|
| `sort_values.ascending` | `bool` / `list[bool]` | `True` | 多列混合升降序时；长度必须等于 `by` |
| `sort_values.kind` | `quicksort` / `mergesort` / `heapsort` / `stable` | `quicksort` | **有并列且要求结果可复现时**，改 `"stable"` |
| `sort_values.na_position` | `first` / `last` | `last` | 想让缺失排最前时；**不受 `ascending` 影响** |
| `sort_values.ignore_index` | `bool` | `False` | 交付前想要干净的 `0..n-1` 索引时 |
| `rank.method` | `average` / `min` / `max` / `first` / `dense` | `average` | **几乎总要改**：业务名次用 `min`，等级用 `dense` |
| `rank.na_option` | `keep` / `top` / `bottom` | `keep` | 想让缺失行也拿到名次时（`keep` = 保持 NaN） |
| `rank.pct` | `bool` | `False` | 要百分位（组内相对位置）而不是名次时 |
| `nlargest.keep` | `first` / `last` / `all` | `first` | 并列时要不要全给出来 |
| `sample.replace` | `bool` | `False` | 自助法（bootstrap）重采样时设 `True` |
| `value_counts.dropna` | `bool` | `True` | 想看到"缺失有多少个"时必须设 `False` |
| `value_counts.normalize` | `bool` | `False` | 要看占比时；**分母是非缺失计数** |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
---

## 三、正文

### 3.1 `sort_values`：排序的三个必填心态

排序本身一行就会写，但**三个参数决定结果对不对**：
`ascending`（方向）、`kind`（稳定性）、`na_position`（缺失位置）。

先看最朴素的升序 / 降序，以及 `NaN` 落在哪：
"""
    ),
    code(
        """asc = dev.sort_values(VAL)
desc = dev.sort_values(VAL, ascending=False)

print("① 升序前 3：", asc[VAL].head(3).tolist(), "  ← 负数哨兵 -1 排最前")
print("   升序末 3 是否 NaN：", asc[VAL].tail(3).isna().tolist())
print("② 降序前 3：", desc[VAL].head(3).tolist(), "  ← 哨兵 9999 排最前")
print("   降序末 3 是否 NaN：", desc[VAL].tail(3).isna().tolist())
print("\\n→ 结论：na_position 默认 'last'，且**与 ascending 无关**：")
print("  升序和降序下，NaN 都待在末尾。这是最容易记反的一点。")

naf = dev.sort_values(VAL, ascending=False, na_position="first")
print("\\n③ na_position='first' 前 3 是否 NaN：", naf[VAL].head(3).isna().tolist())

print("\\n④ 排序后索引原样保留（没有变成 0,1,2...）：")
print("   desc.index[:6] =", desc.index[:6].tolist())
print("   索引是否单调递增：", bool(desc.index.is_monotonic_increasing))

print("\\n⑤ ignore_index=True 才是干净索引：")
print("   ", dev.sort_values(VAL, ascending=False, ignore_index=True).index[:6].tolist())"""
    ),
    md(
        """
### 3.2 `sort_index`：排的是标签，不是值

`groupby` / `pivot` 之后拿到的索引往往是**业务键**（台区编号、日期），
这时"按标签排队"要用 `sort_index`。它和 `sort_values` 是两件事：
"""
    ),
    code(
        """status_by_key = dev.groupby(KEY)["记录ID"].count()
print("① 按台区聚合后，索引是台区编号（字符串），排序用 sort_index：")
print(status_by_key.sort_index().head(6).to_string())

print("\\n② 同一个 Series，换成 sort_values 排的是「记录数」而不是「台区名」：")
print(status_by_key.sort_values(ascending=False).head(6).to_string())

print("\\n③ 索引是日期 / 数字时同理：")
demo = pd.Series([30, 10, 20], index=[2, 0, 1])
print("   原始      :", demo.to_dict())
print("   sort_index:", demo.sort_index().to_dict())
print("   sort_values:", demo.sort_values().to_dict())
print("   sort_index(ascending=False):", demo.sort_index(ascending=False).to_dict())"""
    ),
    md(
        """
---

### 3.3 难点深挖①：`sort_values` 默认 **不稳定**

**为什么难**：`sort_values` 的 `kind` 默认是 `"quicksort"`，而快排**不保证稳定**。
"不稳定"的意思是：**两行取值相同时，谁在前谁在后是不确定的**。

后果不是报错，而是：你排出来的 TopN 名单里，并列的那几行**顺序可能每次都不一样**。
竞赛按提交结果评分时，这就等于"答案不可复现"。

**错误示范**：直接默认排序，并列行的先后完全看运气：

```python
big.sort_values("分组")            # kind="quicksort"
# → 同一组内部的行号是乱序的
```

**实测**（2 万行、只有 5 个取值，所以并列极多）：

```text
kind=quicksort  稳定？False
kind=heapsort   稳定？False
kind=mergesort  稳定？True
kind=stable     稳定？True
```

**正误对照**：

```python
bottom.sort_values("值", kind="quicksort")   # 并列 1 的行：D, B, F（可能变）
bottom.sort_values("值", kind="stable")      # 并列 1 的行：B, D, F（原序）
```

**判定规则**：

> 只要数据里**可能出现并列**（分类列、计数列、评级列……），并且你需要
> **结果可复现**，就写 `kind="stable"`。
> 只有"取值全部唯一"的连续列（如记录 ID）才可以用默认的 `quicksort`。
> 另外：多列排序时，只要把"唯一键"一起写进 `by`，并列自然被拆开，也不依赖稳定性。

用脚手架里的 `stable_check` 实测四种 `kind`：
"""
    ),
    code(
        """results = {k: stable_check(BIG, "分组", k)
           for k in ["quicksort", "heapsort", "mergesort", "stable"]}
for k, ok in results.items():
    mark = "稳定" if ok else "不稳定 ⚠️"
    print(f"  kind={k:10s} -> {mark}")

print("\\n→ 默认的 quicksort 与 heapsort 都不稳定；只有 mergesort / stable 稳定。")
print("Big 形状:", BIG.shape, "| 取值只有", BIG["分组"].nunique(), "种，所以并列成片。")

print("\\n并列到底长什么样（构造一个 6 行的小例子看得更清楚）：")
tie = pd.DataFrame({"键": ["A", "B", "C", "D", "E", "F"],
                    "值": [2, 1, 2, 1, 2, 1],
                    "序号": list(range(6))})
print(tie.sort_values("值", kind="stable")[["键", "值", "序号"]].to_string(index=False))
print("\\n→ 稳定排序下，值为 1 的三行仍按原始序号 1,3,5 排列。")"""
    ),
    md(
        """
### 3.4 多列排序与混合升降序

`by` 传列表就是**字典序**：先比第一列，第一列相等再比第二列。
`ascending` 也传列表时，就能实现"第一列升序、第二列降序"这种需求——但**长度必须对上**：

```python
dev.sort_values([CAT, VAL], ascending=[True, False])   # 5 类 × 组内负荷降序
dev.sort_values([CAT, VAL], ascending=[True])          # ValueError ⚠️
```

实测：
"""
    ),
    code(
        """multi = dev.sort_values([CAT, VAL], ascending=[True, False])
champ = multi.groupby(CAT, sort=False).head(1)
print("① 「设备类型升序 + 组内负荷降序」后每组第一行 = 该类型的负荷冠军：")
print(champ[[CAT, KEY, VAL]].to_string(index=False))

print("\\n② ascending 长度不匹配：",
      probe(dev.sort_values, [CAT, VAL], ascending=[True])[0],
      probe(dev.sort_values, [CAT, VAL], ascending=[True])[1])
print("   实测消息：Length of ascending (1) != length of by (2)")

print("\\n③ na_position 在多列排序下是「逐列生效」的：")
m2 = dev.sort_values([KEY, VAL], ascending=[True, False])
ok = m2.groupby(KEY, sort=False)[VAL].apply(
    lambda s: bool(s.isna().tolist() == sorted(s.isna().tolist()))
).all()
print("   各台区内部，负荷值的 NaN 都排在末尾：", bool(ok))

print("\\n④ 中文默认按 Unicode 码位排，不是拼音：")
print("   原顺序:", dev[CAT].unique().tolist())
print("   排序后:", sorted(dev[CAT].unique().tolist()))
print("   拼音序应是 避雷器 < 变压器 < 断路器 < 互感器 < 绝缘子 —— 完全不同。")
print("   要按业务顺序排，见 §3.16 的 ordered Categorical。")"""
    ),
    md(
        """
---

### 3.5 难点深挖②：排序**不会**重置索引

**为什么难**：`sort_values` 只动行序、不动索引。排完之后"第一个"到底是哪一个？
`loc[0]` 走的是**标签**，`iloc[0]` 走的是**位置**——两者指向完全不同的行，
而且**两个都不报错**。

**错误示范**：

```python
top = dev.sort_values("负荷值", ascending=False)
top.loc[0, "负荷值"]      # 89.96   ← 这不是最大值！这是原始第 0 行
top.iloc[0, "负荷值"]     # 9999.0  （但列位置也得用整数，写列名会报错）
```

**正误对照**：

| 写法 | 取值 | 含义 |
|---|---|---|
| `top.iloc[0][VAL]` | `9999.0` | ✅ 排完序的第一行 = 全局最大 |
| `top.loc[0][VAL]` | `89.96` | ⚠️ 索引标签 `0` 那一行，与排序无关 |
| `top.head(1)[VAL].iloc[0]` | `9999.0` | ✅ 最稳的写法（`head` 后仍按位置取） |
| `top.reset_index(drop=True).loc[0][VAL]` | `9999.0` | ✅ 重置索引后 `loc[0]` 才有"第一名"的含义 |

**判定规则**：

> 排序之后**不要再对原表用 `loc[整数]`**。要取"第 k 名"用
> `iloc[k]` 或 `head(k)`；要交付干净索引，先 `reset_index(drop=True)` 或
> 排序时直接加 `ignore_index=True`。
> 顺带一个语法坑：`iloc` 的**列也必须用整数位置**，`df.iloc[0]["列名"]` 可以，
> `df.iloc[0, "列名"]` 直接 `ValueError`。

实测：
"""
    ),
    code(
        """desc = dev.sort_values(VAL, ascending=False)
print("① 排序后索引没变：", desc.index[:6].tolist(), "（不是 0,1,2,3,4,5）")

print("\\n② 两种取「第一行」的方式，结果完全不同：")
print("   desc.iloc[0][VAL]        =", desc.iloc[0][VAL], "  ✅ 真的最大")
print("   desc.iloc[0]['记录ID']   =", desc.iloc[0]["记录ID"])
print("   desc.loc[0, VAL]         =", desc.loc[0, VAL], "  ⚠️ 这是原始第 0 行")
print("   desc.loc[0, '记录ID']    =", desc.loc[0, "记录ID"])
print("   dev.loc[0, '记录ID']     =", dev.loc[0, "记录ID"], "  ← 和原表一致，说明确实没动")

print("\\n③ 想用 loc[0] 取第一名，必须先重排索引：")
desc_ri = desc.reset_index(drop=True)
print("   reset 后 desc_ri.loc[0, VAL] =", desc_ri.loc[0, VAL], "  ✅")

print("\\n④ iloc 的列也必须是位置，写列名直接报错：")
print("   probe(lambda: desc.iloc[0, VAL]) ->",
      probe(lambda: desc.iloc[0, VAL])[1], "⚠️")
print("   正确写法：desc.iloc[0][VAL] 或 desc[VAL].iloc[0] =", desc[VAL].iloc[0])"""
    ),
    md(
        """
---

### 3.6 `rank`：输出的是"名次"，不是"排名单"

`rank` 返回一个**与输入等长、索引相同**的 Series，值是该行的名次。
它最常见的用途是给分组加一列"组内第几名"：

```python
dev["组内排名"] = dev.groupby("台区编号")["负荷值"].rank(ascending=False, method="min")
```

注意 `ascending=False` 时名次 1 = **最大值**。
默认 `ascending=True`，名次 1 = 最小值——这一点在"缺陷严重度""负荷高低"场景下经常写反。
"""
    ),
    code(
        """print("① 升序排名（默认）：值越小名次越小")
print("   vals =", vals.tolist())
print("   rank()          =", vals.rank().tolist())

print("\\n② 降序排名：名次 1 = 最大值")
print("   rank(ascending=False) =", vals.rank(ascending=False).tolist())

print("\\n③ pct=True 给的是百分位（(0,1]），不是名次")
print("   rank(pct=True)  =", vals.rank(pct=True).tolist())

print("\\n④ rank 与 sort_values 的区别：rank 保持原行序")
print("   vals.rank() 的前 3 个（对应原值 10,20,20）:", vals.rank().head(3).tolist())
print("   vals.sort_values() 的前 3 个:", vals.sort_values().head(3).tolist())"""
    ),
    md(
        """
---

### 3.7 难点深挖③：`rank` 的 `method` 默认是 **`average`**

**为什么难**：数据里只要有并列，五种 `method` 会给出**五套不同的名次**，
而且都不报错。竞赛里"给每个台区按负荷高低排名次"这种题，
如果并列时交了 `2.5` 这种半个名次，业务上就是错的。

用 `vals = [10, 20, 20, 30, 40, 40, 40, 50]` 实测（两个 20、三个 40）：

```text
method=average  [1.0, 2.5, 2.5, 4.0, 6.0, 6.0, 6.0, 8.0]
method=min      [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 5.0, 8.0]
method=max      [1.0, 3.0, 3.0, 4.0, 7.0, 7.0, 7.0, 8.0]
method=first    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
method=dense    [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0]
```

**怎么读这五行**：

| method | 并列的 20 拿 | 并列的 40 拿 | 语义 | 典型场景 |
|---|---|---|---|---|
| `average`（默认） | 2.5 | 6.0 | 占用名次的**平均** | 统计上的"秩"，竞赛里通常**不要** |
| `min` | 2.0 | 5.0 | 并列全给**最好**名次 | **业务名次**：并列第 2 名 |
| `max` | 3.0 | 7.0 | 并列全给**最差**名次 | 保守口径的名次 |
| `first` | 2.0 / 3.0 | 5.0 / 6.0 / 7.0 | 按**出现顺序**强行拆开 | 必须无并列（如派奖） |
| `dense` | 2.0 | 4.0 | 名次**连续不跳号** | **等级 / 档次**（第 1 档、第 2 档） |

**两个可直接背的结论**：

- `min` 的最大值 = **行数**（本数据 49 行 → 最大名次 49 附近）
- `dense` 的最大值 = **不同取值的个数**（本数据 46 个不同值 → 最大名次 46）

**判定规则**：

> 业务名次 → `method="min"`；分等级/档位 → `method="dense"`；
> 必须唯一且可复现 → `method="first"`（但要配合稳定排序，否则 `first` 依赖的"出现顺序"本身不稳定）。
> **默认的 `average` 只在纯统计场景下才是对的。**

实测三种 `method` 在真实数据上的差异：
"""
    ),
    code(
        """tbl = pd.DataFrame({m: vals.rank(method=m) for m in
                    ["average", "min", "max", "first", "dense"]})
tbl.insert(0, "原值", vals)
print("① 五种 method 并排：")
print(tbl.to_string(index=False))

print("\\n② 只有 average 会产出「半个名次」（.5）：")
print("   average 里出现 .5 的个数:", int((vals.rank(method='average') % 1 != 0).sum()))
print("   min/max/dense/first 里 .5 的个数:",
      [int((vals.rank(method=m) % 1 != 0).sum()) for m in ["min", "max", "dense", "first"]])

print("\\n③ min 的最大值 = 行数，dense 的最大值 = 不同取值个数：")
lv = dev.loc[dev[CAT] == "变压器", VAL].dropna()
print("   变压器负荷值：行数", len(lv), "| 不同取值", lv.nunique())
print("   rank(method='min')  最大 =", float(lv.rank(method="min").max()))
print("   rank(method='dense') 最大 =", float(lv.rank(method="dense").max()))

print("\\n④ na_option 决定缺失行怎么处理（默认 keep = 保持 NaN）：")
sn = pd.Series([5.0, np.nan, 3.0, 1.0, np.nan, 7.0])
print("   原值            :", sn.tolist())
print("   rank() 默认     :", sn.rank().tolist())
print("   rank(na_option='top')   :", sn.rank(na_option="top").tolist())
print("   rank(na_option='bottom'):", sn.rank(na_option="bottom").tolist())"""
    ),
    md(
        """
### 3.8 `rank` 实战：给每个台区加一列组内名次

这是竞赛里 `rank` 的最高频用法：**在明细表里标出"这条记录在本台区排第几"**，
再据此挑出"每台区的负荷 Top3"或"本台区最严重的缺陷"。

关键点：`groupby(...)[列].rank(...)` 返回的 Series **索引与 `dev` 对齐**，
所以可以直接赋值成新列。
"""
    ),
    code(
        """dev["组内排名"] = dev.groupby(KEY)[VAL].rank(ascending=False, method="min")
dev["组内百分位"] = dev.groupby(KEY)[VAL].rank(ascending=False, pct=True)

print("① 缺失值的名次也是缺失（na_option 默认 keep）：")
print("   负荷值缺失行数:", int(dev[VAL].isna().sum()),
      "| 名次缺失行数:", int(dev["组内排名"].isna().sum()))

print("\\n② 「台区冠军」有多少行？不是 18 行：")
print("   台区数:", dev[KEY].nunique(), "| 名次 == 1 的行数:", int((dev["组内排名"] == 1).sum()))
print("   → 差额来自并列冠军：STATION_E_03 有两个 9999")

print("\\n③ 组内前 3 名合计：", int((dev["组内排名"] <= 3).sum()),
      "= 18 台区 x 3（该台区记录都够 3 条）")

print("\\n④ 组内百分位最小 = 1 / 「非缺失数最多的组」的个数：")
print("   最小值:", round(float(dev["组内百分位"].min()), 6),
      "| 各组非缺失负荷值个数的最大值:", int(dev.groupby(KEY)[VAL].count().max()))
print("   ⚠️ 注意分母是**非缺失数**（22），不是行数（25）—— 缺失行根本不参与 rank")

print("\\n⑤ 两种等价写法（groupby.rank 与 transform(rank)）：")
r1 = dev.groupby(KEY)[VAL].rank(ascending=False, method="min")
r2 = dev.groupby(KEY)[VAL].transform(lambda s: s.rank(ascending=False, method="min"))
print("   完全一致:", bool((r1.fillna(-1) == r2.fillna(-1)).all()))"""
    ),
    md(
        """
### 3.9 `nlargest` / `nsmallest`：比"排序后取前几行"更好

| 写法 | 复杂度 | NaN | 并列 | 索引顺序 |
|---|---|---|---|---|
| `df.nlargest(k, col)` | O(n log k)，k 小则很快 | **跳过** | 由 `keep` 决定 | 按值降序 |
| `df.sort_values(col, ascending=False).head(k)` | O(n log n) | 排到末尾，`head` 自然避开 | 有多少给多少 | 按值降序（并列按稳定序） |

`nlargest` 的 `keep` 参数是关键：

- `keep="first"`（默认）：**严格给 k 行**，并列的先到先得
- `keep="all"`：并列的全部给出，**行数可能 > k**

本数据里 `9999` 有 4 个，所以 `nlargest(3, keep="all")` 会给你 **4 行**。
"""
    ),
    code(
        """print("① nlargest(5)：", dev.nlargest(5, VAL)[VAL].tolist())
print("   nsmallest(5)：", dev.nsmallest(5, VAL)[VAL].tolist())

print("\\n② keep 决定并列时给几行：")
print("   nlargest(3, keep='first') 行数:", len(dev.nlargest(3, VAL)), "（默认，严格 3 行）")
print("   nlargest(3, keep='all')   行数:", len(dev.nlargest(3, VAL, keep="all")), "（4 个 9999 并列）")

print("\\n③ nlargest 永远不返回 NaN；sort+head 要靠 na_position 兜底：")
sn2 = pd.Series([5.0, np.nan, 3.0, 9.0, np.nan, 1.0], index=list("abcdef"))
print("   nlargest(3)                        :", sn2.nlargest(3).to_dict())
print("   sort desc + head(3)                :", sn2.sort_values(ascending=False).head(3).to_dict())
print("   sort desc + na_position='first'    :",
      sn2.sort_values(ascending=False, na_position="first").head(3).to_dict(),
      "⚠️ 前 3 个里混进了 2 个 NaN")

print("\\n④ 并列时的行序：nlargest 与 sort+head 不保证一致")
print("   sort+head 索引:", dev.sort_values(VAL, ascending=False).index[:6].tolist())
print("   nlargest  索引:", dev.nlargest(6, VAL).index.tolist())

print("\\n⑤ 分组 TopN：`groupby.nlargest` 返回多层索引，交付前必须 reset_index")
gn = dev.groupby(CAT)[VAL].nlargest(1)
print("   形状:", gn.shape, "| 索引层级数:", gn.index.nlevels, "| 索引名:", list(gn.index.names))
print("   推荐写法（先整体稳定排序再 head，索引干净）：")
print("   topn_by_sort 行数:", len(topn_by_sort(dev, VAL, 2)))"""
    ),
    md(
        """
---

### 3.10 `sample`：抽样的第一要务是**可复现**

```python
dev.sample(5, random_state=42)      # 每次都是同一批
dev.sample(5)                       # 每次都不一样 ⚠️
```

`random_state` 就是"种子"。竞赛里凡是涉及随机（抽样、划分训练集、初始化），
**不写种子等于交了一份无法复核的答案**。

`n` 与 `frac` 二选一（同时给会 `ValueError`）；`frac > 1` 必须开 `replace=True`。
"""
    ),
    code(
        """print("① 固定种子 -> 完全可复现：")
print("   sample(5, random_state=42):", dev.sample(5, random_state=42).index.tolist())
print("   再跑一次                  :", dev.sample(5, random_state=42).index.tolist())

print("\\n② 不写种子 -> 每次都不同：")
print("   ", dev.sample(5).index.tolist(), dev.sample(5).index.tolist())

print("\\n③ frac 的行数是 round(n * frac)，可能被静默舍成 0：")
print("   270 x 0.01  = 2.7 -> round =", round(270 * 0.01), "| 实际行数:",
      len(dev.sample(frac=0.01, random_state=42)))
print("   270 x 0.001 = 0.27 -> round =", round(270 * 0.001), "| 实际行数:",
      len(dev.sample(frac=0.001, random_state=42)), "⚠️ 静默空表")

print("\\n④ n 与 frac 同时给：", probe(dev.sample, n=5, frac=0.1, random_state=42))

print("\\n⑤ frac > 1 必须放回：", probe(dev.sample, frac=2.0, random_state=42))
boot = dev.sample(frac=2.0, replace=True, random_state=42)
print("   replace=True 后行数:", len(boot), "| 重复索引数:", int(boot.index.duplicated().sum()))

print("\\n⑥ 分层抽样：按台区（或设备类型）各组等量抽")
strat = dev.groupby(CAT, group_keys=False).sample(2, random_state=42)
print("   形状:", strat.shape, "| 每类型行数一致:", set(strat.groupby(CAT).size().tolist()))

print("\\n⑦ 抽列而不是抽行（axis=1）：", dev.sample(3, axis=1, random_state=42).columns.tolist())"""
    ),
    md(
        """
---

### 3.11 难点深挖④：`sample(weights=...)` 的**两道硬门槛**

**为什么难**：加权抽样看起来只是多传一个列名，但 pandas 会**直接抛 `ValueError`**，
而且两条报错信息都很难懂。更糟的是第一条报错的文案里还有个官方笔误（`many not`），
搜索引擎都不好搜。

**门槛一：权重不能为负。**

```python
dev.sample(10, weights="负荷值")
# ValueError: weight vector many not include negative values
```

本表的 `负荷值` 里有 **4 个 `-1` 哨兵**，所以直接拿它当权重必然失败。

**门槛二：不放回时要求 `n * max(归一化权重) <= 1`。**

```python
dev_pos.sample(10, weights="权重")     # 仍然 ValueError
# ValueError: Weighted sampling cannot be achieved with replace=False.
#             Either set replace=True or use smaller weights.
```

原因在源码里：pandas 先把权重归一化成概率 `p`，然后检查

```python
if not replace and size * weights.max() > 1:   # weights 已归一化
    raise ValueError(...)
```

本表归一化后最大权重 ≈ `0.105759`，`10 × 0.105759 = 1.0576 > 1` → 报错。

**正误对照**：

| 写法 | 结果 |
|---|---|
| `sample(10, weights="负荷值")` | ❌ `ValueError`（含负值） |
| `sample(10, weights="权重")` | ❌ `ValueError`（`n * max(p) > 1`） |
| `sample(10, weights="权重", replace=True)` | ✅ 可行，但同一行可能被抽中多次 |
| `sample(2, weights="权重")` | ✅ `2 × 0.105759 = 0.2115 <= 1`，可行 |

**判定规则**：

> 用 `weights` 前先做两件事：
> ① `weights = 目标列.clip(lower=0).fillna(0)`（清掉负值与缺失）；
> ② 算一下 `n * 权重.max() / 权重.sum()` 是否 ≤ 1，超了就**减少 n 或改用 `replace=True`**。
> 另外注意：加权抽样会把你抽出来那几行的均值**严重拉高**（本次实测 `5666.52` vs 全表 `379.68`），
> 这不是 bug——它本来就是按权重大小优先抽的。

实测三档：
"""
    ),
    code(
        """print("① 权重含负值 ->", probe(dev.sample, 10, weights=VAL, random_state=42))

dpos = dev.assign(权重=dev[VAL].clip(lower=0).fillna(0))
wsum, wmax = float(dpos["权重"].sum()), float(dpos["权重"].max())
print("\\n② 清掉负值后：权重和 =", round(wsum, 2), "| 最大 =", round(wmax, 2),
      "| max/sum =", round(wmax / wsum, 6))
print("   n=10 时 n * max(p) =", round(10 * wmax / wsum, 6), "> 1 ->",
      probe(dpos.sample, 10, weights="权重", random_state=42))

print("\\n③ replace=True 就可行：", dpos.sample(10, weights="权重", replace=True,
                                              random_state=42).index.tolist())
print("   加权抽样均值:", round(float(dpos.sample(10, weights="权重", replace=True,
                                              random_state=42)[VAL].mean()), 4))
print("   等权抽样均值:", round(float(dev.sample(10, random_state=42)[VAL].mean()), 4))
print("   全表均值    :", round(float(dev[VAL].mean()), 4))

print("\\n④ 减小 n 到 2 也可以不放回：", dpos.sample(2, weights="权重", random_state=42).index.tolist(),
      "| 2 * max(p) =", round(2 * wmax / wsum, 6))"""
    ),
    md(
        """
---

### 3.12 `value_counts`：看分布的第一选择

`value_counts` 是"看这一列都取哪些值、各有多少个"，比 `groupby().size()` 更好用
（不需要先设索引、自带排序）。

三个使用频率最高的参数：

| 参数 | 作用 | 默认值的坑 |
|---|---|---|
| `dropna` | 是否统计缺失 | 默认 `True` → **看不到"缺失有几个"** |
| `normalize` | 输出占比还是计数 | **分母是非缺失计数**，不是总行数 |
| `ascending` / `sort` | 排序 | 默认按**计数降序**；`sort=False` 按**出现顺序** |

本表 `缺陷类型` 是个好例子——它有 **10 种写法**（`锈蚀` 和 `锈蚀 `、`渗油` 和 `渗漏油` 都算不同值），
这类"同义不同写法"必须先做 §ch06 的字符串归一，才能得到正确的分布。
"""
    ),
    code(
        """print("① 设备类型：5 类（默认按计数降序）")
print(dev[CAT].value_counts().to_string())

print("\\n② 缺陷类型：原样 10 种写法（含尾随空格）")
vc_def = dev["缺陷类型"].value_counts()
print(vc_def.to_string())
print("\\n   总和 =", int(vc_def.sum()), "= dev 行数", len(dev), "（本列无缺失）")

print("\\n③ 去掉尾随空格后立刻少一种：", dev["缺陷类型"].str.strip().nunique(), "种")
print(dev["缺陷类型"].str.strip().value_counts().head(6).to_string())

print("\\n④ dropna=False 才能看到缺失：")
print("   备注 value_counts() 条目数:", dev["备注"].value_counts().shape[0])
print("   备注 value_counts(dropna=False) 条目数:", dev["备注"].value_counts(dropna=False).shape[0],
      "← 多出来的就是 NaN")
print("   缺陷等级无缺失，两者相同:", dev["缺陷等级"].value_counts(dropna=False).to_dict())

print("\\n⑤ normalize 的分母：")
print("   发热占比:", round(float(vc_def.iloc[0]) / 270, 6),
      "| normalize 结果:", round(float(vc_def.loc['发热'] / vc_def.sum()), 6))

print("\\n⑥ sort=False 改为「出现顺序」：")
print("   unique 顺序:", dev["缺陷类型"].dropna().unique().tolist()[:4], "...")
print("   sort=False :", dev["缺陷类型"].value_counts(sort=False).head(4).to_dict())

print("\\n⑦ DataFrame.value_counts 看多维组合：")
print("   设备类型 x 缺陷等级 组合数:", dev[[CAT, "缺陷等级"]].value_counts().shape[0])
print(dev[[CAT, "缺陷等级"]].value_counts().head(6).to_string())

print("\\n⑧ 索引名 / name 都有含义：",
      "index.name =", dev[CAT].value_counts().index.name,
      "| name =", dev[CAT].value_counts().name)"""
    ),
    md(
        """
---

### 3.13 难点深挖⑤：`value_counts` 默认**丢掉缺失**、默认**降序**

**为什么难**：这两件事都能让"分布表"看起来完全正常，但数字对不上。

**坑一：默认丢 NaN。**

```python
dev["备注"].value_counts().shape[0]              # 2    ← 只有 2 个非缺失取值
dev["备注"].value_counts(dropna=False).shape[0]  # 3    ← 多的一项是 NaN（196 个）
```

后果：你算"各类占比"时，分母只有 `270 - 196 = 74`，但写报告时容易当成"全量 270"。
**要么显式 `dropna=False`，要么在报告里写清分母是"非缺失样本数"。**

**坑二：默认按计数降序。**

```python
dev["缺陷类型"].value_counts(sort=False)   # 按"出现顺序"，不是字母序也不是原始顺序
```

`sort=False` 的顺序是**首次出现的顺序**，这既不是"字典序"也不是"稳定顺序"，
跨版本 / 跨数据子集都会变。想按标签排序要**显式**再 `.sort_index()`。

**正误对照**：

| 需求 | 写法 | 结果 |
|---|---|---|
| 看取值分布（含缺失） | `s.value_counts(dropna=False)` | 缺失单独一项 |
| 看占比（分母=非缺失） | `s.value_counts(normalize=True)` | 和 = 1.0 |
| 按标签（台区编号）排序 | `s.value_counts().sort_index()` | 按编号自然序 |
| 按频次升序看长尾 | `s.value_counts(ascending=True)` | 最少的排最前 |

实测：
"""
    ),
    code(
        """vc = dev["缺陷类型"].value_counts()
print("① 默认降序：第一个就是最多的")
print("   第一个:", vc.index[0], "=", int(vc.iloc[0]), "| 是否是最大值:",
      bool(vc.iloc[0] == vc.max()))

print("\\n② ascending=True 看长尾（先看到最少的）：")
print(dev["缺陷类型"].value_counts(ascending=True).head(4).to_string())

print("\\n③ sort=False 是「首次出现顺序」，不是字典序：")
print("   出现顺序:", dev["缺陷类型"].value_counts(sort=False).head(4).to_dict())
print("   字典序  :", dev["缺陷类型"].value_counts().sort_index().head(4).to_dict())

print("\\n④ 台区编号是字符串，'STATION_A_10' 会排在 'STATION_A_9' 前面：")
sidx = dev[KEY].value_counts().sort_index()
print("   sort_index 末 4 个:", sidx.index[-4:].tolist())

print("\\n⑤ normalize 之和恒为 1（分母是**非缺失**计数，这点对不上就说明有缺失）：")
print("   缺陷类型 normalize 之和:", round(float(dev['缺陷类型'].value_counts(normalize=True).sum()), 10))
print("   缺陷类型 非缺失计数:", int(dev['缺陷类型'].notna().sum()), "| 总行数:", len(dev))

print("\\n⑥ bins 直接把数值列分箱计数（不必先 cut）：")
print(dev[VAL].value_counts(bins=3).to_string())"""
    ),
    md(
        """
---

### 3.14 分组 TopN：两种写法与各自的坑

"每个台区负荷最高的 N 条"是竞赛里的标准题。两种写法：

| 写法 | 返回 | 坑 |
|---|---|---|
| `df.groupby(g).apply(lambda x: x.nlargest(n, col), include_groups=False)` | 依赖子表列 | 慢；返回结构会随 `nlargest` 的输出变化；列会被 `include_groups` 影响 |
| **`df.sort_values(...).groupby(g, sort=False).head(n)`** | 与 `df` 同结构 | ✅ 快、列完整、索引可直接用 |

第二种更好用的原因是：`groupby(...).head(n)` 取的是"每组前 n **行**"，
配合"先按目标列整体排好序"，就自然得到"每组目标列最大的 n 行"。

**注意 `sort=False`**：不加它，`groupby` 会按组名重新排序，打乱你辛苦排好的顺序。
"""
    ),
    code(
        """print("① 推荐写法：sort_values(稳定) + groupby(sort=False).head(n)")
top2 = topn_by_sort(dev, VAL, 2)
print("   行数:", len(top2), "= 18 台区 x 2")
print("   列完整（与 dev 同列数）:", top2.shape[1] == dev.shape[1])
print(top2.sort_values([KEY, VAL], ascending=[True, False])[[KEY, VAL]].head(6).to_string(index=False))

print("\\n② groupby.nlargest 的返回是多层索引，交付前必须 reset_index：")
gn = dev.groupby(CAT)[VAL].nlargest(1)
print("   形状:", gn.shape, "| 索引层级:", gn.index.nlevels, "| 索引名:", list(gn.index.names))
gn_flat = gn.reset_index()
print("   reset_index 后列名:", list(gn_flat.columns))

print("\\n③ 组内名次路线（另一种等价做法）：")
by_rank = dev[dev["组内排名"] <= 2]
print("   组内排名 <= 2 的行数:", len(by_rank))

print("\\n④ 两种路线的差异：并列时「名次路线」会多给行")
champ_rows = dev[dev["组内排名"] == 1]
print("   名次路线（rank == 1）给出:", len(champ_rows), "行 —— 并列全算进去")
print("   head(1) 路线给出:", len(dev.sort_values(VAL, ascending=False, kind='stable')
                                .groupby(KEY, sort=False).head(1)), "行 —— 每组严格 1 行")
print("   → 差额 1 行 = STATION_E_03 的两个 9999 并列冠军")
print("   → 想要「每组恰好 1 行」用 head(1)；想要「所有并列冠军」用 rank == 1")"""
    ),
    md(
        """
### 3.15 实战：在 `load_curve` 上找峰值时刻

`load_curve.csv` 是 3 个台区 × 90 天 × 24 小时的负荷曲线（6480 行）。
"找全网峰值时刻""找每个台区最尖峰的 3 个时刻"，正是排序 + `nlargest` 的主场。

一个必踩的坑：**`时间戳` 读进来是字符串**，`sort_values("时间戳")` 排的是字典序。
本数据的格式 `2026-01-01 00:00:00` 恰好是"字典序 = 时间序"，
但换成 `2026/1/1` 这种格式就会排错——所以排序前应该先 `pd.to_datetime`。
"""
    ),
    code(
        """print("① 时间戳读进来是字符串：", load["时间戳"].dtype)
print("   本数据格式恰好「字典序 = 时间序」，但仍应显式解析：")
load["时间戳"] = pd.to_datetime(load["时间戳"])
print("   解析后 dtype:", load["时间戳"].dtype)

print("\\n② 全网负荷最高的 5 个时刻：")
print(load.nlargest(5, VAL)[["时间戳", KEY, VAL]].to_string(index=False))

print("\\n③ 全网负荷最低的 5 个时刻：")
print(load.nsmallest(5, VAL)[["时间戳", KEY, VAL]].to_string(index=False))
print("   曲线负荷值缺失:", int(load[VAL].isna().sum()), "（nlargest / nsmallest 都会跳过）")

print("\\n④ 每个台区的当日峰值（先按天聚合，再对聚合结果排序）：")
load["日期"] = load["时间戳"].dt.date
daily = load.groupby(["日期", KEY])[VAL].max().reset_index()
print("   逐日峰值表形状:", daily.shape, "= 90 天 x 3 台区")
print(daily.nlargest(3, VAL).to_string(index=False))

print("\\n⑤ 单台区最尖峰的 3 个时刻：")
one = load[load[KEY] == "STATION_A_01"]
print(one.nlargest(3, VAL)[["时间戳", VAL]].to_string(index=False))"""
    ),
    md(
        """
### 3.16 附：用 `ordered Categorical` 指定**业务顺序**

`处理状态` 是本数据里最好的例子：业务顺序应该是
**未处理 → 处理中 → 处理完成**，但它的 Unicode 码位序是
`处理中 < 处理完成 < 未处理`（"未"字的码位 U+672A 大于"处"字 U+5904）——
**排序结果和业务完全相反**。

修法是把这一列变成**有序类别**，让 `categories` 的顺序说了算：

```python
order = ["未处理", "处理中", "处理完成"]
ordered = s.astype(pd.CategoricalDtype(order, ordered=True))
ordered.sort_values()      # → 未处理, 未处理, 处理中, 处理完成  ✅
```

三个必须知道的点：

| 点 | 说明 |
|---|---|
| `astype("category")` 不够 | 它把 `categories` 设成**排序后的唯一值**，顺序就是码位序，等于没改 |
| 必须显式给 `categories=` | 只有自己写的顺序才是业务顺序 |
| `ordered=True` 才有比较运算 | 无序类别能排序，但 `min()` / `max()` / `<` 直接 `TypeError` |

顺带一个好消息：`rank()` 对 `Categorical` 也能正常工作（内部先转成类别码）。
"""
    ),
    code(
        """order = ["未处理", "处理中", "处理完成"]
s_status = pd.Series(["处理完成", "未处理", "处理中", "未处理"])

print("① 原始顺序          :", s_status.tolist())
print("② 普通字符串排序（码位序）:", s_status.sort_values().tolist(), "⚠️ 语义完全反了")

ordered = s_status.astype(pd.CategoricalDtype(order, ordered=True))
print("\\n③ 显式给 categories + ordered=True：")
print("   类别顺序:", list(ordered.cat.categories), "| ordered =", bool(ordered.cat.ordered))
print("   sort_values:", ordered.sort_values().tolist(), "✅")
print("   类别码    :", ordered.cat.codes.tolist(), "（未处理=0, 处理中=1, 处理完成=2）")
print("   min / max :", ordered.min(), "/", ordered.max())
print("   rank(dense):", ordered.rank(method="dense").tolist())

print("\\n④ 只用 astype('category') 等于没改（categories 自动按码位序）：")
plain_cat = s_status.astype("category")
print("   类别:", list(plain_cat.cat.categories), "| ordered =", bool(plain_cat.cat.ordered))
print("   sort_values:", plain_cat.sort_values().tolist(), "⚠️ 还是码位序")

print("\\n⑤ 无序类别能排序，但不能比大小：")
print("   probe(min) ->", probe(plain_cat.min), "| probe(sort_values) ->", probe(plain_cat.sort_values)[0])
print("   实测消息：Categorical is not ordered for operation min")"""
    ),
    md(
        """
---

## 四、易错点清单

1. **`sort_values` 默认 `kind="quicksort"` 不稳定**——有并列且要可复现时写 `kind="stable"`。
2. **`na_position` 不受 `ascending` 影响**——升序降序下 NaN 都默认在末尾。
3. **`ascending` 是列表时长度必须等于 `by`**，否则 `ValueError`。
4. **排序不重置索引**——`loc[0]` 拿的是原始第 0 行，不是第一名。
5. **`df.iloc[0, "列名"]` 报 `ValueError`**——`iloc` 的列也必须是整数位置。
6. **`rank` 默认 `method="average"`**——并列会给 `2.5` 这样的半个名次。
7. **`rank` 默认 `ascending=True`**——名次 1 是**最小**值，不是最大值。
8. **`rank` 默认 `na_option="keep"`**——缺失行的名次也是缺失，不会自动排在最后。
9. **`nlargest` 会跳过 NaN**，而 `sort_values().head()` 靠 `na_position` 兜底。
10. **`nlargest(k, keep="all")` 可能返回多于 k 行**——并列全部给出。
11. **`groupby().nlargest()` 返回多层索引**，交付前必须 `reset_index`。
12. **`sample` 不写 `random_state` 不可复现**——凡是随机都要写死种子。
13. **`sample(frac=小值)` 可能返回空表**——行数按 `round(n * frac)` 计算，可能舍成 0。
14. **`sample(n=..., frac=...)` 同时给会 `ValueError`**，只能给一个。
15. **`sample(frac>1)` 必须 `replace=True`**。
16. **`sample(weights=...)` 有负值直接 `ValueError`**——先 `clip(lower=0).fillna(0)`。
17. **加权 + 不放回还要求 `n * max(归一化权重) <= 1`**，否则同样 `ValueError`。
18. **加权抽样会显著拉高样本均值**（本次 5666.52 vs 全量 379.68），不是 bug。
19. **`value_counts` 默认丢 NaN**——要看缺失必须 `dropna=False`。
20. **`value_counts(normalize=True)` 的分母是非缺失计数**，不是总行数。
21. **`value_counts(sort=False)` 是"首次出现顺序"**，既非字典序也非稳定序。
22. **字符串编号排序是字典序**，`STATION_A_10` 会排在 `STATION_A_9` 之前。
23. **`时间戳` 读进来是字符串**，务必 `pd.to_datetime` 后再排。
24. **`groupby(...).head(n)` 不加 `sort=False` 会按组名重排**，打乱你的顺序。
25. **`astype("category")` 不会给你业务顺序**——它按码位序设 `categories`，必须显式写 `categories=`。
26. **无序 `Categorical` 能排序但不能比大小**——`min()` / `max()` / `<` 报 `TypeError`，要加 `ordered=True`。
27. **中文按 Unicode 码位排 ≠ 业务序**——`处理状态` 的码位序恰好和业务序相反，必须用有序类别。
"""
    ),
    md(
        """
## 五、本章小结

**一句话记住这一章**：

> 排序管"谁在前"，`rank` 管"排第几"，`nlargest` 管"前几个"，`sample` 管"抽哪几个"。
> 四者最容易丢分的地方分别是：**稳定性**、**method**、**并列**、**种子**。

**四件必须做对的事**：

| 场景 | 必须写的参数 |
|---|---|
| 有并列 + 结果要可复现 | `sort_values(..., kind="stable")` |
| 业务名次 / 等级 | `rank(method="min")` / `rank(method="dense")` |
| 抽样（任何随机） | `random_state=<固定整数>` |
| 分布表要含缺失 | `value_counts(dropna=False)` |

**数据真值对账**（练习里的 `assert` 都指向这里）：

| 指标 | 真值 |
|---|---|
| `dev` 形状 / 负荷值缺失 | `(270, 13)` / 21 |
| 降序前 3 / 升序前 3 | 9999 / -1 |
| 降序索引前 3 | `[50, 46, 89]` |
| `desc.loc[0, VAL]` vs `desc.iloc[0][VAL]` | `89.96` vs `9999.0` |
| 稳定性 | quicksort/heapsort 不稳定；mergesort/stable 稳定 |
| 设备类型每个组的负荷冠军值 | 9999, 9999, 9999, 118.01, 103.81 |
| `sample(5, random_state=42)` 索引 | `[30, 116, 79, 127, 196]` |
| `sample(frac=0.001)` 行数 | 0 |
| `replace=True, frac=2.0` 行数 / 重复索引 | 540 / 309 |
| 权重和 / 最大权重 | 94545.46 / 9999.0 |
| 加权抽样均值 vs 全量均值 | 5666.52 vs 379.68 |
| 组内排名缺失 / 第 1 名行数 / 前 3 名行数 | 21 / 19 / 54 |
| 组内百分位最小值 | 0.045455（= 1/22） |
| `nlargest(5)` / `nsmallest(5)` | `[9999,9999,9999,9999,5000]` / `[-1,-1,-1,-1,5]` |
| `nlargest(3, keep="all")` 行数 | 4 |
| 缺陷类型 value_counts 取值数 / 总和 | 10 / 270 |
| 备注 value_counts 条目数（默认 / dropna=False） | 2 / 3 |
| 设备类型 × 缺陷等级 组合数 | 15 |
| 冠军表形状 / 冠军负荷之和 | `(18, 2)` / 46672.25 |
| 逐日峰值表形状 | `(270, 3)` |

**复盘问自己**（六题都能答上来才算过）：

1. `sort_values` 的 `kind` 默认值是什么？什么时候必须改？改了会影响什么？
2. 排序之后为什么 `loc[0]` 和 `iloc[0]` 取到不同的行？`iloc[0, "列名"]` 为什么报错？
3. `rank` 的五种 `method` 分别在什么业务场景下用？`average` 为什么通常是错的？
4. `nlargest(k)` 与 `sort_values().head(k)` 在**并列**和 **NaN** 上分别差在哪？
5. 加权抽样为什么"不放回"会失败？两条硬门槛分别是什么？怎么绕过？
6. `value_counts` 的 `dropna` / `normalize` / `sort` 三个默认值各是什么坑？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch08 练习 —— 排序、排名与采样

> 数据：`device_defects.csv`（270 x 13）、`load_curve.csv`（6480 x 5）
> 已备好：`BIG`（2 万行 / 5 组，专门用来测稳定性）、`vals`（含并列的小样本）、
> `stable_check()`、`topn_by_sort()`、`probe()`

**做题规则**：从上往下依次填，每个 `____` 处跑通再往下。
验收块 `assert` 不要改，它是你自己对账用的。

| 题号 | 考点 |
|---|---|
| TODO(1) | `sort_values` 方向与 `na_position` |
| TODO(2) | 排序稳定性（`kind`） |
| TODO(3) | 多列排序 + 混合升降序 + 长度校验 |
| TODO(4) | 排序后索引：`loc` vs `iloc` |
| TODO(5) | `rank` 的五种 `method` |
| TODO(6) | 组内名次与组内百分位 |
| TODO(7) | `nlargest` / `nsmallest` / `keep` |
| TODO(8) | `sample` 的可复现性与 `frac` 取整 |
| TODO(9) | `sample(weights=...)` 的两道门槛 |
| TODO(10) | `value_counts` 的 `dropna` / `normalize` |
| 综合① | 组内冠军表 + 三方对账 |
| 综合② | 可复现采样报告 |

共 12 个挖空块。
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
---

## 基础题
"""
    ),
    code(
        """# @@todo(1) 排序方向与缺失位置：
#           by_val_desc = 按 负荷值 降序；by_val_asc = 升序
#           by_val_first = 降序 + na_position="first"
#           三者前 3 个负荷值分别存入 head_desc / head_asc
#           再记录 desc 末 3 行里 NaN 的个数 → tail_nan_desc
# @@hint dev.sort_values(VAL, ascending=False) ；na_position="first"
by_val_desc = dev.sort_values(VAL, ascending=False)
by_val_asc = dev.sort_values(VAL)
by_val_first = dev.sort_values(VAL, ascending=False, na_position="first")
head_desc = by_val_desc[VAL].head(3).tolist()
head_asc = by_val_asc[VAL].head(3).tolist()
tail_nan_desc = int(by_val_desc[VAL].tail(3).isna().sum())
# @@end
# ---- 验收 ----
assert head_desc == [9999.0, 9999.0, 9999.0], f"降序前 3 应是哨兵 9999，实际 {head_desc}"
assert head_asc == [-1.0, -1.0, -1.0], f"升序前 3 应是哨兵 -1，实际 {head_asc}"
assert tail_nan_desc == 3, "na_position 默认 'last'，降序时 NaN 仍在末尾"
assert int(by_val_first[VAL].head(3).isna().sum()) == 3, "na_position='first' 时 NaN 排最前"
assert by_val_desc.index[:3].tolist() == [50, 46, 89], "排序不重置索引"
assert bool(by_val_desc.index.is_monotonic_increasing) is False
assert int(dev[VAL].isna().sum()) == 21
print("TODO(1) 通过 ->", head_desc, "|", head_asc, "| 索引", by_val_desc.index[:3].tolist())"""
    ),
    code(
        """# @@todo(2) 排序稳定性：用 stable_check 检测四种 kind 在 BIG 上的表现，
#           结果存入 quick_stable / heap_stable / merge_stable / stable_stable
# @@hint stable_check(BIG, "分组", "quicksort")
quick_stable = stable_check(BIG, "分组", "quicksort")
heap_stable = stable_check(BIG, "分组", "heapsort")
merge_stable = stable_check(BIG, "分组", "mergesort")
stable_stable = stable_check(BIG, "分组", "stable")
# @@end
# ---- 验收 ----
assert quick_stable is False, "quicksort 是默认值，但不稳定 —— 这是本章第一大坑"
assert heap_stable is False, "heapsort 同样不稳定"
assert merge_stable is True, "mergesort 稳定"
assert stable_stable is True, "kind='stable' 稳定"
assert BIG.shape == (20_000, 2) and BIG["分组"].nunique() == 5
print("TODO(2) 通过 -> quicksort", quick_stable, "| heapsort", heap_stable,
      "| mergesort", merge_stable, "| stable", stable_stable)"""
    ),
    code(
        """# @@todo(3) 多列排序与混合升降序：
#           multi = 先按 设备类型 升序、再按 负荷值 降序
#           champ = multi 里每个 设备类型 的第一行（组内冠军）
#           champ_vals = champ 的 负荷值 列表
#           bad_asc = ascending 长度不匹配时的异常类型名
# @@hint dev.sort_values([CAT, VAL], ascending=[True, False])
#        multi.groupby(CAT, sort=False).head(1) ；probe(dev.sort_values, [CAT, VAL], ascending=[True])[1]
multi = dev.sort_values([CAT, VAL], ascending=[True, False])
champ = multi.groupby(CAT, sort=False).head(1)
champ_vals = champ[VAL].tolist()
bad_asc = probe(dev.sort_values, [CAT, VAL], ascending=[True])[1]
# @@end
# ---- 验收 ----
assert multi.shape[0] == len(dev), "排序不改变行数"
assert champ.shape[0] == 5, f"5 个设备类型应各出 1 行，实际 {champ.shape[0]}"
assert champ_vals == [9999.0, 9999.0, 9999.0, 118.01, 103.81], f"实际 {champ_vals}"
assert bad_asc == "ValueError", "ascending 长度必须等于 by 长度"
assert bool(multi.groupby(CAT, sort=False)[VAL].first().tolist() == champ_vals), \\
    "champ 的值应等于各组排序后的首值"
print("TODO(3) 通过 -> 冠军值", champ_vals, "| 长度不匹配 ->", bad_asc)"""
    ),
    code(
        """# @@todo(4) 排序后取行：loc 与 iloc 的分叉
#           top_iloc = by_val_desc.iloc[0][VAL]
#           top_loc  = by_val_desc.loc[0][VAL]
#           desc_ri  = by_val_desc.reset_index(drop=True)，其 loc[0][VAL] 存入 top_ri
# @@hint 用 TODO(1) 排好的 by_val_desc
top_iloc = by_val_desc.iloc[0][VAL]
top_loc = by_val_desc.loc[0][VAL]
desc_ri = by_val_desc.reset_index(drop=True)
top_ri = desc_ri.loc[0][VAL]
# @@end
# ---- 验收 ----
assert top_iloc == 9999.0, "iloc[0] = 排完序的第一行 = 全局最大"
assert top_loc == 89.96, f"loc[0] 取的是索引标签 0，与排序无关，实际 {top_loc}"
assert top_ri == 9999.0, "reset_index(drop=True) 之后标签才是 0..n-1"
assert by_val_desc.loc[0, "记录ID"] == 31, "标签 0 那一行的记录 ID"
assert by_val_desc.iloc[0]["记录ID"] == 16, "排完序第一行的记录 ID"
assert probe(lambda: by_val_desc.iloc[0, VAL])[1] == "ValueError", \\
    "iloc 的列也必须是整数位置，写列名直接 ValueError"
print("TODO(4) 通过 -> iloc", top_iloc, "| loc", top_loc, "| reset 后", top_ri)"""
    ),
    code(
        """# @@todo(5) rank 的五种 method：
#           rank_tbl = {"average": [...], "min": [...], "max": [...], "first": [...], "dense": [...]}
#           dense_max = dense 法的最大值；min_max = min 法的最大值
# @@hint vals.rank(method="dense") ；用 METHODS 里的五个名字做循环键
rank_tbl = {m: vals.rank(method=m).tolist() for m in METHODS}
dense_max = float(vals.rank(method="dense").max())
min_max = float(vals.rank(method="min").max())
# @@end
# ---- 验收 ----
assert rank_tbl["average"] == [1.0, 2.5, 2.5, 4.0, 6.0, 6.0, 6.0, 8.0], "默认 method"
assert rank_tbl["min"] == [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 5.0, 8.0]
assert rank_tbl["max"] == [1.0, 3.0, 3.0, 4.0, 7.0, 7.0, 7.0, 8.0]
assert rank_tbl["first"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
assert rank_tbl["dense"] == [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0]
assert dense_max == 5.0, "dense 数的是「不同取值个数」"
assert min_max == 8.0, "min 数的是「行号」"
assert pd.Series([1, 2, 2]).rank().tolist() == [1.0, 2.5, 2.5], "默认 method 是 average"
assert int((vals.rank(method="average") % 1 != 0).sum()) == 2, "只有 average 会产出半个名次（两个并列的 20）"
print("TODO(5) 通过 -> dense max", dense_max, "| min max", min_max)"""
    ),
    code(
        """# @@todo(6) 组内名次：
#           dev["组内排名"] = 按 KEY 分组、对 VAL 降序的 min 法名次
#           dev["组内百分位"] = 同样分组、pct=True 的名次
#           rank_nan / n_first / n_top3 分别统计名次缺失数、第 1 名行数、前 3 名行数
# @@hint dev.groupby(KEY)[VAL].rank(ascending=False, method="min")
dev["组内排名"] = dev.groupby(KEY)[VAL].rank(ascending=False, method="min")
dev["组内百分位"] = dev.groupby(KEY)[VAL].rank(ascending=False, pct=True)
rank_nan = int(dev["组内排名"].isna().sum())
n_first = int((dev["组内排名"] == 1).sum())
n_top3 = int((dev["组内排名"] <= 3).sum())
# @@end
# ---- 验收 ----
assert rank_nan == 21, f"负荷值缺失 21 行，名次也应缺 21，实际 {rank_nan}"
assert n_first == 19, f"18 个台区 + 1 个并列冠军 = 19，实际 {n_first}"
assert n_top3 == 54, f"18 台区 x 3 = 54，实际 {n_top3}"
assert int(dev[KEY].nunique()) == 18
assert round(float(dev["组内百分位"].min()), 6) == 0.045455, "1/22 —— 最大的组有 22 行"
assert round(float(dev["组内百分位"].max()), 6) == 1.0, "每组都有一行拿到 1.0"
print("TODO(6) 通过 -> 缺失", rank_nan, "| 冠军行", n_first, "| 前 3 行", n_top3)"""
    ),
    code(
        """# @@todo(7) nlargest / nsmallest 与 keep：
#           top5 = dev.nlargest(5, VAL) 的负荷值列表
#           bot5 = dev.nsmallest(5, VAL) 的负荷值列表
#           n_keep_all = dev.nlargest(3, VAL, keep="all") 的行数
# @@hint dev.nlargest(5, VAL)[VAL].tolist()
top5 = dev.nlargest(5, VAL)[VAL].tolist()
bot5 = dev.nsmallest(5, VAL)[VAL].tolist()
n_keep_all = len(dev.nlargest(3, VAL, keep="all"))
# @@end
# ---- 验收 ----
assert top5 == [9999.0, 9999.0, 9999.0, 9999.0, 5000.0], f"实际 {top5}"
assert bot5 == [-1.0, -1.0, -1.0, -1.0, 5.0], f"实际 {bot5}"
assert n_keep_all == 4, f"4 个哨兵 9999 并列第 1，keep='all' 全给，实际 {n_keep_all}"
assert len(dev.nlargest(3, VAL)) == 3, "keep='first'（默认）严格给 3 行"
assert int(dev.nlargest(3, VAL)[VAL].isna().sum()) == 0, "nlargest 永远不返回 NaN"
assert int(dev.nsmallest(3, VAL)[VAL].isna().sum()) == 0
print("TODO(7) 通过 -> top5", top5, "| n_keep_all", n_keep_all)"""
    ),
    code(
        """# @@todo(8) 可复现抽样：
#           s42a / s42b = 两次 dev.sample(5, random_state=42) 的索引列表
#           n_frac_small = dev.sample(frac=0.001, random_state=42) 的行数
#           n_frac_big   = dev.sample(frac=1.0, random_state=1) 的行数
#           bad_both = 同时给 n 和 frac 时的异常类型名
# @@hint frac 的行数是 round(n * frac)；probe(dev.sample, n=5, frac=0.1)
s42a = dev.sample(5, random_state=42).index.tolist()
s42b = dev.sample(5, random_state=42).index.tolist()
n_frac_small = len(dev.sample(frac=0.001, random_state=42))
n_frac_big = len(dev.sample(frac=1.0, random_state=1))
bad_both = probe(dev.sample, n=5, frac=0.1, random_state=42)[1]
# @@end
# ---- 验收 ----
assert s42a == s42b == [30, 116, 79, 127, 196], f"同一 random_state 必须完全复现，实际 {s42a}"
assert n_frac_small == 0, "270 x 0.001 = 0.27 → round 到 0，静默返回空表"
assert n_frac_big == 270, "frac=1.0 不放回 = 全表的一个随机排列"
assert bad_both == "ValueError", "n 与 frac 只能给一个"
assert probe(dev.sample, frac=2.0, random_state=42)[1] == "ValueError", "frac>1 必须 replace=True"
assert len(dev.sample(frac=2.0, replace=True, random_state=42)) == 540
print("TODO(8) 通过 ->", s42a, "| frac=0.001 行数", n_frac_small)"""
    ),
    code(
        """# @@todo(9) sample(weights=...) 的两道门槛：
#           probe_neg    = 直接拿 VAL 当权重的不放回抽样的异常类型名
#           dpos         = dev 加上一列「权重」= VAL 截断负值并填 0
#           probe_norep  = 对 dpos 做 sample(10, weights="权重")（不放回）的异常类型名
#           ok_idx       = dpos.sample(10, weights="权重", replace=True, random_state=42) 的索引
#           ok_mean      = 上面这次加权抽样的 负荷值 均值（保留 4 位）
# @@hint dpos = dev.assign(权重=dev[VAL].clip(lower=0).fillna(0))
probe_neg = probe(dev.sample, 10, weights=VAL, random_state=42)[1]
dpos = dev.assign(权重=dev[VAL].clip(lower=0).fillna(0))
probe_norep = probe(dpos.sample, 10, weights="权重", random_state=42)[1]
sampled = dpos.sample(10, weights="权重", replace=True, random_state=42)
ok_idx = sampled.index.tolist()
ok_mean = round(float(sampled[VAL].mean()), 4)
# @@end
# ---- 验收 ----
assert probe_neg == "ValueError", "权重不能为负 —— 本表有 4 个 -1 哨兵"
assert probe_norep == "ValueError", "不放回加权采样要求 n * max(归一化权重) <= 1"
assert ok_idx == [56, 237, 154, 89, 46, 46, 40, 219, 89, 142], f"实际 {ok_idx}"
assert len(set(ok_idx)) < len(ok_idx), "replace=True 时同一行可能被抽中多次"
assert abs(float(dpos["权重"].sum()) - 94545.46) < 0.01, f"权重和 {float(dpos['权重'].sum())}"
assert round(float(dpos["权重"].max()), 2) == 9999.0
assert 10 * float(dpos["权重"].max()) / float(dpos["权重"].sum()) > 1, "这正是报错的原因"
assert ok_mean == 5666.517, f"加权抽样会把样本均值显著拉高，实际 {ok_mean}"
assert ok_mean > round(float(dev[VAL].mean()), 4), "远高于全表均值 379.6846"
print("TODO(9) 通过 ->", probe_neg, probe_norep, "| 加权均值", ok_mean)"""
    ),
    code(
        """# @@todo(10) value_counts 的默认行为：
#           vc_cat = dev[CAT].value_counts()；vc_def = dev["缺陷类型"].value_counts()
#           n_def  = vc_def 的取值个数
#           vc2_shape = dev[[CAT, "缺陷等级"]].value_counts() 的形状
#           extra_na = 备注 列 dropna=False 与默认的条目数之差
# @@hint dev[CAT].value_counts() ；dev["备注"].value_counts(dropna=False).shape[0]
vc_cat = dev[CAT].value_counts()
vc_def = dev["缺陷类型"].value_counts()
n_def = int(vc_def.shape[0])
vc2_shape = dev[[CAT, "缺陷等级"]].value_counts().shape
extra_na = int(dev["备注"].value_counts(dropna=False).shape[0]
               - dev["备注"].value_counts().shape[0])
# @@end
# ---- 验收 ----
assert vc_cat.shape[0] == 5, f"设备类型 5 类，实际 {vc_cat.shape[0]}"
assert vc_cat.to_dict() == {"避雷器": 59, "互感器": 58, "绝缘子": 57, "变压器": 55, "断路器": 41}, \\
    f"实际 {vc_cat.to_dict()}"
assert vc_def.index[0] == "发热" and int(vc_def.iloc[0]) == 51, "默认按计数降序，最多的是发热 51"
assert int(vc_def.sum()) == 270, "本列无缺失，计数之和 = 行数"
assert n_def == 10, f"缺陷类型原样有 10 种写法（含 '锈蚀 ' 尾随空格），实际 {n_def}"
assert int(dev["缺陷类型"].str.strip().nunique()) == 9, "strip 后只剩 9 种"
assert vc2_shape == (15,), f"5 类型 x 3 等级共出现 15 个组合，实际 {vc2_shape}"
assert extra_na == 1, "备注有 196 个缺失，dropna=False 会多出 NaN 这一项"
assert round(float(vc_def.iloc[0]) / 270, 6) == round(float(vc_def.loc["发热"] / vc_def.sum()), 6)
print("TODO(10) 通过 -> 缺陷类型", n_def, "种 | 组合数", vc2_shape, "| 缺失项", extra_na)"""
    ),
    md(
        """
---

## 进阶题
"""
    ),
    code(
        """# @@todo 综合①：一条链做出「每个台区的负荷冠军」并三方对账
#           champ_tbl = dev 按 VAL 降序（kind="stable"）排序后，groupby(KEY, sort=False).head(1)，
#                       取 [KEY, VAL] 两列，再 reset_index(drop=True)
#           champ_sum = champ_tbl 的 负荷值 之和（保留 2 位）
#           topn2     = topn_by_sort(dev, VAL, 2) 的行数
#           three_way = (champ_tbl 行数, dev[KEY].nunique(), 组内排名 == 1 的行数)
# @@hint dev.sort_values(VAL, ascending=False, kind="stable").groupby(KEY, sort=False).head(1)
champ_tbl = (
    dev.sort_values(VAL, ascending=False, kind="stable")
    .groupby(KEY, sort=False)
    .head(1)[[KEY, VAL]]
    .reset_index(drop=True)
)
champ_sum = round(float(champ_tbl[VAL].sum()), 2)
topn2 = len(topn_by_sort(dev, VAL, 2))
three_way = (champ_tbl.shape[0], int(dev[KEY].nunique()),
             int((dev["组内排名"] == 1).sum()))
# @@end
# ---- 验收 ----
assert champ_tbl.shape == (18, 2), f"冠军表应为 (18, 2)，实际 {champ_tbl.shape}"
assert list(champ_tbl.columns) == [KEY, VAL], f"列名应为 [{KEY}, {VAL}]，实际 {list(champ_tbl.columns)}"
assert champ_tbl.index.tolist() == list(range(18)), "reset_index(drop=True) 后索引应为 0..17"
assert int(champ_tbl[VAL].isna().sum()) == 0, "每个台区至少有一条非缺失的负荷记录"
assert champ_sum == 46672.25, f"冠军负荷之和应为 46672.25，实际 {champ_sum}"
assert topn2 == 36, f"18 台区 x 2 = 36，实际 {topn2}"
assert three_way == (18, 18, 19), f"(冠军行 18, 台区 18, 第 1 名 19) 实际 {three_way}"
assert three_way[2] - three_way[0] == 1, "差的这 1 行 = STATION_E_03 的并列冠军（两个 9999）"
assert int(dev["组内排名"].isna().sum()) == 21, "名次缺失的行不会进入冠军表"
print("TODO(综合①) 通过 -> 冠军表", champ_tbl.shape, "| 之和", champ_sum, "| 三方", three_way)"""
    ),
    code(
        """# @@todo 综合②：可复现采样报告
#           smp   = dev.sample(50, random_state=7)
#           strat = dev.groupby(CAT, group_keys=False).sample(2, random_state=7)
#           boot  = dev.sample(frac=0.5, replace=True, random_state=7)
#           report 字典键（顺序照抄）：
#             "样本行数" / "采样负荷均值"（4 位） / "全表负荷均值"（4 位）
#             / "分层行数" / "分层后组数" / "重采样行数" / "重采样重复索引数"
# @@hint 每一项都要写死 random_state，报告才能复现；bootstrap 的重复索引数用 index.duplicated().sum()
smp = dev.sample(50, random_state=7)
strat = dev.groupby(CAT, group_keys=False).sample(2, random_state=7)
boot = dev.sample(frac=0.5, replace=True, random_state=7)
report = {
    "样本行数": len(smp),
    "采样负荷均值": round(float(smp[VAL].mean()), 4),
    "全表负荷均值": round(float(dev[VAL].mean()), 4),
    "分层行数": len(strat),
    "分层后组数": int(strat[CAT].nunique()),
    "重采样行数": len(boot),
    "重采样重复索引数": int(boot.index.duplicated().sum()),
}
# @@end
# ---- 验收 ----
assert report["样本行数"] == 50
assert report["采样负荷均值"] == 569.6152, f"实际 {report['采样负荷均值']}"
assert report["全表负荷均值"] == 379.6846, f"实际 {report['全表负荷均值']}"
assert report["分层行数"] == 10, "5 个设备类型 x 2 = 10"
assert report["分层后组数"] == 5
assert report["重采样行数"] == 135, f"270 x 0.5 = 135，实际 {report['重采样行数']}"
assert report["重采样重复索引数"] == 28, f"实际 {report['重采样重复索引数']}"
assert report["采样负荷均值"] > report["全表负荷均值"], \\
    "50 行的样本均值偏离全量均值很远 —— 这正是「抽样要看的东西」"
assert len(report) == 7 and list(report)[0] == "样本行数"
print("TODO(综合②) 通过：")
for k, v in report.items():
    print(f"  {k:14s} = {v}")"""
    ),
    md(
        """
---

## 自查清单

跑完全部 `assert` 之后，回看这几件事：

1. 每个 `random_state` 都写死了吗？（不写就不可复现）
2. 涉及并列的排序都加 `kind="stable"` 了吗？
3. `rank` 的 `method` 是按业务语义选的吗？（名次用 `min`，等级用 `dense`）
4. `value_counts` 需要看缺失时，加 `dropna=False` 了吗？
5. 交付前索引都复位了吗？（`reset_index(drop=True)` 或 `ignore_index=True`）
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
