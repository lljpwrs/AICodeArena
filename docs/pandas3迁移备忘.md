# pandas 3.0 迁移备忘

> 本机环境是 **pandas 3.0.6**。网上大量教程和竞赛题库还停留在 1.x / 2.x，直接抄会踩坑。
> 下面全部是**本机实测结论**，不是文档推测。

---

## 一、破坏性变更（会直接报错或静默出错）

### 1. `fillna(method=...)` 已移除

```python
# ❌ pandas 3.0 报错：NDFrame.fillna() got an unexpected keyword argument 'method'
s.fillna(method="ffill")

# ✅ 改用专用方法
s.ffill()          # 前向填充
s.bfill()          # 后向填充
df.ffill()         # DataFrame 同样适用
```

同类：`bfill` / `pad` / `backfill` 都是独立方法，不再走 `method` 参数。

### 2. `DataFrame.applymap` 已移除

```python
# ❌ AttributeError
df.applymap(lambda x: x * 2)

# ✅ 改名为 map
df.map(lambda x: x * 2)
```

注意区别：`DataFrame.map` 是**逐元素**，`DataFrame.apply` 是**逐行/逐列**，`Series.map` 是逐元素。

### 3. 写时复制（Copy-on-Write）默认开启，链式赋值彻底失效

```python
# ❌ 不报错，但值根本改不进去（最坑的一种失败）
df[df["负荷值"] > 100]["状态"] = "异常"
print(df["状态"])        # 原样没变

# ✅ 用 loc 一次性定位
df.loc[df["负荷值"] > 100, "状态"] = "异常"
```

**判断方法**：看到 `df[...][...] = ...` 这种两层方括号赋值，一定是错的。
（`SettingWithCopyWarning` 现在只在极少数场景出现，别指望它提醒你。）

### 4. 字符串列 dtype 变成 `str`，不再是 `object`

```python
df = pd.DataFrame({"台区编号": ["STATION_A_01", "STATION_B_02"]})
print(df.dtypes)
# 台区编号    str        ← 3.0
# 台区编号    object     ← 2.x
```

对复习的影响：

- `df.select_dtypes(include="object")` **目前仍能**选到 `str` 列（向后兼容），
  但会抛 `Pandas4Warning` 并声明未来版本移除 —— **不要依赖它**。
  实测（本机 pandas 3.0.6）：
  ```python
  len(dev.select_dtypes(include="object").columns)   # 9  ← 能选到，但带警告
  len(dev.select_dtypes(include="str").columns)      # 9  ← 正确写法
  len(dev.select_dtypes(exclude="number").columns)   # 9  ← 最稳，不报警告
  ```
  推荐写 `include="str"`（精确）或 `exclude="number"`（不需要关心具体类型时）
- 判断"这列是不是字符串"要写 `pd.api.types.is_string_dtype(col)`，别用 `dtype == object`
- 本机 **未安装 pyarrow**，字符串底层是 `StringDtype(storage='python')`，缺失值仍显示为 `NaN`

### 5. `pd.to_datetime(format="mixed")` 解析不了中文日期

```python
# ❌ DateParseError: Unknown datetime string format, unable to parse: 2026年3月6日
pd.to_datetime(pd.Series(["2026/03/05", "2026年3月6日"]), format="mixed")

# ✅ 先把中文年月日替换成连字符，再统一解析
s = pd.Series(["2026/03/05", "2026年3月6日"])
s = s.str.replace("年", "-").str.replace("月", "-").str.replace("日", "")
pd.to_datetime(s, format="mixed")
```

竞赛数据里出现「2026年3月5日」这类中文日期是常态，**必须会手写清洗**。

---

## 二、行为变化（不报错，但结果和教程对不上）

### 1. `groupby(...).apply()` 不再默认带上分组列

返回结果里不会有分组键那一列，需要的话用 `include_groups=True`（会告警）或改用 `agg` / `transform`。
**竞赛里优先用 `agg` + `reset_index()`**，结果稳定、易读、不易错。

### 2. `pd.concat` 的 `copy` 参数

本机实测仍接受 `copy=False`，但已是过渡状态，新代码不必写。

### 3. `describe(include="all")` 返回行数

会包含 `unique` / `top` / `freq`，对混合类型 DataFrame 是 4 行而非 8 行，别背死。

---

## 三、本机实测可用清单（放心用）

| 操作 | 状态 |
|---|---|
| `pd.cut` / `pd.qcut(duplicates="drop")` | ✅ |
| `groupby().agg([...])` / `transform` / `observed=` | ✅ |
| `merge(how="inner/left/right/outer")` | ✅ |
| `pivot_table` / `melt` / `stack` / `unstack` | ✅ |
| `resample("D")` / `rolling(2).mean()` / `shifting` | ✅ |
| `clip(lower=, upper=)` | ✅ |
| `query("列名 > 1")`（支持中文列名）/ `between` / `isin` | ✅ |
| `to_numeric(errors="coerce")` | ✅ |
| `interpolate()` | ✅ |
| `value_counts(normalize=)` / `nlargest` / `idxmax` | ✅ |
| `memory_usage(deep=True)` | ✅ |
| `astype("category")` / `astype("str")` | ✅ |

---

## 四、一个反直觉的坑

```python
s = pd.DataFrame({"缺陷类型": ["渗漏油"]})["缺陷类型"]
s.str.replace("", "未知")
# 结果：'未知渗未知漏未知油未知'  ← 空串在每个字符之间都匹配
```

空字符串正则匹配所有位置。要填充空值用 `fillna` 或 `replace("", pd.NA)`，**不要用 `str.replace("", ...)`**。

---

## 五、窗口族（ch12）实测补充

### 5.1 `rolling("24h")` 在非 `DatetimeIndex` 上直接报错

```python
# RangeIndex 上：
load.rolling("24h").mean()
# ValueError: window must be an integer 0 or greater
```

时间窗口**只认 `DatetimeIndex`**。报错信息完全不提"索引类型"，容易误判成参数写错。

### 5.2 时间窗口的 `min_periods` 默认值是 1（不是窗口长度）

| 写法 | `min_periods` 默认 | 本仓数据 NaN 数 |
|---|---|---|
| `rolling(24).mean()` | **24**（= `window`） | **319** |
| `rolling("24h").mean()` | **1** | **0** |
| `rolling("24h", min_periods=24).mean()` | 显式传 24 | **319** |

同一份数据、同样的窗口跨度，**默认值不同 → NaN 差 319 个**。
这是固定窗口与时间窗口最容易踩的不一致点。

### 5.3 `gb.rolling()` 结果直接赋回原表会报 `TypeError`（好事）

```python
gb_roll = curve.groupby("台区编号")["负荷值"].rolling(24).mean()
curve["滑动均值"] = gb_roll
# TypeError: incompatible index of inserted column with frame index
```

pandas 3.0 会明确拒绝，**不会静默变 NaN**。正确做法：

```python
curve["滑动均值"] = gb_roll.reset_index(drop=True).to_numpy()
```

`gb.rolling()` 的索引是 `['台区编号', None]` 两级 `MultiIndex`，
`reset_index()` 后的列名是 `['台区编号', 'level_1', '负荷值']`。

### 5.4 真正危险的是"顺序错了但索引看起来正常"

```python
# 行数不变、索引仍是 0..n-1、NaN 数完全正常 —— 但值全贴在错行上
bad = (gb_roll.reset_index(drop=True)
       .sort_values(ascending=False, na_position="last")
       .reset_index(drop=True))
```

本仓实测：错位率 **99.96%**（4942 个非空行里只有 2 行碰巧对上），
而 `isna().sum()` 与正确写法**完全相同**。
唯一可靠的检查方式是**抽 2~3 行手算窗口值对账**。

### 5.5 `pct_change` 的 `fill_method`

```python
s.pct_change(24, fill_method=None)   # 明确"不做填充"
```

`fill_method` 的默认值在 3.0 已弃用，显式传 `None` 才是干净语义。

### 5.6 窗口族的 `NaN` 会"指数放大"

1 个缺失值 → 它后面 `window-1` 个窗口结果都是 `NaN`（因为窗口里含它）。
本仓实测：23 个缺失 → `rolling(24).mean()` 出现 **319** 个 `NaN`。

**推论**：`dropna()` 之前务必先看 `NaN` 是怎么来的，
否则会连带丢掉大量本来完好的样本。

---

## 六、连接与重塑（ch13 / ch14）实测补充

### 6.1 `merge` 的 `indicator` 列是 `category`

```python
flagged = dev.merge(info, on="台区编号", how="left", indicator=True)
flagged["_merge"].dtype          # category
flagged["_merge"].value_counts()
# both 247 / left_only 23 / right_only 0   ← 0 计数也会列出来
```

`how="left"` 时 `right_only` 恒为 0，但因为是 `category`，
`value_counts()` **仍会列出该类**。想只看非零：

```python
flagged["_merge"].value_counts().pipe(lambda s: s[s > 0])
```

### 6.2 `merge_asof` 的三条约束

| 项 | 实测 |
|---|---|
| 左键未排序 | `ValueError: left keys must be sorted` |
| `direction="backward"`（默认） | 取 ≤ 当前时刻的最近一条（**用过去**） |
| `direction="forward"` | 取 ≥ 当前时刻的最近一条（**用未来**） |

同一份数据（2160 小时负荷 + 6 条限值）实测：
`backward` 的匹配缺失是 **0**，`forward` 是 **383**。

### 6.3 `pivot` / `unstack` 遇重复索引直接报错

```python
dev.pivot(index="台区编号", columns="设备类型", values="负荷值")
# ValueError: Index contains duplicate entries, cannot reshape
```

`pivot` 是**纯位置搬运**，不做聚合；`pivot_table` 会先 `groupby`，
所以同样的数据它能正常返回 `(18, 5)`。

### 6.4 `DataFrame.unstack` 会产生**两级列名**（`Series.unstack` 不会）

```python
# 长表是 DataFrame（多列聚合）→ 列名被带进列轴
long_df.unstack(1).columns          # [('负荷值', '互感器'), ('负荷值', '变压器'), ...]
long_df.unstack(1).columns.nlevels  # 2
long_df.unstack(1)["互感器"]        # KeyError

# 长表是 Series（单列）→ 列名干净
long_s.unstack("设备类型").columns.nlevels   # 1
```

**规避方式**：`dev.groupby([k1, k2])[单列].mean()`（不要加 `as_index=False`
或 `[[多列]]`），得到的就是 Series。

### 6.5 `stack` 在 3.0 换实现：两处破坏性变更

| 老写法 | pandas 3.0 |
|---|---|
| `df.stack(dropna=False)` | **`ValueError: dropna must be unspecified as the new implementation does not introduce rows of NA values`** |
| `series.stack()` | **`AttributeError`**（`Series` 已无 `stack` 方法，改 `s.to_frame().stack()`） |
| `df.stack(future_stack=True)` | ✅ 仍可用（新实现即该语义） |
| `df.stack()` 是否丢 `NaN` | **不再丢**：18×5 的宽表 → 90 行（含 3 个 `NaN` 格），格数守恒 |

### 6.6 宽表 `NaN` 的两张面孔

```python
long.unstack(1)                  # NaN 3 个 —— 两种成因混在一起
long.unstack(1, fill_value=-1)   # NaN 1 个 + (-1) 2 个
```

**3 = 2（组合不存在）+ 1（组合存在但值全缺）**。
`fill_value` 只填"搬出来的空格"，填不掉"本来就缺的值"——
所以它同时也是一个**诊断工具**。

### 6.7 重塑闭环的行数有三个口径

```text
dev 270 行  →  groupby  →  long 88 组合（其中 1 个值全缺 → 87 个有值）
                        →  unstack →  wide 18×5 = 90 格
                        →  melt    →  long_back 90 行（满格）
                        →  dropna  →  87 行（= long.dropna()）
```

`90 ≠ 88 ≠ 87`。对账时**必须写清用的是哪个口径**，
否则会得出"重塑丢了数据"的错误结论。

### 6.8 `explode` 的前置条件

```python
s.str.split("/").explode()     # ✅ 先切成列表再炸
s.explode()                    # ❌ 字符串没被切，行数不变，静默无效
```

- 无分隔符时 `split` 返回**单元素列表**，展开后仍是 1 行（不会产生 `NaN`）
- 多分隔符必须 `regex=True`：`s.str.split("[/,]", regex=True)`
- 展开后索引必然重复 → 用 `ignore_index=True`
- `NaN` 会保留为一行 `NaN`（行数不丢）

---

## 七、排序 / 排名 / 采样实测补充

### 7.1 `sort_values` 默认 `kind="quicksort"` 是**不稳定**排序

签名实测：`kind: SortKind = 'quicksort'`。

```python
big = pd.DataFrame({"分组": np.random.default_rng(0).integers(0, 5, 20_000)})
big["序号"] = np.arange(len(big))

# 稳定判据：同一取值内部，原始序号必须严格递增
def stable_check(frame, by, kind):
    o = frame.sort_values(by, kind=kind)
    pos, keys = o["序号"].to_numpy(), o[by].to_numpy()
    same = keys[1:] == keys[:-1]
    return bool(np.all(pos[1:][same] > pos[:-1][same]))
```

实测结果（2 万行 / 5 组，并列极多）：

| `kind` | 稳定 |
|---|---|
| `"quicksort"`（默认） | ❌ |
| `"heapsort"` | ❌ |
| `"mergesort"` | ✅ |
| `"stable"` | ✅ |

**结论**：只要数据里可能出现并列（分类列、计数列、评级列），
且结果需要**可复现**，就必须写 `kind="stable"`。
或者把唯一键一起放进 `by`，靠多列排序拆开并列。

### 7.2 `na_position` **不受** `ascending` 影响

```python
dev["负荷值"].sort_values().tail(3).isna().tolist()                       # [True, True, True]
dev["负荷值"].sort_values(ascending=False).tail(3).isna().tolist()        # [True, True, True]
dev["负荷值"].sort_values(ascending=False, na_position="first").head(3)   # 前 3 全是 NaN
```

多列排序时 `na_position` 是**逐列生效**的：`sort_values([台区编号, 负荷值], ascending=[True, False])`
之后，每个台区内部的负荷值 `NaN` 都落在该组末尾。

### 7.3 排序**不重置索引**：`loc` 与 `iloc` 语义分叉

```python
desc = dev.sort_values("负荷值", ascending=False)

desc.index[:6].tolist()          # [50, 46, 89, 56, 68, 219]  ← 没变
desc.iloc[0]["负荷值"]           # 9999.0   ✅ 排完序的第一行
desc.loc[0, "负荷值"]            # 89.96    ⚠️ 索引标签 0 那一行
desc.iloc[0]["记录ID"]           # 16
desc.loc[0, "记录ID"]            # 31
```

- `df.iloc[0, "列名"]` → **`ValueError`**（`iloc` 的列也必须是整数位置）
- 要交付干净索引：`sort_values(..., ignore_index=True)` 或 `reset_index(drop=True)`

### 7.4 `rank` 默认 `method="average"`

`vals = [10, 20, 20, 30, 40, 40, 40, 50]`：

| `method` | 结果 | 语义 |
|---|---|---|
| `"average"`（默认） | `[1, 2.5, 2.5, 4, 6, 6, 6, 8]` | 占用名次的平均 → **会产出半个名次** |
| `"min"` | `[1, 2, 2, 4, 5, 5, 5, 8]` | 并列都给最好名次 |
| `"max"` | `[1, 3, 3, 4, 7, 7, 7, 8]` | 并列都给最差名次 |
| `"first"` | `[1, 2, 3, 4, 5, 6, 7, 8]` | 按出现顺序强行拆开 |
| `"dense"` | `[1, 2, 2, 3, 4, 4, 4, 5]` | 名次连续不跳号 |

两条可背的关系（实测：变压器组 49 行 / 46 个不同取值）：

- `rank(method="min").max()` = **48**（≈ 非缺失行数）
- `rank(method="dense").max()` = **46**（= 不同取值个数）

其他默认值：

- `ascending=True` → 名次 1 是**最小**值
- `na_option="keep"` → 缺失行的名次仍是缺失（`dev["负荷值"].rank()` 缺失 **21** 个，
  且 `rank(pct=True)` 的分母是**非缺失计数**，不是行数）
- `na_option="top"` / `"bottom"` 才会把缺失纳入名次

### 7.5 `nlargest` 的 `keep` 与"并列"

```python
dev["负荷值"].value_counts().get(9999.0)   # 4
len(dev.nlargest(3, "负荷值"))             # 3   ← 默认 keep="first"
len(dev.nlargest(3, "负荷值", keep="all")) # 4   ← 并列全部给出
```

另外，**并列行的输出顺序与 `sort_values().head()` 不保证一致**：

```python
dev.sort_values("负荷值", ascending=False).index[:6].tolist()  # [50, 46, 89, 56, 68, 219]
dev.nlargest(6, "负荷值").index.tolist()                       # [46, 50, 56, 89, 55, 68]
```

`nlargest` / `nsmallest` 永远**不返回 NaN**（`sort+head` 要靠 `na_position` 兜底）。
`groupby(...).nlargest(n)` 返回**多层索引**（索引名 `['设备类型', None]`），交付前必须 `reset_index`。

### 7.6 `sample` 的三个静默陷阱

```python
len(dev.sample(frac=0.01,  random_state=42))   # 3   ← 270 × 0.01  = 2.7  → round 3
len(dev.sample(frac=0.001, random_state=42))   # 0   ⚠️ 270 × 0.001 = 0.27 → round 0，静默空表
dev.sample(n=5, frac=0.1)                      # ValueError: Please enter a value for `frac` OR `n`
dev.sample(frac=2.0)                           # ValueError: Replace has to be set to `True` when upsampling
```

- 行数按 `round(n * frac)` 计算，**可能舍成 0**
- 不写 `random_state` 就不可复现（`sample(5, random_state=42)` 恒为 `[30, 116, 79, 127, 196]`）
- `sample(..., axis=1)` 抽的是**列**

### 7.7 `sample(weights=...)` 的两道硬门槛

第一条报错文案里有官方笔误（`many not`），搜索时注意：

```python
dev.sample(10, weights="负荷值")
# ValueError: weight vector many not include negative values     ← 本表有 4 个 -1 哨兵
```

pandas 源码 `core/sample.py` 的校验顺序：

```python
if (weights < 0).any():
    raise ValueError("weight vector many not include negative values")
...
if not replace and size * weights.max() > 1:      # weights 已归一化
    raise ValueError("Weighted sampling cannot be achieved with replace=False. ...")
```

本数据实测：非负权重和 **94545.46**、最大权重 **9999.0** → 归一化后 `max(p) = 0.105759`，
`10 × 0.105759 = 1.0576 > 1` → 报错。

| 写法 | 结果 |
|---|---|
| `sample(10, weights="负荷值")` | ❌ 负值 |
| `sample(10, weights="权重")`（非负，不放回） | ❌ `n × max(p) > 1` |
| `sample(2, weights="权重")` | ✅ `2 × 0.105759 = 0.2115 ≤ 1` |
| `sample(10, weights="权重", replace=True)` | ✅ 但会抽出重复行 |

**加权抽样会把样本均值显著拉高**（实测 **5666.517** vs 全表 **379.6846**）——这是设计使然，不是 bug。

### 7.8 `value_counts` 的默认值

```python
dev["备注"].value_counts().shape[0]              # 2   ← 默认丢 NaN（备注有 196 个缺失）
dev["备注"].value_counts(dropna=False).shape[0]  # 3   ← NaN 单独一项
dev["缺陷类型"].value_counts(sort=False)         # 按「首次出现顺序」，非字典序
```

- `normalize=True` 的分母是**非缺失计数**（`dev["缺陷类型"]` 无缺失，故和 = 1.0）
- 默认按**计数降序**；`ascending=True` 看长尾
- `value_counts(bins=3)` 可直接对数值列分箱计数，不必先 `pd.cut`
- `index.name` = 原列名，`name` = `"count"`
- `DataFrame.value_counts(subset=...)` 可做多维组合频次

### 7.9 `ordered Categorical` 才有业务顺序

```python
order = ["未处理", "处理中", "处理完成"]
s.astype(pd.CategoricalDtype(order, ordered=True)).sort_values()
```

- `astype("category")` 的 `categories` 是**按码位序自动生成**的，给不了业务顺序
  （`处理状态` 自动类别 = `['处理中', '处理完成', '未处理']`，与业务序完全相反）
- 无序 `Categorical` **能** `sort_values`（按 `categories` 顺序），
  但 `min()` / `max()` / `<` 直接 **`TypeError: Categorical is not ordered for operation min`**
- `rank()` 对 `Categorical` 可用（内部转类别码）
- pandas 3.0 起，用 `Categorical(values, categories=...)` 传入**不在 categories 里的值**
  已发 `Pandas4Warning`（提示未来会报错），建议改用 `astype(CategoricalDtype(...))`
