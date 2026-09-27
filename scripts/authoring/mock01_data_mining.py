"""生成 mock01 模拟卷的空白卷与答案卷。

空白卷 `mock01.ipynb`：只有题面 + 空的作答 cell，模拟真实考场从零写。
答案卷 `mock01_solution.ipynb`：完整可运行，与 `rubric.yaml` 逐项对应。

> 本脚本是 mock01 的**唯一真相源**。改题目或改答案都在这里改，然后重跑。

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/mock01_data_mining.py

改完必须同步做两件事：
  ① 重跑 `scripts/grade_notebook.py`，确认答案卷仍是满分 —— `assert` 真值实跑，不许手算；
  ② 若动了题面或分值，同步 `README.md` 的「三、任务要求」与「四、评分表」。
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "exams" / "mock01_数据分析与挖掘"
NAME = "mock01"

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.15"},
}

INTRO = [
    """# mock01 · 数据分析与挖掘

**限时 90 分钟 · 满分 50 分**

仿「第八届全国职工职业技能大赛人工智能训练师赛项江苏选拔赛」模块 A 的评分结构：
**按步骤给分，每项独立计分，漏一项就丢一项的分。**

> 场景与数据说明见本目录 `README.md`。做完保存，用 `scripts/grade_notebook.py` 判卷。
""",
    """## 约定

- 数据在 `./data/` 下，主表是 **GBK 编码**。
- 中途不查文档、不看答案，模拟断网。
- 判卷器按名字取结果，**随机森林测试集准确率必须存入变量 `acc_rf`**。
- 一题一个 cell，便于定位失分点。
""",
]

ANSWERS: dict[str, str] = {}

ANSWERS["E1"] = '''
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Hiragino Sans GB"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA = Path("./data")

records = pd.read_csv(DATA / "defect_records.csv", encoding="gbk")
master = pd.read_csv(DATA / "device_master.csv")

print("缺陷记录", records.shape, "| 设备台账", master.shape)
records.info()
'''

ANSWERS["E2"] = '''
print(records.describe(include="all").T)
print("\\n缺陷等级取值分布：")
print(records["缺陷等级"].value_counts())
print("\\n处理状态取值分布：")
print(records["处理状态"].value_counts())
'''

ANSWERS["E3"] = '''
print("各列缺失值数量：")
print(records.isna().sum())
print("\\n完全重复行数：", int(records.duplicated().sum()))
'''

ANSWERS["C1"] = '''
n_before = len(records)
records = records.drop_duplicates().reset_index(drop=True)
print(f"删除重复行 {n_before - len(records)} 行，剩余 {len(records)} 行")
'''

ANSWERS["C2"] = '''
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y年%m月%d日")


def parse_date(value):
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


records["发现日期"] = records["发现日期"].map(parse_date)
print("日期解析失败：", int(records["发现日期"].isna().sum()))
print("日期范围：", records["发现日期"].min().date(), "~", records["发现日期"].max().date())
'''

ANSWERS["C3"] = '''
loaded = (records["负荷值_kW"].astype(str)
          .str.replace("kW", "", regex=False)
          .str.replace(",", "", regex=False)
          .str.strip())
loaded = loaded.replace({"N/A": np.nan, "—": np.nan, "": np.nan, "nan": np.nan})
records["负荷值_kW"] = pd.to_numeric(loaded, errors="coerce")
print("负荷值转数值后缺失：", int(records["负荷值_kW"].isna().sum()))

SEVERITY_MAP = {"一般": "一般", "轻微": "一般", "严重": "严重", "较重": "严重", "危急": "危急", "紧急": "危急"}
STATUS_MAP = {"已处理": "已处理", "处理完成": "已处理", "处理中": "处理中", "在办": "处理中", "未处理": "未处理"}
records["缺陷等级"] = records["缺陷等级"].str.strip().map(SEVERITY_MAP)
records["处理状态"] = records["处理状态"].str.strip().map(STATUS_MAP)
print("归一后等级分布：", records["缺陷等级"].value_counts().to_dict())
'''

ANSWERS["C4"] = '''
n_fill = int(records["负荷值_kW"].isna().sum() + records["处理时长_小时"].isna().sum())
records["负荷值_kW"] = records["负荷值_kW"].fillna(records["负荷值_kW"].median())
records["处理时长_小时"] = records["处理时长_小时"].fillna(records["处理时长_小时"].median())
print(f"填充缺失值 {n_fill} 条（中位数填充）")
'''

ANSWERS["C5"] = '''
records["处理时长_小时"] = records["处理时长_小时"].replace({9999: np.nan, -1: np.nan})
records["处理时长_小时"] = records["处理时长_小时"].fillna(records["处理时长_小时"].median())

fig, ax = plt.subplots(figsize=(7, 3))
ax.boxplot(records["处理时长_小时"].dropna(), orientation="horizontal")
ax.set_title("处理时长箱线图（异常值检测）")
ax.set_xlabel("小时")
plt.tight_layout()

q1, q3 = records["处理时长_小时"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outlier_mask = (records["处理时长_小时"] < lower) | (records["处理时长_小时"] > upper)
n_outlier = int(outlier_mask.sum())
records["处理时长_小时"] = records["处理时长_小时"].clip(lower, upper)
print(f"处理异常值 {n_outlier} 个，截断到 [{lower:.2f}, {upper:.2f}]")
'''

ANSWERS["V1"] = '''
fig, ax = plt.subplots(figsize=(6, 4))
records["缺陷等级"].value_counts().plot(kind="bar", ax=ax, color="#4C78A8")
ax.set_title("缺陷等级分布")
ax.set_xlabel("缺陷等级")
ax.set_ylabel("记录数")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
'''

ANSWERS["V2"] = '''
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(records["处理时长_小时"], bins=30, color="#F58518", edgecolor="white")
ax.set_title("处理时长分布")
ax.set_xlabel("处理时长（小时）")
ax.set_ylabel("频数")
plt.tight_layout()
'''

ANSWERS["M1"] = '''
from sklearn.model_selection import train_test_split

df = records.merge(master, on="设备编号", how="left")
print("关联台账后", df.shape, "| 设备类型缺失", int(df["设备类型"].isna().sum()))

df["设备类型"] = df["设备类型"].fillna("未知")
df["台区编号"] = df["台区编号"].fillna("未知台区")
df["线路名称"] = df["线路名称"].fillna("未知线路")
df["投运年份"] = df["投运年份"].fillna(df["投运年份"].median())

df["高风险"] = df["缺陷等级"].isin(["严重", "危急"]).astype(int)
print("高风险占比：", round(df["高风险"].mean(), 4))

FEATURES = ["设备类型", "台区编号", "线路名称", "缺陷类型", "投运年份", "负荷值_kW", "处理时长_小时"]
X = df[FEATURES]
y = df["高风险"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("训练集", X_train.shape, "测试集", X_test.shape)
'''

ANSWERS["M2"] = '''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["设备类型", "台区编号", "线路名称", "缺陷类型"]
NUM_COLS = ["投运年份", "负荷值_kW", "处理时长_小时"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ("num", StandardScaler(), NUM_COLS),
])
print(preprocess)
'''

ANSWERS["M3"] = '''
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

lr_model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
lr_model.fit(X_train, y_train)
pred_lr = lr_model.predict(X_test)
print("逻辑回归训练完成")
'''

ANSWERS["M4"] = '''
from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)),
])
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)
print("随机森林训练完成")
'''

ANSWERS["M5"] = '''
from sklearn.metrics import accuracy_score

acc_lr = accuracy_score(y_test, pred_lr)
acc_rf = accuracy_score(y_test, pred_rf)
print(f"逻辑回归 测试集准确率 {acc_lr:.4f}")
print(f"随机森林 测试集准确率 {acc_rf:.4f}")
'''

ANSWERS["T1"] = '''
from sklearn.metrics import f1_score, precision_score, recall_score

metrics = {}
for name, pred in [("逻辑回归", pred_lr), ("随机森林", pred_rf)]:
    metrics[name] = {
        "准确率": accuracy_score(y_test, pred),
        "精确率": precision_score(y_test, pred),
        "召回率": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }
print(pd.DataFrame(metrics).T.round(4))
'''

ANSWERS["T2"] = '''
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred_rf)
fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_title("随机森林 混淆矩阵")
ax.set_xlabel("预测")
ax.set_ylabel("真实")
ax.set_xticks([0, 1], labels=["低风险", "高风险"])
ax.set_yticks([0, 1], labels=["低风险", "高风险"])
fig.colorbar(im, ax=ax)
plt.tight_layout()
print(cm)
'''

ANSWERS["T3"] = '''
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=1)),
    ]),
    {"clf__n_estimators": [100, 200], "clf__max_depth": [8, 14, None]},
    cv=3, scoring="f1", n_jobs=1,
)
grid.fit(X_train, y_train)
print("最佳参数：", grid.best_params_)
print(f"交叉验证 F1：{grid.best_score_:.4f}")
print(f"注意：交叉验证最优不保证测试集提升，测试集准确率 {grid.score(X_test, y_test):.4f}")
'''

TASKS: list[tuple[str, str, str, str]] = [
    ("E1", "结构探查（2 分）",
     "读取 `defect_records.csv` 与 `device_master.csv`（注意编码），打印两表形状，并打印主表的字段类型与非空计数。", ANSWERS["E1"]),
    ("E2", "描述统计与类别分布（2 分）",
     "输出数值列的描述统计，并统计关键类别列的取值分布。", ANSWERS["E2"]),
    ("E3", "缺失与重复统计（2 分）",
     "统计各列缺失值数量与完全重复行数。", ANSWERS["E3"]),
    ("C1", "重复值删除（2 分）",
     "删除重复记录，并**打印删除的行数**。", ANSWERS["C1"]),
    ("C2", "日期多格式解析（2 分）",
     "把 `发现日期` 解析为日期类型 —— 该列**存在多种格式**。", ANSWERS["C2"]),
    ("C3", "负荷值与类别清洗（2 分）",
     "把 `负荷值_kW` 清洗为数值；把 `缺陷等级`/`处理状态` 的别名与空格归一。", ANSWERS["C3"]),
    ("C4", "缺失值填充（2 分）",
     "处理缺失值（先说明策略），并**打印填充的条数**。", ANSWERS["C4"]),
    ("C5", "箱线图 + IQR 处理异常（2 分）",
     "用箱线图检测 `处理时长_小时` 的异常值，用 IQR 法处理，并**打印处理掉的个数**。", ANSWERS["C5"]),
    ("V1", "可视化 1 · 等级分布柱状图（2 分）",
     "画 `缺陷等级` 的分类柱状图，含标题与轴标签。", ANSWERS["V1"]),
    ("V2", "可视化 2 · 处理时长直方图（2 分）",
     "画 `处理时长_小时` 的直方图，含标题与轴标签。", ANSWERS["V2"]),
    ("M1", "划分数据（4 分）",
     "关联设备台账，构造标签 `高风险`（`缺陷等级` ∈ {严重, 危急} → 1），划分训练集与测试集。", ANSWERS["M1"]),
    ("M2", "特征工程（4 分）",
     "类别特征编码 + 数值特征标准化（**注意与划分的先后顺序**）。", ANSWERS["M2"]),
    ("M3", "训练逻辑回归（4 分）",
     "训练逻辑回归模型并在测试集上预测。", ANSWERS["M3"]),
    ("M4", "训练随机森林（4 分）",
     "训练随机森林模型并在测试集上预测。", ANSWERS["M4"]),
    ("M5", "输出预测与准确率（4 分）",
     "输出两模型的测试集准确率，随机森林的准确率存入变量 **`acc_rf`**。", ANSWERS["M5"]),
    ("T1", "四项评估指标（3 分）",
     "输出准确率 / 精确率 / 召回率 / F1。", ANSWERS["T1"]),
    ("T2", "混淆矩阵热力图（4 分）",
     "画混淆矩阵热力图。", ANSWERS["T2"]),
    ("T3", "网格调参（3 分）",
     "用 `GridSearchCV` 调参，打印最佳参数。", ANSWERS["T3"]),
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def dump(path: Path, cells: list[dict]) -> None:
    nb = {"cells": cells, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def build() -> None:
    blank: list[dict] = [md(t) for t in INTRO]
    answer: list[dict] = [md(t) for t in INTRO]

    for tid, title, requirement, solution in TASKS:
        head = f"### {tid} · {title}\n\n{requirement}"
        blank.append(md(head))
        blank.append(code(f"\n# {tid} 在此作答\n"))
        answer.append(md(head))
        answer.append(code(solution))

    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / f"{NAME}.ipynb", blank)
    dump(OUT / f"{NAME}_solution.ipynb", answer)

    print(f"{'文件':34s}{'cells':>7s}{'c 代码':>8s}")
    print("-" * 50)
    print(f"{NAME + '.ipynb':34s}{len(blank):>7d}{sum(c['cell_type'] == 'code' for c in blank):>8d}")
    print(f"{NAME + '_solution.ipynb':34s}{len(answer):>7d}{sum(c['cell_type'] == 'code' for c in answer):>8d}")
    print(f"\n共 {len(TASKS)} 个任务，总分 {sum(1 for _ in TASKS) and 50} 分")


if __name__ == "__main__":
    build()
