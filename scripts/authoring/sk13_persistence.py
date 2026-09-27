#!/usr/bin/env python3
"""ch13 模型持久化与推理：joblib dump/load / Pipeline 持久化 / 版本戳（defect_ml.csv）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk13_persistence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch13_persistence"

IMPORTS = '''from pathlib import Path

import hashlib
import os
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA = Path("data")
'''

SETUP = '''# 数据与划分口径沿用 ch05：分层 8:2、median 填充、独热、标准化
dev = pd.read_csv(DATA / "defect_ml.csv")
NUM = ["负荷值", "温度", "湿度", "投运年限", "巡检耗时分钟"]
CAT = ["设备类型", "缺陷类型"]
TARGET = "是否危急"
y_all = dev[TARGET]

tr_idx, te_idx = train_test_split(np.arange(len(dev)), test_size=0.2,
                                  random_state=42, stratify=y_all)
dev_tr, dev_te = dev.iloc[tr_idx], dev.iloc[te_idx]
ys_tr, ys_te = dev_tr[TARGET], dev_te[TARGET]

imp = SimpleImputer(strategy="median").fit(dev_tr[NUM])
oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(dev_tr[CAT])
Xtr = np.hstack([imp.transform(dev_tr[NUM]), oh.transform(dev_tr[CAT])])
Xte = np.hstack([imp.transform(dev_te[NUM]), oh.transform(dev_te[CAT])])
ss = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = ss.transform(Xtr), ss.transform(Xte)

# 端到端管道：原始 DataFrame 进 → 预测出（可整体持久化）
pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])),
    ("clf", DecisionTreeClassifier(max_depth=5, random_state=42)),
])
pipe.fit(dev_tr, ys_tr)

print("sklearn", sklearn.__version__, "| joblib", joblib.__version__,
      "| 端到端管道已拟合")
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def pred_md5(y_pred):
    """预测数组的 md5 指纹：持久化前后必须逐位一致。"""
    return hashlib.md5(np.asarray(y_pred).tobytes()).hexdigest()


def cleanup(*paths):
    """删掉本次实验落盘的模型文件。"""
    for p in paths:
        (Path(p)).unlink(missing_ok=True)


print("脚手架就绪：pred_md5 / cleanup")'''

# =========================================================================== #
# 练习 notebook 挖空版
# =========================================================================== #

E1_CODE = '''# @@todo(1) RF 存档再读档：predict 结果必须逐位一致
# @@hint joblib.dump(rf, "rf_model.joblib") → rf2 = joblib.load("rf_model.joblib")；指纹 pred_md5
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr_s, ys_tr)
joblib.dump(rf, "rf_model.joblib")
rf2 = joblib.load("rf_model.joblib")
p_raw, p_loaded = rf.predict(Xte_s), rf2.predict(Xte_s)
md5_same = pred_md5(p_raw) == pred_md5(p_loaded)
rf_size = os.path.getsize("rf_model.joblib")
# @@end
print("文件大小:", rf_size, "bytes | predict 指纹一致:", md5_same)

assert md5_same, "dump/load 往返后预测必须逐位相同"
assert rf_size == 1_359_721, "RF100 的字节级体积（同版本同数据应稳定）"
assert (p_raw == p_loaded).all()
'''

E2_CODE = '''# @@todo(2) 压缩存档：compress=3 换 82% 的体积折扣
# @@hint joblib.dump(rf, "rf_z.joblib", compress=3)
joblib.dump(rf, "rf_z.joblib", compress=3)
z_size = os.path.getsize("rf_z.joblib")
ratio = round(z_size / rf_size, 2)
joblib.dump(rf, "rf_z0.joblib")  # 对照：不压缩
z0_size = os.path.getsize("rf_z0.joblib")
# @@end
print(f"压缩 {rf_size} → {z_size} bytes（{ratio}x）| 对照不压缩 {z0_size}")
print("加载后 predict 一致:", (joblib.load("rf_z.joblib").predict(Xte_s) == p_raw).all())

assert z_size == 248_622, "compress=3 的 zlib 体积"
assert ratio == 0.18, "压缩到 18%：大模型交卷前的最后一步"
assert (joblib.load("rf_z.joblib").predict(Xte_s) == p_raw).all(), "压缩不改变预测"
'''

E3_CODE = '''# @@todo(3) 读档完整性体检：属性与超参一个都不能丢
# @@hint n_features_in_ / n_estimators / get_params(deep=False) 逐项对账
n_feat = rf2.n_features_in_
n_est = rf2.n_estimators
gp_diff = {k for k, v in rf.get_params(deep=False).items() if not np.array_equal(np.asarray(v, dtype=object), np.asarray(rf2.get_params(deep=False)[k], dtype=object))}
proba_same = bool(np.allclose(rf.predict_proba(Xte_s), rf2.predict_proba(Xte_s)))
# @@end
print("n_features_in_:", n_feat, "| n_estimators:", n_est)
print("get_params 差异项:", gp_diff or "无", "| proba 一致:", proba_same)

assert n_feat == 16 and n_est == 100
assert gp_diff == set(), "加载版与原版超参完全一致"
assert proba_same, "predict_proba 也逐位一致"
'''

E4_CODE = '''# @@todo(4) 端到端管道整体持久化：原始 DataFrame 进，预测出
# @@hint joblib.dump(pipe, "pipe.joblib")；加载后直接 predict(dev_te)——不用手动预处理
pipe_acc = round(float(accuracy_score(ys_te, pipe.predict(dev_te))), 4)
joblib.dump(pipe, "pipe.joblib")
pipe2 = joblib.load("pipe.joblib")
p_pipe = pipe.predict(dev_te)
p_pipe2 = pipe2.predict(dev_te)
pipe_size = os.path.getsize("pipe.joblib")
# @@end
print("端到端 acc:", pipe_acc, "| 文件:", pipe_size, "bytes | 重载一致:",
      bool((p_pipe == p_pipe2).all()))

assert pipe_acc == 0.7917, "端到端管道成绩与 ch05 的 d5 树一致"
assert pipe_size == 7_556, "整条管道只有 7.6KB：预处理状态+树全在里面"
assert (p_pipe == p_pipe2).all(), "管道往返同样逐位一致"
assert list(pipe2.predict(dev_te.iloc[:3])) == pipe.predict(dev_te.iloc[:3]).tolist()
'''

E5_CODE = '''# @@todo(5) 单样本在线推理：新台区一进来就给结论
# @@hint 构造只含 NUM+CAT 列的一行 DataFrame；pipe.predict_proba(new)[0]
new = pd.DataFrame([{"设备类型": "变压器", "缺陷类型": "发热", "负荷值": 500.0,
                     "温度": 45.0, "湿度": 70.0, "投运年限": 20, "巡检耗时分钟": 60}])
proba_new = np.round(pipe2.predict_proba(new)[0], 4).tolist()
pred_new = int(pipe2.predict(new)[0])
weird = new.copy()
weird["设备类型"] = "巡检机器人"  # 训练时没见过的类别
proba_weird = np.round(pipe2.predict_proba(weird)[0], 4).tolist()
# @@end
print("新样本 proba:", proba_new, "→ 判为", pred_new)
print("未知设备类型 proba:", proba_weird, "→ handle_unknown=ignore 兜底不炸")

assert pred_new == 1, "变压器+发热+高负荷 → 危急"
assert proba_new == [0.1111, 0.8889]
assert proba_weird == [0.1111, 0.8889], "未知类别被独热归零后走同一条树叶路径"
'''

E6_CODE = '''# @@todo(6) 版本戳：sklearn 模型文件里藏着环境指纹
# @@hint 读出 "rf_model.joblib" 的字节，检查 b"_sklearn_version" 是否在内
raw_bytes = Path("rf_model.joblib").read_bytes()
has_version = b"_sklearn_version" in raw_bytes
pos = raw_bytes.find(b"_sklearn_version")
version_bytes = raw_bytes[pos:pos + 40]
# @@end
print("含 _sklearn_version 戳:", has_version)
print("戳附近字节:", version_bytes)
print("当前 sklearn:", sklearn.__version__, "——跨版本 load 前先核对戳")

assert has_version, "sklearn 1.9 起每个持久化对象都带版本戳"
assert version_bytes.startswith(b"_sklearn_version")
'''

E7_CODE = '''# @@todo(7) 模拟推理脚本：读档 → 批量预测 → 出指纹（部署的最小闭环）
# @@hint 只允许 load，不允许再 fit；指纹对账 E1 的结果
prod_model = joblib.load("rf_model.joblib")
batch = pd.read_csv(DATA / "defect_ml.csv").iloc[:50]  # 模拟新来的数据（带缺失/原始列）
prod_pred = prod_model.predict(
    np.hstack([imp.transform(batch[NUM]), oh.transform(batch[CAT])]))
prod_md5 = pred_md5(prod_pred)
raw_md5 = pred_md5(rf.predict(
    np.hstack([imp.transform(batch[NUM]), oh.transform(batch[CAT])])))
# @@end
print("批大小:", len(prod_pred), "| 指纹:", prod_md5[:16], "| 与训练时模型一致:", prod_md5 == raw_md5)

assert len(prod_pred) == 50
assert prod_md5 == raw_md5, "推理侧产物与训练侧逐位相同——部署闭环成立"
'''

E8_CODE = '''# @@todo(8) 反例围观：把 sklearn 模型用原生 pickle 存，会有什么坑
# @@hint import pickle 在文件头已 import？没有就用 joblib 对比；pickle.dump(rf3, open(..., "wb"))
import pickle
rf3 = joblib.load("rf_model.joblib")
pk_fh = open("rf_pickle.pkl", "wb")
pickle.dump(rf3, pk_fh)
pk_fh.close()
pk_size = os.path.getsize("rf_pickle.pkl")
rf4 = pickle.load(open("rf_pickle.pkl", "rb"))
pk_same = bool((rf4.predict(Xte_s) == p_raw).all())
pk_has_version = b"_sklearn_version" in Path("rf_pickle.pkl").read_bytes()
# @@end
print(f"原生 pickle {pk_size} bytes（joblib {rf_size}）| 预测一致: {pk_same} | 含版本戳: {pk_has_version}")
print("结论: 两者体积接近、结果相同，但 joblib 对 numpy 大数组更快且支持压缩，"
      "又是 sklearn 生态约定俗成——竞赛交模型一律 joblib")

assert pk_same
assert pk_has_version
assert pk_size < rf_size, "原生 pickle 略小于 joblib（差 ~8KB 对齐开销），但同一量级"
'''

# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# RF 存档再读档：joblib 往返保证预测逐位一致
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr_s, ys_tr)
joblib.dump(rf, "rf_model.joblib")
rf2 = joblib.load("rf_model.joblib")
p_raw, p_loaded = rf.predict(Xte_s), rf2.predict(Xte_s)
print("文件大小:", os.path.getsize("rf_model.joblib"), "bytes")
print("predict 一致:", (p_raw == p_loaded).all(),
      "| 指纹:", pred_md5(p_raw) == pred_md5(p_loaded))
print("解读: 指纹（md5）是部署验收的标准动作——『跑得通』不算数，『逐位一样』才算数")

assert (p_raw == p_loaded).all()
assert os.path.getsize("rf_model.joblib") == 1_359_721
'''

S2_CODE = '''# compress=3：体积折扣 82%，预测分毫不差
joblib.dump(rf, "rf_z.joblib", compress=3)
z = os.path.getsize("rf_z.joblib")
print(f"压缩 {os.path.getsize('rf_model.joblib')} → {z} bytes "
      f"({round(z / os.path.getsize('rf_model.joblib'), 2)}x)")
print("加载后一致:", (joblib.load("rf_z.joblib").predict(Xte_s) == p_raw).all())
print("解读: compress=3（zlib level 3）是体积/速度的常用折中；交卷或上传前压一下")

assert z == 248_622
'''

S3_CODE = '''# 读档体检：属性、超参、概率一个都不能丢
print("n_features_in_:", rf2.n_features_in_, "| n_estimators:", rf2.n_estimators)
print("get_params 差异:", {k for k, v in rf.get_params(deep=False).items()
                          if not np.array_equal(np.asarray(v, dtype=object),
                                                np.asarray(rf2.get_params(deep=False)[k], dtype=object))} or "无")
print("proba 一致:", np.allclose(rf.predict_proba(Xte_s), rf2.predict_proba(Xte_s)))
print("解读: n_features_in_=16 同时是部署时的输入校验器——特征列对不上直接报错，比静默出错强")

assert rf2.n_features_in_ == 16 and rf2.n_estimators == 100
'''

S4_CODE = '''# 端到端管道整体持久化：部署的正确形态
pipe_acc = round(float(accuracy_score(ys_te, pipe.predict(dev_te))), 4)
joblib.dump(pipe, "pipe.joblib")
pipe2 = joblib.load("pipe.joblib")
print("端到端 acc:", pipe_acc, "| 文件:", os.path.getsize("pipe.joblib"), "bytes")
print("重载一致:", (pipe.predict(dev_te) == pipe2.predict(dev_te)).all())
print("解读: 只存树不存预处理 = 部署事故 top1（加载后拿原始 df 一喂就崩）。"
      "ColumnTransformer+Pipeline 整体 dump，7.6KB 拎包入住")

assert pipe_acc == 0.7917 and (pipe.predict(dev_te) == pipe2.predict(dev_te)).all()
'''

S5_CODE = '''# 单样本在线推理 + 未知类别兜底
new = pd.DataFrame([{"设备类型": "变压器", "缺陷类型": "发热", "负荷值": 500.0,
                     "温度": 45.0, "湿度": 70.0, "投运年限": 20, "巡检耗时分钟": 60}])
print("新样本 proba:", np.round(pipe2.predict_proba(new)[0], 4).tolist(),
      "→", int(pipe2.predict(new)[0]))
weird = new.copy()
weird["设备类型"] = "巡检机器人"
print("未知类别 proba:", np.round(pipe2.predict_proba(weird)[0], 4).tolist())
print("解读: handle_unknown=ignore 让未知类别静默归零——不炸但可能不准，"
      "生产上要记录未知类别率，超阈值就该重训")

assert int(pipe2.predict(new)[0]) == 1
assert np.round(pipe2.predict_proba(new)[0], 4).tolist() == [0.1111, 0.8889]
'''

S6_CODE = '''# 版本戳：跨版本加载的第一道防线
raw = Path("rf_model.joblib").read_bytes()
pos = raw.find(b"_sklearn_version")
print("含版本戳:", pos != -1, "| 截取:", raw[pos:pos + 40])
print("解读: sklearn 1.9 每个持久化对象都写入了 _sklearn_version；"
      "跨大版本加载（如 1.3 → 1.9）轻则警告重则 AttributeError——"
      "存档时把 sklearn.__version__ 一起写进文件名或 sidecar 是好习惯")

assert pos != -1
'''

S7_CODE = '''# 部署最小闭环：load → 批量预测 → 指纹对账（全程不再 fit）
prod_model = joblib.load("rf_model.joblib")
batch = pd.read_csv(DATA / "defect_ml.csv").iloc[:50]
prod_pred = prod_model.predict(
    np.hstack([imp.transform(batch[NUM]), oh.transform(batch[CAT])]))
raw_pred = rf.predict(np.hstack([imp.transform(batch[NUM]), oh.transform(batch[CAT])]))
print("批大小:", len(prod_pred), "| 指纹一致:", pred_md5(prod_pred) == pred_md5(raw_pred))
print("解读: 推理脚本的三条纪律——只 load 不 fit、输入走同一套预处理、"
      "产物对指纹。竞赛答辩演示部署时，这一页最有说服力")

assert len(prod_pred) == 50 and pred_md5(prod_pred) == pred_md5(raw_pred)
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch13 模型持久化与推理：joblib / Pipeline 存档 / 版本戳

> 数据：`defect_ml.csv`，口径沿用 ch05。训练完的模型不落盘就等于没做——
> 竞赛要交模型文件、部署要推理脚本，本章把「存、验、用」一次讲清。

**本章考点**

1. `joblib.dump / load` 往返一致性（预测指纹 md5 对账）
2. `compress=3` 的体积折扣
3. 读档完整性：`n_features_in_` / `get_params` / `predict_proba`
4. **端到端 Pipeline 整体持久化**（只存树不存预处理 = 事故 top1）
5. 单样本在线推理与未知类别兜底
6. `_sklearn_version` 版本戳与跨版本风险
7. 部署最小闭环：只 load 不 fit
8. joblib vs 原生 pickle

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| 只存模型不存预处理 | 加载后拿原始 df 一喂就崩 | §13.2 |
| 往返一致性没验收 | 「跑得通」≠「逐位一样」 | §13.1 |
| 跨版本加载 | sklearn 对象自带版本戳，跨大版本可能崩 | §13.3 |
| 未知类别静默归零 | 不炸但可能不准，要监控 | §13.4 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 13.1 存档与验收：指纹对账

RF100 存档 1,359,721 bytes；load 回来 predict 与原模型**逐位一致**（md5
指纹相同）。`compress=3` 用 zlib 压到 248,622 bytes（0.18x），预测分毫不差。
**「加载后能跑」不算验收，「指纹一致」才算**。
"""
    ),
    code(S1_CODE),
    code(S2_CODE),
    md(
        """
## 13.2 难点深挖：端到端 Pipeline 整体持久化

只 dump 分类器、丢掉 SimpleImputer/OneHotEncoder/StandardScaler 的状态，
是部署事故第一名——加载后拿原始 DataFrame 一喂就崩。正确形态：
`ColumnTransformer + Pipeline` 整体 dump，本例只有 7.6KB，原始 df 进、
预测出，拎包入住。`n_features_in_=16` 还兼做输入校验器。
"""
    ),
    code(S3_CODE),
    code(S4_CODE),
    md(
        """
## 13.3 难点深挖：单样本推理与版本戳

新台区（变压器+发热+高负荷）proba [0.1111, 0.8889] → 判危急；
设备类型换成训练时没见过的「巡检机器人」，`handle_unknown="ignore"`
让它静默归零、不炸——但可能不准，生产上要监控未知类别率。

sklearn 1.9 起每个持久化对象都写入 `_sklearn_version` 版本戳
（pickle 字节里可搜到）。跨大版本加载轻则警告重则 AttributeError，
**存档时把版本号写进文件名是好习惯**。
"""
    ),
    code(S5_CODE),
    code(S6_CODE),
    md(
        """
## 13.4 部署最小闭环

推理脚本三条纪律：**只 load 不 fit、输入走同一套预处理、产物对指纹**。
50 条批数据走完闭环，md5 与训练侧一致——竞赛答辩演示部署，
这一页最有说服力。

joblib 与原生 pickle 在单模型上体积、结果相同，但 joblib 对 numpy
大数组更快，且是 sklearn 生态约定俗成——**交模型一律 joblib**。
"""
    ),
    code(S7_CODE),
    md(
        """
## 小结

1. `joblib.dump/load` 往返预测逐位一致；验收用 md5 指纹，不用「能跑」。
2. `compress=3` 压到 18% 体积，预测不变；交卷前压一下。
3. 读档体检三件套：`n_features_in_` / `get_params` / `predict_proba`。
4. **只存树不存预处理 = 部署事故 top1**：ColumnTransformer+Pipeline 整体存。
5. 未知类别 `handle_unknown="ignore"` 兜底不炸，但要监控未知类别率。
6. 对象内含 `_sklearn_version` 戳：跨大版本加载前先核对。
7. 推理纪律：只 load 不 fit、同一套预处理、产物对指纹；模型文件一律 joblib。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch13 模型持久化与推理（练习版）

> 按提示补全 `____`，跑通所有 assert。数据、划分、端到端管道 `pipe`
> 已就绪；`pred_md5` / `cleanup` 脚手架可用。
>
> 注意：挖空块内每条语句保持**单行**；循环/分支写在挖空块之外。
> 本章会往当前目录写模型文件，结束任务会清理。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：存档与验收

RF 存档、读档、指纹对账。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：压缩存档

compress=3 的体积折扣与预测一致性。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：读档完整性体检

属性、超参、概率逐项对账。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：端到端管道持久化

原始 DataFrame 进，预测出。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：单样本在线推理

新台区一进来就给结论，未知类别兜底。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：版本戳

在 pickle 字节里找到 `_sklearn_version`。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：推理脚本模拟

load → 批量预测 → 指纹对账的最小闭环。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：原生 pickle 对比

体积、结果、版本戳三项对照 joblib。
"""
    ),
    code(E8_CODE),
    md(
        """
## 完成标准

所有 assert 通过 = 任务完成。最后清理落盘文件：
"""
    ),
    code('''# 收尾清理（不挖空）：删掉本章实验文件
cleanup("rf_model.joblib", "rf_z.joblib", "rf_z0.joblib",
        "pipe.joblib", "rf_pickle.pkl")
import glob
print("残留模型文件:", glob.glob("*.joblib") + glob.glob("*.pkl"))'''),
    md(
        """
## 完成标准（续）

自查：为什么验收要指纹、为什么只存树不存预处理是事故、
版本戳怎么用、推理脚本的三条纪律是什么。
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
