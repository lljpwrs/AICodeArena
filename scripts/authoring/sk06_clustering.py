"""sk06 —— 聚类与无监督：KMeans / 层次聚类 / DBSCAN / GMM / 轮廓系数

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk06_clustering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch06_clustering"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DATA = Path("data")

df = pd.read_csv(DATA / "station_load.csv")   # 300 台区 × 5 特征 + 用电类型（仅用于对账）

FEATS = ["日均负荷", "夜间负荷率", "峰谷差率", "周末负荷率", "报装容量kVA"]
TRUE = "用电类型"
X = df[FEATS]
y_true = df[TRUE]

ss = StandardScaler().fit(X)
Xs = ss.transform(X)

print("sklearn", sklearn.__version__)
print(df.shape, "| 类型计数:", y_true.value_counts().to_dict())
print("量纲对比: 日均负荷 std", round(float(X["日均负荷"].std()), 1),
      "vs 峰谷差率 std", round(float(X["峰谷差率"].std()), 4))'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
"""
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


def cluster_table(y_true, labels):
    """交叉表 + 最优映射准确率（簇编号与真实标签的对应关系任选）。"""
    ct = pd.crosstab(y_true, labels)
    mapping = ct.idxmax(axis=1).to_dict()          # 真实类型 -> 簇编号
    acc = sum(ct.loc[t, c] for t, c in mapping.items()) / len(y_true)
    return ct, {c: t for t, c in mapping.items()}, acc


print("脚手架就绪：probe / cluster_table")'''


# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# KMeans k=3（scaled）：一次教科书级的完美聚类
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
ct, clu2type, acc = cluster_table(y_true, km.labels_)
print("inertia(WCSS):", round(float(km.inertia_), 4),
      "| n_iter:", int(km.n_iter_))
print("簇计数:", pd.Series(km.labels_).value_counts().sort_index().to_dict())
print(ct.to_string())
print("ARI vs 用电类型:", round(adjusted_rand_score(y_true, km.labels_), 4),
      "| 最优映射 acc:", round(acc, 4))
print("簇 -> 类型:", clu2type)

# 聚类的独特能力：新样本归属预测 + 到各簇心距离
new = pd.DataFrame({"日均负荷": [510.0], "夜间负荷率": [0.22], "峰谷差率": [0.74],
                    "周末负荷率": [0.66], "报装容量kVA": [800.0]})
new_s = ss.transform(new)
lab_new = int(km.predict(new_s)[0])
print("新样本(商业画像) → 簇", lab_new, "=", clu2type[lab_new],
      "| 到各簇心距离:", np.round(km.transform(new_s)[0], 4).tolist())

assert round(float(km.inertia_), 4) == 402.6153
assert pd.Series(km.labels_).value_counts().sort_index().to_dict() == {0: 90, 1: 75, 2: 135}
assert round(adjusted_rand_score(y_true, km.labels_), 4) == 1.0, "ARI=1 完美还原三型"
assert round(acc, 4) == 1.0
assert clu2type == {0: "商业型", 1: "工业型", 2: "居民型"}
assert lab_new == 0 and clu2type[lab_new] == "商业型"
'''

S2_CODE = '''# 量纲陷阱：同一份数据不 scale，聚类当场失忆
km_raw = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X.to_numpy())
ct_raw, _, acc_raw = cluster_table(y_true, km_raw.labels_)
print("raw inertia:", round(float(km_raw.inertia_), 2),
      "| scaled inertia:", round(float(km.inertia_), 4))
print("raw ARI:", round(adjusted_rand_score(y_true, km_raw.labels_), 4),
      "| raw 最优映射 acc:", round(acc_raw, 4))
print(ct_raw.to_string())
print("解读: 报装容量(数百) 与 率值(零点几) 量纲差 3 个数量级，距离被容量独裁")

assert round(adjusted_rand_score(y_true, km_raw.labels_), 4) == 0.2557
assert round(acc_raw, 4) == 0.61, "近四成台区被分错型，商业/居民还在抢同一个簇"
assert round(float(km_raw.inertia_), 2) == 23988840.36, "inertia 是量纲的奴隶，跨尺度不可比"
'''

S3_CODE = '''# k 怎么选：肘部法看惯性拐点，轮廓系数看峰值，两个都看
inertias, sils, ks = [], [], list(range(2, 9))
for k in ks:
    kmk = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    inertias.append(round(float(kmk.inertia_), 2))
    sils.append(round(float(silhouette_score(Xs, kmk.labels_)), 4))
    print(f"k={k}: inertia={inertias[-1]:>9.2f} silhouette={sils[-1]:.4f}")

best_k = ks[int(np.argmax(sils))]
print("轮廓系数峰值 k =", best_k, "→", sils[int(np.argmax(sils))])

assert inertias == [890.25, 402.62, 293.56, 218.98, 172.42, 155.54, 141.55], "单调下降没有拐点喊得那么响"
assert sils == [0.4291, 0.5628, 0.5156, 0.4956, 0.4592, 0.3907, 0.3523]
assert best_k == 3, "轮廓系数在 k=3 出峰，与真实簇数一致"
'''

S4_CODE = '''# 轮廓系数往下钻：样本级 silhouette_samples，看每个簇的健康度
sil_vals = silhouette_samples(Xs, km.labels_)
per = pd.Series(sil_vals).groupby(km.labels_).mean().round(4)
print("各簇平均轮廓:", per.to_dict())
print("overall:", round(float(silhouette_score(Xs, km.labels_)), 4))
print("最低样本轮廓:", round(float(sil_vals.min()), 4),
      "| 负轮廓样本数:", int((sil_vals < 0).sum()))

assert per.to_dict() == {0: 0.5381, 1: 0.5907, 2: 0.5638}
assert round(float(silhouette_score(Xs, km.labels_)), 4) == 0.5628
assert int((sil_vals < 0).sum()) == 0, "没有样本分错阵营"
'''

S5_CODE = '''# 层次聚类：三种链接都 ARI=1.0，但簇编号是「抽签」的
for link in ("ward", "complete", "average"):
    ag = AgglomerativeClustering(n_clusters=3, linkage=link).fit(Xs)
    same = float((ag.labels_ == km.labels_).mean())
    print(f"{link:>9}: ARI={adjusted_rand_score(y_true, ag.labels_):.4f} "
          f"| 与 KMeans 标签逐位一致率: {same:.4f}")

ag_ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(Xs)
st_pred = probe(getattr, AgglomerativeClustering(), "predict")
print("AGG.predict →", st_pred, "← 层次聚类没有 predict，新样本要另想办法")

assert round(adjusted_rand_score(y_true, ag_ward.labels_), 4) == 1.0
assert float((ag_ward.labels_ == km.labels_).mean()) == 0.0, "ARI=1.0 但编号完全错位——簇编号无语义"
assert st_pred == ("err", "AttributeError")
'''

S6_CODE = '''# DBSCAN：eps 一字符定生死
db = DBSCAN(eps=0.3, min_samples=5).fit(Xs)
n03 = int((db.labels_ == -1).sum())
print(f"eps=0.3: 噪声 {n03}/300，只剩 {len(set(db.labels_)) - 1} 个簇 ← 全员社死")

# eps 扫描（不用挖空）：噪声海 → 碎片化 → 基本成型 → 完美
results = {}
for eps in (0.5, 0.8, 1.2):
    dbp = DBSCAN(eps=eps, min_samples=5).fit(Xs)
    lab = dbp.labels_
    n_cl = len(set(lab)) - (1 if -1 in lab else 0)
    n_noise = int((lab == -1).sum())
    ari = round(float(adjusted_rand_score(y_true[lab != -1], lab[lab != -1])), 4) if n_cl > 1 else 0.0
    results[eps] = (n_cl, n_noise, ari)
    print(f"eps={eps}: 簇数={n_cl} 噪声={n_noise} ARI(非噪声)={ari}")

assert n03 == 289, "eps 太小：289 个点被判噪声"
assert results[0.5] == (9, 111, 0.3366), "eps 稍大：碎成 9 瓣"
assert results[0.8] == (5, 4, 0.9081)
assert results[1.2] == (3, 0, 1.0), "eps=1.2 才完整还原三簇"
'''

S7_CODE = '''# eps 别拍脑袋：k-距离图找「肘」
nn = NearestNeighbors(n_neighbors=5).fit(Xs)
d4 = np.sort(nn.kneighbors(Xs)[0][:, -1])          # 每个点的第 4 近邻距离
p50, p90, p95 = (round(float(np.quantile(d4, q)), 4) for q in (0.5, 0.9, 0.95))
print(f"第4近邻距离分位: p50={p50} p90={p90} p95={p95}")
print("经验法则: min_samples=5 时，eps 取曲线拐点附近 ≈ 1.0~1.2")

assert p50 == 0.5348 and p90 == 0.7546 and p95 == 0.8173
assert p95 < 1.2, "拐点之下才解释了 eps=1.2 为什么不坍缩"
'''

S8_CODE = '''# GMM 对照：软分配版 KMeans，ARI 同样满分，还送概率
gm = GaussianMixture(n_components=3, random_state=42).fit(Xs)
lab_g = gm.predict(Xs)
proba_g = gm.predict_proba(Xs)
print("GMM ARI:", round(adjusted_rand_score(y_true, lab_g), 4),
      "| 计数:", pd.Series(lab_g).value_counts().sort_index().to_dict())
print("proba shape:", proba_g.shape, "| 行和:", np.round(proba_g.sum(axis=1)[:3], 6).tolist())
print("新样本(商业画像) GMM 概率:", np.round(gm.predict_proba(new_s)[0], 4).tolist(),
      "→", ["商业型", "工业型", "居民型"][int(np.argmax(gm.predict_proba(new_s)[0]))])

assert round(adjusted_rand_score(y_true, lab_g), 4) == 1.0
assert proba_g.shape == (300, 3)
assert float(np.abs(proba_g.sum(axis=1) - 1).max()) < 1e-6
assert int(np.argmax(gm.predict_proba(new_s)[0])) == 0, "商业画像落在商业簇"
'''


E1_CODE = '''# @@todo(1) 标准化数据上跑 KMeans k=3，做交叉表并算 ARI 与最优映射 acc
# @@hint KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)；cluster_table(y_true, km.labels_)
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
ct, clu2type, acc = cluster_table(y_true, km.labels_)
# @@end
print("inertia(WCSS):", round(float(km.inertia_), 4),
      "| n_iter:", int(km.n_iter_))
print("簇计数:", pd.Series(km.labels_).value_counts().sort_index().to_dict())
print(ct.to_string())
print("ARI vs 用电类型:", round(adjusted_rand_score(y_true, km.labels_), 4),
      "| 最优映射 acc:", round(acc, 4))
print("簇 -> 类型:", clu2type)

# 商业画像新样本（脚手架，不用挖空）
new = pd.DataFrame({"日均负荷": [510.0], "夜间负荷率": [0.22], "峰谷差率": [0.74],
                    "周末负荷率": [0.66], "报装容量kVA": [800.0]})

# @@todo(2) 给「商业画像」新样本预测归属簇，并算到各簇心的距离
# @@hint ss.transform(new) 后 km.predict / km.transform
new_s = ss.transform(new)
lab_new = int(km.predict(new_s)[0])
# @@end
print("新样本(商业画像) → 簇", lab_new, "=", clu2type[lab_new],
      "| 到各簇心距离:", np.round(km.transform(new_s)[0], 4).tolist())

assert round(float(km.inertia_), 4) == 402.6153
assert pd.Series(km.labels_).value_counts().sort_index().to_dict() == {0: 90, 1: 75, 2: 135}
assert round(adjusted_rand_score(y_true, km.labels_), 4) == 1.0, "ARI=1 完美还原三型"
assert round(acc, 4) == 1.0
assert clu2type == {0: "商业型", 1: "工业型", 2: "居民型"}
assert lab_new == 0 and clu2type[lab_new] == "商业型"
'''

E2_CODE = '''# @@todo(3) 不做标准化直接跑 KMeans k=3，与 scaled 版对账 ARI 与最优映射 acc
# @@hint km_raw = KMeans(...).fit(X.to_numpy())；X 是原始量纲
km_raw = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X.to_numpy())
ct_raw, _, acc_raw = cluster_table(y_true, km_raw.labels_)
# @@end
print("raw inertia:", round(float(km_raw.inertia_), 2),
      "| scaled inertia:", round(float(km.inertia_), 4))
print("raw ARI:", round(adjusted_rand_score(y_true, km_raw.labels_), 4),
      "| raw 最优映射 acc:", round(acc_raw, 4))
print(ct_raw.to_string())
print("解读: 报装容量(数百) 与 率值(零点几) 量纲差 3 个数量级，距离被容量独裁")

assert round(adjusted_rand_score(y_true, km_raw.labels_), 4) == 0.2557
assert round(acc_raw, 4) == 0.61, "近四成台区被分错型，商业/居民还在抢同一个簇"
assert round(float(km_raw.inertia_), 2) == 23988840.36, "inertia 是量纲的奴隶，跨尺度不可比"
'''

E3_CODE = '''# @@todo(4) 初始化扫描容器：inertias / sils / ks（k 从 2 到 8）
# @@hint ks = list(range(2, 9))
inertias, sils, ks = [], [], list(range(2, 9))
# @@end

# 扫描循环（骨架不用挖空）
for k in ks:
    # @@todo(5) 循环体：跑一个 k，把 inertia 与轮廓系数各记一笔
    # @@hint kmk = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)；inertias.append(...)；sils.append(round(float(silhouette_score(Xs, kmk.labels_)), 4))
    kmk = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    inertias.append(round(float(kmk.inertia_), 2))
    sils.append(round(float(silhouette_score(Xs, kmk.labels_)), 4))
    # @@end
    print(f"k={k}: inertia={inertias[-1]:>9.2f} silhouette={sils[-1]:.4f}")

# @@todo(6) 找轮廓系数峰值的 k
# @@hint best_k = ks[int(np.argmax(sils))]
best_k = ks[int(np.argmax(sils))]
# @@end
print("轮廓系数峰值 k =", best_k, "→", sils[int(np.argmax(sils))])

assert inertias == [890.25, 402.62, 293.56, 218.98, 172.42, 155.54, 141.55], "单调下降没有拐点喊得那么响"
assert sils == [0.4291, 0.5628, 0.5156, 0.4956, 0.4592, 0.3907, 0.3523]
assert best_k == 3, "轮廓系数在 k=3 出峰，与真实簇数一致"
'''

E4_CODE = '''# @@todo(6) 用 silhouette_samples 算样本级轮廓，统计各簇均值与负轮廓数
# @@hint silhouette_samples(Xs, km.labels_)；按簇 groupby 求均值
sil_vals = silhouette_samples(Xs, km.labels_)
per = pd.Series(sil_vals).groupby(km.labels_).mean().round(4)
# @@end
print("各簇平均轮廓:", per.to_dict())
print("overall:", round(float(silhouette_score(Xs, km.labels_)), 4))
print("最低样本轮廓:", round(float(sil_vals.min()), 4),
      "| 负轮廓样本数:", int((sil_vals < 0).sum()))

assert per.to_dict() == {0: 0.5381, 1: 0.5907, 2: 0.5638}
assert round(float(silhouette_score(Xs, km.labels_)), 4) == 0.5628
assert int((sil_vals < 0).sum()) == 0, "没有样本分错阵营"
'''

E5_CODE = '''# 循环骨架不用挖空；挖空的是循环体
for link in ("ward", "complete", "average"):
    # @@todo(7) 对当前链接跑层次聚类，算 ARI 与「与 KMeans 逐位一致率」
    # @@hint AgglomerativeClustering(n_clusters=3, linkage=link).fit(Xs)；same = float((ag.labels_ == km.labels_).mean())
    ag = AgglomerativeClustering(n_clusters=3, linkage=link).fit(Xs)
    same = float((ag.labels_ == km.labels_).mean())
    # @@end
    print(f"{link:>9}: ARI={adjusted_rand_score(y_true, ag.labels_):.4f} "
          f"| 与 KMeans 标签逐位一致率: {same:.4f}")

ag_ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(Xs)
st_pred = probe(getattr, AgglomerativeClustering(), "predict")
print("AGG.predict →", st_pred, "← 层次聚类没有 predict，新样本要另想办法")

assert round(adjusted_rand_score(y_true, ag_ward.labels_), 4) == 1.0
assert float((ag_ward.labels_ == km.labels_).mean()) == 0.0, "ARI=1.0 但编号完全错位——簇编号无语义"
assert st_pred == ("err", "AttributeError")
'''

E6_CODE = '''# @@todo(8) 用 eps=0.3 跑 DBSCAN，看它把多少点判成噪声
# @@hint DBSCAN(eps=0.3, min_samples=5).fit(Xs)；噪声 = (db.labels_ == -1).sum()
db = DBSCAN(eps=0.3, min_samples=5).fit(Xs)
n03 = int((db.labels_ == -1).sum())
# @@end
print(f"eps=0.3: 噪声 {n03}/300，只剩 {len(set(db.labels_)) - 1} 个簇 ← 全员社死")

# eps 扫描（不用挖空）：噪声海 → 碎片化 → 基本成型 → 完美
results = {}
for eps in (0.5, 0.8, 1.2):
    dbp = DBSCAN(eps=eps, min_samples=5).fit(Xs)
    lab = dbp.labels_
    n_cl = len(set(lab)) - (1 if -1 in lab else 0)
    n_noise = int((lab == -1).sum())
    ari = round(float(adjusted_rand_score(y_true[lab != -1], lab[lab != -1])), 4) if n_cl > 1 else 0.0
    results[eps] = (n_cl, n_noise, ari)
    print(f"eps={eps}: 簇数={n_cl} 噪声={n_noise} ARI(非噪声)={ari}")

assert n03 == 289, "eps 太小：289 个点被判噪声"
assert results[0.5] == (9, 111, 0.3366), "eps 稍大：碎成 9 瓣"
assert results[0.8] == (5, 4, 0.9081)
assert results[1.2] == (3, 0, 1.0), "eps=1.2 才完整还原三簇"
'''

E7_CODE = '''# @@todo(9) 用 NearestNeighbors 算每点第 4 近邻距离，报 p50/p90/p95 分位
# @@hint nn.kneighbors(Xs)[0][:, -1]；np.quantile(d4, q)
nn = NearestNeighbors(n_neighbors=5).fit(Xs)
d4 = np.sort(nn.kneighbors(Xs)[0][:, -1])          # 每个点的第 4 近邻距离
p50, p90, p95 = (round(float(np.quantile(d4, q)), 4) for q in (0.5, 0.9, 0.95))
# @@end
print(f"第4近邻距离分位: p50={p50} p90={p90} p95={p95}")
print("经验法则: min_samples=5 时，eps 取曲线拐点附近 ≈ 1.0~1.2")

assert p50 == 0.5348 and p90 == 0.7546 and p95 == 0.8173
assert p95 < 1.2, "拐点之下才解释了 eps=1.2 为什么不坍缩"
'''

E8_CODE = '''# @@todo(10) 训练 GMM(n_components=3)，对账 ARI 并看概率输出形状
# @@hint GaussianMixture(n_components=3, random_state=42).fit(Xs)；gm.predict / gm.predict_proba
gm = GaussianMixture(n_components=3, random_state=42).fit(Xs)
lab_g = gm.predict(Xs)
proba_g = gm.predict_proba(Xs)
# @@end
print("GMM ARI:", round(adjusted_rand_score(y_true, lab_g), 4),
      "| 计数:", pd.Series(lab_g).value_counts().sort_index().to_dict())
print("proba shape:", proba_g.shape, "| 行和:", np.round(proba_g.sum(axis=1)[:3], 6).tolist())
print("新样本(商业画像) GMM 概率:", np.round(gm.predict_proba(new_s)[0], 4).tolist(),
      "→", ["商业型", "工业型", "居民型"][int(np.argmax(gm.predict_proba(new_s)[0]))])

assert round(adjusted_rand_score(y_true, lab_g), 4) == 1.0
assert proba_g.shape == (300, 3)
assert float(np.abs(proba_g.sum(axis=1) - 1).max()) < 1e-6
assert int(np.argmax(gm.predict_proba(new_s)[0])) == 0, "商业画像落在商业簇"
'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch06 讲解：聚类与无监督

任务：在 `station_load.csv`（300 台区 × 5 特征）上把台区按**用电形态**分组。
「用电类型」列**只用于对账**——真实聚类任务里没有它。

本章四个必考现场：

- **scaled ARI=1.0 vs raw ARI=0.2557**——量纲差 3 个数量级，距离被容量独裁
- **肘部法看不出拐点，轮廓系数在 k=3 出峰**——两把尺子一起带
- **三种链接 ARI 全是 1.0，但 ward 与 KMeans 标签一致率 0.0**——簇编号是抽签
- **DBSCAN 的 eps 从 0.3 到 1.2：全员社死 → 碎成 9 瓣 → 完美三簇**

> 本章 `assert` 真值全部沉淀在方向 README 的「ch06 专项真值」表里。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 无监督建模 | KMeans / 层次 / DBSCAN / GMM 的 fit-predict | 簇分配、簇心、新样本归属 |
| 聚类评估 10% | 无标签与有标签两类指标 | inertia / 轮廓系数 / ARI |

**学习目标**：跑通四族聚类算法；能解释「为什么聚类必 scale」「簇编号为什么
不能直接当标签用」「DBSCAN 的 eps 怎么选」。

## 二、API 速查表

| 类 / 函数 | 关键参数 | 一句话说明 |
|---|---|---|
| `KMeans` | `n_clusters` `n_init` `random_state` | 质心聚类；`inertia_` / `predict` / `transform` |
| `AgglomerativeClustering` | `n_clusters` `linkage` | 层次聚类；**无 `predict`** |
| `DBSCAN` | `eps` `min_samples` | 密度聚类；噪声=-1；形状自由但 eps 敏感 |
| `GaussianMixture` | `n_components` `random_state` | 软分配；`predict_proba` / AIC / BIC |
| `silhouette_score` / `silhouette_samples` | — | 整体 / 样本级轮廓系数（-1~1） |
| `adjusted_rand_score` | — | 有真标签时的对账指标（对编号置换免疫） |
| `NearestNeighbors` | `n_neighbors` | k-距离图选 eps 的工具 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 `KMeans`：教科书级起点

标准化后 k=3 一次跑出 **ARI=1.0**：90/75/135 三个簇与真实三型一一对应。
`predict` 给新样本归属、`transform` 给到各簇心的距离——这是 KMeans 独有的
「可服务化」能力。
"""
    ),
    code(S1_CODE),
    md(
        """
### 3.2 难点深挖：不 scale 的聚类是「容量独裁」

**为什么难**：报装容量（数百 kVA）与负荷率（零点几）量纲差 3 个数量级。
欧氏距离里，率值列的差异完全被容量列淹没——**raw 版 ARI 0.2557，
近四成台区被分错型**，而且不报任何错。

**正误对照**（同一数据、同一 k=3、同一随机种子）：

| 输入 | ARI | 最优映射 acc | inertia |
|---|---|---|---|
| 标准化 | **1.0000** | 1.0000 | 402.62 |
| 原始 | 0.2557 | 0.6100 | 23988840.36 |

**判定规则**：**KMeans / 层次 / DBSCAN / GMM 全家都吃欧氏距离，一律先 scale**。
inertia 是量纲的奴隶——raw 的 2398 万和 scaled 的 402.6 不可比。
"""
    ),
    code(S2_CODE),
    md(
        """
## 3.3 k 怎么选：肘部失灵，轮廓出峰

inertia 随 k 单调下降，本数据的「肘」并不明显；轮廓系数在 **k=3** 出峰
0.5628，与真实簇数一致。竞赛口径：**肘部法 + 轮廓系数一起报**。
"""
    ),
    code(S3_CODE),
    md(
        """
## 3.4 轮廓系数往下钻一层

`silhouette_samples` 给样本级健康度：三簇平均 0.54 / 0.59 / 0.56，
最低样本 0.2786、负轮廓 **0** 个——没有样本被分错阵营。
"""
    ),
    code(S4_CODE),
    md(
        """
### 3.5 难点深挖：簇编号是抽签，ARI 才是对账尺

ward / complete / average 三种链接 ARI 全部 1.0，但 ward 的标签编号与
KMeans **逐位一致率 0.0**——居民被编成 0、商业 1、工业 2，和 KMeans
完全错位。**簇编号没有语义**：跨算法、跨运行比较聚类必须用对编号置换
免疫的 ARI（或互信息），不能直接比标签。

另一个硬约束：`AgglomerativeClustering` **没有 `predict`**——层次聚类
只对训练数据负责，新样本归属要靠 `fit` 后手工找最近簇心或换 KMeans。
"""
    ),
    code(S5_CODE),
    md(
        """
## 3.6 `DBSCAN`：eps 一字符定生死

eps=0.3 → 289/300 全员「噪声」；0.5 → 碎成 9 瓣；0.8 → 5 簇 4 噪声；
**1.2 → 恰好 3 簇 0 噪声、ARI=1.0**。DBSCAN 的诱人之处（形状自由、
自带噪声检测）与危险之处（eps 极敏感）是同一枚硬币。
"""
    ),
    code(S6_CODE),
    md(
        """
## 3.7 eps 别拍脑袋：k-距离图找肘

`min_samples=5` 时看每个点的**第 4 近邻距离**：p50=0.53、p95=0.82，
曲线在 1.0 附近抬头——所以 eps 取 1.0~1.2 不坍缩也不碎裂。
"""
    ),
    code(S7_CODE),
    md(
        """
## 3.8 `GaussianMixture`：软分配的 KMeans

GMM 同样 ARI=1.0，且每个样本拿到**三簇隶属概率**（行和为 1）：
新样本「商业画像」落商业簇。KMeans 只会硬投票，GMM 会说
「92% 商业 + 6% 居民 + 2% 工业」——边界样本的场景它更会说话。
"""
    ),
    code(S8_CODE),
    md(
        """
## 四、本章小结

| 事实 | 数字 | 出处 |
|---|---|---|
| scaled 完美聚类 | KMeans k=3 ARI=1.0，簇 90/75/135 | §3.1 |
| 量纲陷阱 | raw ARI 0.2557、acc 0.6333 | §3.2 难点深挖 |
| k 选择 | inertia 单调降；轮廓系数 k=3 出峰 0.5628 | §3.3 |
| 簇健康度 | 各簇轮廓 0.54/0.59/0.56，负轮廓 0 | §3.4 |
| 编号任意性 | ward ARI=1.0 但与 KMeans 一致率 0.0 | §3.5 难点深挖 |
| AGG 无 predict | `predict` → AttributeError | §3.5 |
| DBSCAN eps 三部曲 | 0.3→289 噪声 / 0.5→9 簇 / 1.2→完美 | §3.6 |
| k-距离选 eps | p95=0.8173，拐点 ≈ 1.0~1.2 | §3.7 |
| 软分配 | GMM ARI=1.0 + 概率行和 1 | §3.8 |

### 自测五问

1. 题 1 的簇编号 0/1/2 换个随机种子还会一样吗？汇报结果时该怎么写？
2. 题 2 的 raw inertia 是 2398 万，有人说「远大于 402，说明聚类更差」——错在哪？
3. 题 3 里肘部法和轮廓系数意见不合时听谁的？
4. 题 5 为什么层次聚类 ARI=1.0 也没法给新样本发簇号？
5. 题 6 的 eps=0.8 已经 4 个噪声了，为什么还要放大到 1.2？

全部答得上来，进入 ch07（分类评估）。
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch06 练习：聚类与无监督

与讲解版逐 Cell 对应。核心代码被挖空，`assert` 验收**保留**——
跑通所有 `assert` 即自查通过。卡住就回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 无监督建模 | KMeans / 层次 / DBSCAN / GMM 的 fit-predict |
| 聚类评估 | inertia / 轮廓系数 / ARI |

## 二、API 速查

| 类 | 关键参数 | 一句话 |
|---|---|---|
| `KMeans(n_clusters, n_init)` | `random_state` | `predict` / `transform` 可服务化 |
| `AgglomerativeClustering(linkage)` | — | 无 `predict` |
| `DBSCAN(eps, min_samples)` | — | 噪声=-1，eps 敏感 |
| `GaussianMixture(n_components)` | — | 软分配 `predict_proba` |
| `silhouette_score` / `silhouette_samples` | — | 整体 / 样本级轮廓 |
| `adjusted_rand_score` | — | 编号置换免疫的对账尺 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：KMeans 完美聚类与新样本归属（对应讲解 §3.1）"),
    code(E1_CODE),
    md(
        """
**为什么这样做**：ARI 不受簇编号置换影响，交叉表 + 最优映射给出「哪个簇
是哪类人」的业务语言——两把尺子一起拿，汇报才完整。
"""
    ),
    md("## 题 2：量纲陷阱（对应讲解 §3.2）"),
    code(E2_CODE),
    md("## 题 3：k 扫描——肘部与轮廓（对应讲解 §3.3）"),
    code(E3_CODE),
    md("## 题 4：样本级轮廓（对应讲解 §3.4）"),
    code(E4_CODE),
    md("## 题 5：层次聚类与编号抽签（对应讲解 §3.5）"),
    code(E5_CODE),
    md("## 题 6：DBSCAN 的 eps（对应讲解 §3.6）"),
    code(E6_CODE),
    md("## 题 7：k-距离图选 eps（对应讲解 §3.7）"),
    code(E7_CODE),
    md("## 题 8：GMM 软分配（对应讲解 §3.8）"),
    code(E8_CODE),
    md(
        """
## 收官自查

1. 题 1 的 ARI=1.0 和 acc=1.0 分别在说什么？去掉「用电类型」列还剩哪个能算？
2. 题 2 里 raw 版也报出了 3 个簇，为什么说它「失忆」？
3. 题 3 的轮廓系数 0.5628 离 1 很远——这个聚类到底好不好？
4. 题 5 的 ward 与 KMeans 一致率 0.0，谁对谁错？
5. 题 8 的 GMM 概率 0.92 意味着什么？什么业务场景必须用它而不是 KMeans？

全部答得上来，进入 ch07（分类评估）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
