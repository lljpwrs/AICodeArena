# coding/ —— 手撕代码题库

按竞赛实考方向拆成五个模块。**优先级不等于顺序号**，建议按 §二 的推荐顺序推进。

---

## 一、模块清单

| 模块 | 主题 | 核心库 | 竞赛对应考点 | 章节数 |
|---|---|---|---|---|
| [`01_pandas`](./01_pandas/) | 数据预处理 | pandas / numpy / matplotlib | 数据探索、数据清洗、数据可视化（各自独立计分）；数据准备及处理占技能操作 10% | 18 章 + final（已完成） |
| [`02_sklearn`](./02_sklearn/) | 机器学习 | scikit-learn / matplotlib | 模型选择 10%、模型训练 40%、模型调参 10%、模型性能评估 10% | 13 + 综合 |
| [`03_cv`](./03_cv/) | 计算机视觉 | torch / torchvision / ultralytics | 电力现场作业安全监督场景综合应用 | 11 + 3 个全流程 |
| [`04_nlp`](./04_nlp/) | NLP 与 Transformer | torch / transformers / jieba | 大语言模型相关理论考查 + 大模型微调 / RAG | 10 + 综合 |
| [`05_timeseries`](./05_timeseries/) | 时序预测 | torch / lightgbm / statsmodels | 新能源功率预测、负荷预测 | 8 + 综合 |
| [`99_debug`](./99_debug/) | **改错题**（复用前五方向代码埋雷） | — | 「会调试」——赛场常态是给你跑不通的代码 | 3 题 |

---

## 二、推荐推进顺序

```
01_pandas  →  02_sklearn  →  05_timeseries  →  03_cv  →  04_nlp
   地基         计分主力        与 sklearn 综合题衔接    体量最大   原理性最强
```

理由：

1. **pandas 必须最先做透**。数据探索 / 清洗 / 可视化在竞赛里是**独立计分项**，各 2 分；而且后面所有方向都建立在它之上。
2. **sklearn 是分值主力**（训练 40% + 调参 10% + 评估 10% + 选择 10% = 70%）。
3. **时序提到 CV 之前**：`02_sklearn` 的综合题本身就是时序预测，两章连着做能打通「pandas → sklearn → 时序」链路。
4. **CV 体量最大**（三个全流程模型题），单独排期。
5. **NLP 偏原理**，手写 Self-Attention / Transformer 需要整块时间。

---

## 三、每章四件套

见根目录 [`AGENTS.md`](../AGENTS.md) §二、§三。简版：

```
chNN_<短名>.ipynb             讲解
chNN_<短名>_practice.ipynb    练习（挖空）
chNN_<短名>_solution.ipynb    答案（完整可运行）
```

---

## 四、进度

| 模块 | 已完成章节 | 进度 |
|---|---|---|
| 01_pandas | ch01 ~ ch18 + final 全部 | 19 / 19 |
| 02_sklearn | ch01 ~ ch13 + final | 14 / 14 |
| 03_cv | ch01 ~ ch11 + final_01 ~ final_03 | 14 / 14 |
| 04_nlp | ch01 ~ ch10 + final | 11 / 11 |
| 05_timeseries | ch01 ~ ch08 + final（全流程） | **9 / 9** ✅ |
| **合计** | **五大方向** | **67 / 67** ✅ |
| 99_debug | dbg01 pandas 陷阱 / dbg02 sklearn 泄漏 / dbg03 时序坑 | 3 / 3 |
