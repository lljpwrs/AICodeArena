# AICodeArena

面向 **电力行业 / 国家电网体系人工智能职业技能竞赛** 的知识点复习与手撕代码题库。

这不是教程合集，而是**按竞赛评分点组织的训练场**——每一章都能自查"我漏了哪一步"。

> 竞赛是"按步骤给分"的：江苏选拔赛的数据分析模块把「3 项数据探索 + 5 项数据清洗 + 2 项数据可视化」拆成每项 2 分独立计分。
> 所以本仓库每个章节都带一张迷你评分表，而不是只对最终精度。

---

## 一、目录导航

| 目录 | 作用 | 状态 |
|---|---|---|
| [`coding/`](./coding/) | **手撕代码题库**（本仓库主体）：五大方向 + 改错题 | ✅ 67/67 + 3 道改错题 |
| [`exams/`](./exams/) | 限时全真模拟卷 + 读题拆解训练 | ✅ mock01 已建（可执行评分表） |
| [`theory/`](./theory/) | 理论题库（选择 / 判断 / 简答） | 占位，待参考书籍 |
| [`docs/`](./docs/) | 考纲对照、编写规范、应试清单、技术文档模板 | ✅ |
| [`scripts/`](./scripts/) | Notebook 校验 + 模拟卷判卷器 | ✅ |
| [`_notes/`](./_notes/) | 错题本，复习时优先看 | ✅ |

## 二、coding/ 五大方向

| 目录 | 主题 | 对应竞赛考点 |
|---|---|---|
| [`01_pandas`](./coding/01_pandas/) | 数据预处理 | 数据探索 / 清洗 / 可视化（独立计分项）+ 数据准备及处理 10% |
| [`02_sklearn`](./coding/02_sklearn/) | 机器学习 | 模型选择 10% / 训练 40% / 调参 10% / 评估 10% |
| [`03_cv`](./coding/03_cv/) | 计算机视觉 | 电力现场作业安全监督场景综合应用 |
| [`04_nlp`](./coding/04_nlp/) | NLP 与 Transformer | 大语言模型相关理论考查 + 大模型微调 / RAG |
| [`05_timeseries`](./coding/05_timeseries/) | 时序预测 | 新能源功率预测 / 负荷预测 |

## 三、每章的四件套

```
coding/01_pandas/
├── README.md                              # 考纲、知识点表、易错点、练习指引、迷你评分表
├── data/make_data.py                      # 合成数据生成脚本（固定 seed=42）
├── ch04_missing.ipynb                     # 讲解
├── ch04_missing_practice.ipynb            # 练习（核心代码挖空 + 注释提示）
└── ch04_missing_solution.ipynb            # 答案（完整可直接运行）
```

`practice` 与 `solution` **同级平铺**，Cell 数量、顺序、标题强制一一对应。

## 四、复习四步法

1. **读** `chXX_*.ipynb` —— 只看讲解，不敲代码，建立方法地图
2. **做** `chXX_*_practice.ipynb` —— 按 `TODO` 提示填空题，用 `assert` 自查
3. **对** `chXX_*_solution.ipynb` —— 逐 Cell 比对，差异记入 `_notes/错题本.md`
4. **查** 本章 `README.md` 的迷你评分表 —— 逐项打勾，漏项回头补

限时训练用 `exams/` 下的模拟卷，严格计时，做完再用评分表核对。

## 五、环境

固定使用本机 conda 环境 `self`，不新建 venv：

```bash
/Users/luolinjie/miniconda3/envs/self/bin/python -V
```

依赖清单见 [`docs/环境依赖.md`](./docs/环境依赖.md)。缺包直接装进该环境。

## 六、Notebook 一致性校验

```bash
/Users/luolinjie/miniconda3/envs/self/bin/python scripts/check_notebooks.py
```

校验四件事：① 所有 notebook 是合法 JSON 且结构完整；② 每对 practice / solution 的 Cell 数与各 Cell 的 Markdown / `assert` 一致；③ **逐 Code Cell 语法检查**（防挖空把关键字参数当赋值目标这类生成器 bug）；④ practice 中确实存在挖空标记。

## 七、修改与扩展指引 ★

> **这一节是给未来的自己（和 AI）看的操作手册。**
> 题库不可能一次写到位——真正做题时才会发现「这个知识点该拆开」「那个真值不对」
> 「章节顺序该调」「这章该并进下一章」。下面把每种改动的**影响面**写清楚，照着做就不会漏改。
>
> - **本章节**：全仓通用的操作手册（01~05 所有方向共用）。
> - **方向级差异**：各方向 `coding/NN_*/README.md` 的「修改与扩展指引」一节，
>   写该方向特有的约定（数据源、依赖、章节划分待定项等），**冲突时以方向级为准**。

### 7.1 真相源与派生产物（先理解这条，后面都顺）

| 角色 | 路径 | 能否手改 |
|---|---|---|
| **唯一真相源** | `scripts/authoring/<方向><章>.py` | ✅ 只改这里 |
| 生成器 | `scripts/nb_builder.py` | ❌ 改动需全仓回归 |
| **派生产物** | `coding/NN_*/chNN_*.ipynb`（讲解）· `_practice.ipynb` · `_solution.ipynb` | ❌ **禁止手改**，改脚本重跑 |
| 校验器 | `scripts/check_notebooks.py` / `scripts/exec_notebooks.py` | ❌ |
| 活文档 | 本文件「八、进度」+ `coding/NN_*/README.md` | ✅ |
| 模板 | `scripts/authoring/_TEMPLATE_chapter.py` | ✅ |

一句话：**改内容 → 只动 authoring 脚本；改规划 → 动 README。ipynb 永远是产物。**

### 7.2 三类改动的标准动作

**A. 改一章的内容（最高频）**

```bash
# ① 重建三件套（脚本内部叫 build()）
/Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/<章节脚本>.py

# ② 静态校验：结构 / 配对 / 逐 Cell 语法 / 挖空标记
/Users/luolinjie/miniconda3/envs/self/bin/python scripts/check_notebooks.py -v coding/<方向>

# ③ 讲解版 + 答案版必须**全部跑通**
/Users/luolinjie/miniconda3/envs/self/bin/python scripts/exec_notebooks.py coding/<方向> --exclude-practice

# ④ 练习版必须**失败**，且原因是 NameError: name '____' is not defined
/Users/luolinjie/miniconda3/envs/self/bin/python scripts/exec_notebooks.py coding/<方向> --expect-fail
```

**改了真值 → 必须同步两处**：脚本里的 `assert` + 该方向 `README.md` 的真值备忘表。
（少同步一处，下一个改这章的人就会被旧真值误导。）

**B. 新增一章**

```bash
cp scripts/authoring/_TEMPLATE_chapter.py scripts/authoring/<方向><NN>_<短名>.py
# 改 4 个地方：文件头 docstring / OUT / NAME / 正文 cells
# 然后走 A 的四步校验
```

再补**三处登记**：① 该方向 `README.md` 的章节表加一行；② `coding/README.md` 进度表；③ 本文件「八、进度」表。

**C. 拆分 / 合并 / 重命名章节**

需要同步 **5 处**，顺序别乱：

1. `scripts/authoring/<旧脚本>.py` → 改名（或用 `cp` + `rm` 旧文件）
2. 脚本里的 `OUT` / `NAME` 两个常量
3. `coding/<方向>/` 下 3 个 ipynb 的**文件名**——**不要手工重命名**，改 `NAME` 重跑 `build()` 生成新名，再删旧名三个
4. 该方向 `README.md` 的章节表 + 逐章要点 §
5. `coding/README.md` 与本文件「八、进度」的数字

### 7.3 写作纪律（挖空题专用，违反会直接构建失败）

| 纪律 | 原因 |
|---|---|
| 挖空块内**不得出现控制流或语句头**（`for` / `if` / `while` / `def` / `:` 结尾、续行以 `for` 或 `if` 开头） | `nb_builder` 会抛 `BuildError`；`for` / `if` 骨架要写在块**外** |
| `def` 只能挖**函数体**，不能把整个函数包进块里 | 同上 |
| `assert` 必须写在 `# @@end` **之后** | 否则 assert 会被当成挖空内容 |
| 每个 `@@todo` 至少配一条 `@@hint(...)` | 练习版要能靠提示独立完成 |
| **`assert` 真值必须实跑**，禁止手算填数 | 探针脚本写 `/tmp/`，**不入库** |
| notebook 数据路径按内核 cwd 写 | cwd = 该方向目录；`final/` 子目录要多退一层（`Path("../data")`） |

### 7.4 依赖变化与断网约束

全仓硬约束：**生成脚本断网可跑、notebook 不依赖外部网络**。
装新包后**不要重构已有章节**，按各方向 README 预留的「对照位」把 `try/except ImportError`
分支从「跳过」改成「跑对比」，并补一节 `### 难点深挖` 即可。

若某包（如 `jieba` / `optuna`）因沙箱文件策略装不上——`pip install` 报
`EEXIST ... pip-install-...`——直接把该能力**手写实现**，反而更贴合本仓库「手撕代码」的定位。

### 7.5 不确定项的登记规则

整理时**主动标注**没把握的地方，不要假装定稿。统一登记在**各方向 README** 的
「⚠️ 尚未定稿、可能要改的地方」小节，按 `编号 / 不确定什么 / 两种改法的代价 / 触发条件` 写。
根 README 只保留一句话指路，不重复列条目（避免双份维护走样）。

## 八、进度

| 方向 | 章节 | 进度 |
|---|---|---|
| 01_pandas | ch01 基础 / ch02 读写 / ch03 探索 / ch04 缺失值 / ch05 重复与异常 / ch06 类型与字符串 / ch07 筛选与条件 / ch08 排序与采样 / ch09 分箱 / ch10 分组聚合 / ch11 变换与 apply / ch12 窗口 / ch13 连接与合并 / ch14 重塑与透视 / ch15 多级索引 / ch16 时间序列 / ch17 性能与内存 / ch18 可视化 / final 综合题 | **19/19** ✅ |
| 02_sklearn | ch01 划分与流水线 / ch02 特征工程 / ch03 特征选择降维 / ch04 回归 / ch05 分类 / ch06 聚类 / ch07 分类评估 / ch08 回归评估 / ch09 交叉验证与调参 / ch10 集成与融合 / ch11 评估可视化 / ch12 类别不平衡 / ch13 持久化与推理 / final 时序预测综合题 | **14/14 ✅** |
| 03_cv | ch01 图像读写与数组基础 / ch02 图像清洗与数据增强 / ch03 颜色空间与阈值分割 / ch04 卷积与滤波 / ch05 边缘检测与梯度 / ch06 形态学与轮廓 / ch07 图像特征与分类 / ch08 模板匹配与关键点 / ch09 目标检测基础 / ch10 图像分割进阶 / ch11 深度学习入门 / final_01 安全帽视觉检测全流程 / final_02 手写数字识别全流程 / final_03 分割测量全流程 | **14/14** ✅ |
| 04_nlp | ch01 文本预处理 / ch02 文本表示 / ch03 序列建模动机 / ch04 Self-Attention / ch05 Transformer 架构 / ch06 训练与推理细节 / ch07 微调范式（BERT/WordPiece/LoRA） / ch08 jieba 中文分词实战（前缀词典+DAG+DP 手写复刻 / HMM / 用户词典 / 词性 / 关键词） / ch09 小模型训练与规模判定（参数量恒等式 / 内存四件套 / 激活缩放律 / 梯度累积 / 冻结 / 精度） / ch10 RAG 检索增强（清洗深度 / 手写 BM25 / 稀疏·稠密·融合 / 误差归因 / 定向改进 / 拒答与 prompt 拼装） / final_01 工单文本分类全流程 | **11/11** ✅ |
| 05_timeseries | ch01 时序数据与特征 / ch02 平稳性与统计方法 / ch03 RNN 手写 / ch04 LSTM·GRU / ch05 序列到序列预测 / ch06 数据构造与标准化 / ch07 训练与评估 / ch08 进阶与基线 / final 负荷预测全流程 | **9/9** ✅ |
| 99_debug | 改错题（pandas 陷阱 / sklearn 泄漏 / 时序坑） | **3/3** ✅ |
| exams | mock01 数据分析与挖掘（含可执行评分表）；mock02 ~ mock04 待建 | **1/4** |

## 九、待办

- [x] ~~技术文档撰写模板（部分赛事占技能操作 **20%**）~~ → ✅ **已完成**
      [`docs/技术文档撰写模板.md`](./docs/技术文档撰写模板.md)（骨架 + 逐节写法 + 提交前自查清单）
      + [`exams/mock01_数据分析与挖掘/技术文档示例.md`](./exams/mock01_数据分析与挖掘/技术文档示例.md)（满分档示例，含 4 张真图）
- [x] ~~大模型微调与 RAG（中电联 2025 / 2026 实操重点）~~ → ✅ **已完成**
      `04_nlp` **ch09 小模型训练与规模判定**（参数量恒等式对账 **0.0000%** / 16 B·参数⁻¹ 内存模型 /
      激活 batch 线性·seq 超线性 / 梯度累积等价 **2.1e-07** / 冻结与精度的收益边界）
      + **ch10 RAG 检索增强**（清洗深度 MRR **0.5659 → 0.6111** / 手写 BM25 / 四条路线对照 /
      R@k 上限 / 误差归因四象限 / **定向改进 MRR → 0.9345** / 拒答阈值 4.0 / prompt 拼装与引用）
- [ ] 理论题库：等待提供参考书籍后按书籍目录重建 `theory/`
- [ ] 算法打榜赛提分套路（`coding/06_打榜技巧/`）—— 若转向数字中国类算法赛再建

> 各方向「未定稿、可能要改」的条目，见对应 `coding/NN_*/README.md` 的
> 「修改与扩展指引 → 尚未定稿、可能要改的地方」（如 [`04_nlp`](./coding/04_nlp/README.md)）。

## 十、许可

见 [LICENSE](./LICENSE)。
