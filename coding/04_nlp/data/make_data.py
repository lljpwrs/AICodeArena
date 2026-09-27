#!/usr/bin/env python3
"""生成 04_nlp 用的合成电力设备缺陷工单文本。固定 seed=42，断网可跑、幂等。

设计原则（见 docs/合成数据规范.md）：

- 编号一律虚构：台区 ``STATION_A_01`` ~ ``STATION_F_03``，工单 ``WO_0001``
- 文本里**故意埋噪**：HTML 标签 / URL / 半角标点 / 多余空白 / 重复标点 /
  字间空格 / 空文本 / 无效占位符，供 ch01 清洗与统计
- 五类缺陷的「信号词」互不重叠，保证分类任务既**可学**又**可解释**
- 输出 CSV < 1 MB，直接提交仓库

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python coding/04_nlp/data/make_data.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

SEED = 42
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "synth_text.csv"

HAN_RE = re.compile(r"[\u4e00-\u9fff]")

# --------------------------------------------------------------------------- #
# 词表（信号词按类别互斥，是分类任务的「可解释锚点」）
# --------------------------------------------------------------------------- #

DEFECTS: dict[str, list[str]] = {
    "渗漏油": ["渗油", "漏油", "油迹", "油污", "滴油", "油位下降"],
    "锈蚀": ["锈蚀", "生锈", "铁锈", "漆面剥落", "镀锌层脱落"],
    "破损": ["破损", "裂纹", "碎裂", "缺口", "掉瓷", "箱体变形"],
    "发热": ["发热", "温度偏高", "过热", "烫手", "温升异常"],
    "异物": ["异物", "鸟巢", "塑料袋", "树枝", "风筝线搭挂"],
}

DEVICES = ["变压器", "断路器", "隔离开关", "避雷器", "电流互感器", "绝缘子", "套管", "母线"]
STATIONS = [f"STATION_{c}_{i:02d}" for c in "ABCDEF" for i in (1, 2, 3)]
ACTIONS = ["请安排抢修", "请尽快核实", "已通知运维班", "请列入本周计划", "需停电处理"]
LEVELS = ["一般", "严重", "危急"]

# 六种句式，保证同一标签下文本不重复
TEMPLATES = [
    "{st}{dev}发现{kw}，{lv}缺陷，{act}",
    "{dev}巡检时发现{kw}，位于{st}，{lv}",
    "【{lv}】{st}{dev}存在{kw}，{act}",
    "{st}{dev}{kw}，请核实后反馈，{act}",
    "{kw}（{dev}，{st}），{lv}",
    "接报：{st}{dev}{kw}，{act}",
]

# 发现日期：4 种格式故意混用（含 pandas 3.0 的 to_datetime 解析不了的那种）
DATE_FORMATS = [
    lambda y, m, d: f"{y}-{m:02d}-{d:02d}",
    lambda y, m, d: f"{y}/{m:02d}/{d:02d}",
    lambda y, m, d: f"{m:02d}/{d:02d}/{y}",
    lambda y, m, d: f"{y}年{m}月{d}日",
]

# 处理状态：同一含义 5 种写法
STATUSES = ["已处理", "已 处理", "处理完成", "done", "处理"]


def clean_text(rec: dict[str, str]) -> str:
    """按模板拼一条干净文本。"""
    return rec["_tpl"].format(
        st=rec["台区编号"],
        dev=rec["设备类型"],
        kw=rec["_kw"],
        lv=rec["缺陷等级"],
        act=rec["_act"],
    )


def add_noise(text: str, rng: np.random.Generator) -> str:
    """叠加式注入噪声（各项独立，所以同一条可能带多种噪声）。"""
    if rng.random() < 0.18:                       # HTML 包裹
        text = f"<p>{text}</p>"
    if rng.random() < 0.12:                       # 追加 URL
        text = f"{text} 详见 http://10.0.0.1/wo"
    if rng.random() < 0.25:                       # 全角逗号 → 半角
        text = text.replace("，", ",")
    if rng.random() < 0.30:                       # 多余空白
        text = "  " + text.replace("，", "  ，") + " "
    if rng.random() < 0.15:                       # 重复标点
        text = text.replace("，", "，，，") if rng.random() < 0.5 else text + "！！！"
    if rng.random() < 0.20:                       # 汉字间随机插空格（模拟 OCR / 复制粘贴噪声）
        chars = list(text)
        merged: list[str] = []
        for idx, ch in enumerate(chars):
            merged.append(ch)
            nxt = chars[idx + 1] if idx + 1 < len(chars) else ""
            if HAN_RE.match(ch) and HAN_RE.match(nxt) and rng.random() < 0.35:
                merged.append(" ")
        text = "".join(merged)
    return text


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_rows = 600
    labels = list(DEFECTS)
    rows: list[dict[str, str]] = []

    for i in range(n_rows):
        label = labels[i % len(labels)]          # 先均分保证五类各 120 条
        kw = str(rng.choice(DEFECTS[label]))
        rec = {
            "工单编号": f"WO_{i + 1:04d}",
            "台区编号": str(rng.choice(STATIONS)),
            "设备类型": str(rng.choice(DEVICES)),
            "缺陷类型": label,
            "缺陷等级": str(rng.choice(LEVELS)),
            # 注意：np 的 choice 不能直接挑 callable（会被降级成 str），用整数索引
            "发现日期": DATE_FORMATS[int(rng.integers(len(DATE_FORMATS)))](
                int(rng.integers(2025, 2027)),
                int(rng.integers(1, 13)),
                int(rng.integers(1, 29)),
            ),
            "处理状态": str(rng.choice(STATUSES)),
            "_kw": kw,
            "_act": str(rng.choice(ACTIONS)),
            "_tpl": str(rng.choice(TEMPLATES)),
        }
        text = add_noise(clean_text(rec), rng)
        rec["缺陷描述"] = text
        rows.append(rec)

    # 再注入 30 条「无效文本」（空 / 占位符），供 ch01 做有效性过滤
    for j, bad in enumerate(["", "   ", "N/A", "-", "无"] * 6):
        label = labels[j % len(labels)]
        rows[n_rows - 30 + j]["缺陷描述"] = bad

    cols = [
        "工单编号", "台区编号", "设备类型", "缺陷类型",
        "缺陷等级", "发现日期", "处理状态", "缺陷描述",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for rec in rows:
            writer.writerow({c: rec[c] for c in cols})

    # ---- 统计（真值对账用）----
    texts = [r["缺陷描述"] for r in rows]
    n_empty = sum(1 for t in texts if not t.strip())
    n_html = sum(1 for t in texts if "<p>" in t)
    n_url = sum(1 for t in texts if "http" in t)
    n_half = sum(1 for t in texts if "," in t)
    n_bang = sum(1 for t in texts if "！！" in t)
    n_space = sum(1 for t in texts if "  " in t or t != t.strip())
    n_bad = sum(1 for t in texts if t.strip() in {"", "N/A", "-", "无"})

    print(f"写出 {OUT_CSV}（{OUT_CSV.stat().st_size / 1024:.1f} KB）")
    print(f"总条数：{len(rows)}")
    print(f"缺陷类型分布：{ {k: sum(1 for r in rows if r['缺陷类型'] == k) for k in labels} }")
    print(f"空/空白文本：{n_empty} 个 | 含无效占位符：{n_bad} 个")
    print(f"含 HTML：{n_html} | 含 URL：{n_url} | 含半角逗号：{n_half}")
    print(f"含重复标点：{n_bang} | 含多余/首尾空白：{n_space}")
    print(f"去重后文本数：{len(set(texts))}")


if __name__ == "__main__":
    main()
