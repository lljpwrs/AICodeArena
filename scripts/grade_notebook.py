#!/usr/bin/env python
"""模拟卷自动判卷器。

为什么需要它
------------
竞赛按步骤给分，「漏做一项就丢一项的分」。而练习 notebook 里的 `assert`
只能验证**你写的那一行**对不对，验证不了两件事：

  · **漏项** —— 题目要求 5 项数据清洗，你只做了 3 项；
  · **顺序错** —— 每行单看都对，但你先标准化再划分，造成数据泄漏。

静态的 Markdown 评分表又只能靠自觉打勾，而**自己给自己打分恰恰最不可靠**：
你不知道自己不知道什么。

本脚本把评分表变成**可执行的检查器**，对一份 notebook 逐项判定 ✅ / ❌。

两个判卷通道
------------
① **静态通道**：把所有 code cell 拼成一段源码，扫描你显式写了哪些 API、
   按什么顺序写。用来判「有没有做」「顺序对不对」。
② **动态通道**：执行一遍 notebook，捕获 stdout 与最终命名空间。用来判
   「有没有把删除行数打印出来」「准确率有没有达标」。

能静态判的就不要依赖动态 —— 一旦执行中断，动态项全部失效。

用法
----
    python scripts/grade_notebook.py exams/mock01_数据分析与挖掘/mock01.ipynb
    python scripts/grade_notebook.py <notebook> --rubric <rubric.yaml>
    python scripts/grade_notebook.py <notebook> --static-only
    python scripts/grade_notebook.py <notebook> --json

评分表格式见 `exams/mock01_数据分析与挖掘/rubric.yaml`，检查器语义：

    static_any    正则列表，命中**任一**即算做过
    static_all    正则列表，必须**全部**命中
    static_none   正则列表，必须**全部不出现**（用于「别踩这个坑」）
    order_before  正则列表，必须按此先后顺序出现（用于「先划分再标准化」）
    output_regex  正则，须在 stdout 里命中（用于「打印删除的行数」）
    value_expr    Python 表达式，在最终命名空间里求值须为真（用于指标达标）

同一 item 下的多个检查器是 **AND** 关系；全部通过才给分。
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import io
import json
import os
import re
import sys
import traceback
import unicodedata
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("[错误] 需要 pyyaml：pip install pyyaml", file=sys.stderr)
    raise SystemExit(2) from None

MAGIC_RE = re.compile(r"^\s*[%!]")
DRAW_METHODS = (
    "plot", "boxplot", "hist", "scatter", "bar", "barh",
    "pie", "imshow", "fill_between", "step",
)

OK, FAIL, WARN = "✅", "❌", "⚠️"


# --------------------------------------------------------------------- 读取

def load_notebook(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[错误] {path} 不是合法 JSON：{exc}") from exc


def code_cells(nb: dict) -> list[tuple[int, str]]:
    """返回 [(cell 序号, 源码)]，只含 code cell。"""
    out = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            out.append((idx, "".join(cell.get("source", []))))
    return out


def strip_magics(src: str) -> str:
    """去掉 `%matplotlib inline` / `!pip install` 这类 IPython 魔术命令。"""
    return "\n".join("" if MAGIC_RE.match(line) else line for line in src.splitlines())


# --------------------------------------------------------------------- 执行

def install_draw_hooks(sink: list[str]) -> dict:
    """给 matplotlib 的绘图方法打桩，把「画了什么图」记进 sink。"""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    saved: dict = {}
    for name in DRAW_METHODS:
        original = getattr(Axes, name, None)
        if original is None:
            continue
        saved[name] = original

        def make(orig, method_name):
            def wrapper(self, *args, **kwargs):
                sink.append(method_name)
                return orig(self, *args, **kwargs)

            return wrapper

        setattr(Axes, name, make(original, name))
    return saved


def uninstall_draw_hooks(saved: dict) -> None:
    from matplotlib.axes import Axes

    for name, original in saved.items():
        setattr(Axes, name, original)


def run_notebook(cells: list[tuple[int, str]], cwd: Path) -> dict:
    """执行 notebook 的 code cell，返回 {ns, stdout, figures, error}。

    执行中断时保留已产生的 stdout 与命名空间 —— 静态检查照常工作，
    动态检查会被标记为「未验证」。
    """
    ns: dict = {"__name__": "__main__"}
    chunks: list[str] = []
    figures: list[str] = []
    error = None

    saved = install_draw_hooks(figures)
    previous_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        for idx, src in cells:
            body = strip_magics(src)
            if not body.strip():
                continue
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(body, f"<cell {idx}>", "exec"), ns)
            except Exception:
                chunks.append(buf.getvalue())
                last = traceback.format_exc(limit=2).strip().splitlines()[-1]
                error = f"cell {idx} — {last}"
                break
            chunks.append(buf.getvalue())
    finally:
        uninstall_draw_hooks(saved)
        os.chdir(previous_cwd)

    return {"ns": ns, "stdout": "\n".join(chunks), "figures": figures, "error": error}


# --------------------------------------------------------------------- 判定

def first_pos(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    return match.start() if match else -1


def judge(item: dict, ctx: dict) -> tuple[bool, str, bool]:
    """对单个 item 判分，返回 (是否通过, 失败原因, 是否已验证)。"""
    text, stdout, ns = ctx["text"], ctx["stdout"], ctx["ns"]
    executed = ctx["executed"]
    reasons: list[str] = []
    verified = True

    group = item.get("static_any")
    if group and not any(first_pos(text, p) >= 0 for p in group):
        reasons.append("未检测到 " + " 或 ".join(f"`{p}`" for p in group))

    group = item.get("static_all")
    if group:
        missing = [p for p in group if first_pos(text, p) < 0]
        if missing:
            reasons.append("缺少 " + "、".join(f"`{p}`" for p in missing))

    group = item.get("static_none")
    if group:
        hit = [p for p in group if first_pos(text, p) >= 0]
        if hit:
            reasons.append("出现不该有的写法 " + "、".join(f"`{p}`" for p in hit))

    seq = item.get("order_before")
    if seq:
        positions = [first_pos(text, p) for p in seq]
        if any(p < 0 for p in positions):
            miss = [p for p, pos in zip(seq, positions) if pos < 0]
            reasons.append("无法判定顺序（缺少 " + "、".join(f"`{p}`" for p in miss) + "）")
        elif any(positions[i] >= positions[i + 1] for i in range(len(positions) - 1)):
            reasons.append("顺序错误，应为 " + " → ".join(f"`{p}`" for p in seq))

    pattern = item.get("output_regex")
    if pattern:
        if not executed:
            verified = False
        elif not re.search(pattern, stdout):
            reasons.append(f"输出里没有匹配 `{pattern}` 的内容")

    expr = item.get("value_expr")
    if expr:
        if not executed:
            verified = False
        else:
            try:
                ok = bool(eval(expr, {"__builtins__": builtins}, dict(ns)))  # noqa: S307
            except Exception as exc:
                ok = False
                reasons.append(f"表达式 `{expr}` 求值出错：{exc}")
            if not ok and not reasons:
                reasons.append(f"表达式 `{expr}` 不成立")

    if reasons:
        # 已经有确定结论（缺什么调用 / 顺序错 / 输出没匹配上）→ 直接判未通过。
        # 不要让它被 verified 拉成「未验证」，否则「明确没做」会显示成「无法判定」。
        return False, "；".join(reasons), True
    if not verified:
        return False, "notebook 执行中断，动态检查未能验证", False
    return True, "", True


# --------------------------------------------------------------------- 报告

def pad(text: str, width: int) -> str:
    """按显示宽度左侧补空格（全角字符算 2 列）。"""
    used = sum(2 if unicodedata.east_asian_width(ch) in "WFA" else 1 for ch in text)
    return text + " " * max(0, width - used)


def render(payload: dict, notebook: Path) -> None:
    meta, steps = payload["meta"], payload["steps"]
    line = "=" * 72
    print(line)
    print(f" {pad(meta['title'], 40)}{pad('', 4)}限时 {meta['limit_min']} min · 满分 {meta['total']} 分")
    print(f" 作答文件：{notebook}")
    print(line)

    for step in steps:
        got = sum(i["score"] for i in step["items"] if i["passed"])
        full = sum(i["score"] for i in step["items"])
        print(f"\n{pad(step['name'], 58)}{got} / {full}")
        for item in step["items"]:
            if item["passed"]:
                mark = OK
            elif not item["verified"]:
                mark = WARN
            else:
                mark = FAIL
            head = f"  {mark} {pad(item['id'], 6)}{pad(item['desc'], 44)}"
            print(f"{head}{item['score'] if item['passed'] else 0} / {item['score']}")
            if item["reason"]:
                print(f"        └─ {item['reason']}")

    total = sum(i["score"] for st in steps for i in st["items"] if i["passed"])
    full = sum(i["score"] for st in steps for i in st["items"])
    n_ok = sum(1 for st in steps for i in st["items"] if i["passed"])
    n_all = sum(len(st["items"]) for st in steps)
    missed = [i["id"] for st in steps for i in st["items"] if not i["passed"]]

    print("\n" + "-" * 72)
    if payload.get("mode") == "static":
        status = "未执行（--static-only）"
    elif payload["executed"]:
        status = "正常"
    else:
        status = f"中断 → {payload['error']}"
    print(f" 总分 {total} / {full}  ·  通过 {n_ok} / {n_all} 项  ·  notebook 执行：{status}")
    if missed:
        print(f" 失分项：{'、'.join(missed)}")
    print(line)


# --------------------------------------------------------------------- 入口

def build_payload(rubric: dict, ctx: dict) -> dict:
    steps = []
    for step in rubric["steps"]:
        items = []
        for item in step["items"]:
            passed, reason, verified = judge(item, ctx)
            items.append({
                "id": item["id"],
                "desc": item["desc"],
                "score": float(item["score"]),
                "passed": passed,
                "reason": reason,
                "verified": verified,
            })
        steps.append({"name": step["name"], "items": items})
    return {
        "meta": rubric["meta"],
        "steps": steps,
        "executed": ctx["executed"],
        "error": ctx.get("error"),
        "mode": ctx.get("mode", "exec"),
        "figures": ctx.get("figures", []),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="模拟卷自动判卷器")
    ap.add_argument("notebook", type=Path, help="要判分的 notebook")
    ap.add_argument("--rubric", type=Path, help="评分表 YAML（默认取 notebook 同级的 rubric.yaml）")
    ap.add_argument("--static-only", action="store_true", help="只跑静态检查，不执行 notebook")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args(argv)

    if not args.notebook.exists():
        raise SystemExit(f"[错误] 找不到 notebook：{args.notebook}")
    rubric_path = args.rubric or args.notebook.parent / "rubric.yaml"
    if not rubric_path.exists():
        raise SystemExit(f"[错误] 找不到评分表：{rubric_path}")

    rubric = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    cells = code_cells(load_notebook(args.notebook))
    text = "\n".join(strip_magics(src) for _, src in cells)

    if args.static_only:
        ctx = {"text": text, "stdout": "", "ns": {}, "executed": False,
               "figures": [], "error": None, "mode": "static"}
    else:
        result = run_notebook(cells, args.notebook.parent)
        ctx = {"text": text, "stdout": result["stdout"], "ns": result["ns"],
               "executed": result["error"] is None, "figures": result["figures"],
               "error": result["error"], "mode": "exec"}

    payload = build_payload(rubric, ctx)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        render(payload, args.notebook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
