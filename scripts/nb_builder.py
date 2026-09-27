#!/usr/bin/env python3
"""Notebook 生成器 —— 仓库内部写作工具。

## 为什么需要它

`*_practice.ipynb` 与 `*_solution.ipynb` 必须逐 Cell 对应（见 `AGENTS.md` §三）。
手写两份 JSON 必然会漂。所以这里把 **Python 源文件当作唯一真相**：

    scripts/authoring/<chapter>.py  ──build──▶  讲解.ipynb
                                                _solution.ipynb
                                                _practice.ipynb（自动派生）

派生规则：`_practice` 由 `_solution` 自动挖空而来，因此
Cell 数量 / 顺序 / Markdown 内容 / assert 验收块 **在构造上就不可能不一致**。

## 挖空 DSL

在普通 Python 代码里插入三个指令注释即可：

```python
q1, q3 = df["负荷值"].quantile([0.25, 0.75])      # ← @@@ 块外的行：两版都保留
iqr = q3 - q1
# @@todo(4) 用 IQR 法算出上下界，把超出范围的值截断到边界上
# @@hint Series.clip(lower=..., upper=...)
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["负荷值"] = df["负荷值"].clip(lower, upper)
# @@end
```

生成的练习版：

```python
q1, q3 = df["负荷值"].quantile([0.25, 0.75])
iqr = q3 - q1
# TODO(4)：用 IQR 法算出上下界，把超出范围的值截断到边界上
# 提示：Series.clip(lower=..., upper=...)
lower, upper = ____
df["负荷值"] = ____
```

规则：

- `# @@todo(描述)` 开启一个挖空块，`# @@end` 结束
- `# @@hint(提示)` 写在 `# @@todo` 与**第一行真正代码**之间的「前导注释区」
  里即可（可选）：可以写在 `# @@todo` 的下一行，也可以写在多行描述续行之后，
  **可以有多条**，都会按出现顺序渲染成 `# 提示：`。前导区里的空行与
  描述续行会原样保留在原位置
- 块内**所有非注释代码行**被挖空，块内注释原样保留
- 挖空时自动保留赋值目标：`x = f(...)` → `x = ____`；`df["c"] = g(...)` → `df["c"] = ____`
  没有赋值的形式（如裸表达式、`plt.show()`）→ 整行变成 `____`
- **挖空以「逻辑语句」为单位，不是以物理行为单位**：跨行的表达式（字典字面量、
  括号换行的链式调用）只产出一个占位，避免生成「孤立缩进行」导致的 `IndentationError`
- 题号 `TODO(n)` 从 1 起按出现顺序自动编号，**不要手写编号**
- **assert 验收块必须写在 `@@end` 之后**，否则会被挖掉，导致两版不一致

块内不要写控制流（`for` / `if` / `def`），挖空后会变成语法错误。
"""

from __future__ import annotations

import ast
import io
import json
import re
import textwrap
import tokenize
from pathlib import Path

# --------------------------------------------------------------------------- #
# 指令解析
# --------------------------------------------------------------------------- #

TODO_RE = re.compile(r"^\s*#\s*@@todo(?:\(\s*\d+\s*\))?\s+(.*)$")
HINT_RE = re.compile(r"^\s*#\s*@@hint\s+(.*?)\s*$")
END_RE = re.compile(r"^\s*#\s*@@end\s*$")
# 指令行（元标记）：只该出现在 scripts/authoring/*.py 里，不该进 notebook
DIRECTIVE_RE = re.compile(r"^\s*#\s*@@(?:todo|hint|end)\b")


class BuildError(RuntimeError):
    """源文件写法不符合 DSL 约束时抛出。"""


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _assign_target(stmt_text: str) -> str | None:
    """若该语句是赋值语句，返回 `=` 左侧的目标源码；否则返回 None。

    用 `tokenize` 而不是正则，是为了不把**关键字参数**误判成赋值目标：

        df.to_csv(path, index=False)     # 这里的 index= 在括号内，不是赋值

    正则写法会产出 `df.to_csv(path, index = ____` —— 直接语法错误。
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(stmt_text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return None

    depth = 0
    for tok in tokens:
        if tok.type != tokenize.OP:
            continue
        if tok.string in "([{":
            depth += 1
        elif tok.string in ")]}":
            depth -= 1
        elif tok.string == "=" and depth == 0:
            # 只接受「目标写在第一行」的常规写法；跨行赋值不猜，整行占位
            if tok.start[0] != 1:
                return None
            target = stmt_text.split("\n")[0][: tok.start[1]].strip()
            return target or None
    return None


def blank_statement(stmt_text: str, indent: str) -> str:
    """把一条逻辑语句转成挖空占位。"""
    target = _assign_target(stmt_text)
    return f"{indent}{target} = ____" if target else f"{indent}____"


def blank_line(line: str) -> str:
    """把一行代码转成挖空占位。"""
    stripped = line.strip()
    indent = _indent_of(line)
    if not stripped:
        return line
    if stripped.startswith("#"):
        return line
    return blank_statement(line, indent)


def _statement_spans(body: list[str]) -> list[tuple[int, int]] | None:
    """用 AST 求出块内每条**逻辑语句**占用的物理行范围（1-based，两端含）。

    为什么要按逻辑语句而不是物理行挖空：一个跨行的表达式（字典字面量、
    括号换行的链式调用）如果逐行挖空，会产出

        load_fill_group = ____
            ____
        ____

    其中第二行是「孤立缩进」，直接 `IndentationError`。
    按 AST 分组的正确结果是单行 `load_fill_group = ____`。

    返回 None 表示整块无法解析，此时调用方退回逐行挖空。

    **必须先 dedent**：挖空块常常嵌在 `for` / `while` 里（如「循环骨架外置、
    `@@todo` 放循环体内」的写法），整块都是缩进状态，直接 `ast.parse` 会抛
    `IndentationError` 而退回逐行挖空 —— 于是一个**跨行语句**又被拆成
    「孤立缩进」的占位，生成出语法错误的练习版。

    dedent 只影响缩进、不改动行号，所以返回的物理行范围依然有效。
    """
    try:
        tree = ast.parse(textwrap.dedent("\n".join(body)))
    except SyntaxError:
        return None
    return [(node.lineno, node.end_lineno or node.lineno) for node in tree.body]


def blank_block(body: list[str], where: str = "") -> list[str]:
    """把挖空块的正文转成练习版正文（按逻辑语句分组）。"""
    spans = _statement_spans(body)
    if spans is None:
        # 无法解析（少见）：退回逐行挖空
        return [blank_line(line) for line in body]

    span_by_start = {start: end for start, end in spans}
    owned: set[int] = set()
    for start, end in spans:
        owned.update(range(start, end + 1))

    out: list[str] = []
    for lineno, line in enumerate(body, start=1):
        if lineno not in owned:
            out.append(line)          # 注释 / 空行：原样保留
            continue
        end = span_by_start.get(lineno)
        if end is None:
            continue                  # 多行语句的中间行：由首行的占位代表
        statement = "\n".join(body[lineno - 1 : end])
        out.append(blank_statement(statement, _indent_of(line)))
    return out


def strip_directives(text: str) -> str:
    """删掉 DSL 指令行（`@@todo` / `@@hint` / `@@end`）。

    这三行是**给生成器的指令**，只该出现在 `scripts/authoring/*.py` 里。
    讲解版与答案版走这条路径清理（纯行删除，代码与其余注释原样保留）；
    练习版走 `blank_code`，会把指令渲染成 `# TODO(n)：...` / `# 提示：...`。

    历史坑：`build()` 早期直接 dump 原文本，导致全仓 3831 行 `# @@end`
    之类的元标记进了 notebook（2026-09-27 修复）。
    """
    return "\n".join(line for line in text.split("\n") if not DIRECTIVE_RE.match(line))


def _strip_directive_cells(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(kind, strip_directives(text) if kind == "code" else text) for kind, text in cells]


def blank_code(text: str, counter: list[int], where: str = "") -> str:
    """按 DSL 把一段代码的挖空块替换掉。counter 是跨 cell 共享的题号计数器。"""
    lines = text.split("\n")
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        todo = TODO_RE.match(line)
        if not todo:
            if END_RE.match(line):
                raise BuildError(f"{where}: 出现没有 @@todo 与之配对的 @@end")
            if HINT_RE.match(line):
                raise BuildError(f"{where}: @@hint 必须写在 @@todo 之后、@@end 之前")
            out.append(line)
            idx += 1
            continue

        # ---- 收集块 ----
        desc = todo.group(1).strip()
        if not desc:
            raise BuildError(f"{where}: @@todo 必须带描述文字")
        # ---- 扫描「前导注释区」：@@todo 与第一行真正代码之间的部分 ----
        # 这块里可能混着三类行：`@@hint` 提示、描述续行（`# ...`）、空行。
        # 历史坑（2026-09-27 第二次）：早期只认「紧跟在 @@todo 后面、连续排列」
        # 的 `@@hint`，一旦作者把提示写在多行描述**之后**（如 ch08/ch13 风格），
        # 收集循环会在第一行描述续行处 break，`@@hint` 便落进 body，按
        # 「块内注释原样保留」的规则输出成裸 DSL 指令 `# @@hint ...` 给学生。
        head: list[tuple[str, str]] = []
        idx += 1
        while idx < len(lines):
            raw = lines[idx]
            if END_RE.match(raw) or TODO_RE.match(raw):
                break                       # 空块 / 交给下游报「嵌套 @@todo」
            hint_match = HINT_RE.match(raw)
            if hint_match:
                head.append(("hint", hint_match.group(1).strip()))
                idx += 1
                continue
            if raw.strip() == "" or raw.lstrip().startswith("#"):
                head.append(("raw", raw))   # 描述续行 / 空行：原样保留，位置不变
                idx += 1
                continue
            break                           # 第一行真代码 → 前导区结束

        body: list[str] = []
        closed = False
        while idx < len(lines):
            if END_RE.match(lines[idx]):
                closed = True
                idx += 1
                break
            if TODO_RE.match(lines[idx]):
                raise BuildError(f"{where}: @@todo 嵌套了另一个 @@todo")
            body.append(lines[idx])
            idx += 1
        if not closed:
            raise BuildError(f"{where}: @@todo(「{desc}」) 没有对应的 @@end")

        # ---- 校验块内容 ----
        for raw in body:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                # 兜底：`@@hint` 若被写在第一行代码**之后**，就会走「块内注释原样
                # 保留」这条路漏进练习版。这里直接拦下，让作者把它挪到前导区。
                if DIRECTIVE_RE.match(raw):
                    raise BuildError(
                        f"{where}: `@@hint` 必须写在 `@@todo` 与第一行代码之间"
                        f"（不能夹在代码行之后）：`{stripped}`"
                    )
                continue
            if stripped.endswith(":") or re.match(r"^(for|while|if|elif|else|def|class|try|with|async)\b", stripped):
                raise BuildError(
                    f"{where}: 挖空块里不能出现控制流或语句头（`{stripped}`），"
                    "挖空后会变成语法错误。请把控制流写在 @@todo 之外。"
                )
            if stripped.startswith("assert "):
                raise BuildError(
                    f"{where}: assert 不能放进挖空块（会导致 practice/solution 的验收块不一致），"
                    "请把它移到 @@end 之后。"
                )

        # ---- 生成练习版块 ----
        counter[0] += 1
        # TODO / 提示 注释要跟着块的缩进走：挖空块常常嵌在 `for` / `while` / `def`
        # 体内，若注释贴在 0 列，练习版会出现
        #     for t in CASES:
        #     # TODO(1)：...
        #         a, b = ____
        # 这种「注释比代码还靠左」的排版（语法合法，但看起来像坏了）。
        indent = min(
            (_indent_of(raw) for raw in body if raw.strip()),
            key=len,
            default="",
        )
        out.append(f"{indent}# TODO({counter[0]})：{desc}")
        for kind, text in head:
            # 前导区按作者原始顺序回放：提示翻译成 `# 提示：`，其余原样带走。
            out.append(f"{indent}# 提示：{text}" if kind == "hint" else text)
        out.extend(blank_block(body, where=where))

    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 写作 DSL
# --------------------------------------------------------------------------- #


def md(text: str) -> tuple[str, str]:
    return ("markdown", text.strip("\n"))


def code(text: str) -> tuple[str, str]:
    return ("code", text.strip("\n"))


# --------------------------------------------------------------------------- #
# notebook 写出
# --------------------------------------------------------------------------- #

KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO = {
    "name": "python",
    "version": "3.13.12",
    "mimetype": "text/x-python",
    "file_extension": ".py",
    "pygments_lexer": "ipython3",
    "nbconvert_exporter": "python",
    "codemirror_mode": {"name": "ipython", "version": 3},
}


def _split_source(text: str) -> list[str]:
    if text == "":
        return []
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _to_ipynb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for i, (kind, text) in enumerate(cells):
        cell = {
            "cell_type": kind,
            "id": f"cell-{i:03d}",
            "metadata": {},
            "source": _split_source(text),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {"kernelspec": KERNELSPEC, "language_info": LANGUAGE_INFO},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _dump(path: Path, nb: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #


def build(
    out_dir: Path,
    name: str,
    lesson: list[tuple[str, str]],
    exercise: list[tuple[str, str]],
) -> dict:
    """写出三件套，返回统计信息。

    Args:
        out_dir: 章节所在目录（如 coding/01_pandas）
        name: 章节基名（如 ch01_series_dataframe）
        lesson: 讲解 notebook 的 cells
        exercise: 练习/答案共用的 cells（含 @@todo 挖空标记）

    Notes:
        `exercise` 里写的是**答案版**内容；练习版由它自动派生。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    where = name
    counter = [0]
    practice_cells: list[tuple[str, str]] = []
    for kind, text in exercise:
        if kind == "code":
            practice_cells.append((kind, blank_code(text, counter, where=where)))
        else:
            practice_cells.append((kind, text))

    stats = {
        "name": name,
        "lesson_cells": len(lesson),
        "exercise_cells": len(exercise),
        "blanks": counter[0],
    }

    _dump(out_dir / f"{name}.ipynb", _to_ipynb(_strip_directive_cells(lesson)))
    _dump(out_dir / f"{name}_solution.ipynb", _to_ipynb(_strip_directive_cells(exercise)))
    _dump(out_dir / f"{name}_practice.ipynb", _to_ipynb(practice_cells))
    return stats


def report(stats_list: list[dict]) -> None:
    print("=" * 68)
    print(f"{'章节':34s} {'讲解':>6s} {'练习/答案':>10s} {'挖空数':>7s}")
    print("-" * 68)
    for s in stats_list:
        print(
            f"{s['name']:34s} {s['lesson_cells']:>6d} "
            f"{s['exercise_cells']:>10d} {s['blanks']:>7d}"
        )
    print("=" * 68)
    print(f"共 {len(stats_list)} 章，{sum(s['lesson_cells'] for s in stats_list)} 个讲解 cell，"
          f"{sum(s['blanks'] for s in stats_list)} 个挖空")
