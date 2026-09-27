"""改错题（debug 型练习）的共享生成器。

与「讲解 / 练习 / 答案」三件套的区别：

| | 练习卷 | 改错卷 |
|---|---|---|
| 给什么 | 题面 + 骨架 + `____` | 题面 + **能跑的错代码** |
| 考什么 | 会不会写 | 能不能**看出**哪里错 |
| 失败形态 | `NameError` | **不报错，但结果是错的** |

生成两件套：
    <name>.ipynb           题目卷：题面 + 错误代码 + 空的修复 cell
    <name>_solution.ipynb  答案卷：题面 + 修复代码 + assert 验收

> 错误代码只出现在**题目卷**里。答案卷不放它 —— 否则答案卷自己就跑不通了
> （错误代码是真的会抛异常），也违背「答案卷必须完整可运行」的约定。

> 每道题的「错误代码」都经过实跑确认 —— 要么报错、要么静默出错，
> **不许凭想象编一个 bug**。写完新题请先单独跑一遍确认现象。
"""

from __future__ import annotations

import json
from pathlib import Path

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.15"},
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def _dump(path: Path, cells: list[dict]) -> None:
    path.write_text(
        json.dumps({"cells": cells, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5},
                   ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def build_debug(
    out_dir: Path,
    name: str,
    intro: list[str],
    tasks: list[tuple[str, str, str, str]],
) -> dict:
    """写出改错题两件套。

    Args:
        out_dir: 章节目录（如 coding/99_debug）
        name: 基名（如 dbg01_pandas_traps）
        intro: 两卷共用的开头 Markdown
        tasks: [(标题, 题面, 错误代码, 修复代码)]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quiz: list[dict] = [md(t) for t in intro]
    solution: list[dict] = [md(t) for t in intro]

    for index, (title, statement, buggy, fixed) in enumerate(tasks, start=1):
        head = f"### 题 {index} · {title}\n\n{statement}"
        hint = (
            "\n# 修复后请在这里重写（可以直接改上面那个 cell，也可以写在这里）\n"
        )
        quiz += [md(head), code_cell(buggy), code_cell(hint)]
        solution += [md(head), code_cell(fixed)]

    _dump(out_dir / f"{name}.ipynb", quiz)
    _dump(out_dir / f"{name}_solution.ipynb", solution)

    return {"name": name, "tasks": len(tasks), "quiz_cells": len(quiz), "sol_cells": len(solution)}
