#!/usr/bin/env python3
"""校验 AICodeArena 仓库中 notebook 的结构一致性。

校验三件事：
  1. 所有 .ipynb 都是合法 JSON，且具备最基本的 notebook 结构
  2. 每一对 `*_practice.ipynb` / `*_solution.ipynb` 逐 Cell 对应
     - Cell 数量相同
     - Cell 类型序列相同
     - Markdown Cell 内容完全相同
     - Code Cell 中的 assert 验收块完全相同
  3. 每个 Code Cell 都能通过 `ast.parse` 语法检查
     （防的是「挖空把关键字参数当赋值目标」这类生成器 bug：
      `df.to_csv(path, index = ____` 会直接语法错误，而逐行目视很难发现）
  4. 练习版确实存在挖空标记（TODO / ____）

用法：
    python scripts/check_notebooks.py                # 校验整个仓库
    python scripts/check_notebooks.py coding/01_pandas
    python scripts/check_notebooks.py -v             # 打印每个文件的详情

退出码：0 = 全部通过，1 = 存在问题。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = ("coding", "exams")

BLANK_MARKERS = ("____", "TODO(")
SOLUTION_SUFFIX = "_solution.ipynb"
PRACTICE_SUFFIX = "_practice.ipynb"


def load_notebook(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def assert_lines(text: str) -> list[str]:
    """提取代码中的 assert 行（去掉首尾空白，忽略缩进差异）。"""
    return [line.strip() for line in text.splitlines() if line.strip().startswith("assert ")]


def check_structure(path: Path, nb: dict) -> list[str]:
    problems: list[str] = []
    for key in ("cells", "metadata", "nbformat"):
        if key not in nb:
            problems.append(f"缺少字段 `{key}`")
    if nb.get("nbformat") != 4:
        problems.append(f"nbformat = {nb.get('nbformat')}，期望 4")
    cells = nb.get("cells")
    if not isinstance(cells, list):
        problems.append("`cells` 不是列表")
    else:
        for idx, cell in enumerate(cells):
            ctype = cell.get("cell_type")
            if ctype not in ("code", "markdown", "raw"):
                problems.append(f"cell[{idx}] cell_type 非法：{ctype!r}")
            if "source" not in cell:
                problems.append(f"cell[{idx}] 缺少 source")
    return problems


def check_syntax(nb: dict) -> list[str]:
    """逐 Code Cell 做语法检查。挖空占位 `____` 是合法标识符，所以能解析出来。"""
    problems: list[str] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)
        if not source.strip():
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            line = (exc.text or "").rstrip()
            problems.append(
                f"cell[{idx}] 语法错误：{exc.msg}（第 {exc.lineno} 行"
                + (f"：{line.strip()[:60]}" if line.strip() else "")
                + "）"
            )
    return problems


def compare_pair(practice: Path, solution: Path) -> list[str]:
    problems: list[str] = []
    nb_p = load_notebook(practice)
    nb_s = load_notebook(solution)
    cp, cs = nb_p.get("cells", []), nb_s.get("cells", [])

    if len(cp) != len(cs):
        problems.append(
            f"Cell 数量不一致：practice {len(cp)} 个，solution {len(cs)} 个"
        )

    for idx, (a, b) in enumerate(zip(cp, cs)):
        ta, tb = a.get("cell_type"), b.get("cell_type")
        if ta != tb:
            problems.append(f"cell[{idx}] 类型不一致：practice={ta}，solution={tb}")
            continue

        if ta == "markdown":
            sa, sb = cell_source(a).strip(), cell_source(b).strip()
            if sa != sb:
                head = sa.splitlines()[0][:40] if sa else "(空)"
                problems.append(f"cell[{idx}] Markdown 内容不一致（起始：{head}）")
        elif ta == "code":
            la, lb = assert_lines(cell_source(a)), assert_lines(cell_source(b))
            if la != lb:
                problems.append(
                    f"cell[{idx}] assert 验收块不一致"
                    f"（practice {len(la)} 条，solution {len(lb)} 条）"
                )

    practice_text = "\n".join(cell_source(c) for c in cp)
    if not any(marker in practice_text for marker in BLANK_MARKERS):
        problems.append("practice 中找不到任何挖空标记（____ 或 TODO(）")

    blanks = practice_text.count("____")
    if blanks == 0:
        problems.append("practice 中没有任何 ____ 占位符")

    return problems


# 这些目录下的 *_solution.ipynb 不参与「挖空练习 / 答案」配对校验：
#   · 99_debug  改错卷 —— 给的是能跑的错代码，不是挖空题
#   · exams     模拟卷 —— 空白卷从零作答，正文里没有 ____ 占位符
# 它们仍然参与 ① 结构校验与 ③ 逐 cell 语法校验（这两个才是通用约束）。
SKIP_PAIR_DIRS = {"99_debug", "exams"}


def iter_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for sol in sorted(root.rglob(f"*{SOLUTION_SUFFIX}")):
        if ".ipynb_checkpoints" in sol.parts:
            continue
        if any(part in SKIP_PAIR_DIRS for part in sol.parts):
            continue
        pra = sol.with_name(sol.name.replace(SOLUTION_SUFFIX, PRACTICE_SUFFIX))
        pairs.append((pra, sol))
    return pairs


def main(argv: list[str]) -> int:
    verbose = "-v" in argv or "--verbose" in argv
    args = [a for a in argv if not a.startswith("-")]
    roots = [Path(a) for a in args] if args else [REPO_ROOT / r for r in DEFAULT_ROOTS]

    all_notebooks: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"  [跳过] 目录不存在：{root}")
            continue
        all_notebooks.extend(
            p for p in sorted(root.rglob("*.ipynb")) if ".ipynb_checkpoints" not in p.parts
        )

    errors = 0
    print(f"扫描到 {len(all_notebooks)} 个 notebook\n")

    print("① 结构校验")
    for path in all_notebooks:
        rel = path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path
        try:
            nb = load_notebook(path)
        except json.JSONDecodeError as exc:
            print(f"  [失败] {rel}：JSON 解析错误 -> {exc}")
            errors += 1
            continue
        problems = check_structure(path, nb)
        if problems:
            errors += len(problems)
            print(f"  [失败] {rel}")
            for p in problems:
                print(f"         - {p}")
        elif verbose:
            print(f"  [通过] {rel}（{len(nb.get('cells', []))} cells）")
    print()

    print("② 练习 / 答案 配对校验")
    pairs_found = 0
    for root in roots:
        if not root.exists():
            continue
        for practice, solution in iter_pairs(root):
            pairs_found += 1
            rel = solution.relative_to(REPO_ROOT) if REPO_ROOT in solution.parents else solution
            if not practice.exists():
                print(f"  [失败] {rel}：找不到对应的 {PRACTICE_SUFFIX} 文件")
                errors += 1
                continue
            problems = compare_pair(practice, solution)
            if problems:
                errors += len(problems)
                print(f"  [失败] {rel}")
                for p in problems:
                    print(f"         - {p}")
            elif verbose:
                print(f"  [通过] {rel}")
    if pairs_found == 0:
        print("  （未找到任何 _solution.ipynb，先把章节写出来再跑）")
    print()

    print("③ 语法校验（逐 Code Cell）")
    syntax_errors = 0
    for path in all_notebooks:
        rel = path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path
        try:
            nb = load_notebook(path)
        except json.JSONDecodeError:
            continue          # ① 已经报过
        problems = check_syntax(nb)
        if problems:
            syntax_errors += len(problems)
            errors += len(problems)
            print(f"  [失败] {rel}")
            for p in problems:
                print(f"         - {p}")
        elif verbose:
            print(f"  [通过] {rel}")
    if syntax_errors == 0:
        print("  （全部 Code Cell 均可解析）")
    print()

    if errors:
        print(f"❌ 共发现 {errors} 个问题")
        return 1
    print(f"✅ 全部通过（notebook {len(all_notebooks)} 个，配对 {pairs_found} 组）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
