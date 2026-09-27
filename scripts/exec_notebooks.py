#!/usr/bin/env python3
"""按真实内核执行 notebook，验证「答案能跑通 / 练习必然卡住」。

与 `check_notebooks.py` 的分工：

- `check_notebooks.py`  —— 静态校验（JSON 结构、Cell 对应、assert 一致）
- `exec_notebooks.py`   —— 动态校验（真的起一个 ipykernel 跑一遍）

用法：

    python scripts/exec_notebooks.py <路径...>                 # 只跑 solution，要求全部成功
    python scripts/exec_notebooks.py <路径...> --expect-fail   # 跑 practice，要求全部失败
    python scripts/exec_notebooks.py <路径...> --timeout 300

路径可以是目录（递归找）或单个 .ipynb。
退出码：0 = 全部符合预期，1 = 有不符合的。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent


def collect(targets: list[str], suffix: str) -> list[Path]:
    found: list[Path] = []
    for raw in targets:
        p = Path(raw)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_dir():
            found.extend(sorted(x for x in p.rglob(f"*{suffix}") if ".ipynb_checkpoints" not in x.parts))
        elif p.exists():
            found.append(p)
        else:
            print(f"  [跳过] 路径不存在：{raw}")
    return found


def run_one(path: Path, timeout: int) -> tuple[bool, str]:
    """执行单个 notebook，返回 (是否成功, 失败摘要)。"""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        text = str(exc)
        # 提取最关键的一行报错
        key = ""
        for line in reversed(text.splitlines()):
            if line.strip().startswith(("NameError", "SyntaxError", "ValueError", "KeyError",
                                        "TypeError", "AssertionError", "IndexError",
                                        "AttributeError", "ImportError", "ModuleNotFoundError")):
                key = line.strip()
                break
        if not key:
            tail = [l for l in text.splitlines() if l.strip()]
            key = tail[-1].strip() if tail else "未知错误"
        return False, key[:160]
    except Exception as exc:  # noqa: BLE001 - 启动内核失败等
        return False, f"{type(exc).__name__}: {exc}"[:160]
    return True, ""


def main(argv: list[str]) -> int:
    expect_fail = "--expect-fail" in argv
    timeout = 300
    if "--timeout" in argv:
        timeout = int(argv[argv.index("--timeout") + 1])

    targets = [a for a in argv if not a.startswith("-") and not a.isdigit()]
    if not targets:
        print(__doc__)
        return 1

    suffix = "_practice.ipynb" if expect_fail else "_solution.ipynb"
    files = collect(targets, suffix)
    if not files:
        print(f"没找到任何 {suffix} 文件")
        return 1

    print("=" * 72)
    print(f"执行 {len(files)} 个 {suffix}"
          f"（预期：{'失败（挖空处应报错）' if expect_fail else '成功'}）")
    print("=" * 72)

    bad = 0
    for path in files:
        rel = path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path
        t0 = time.time()
        ok, msg = run_one(path, timeout)
        dt = time.time() - t0
        if ok == expect_fail:
            flag = "✗ 不符合预期" if expect_fail else "✗ 执行失败"
            bad += 1
            print(f"  [{flag}] {rel}  ({dt:.1f}s)")
            print(f"      {msg}")
        else:
            note = "" if ok else f"  卡在：{msg}"
            tag = "预期失败 ✓" if expect_fail else "通过 ✓"
            print(f"  [{tag}] {rel}  ({dt:.1f}s){note}")

    print("=" * 72)
    if bad:
        print(f"❌ {bad}/{len(files)} 个不符合预期")
        return 1
    print(f"✅ {len(files)} 个全部符合预期")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
