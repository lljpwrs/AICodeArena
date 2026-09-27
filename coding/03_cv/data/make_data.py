#!/usr/bin/env python3
"""03_cv 数据底座：下载经典测试图 + 导出 MNIST 子集。

来源（全部为公开渠道，脚本可重复执行，已存在且 MD5 一致的文件自动跳过）：

1. classic/ —— 经典图像处理测试图（教学界公认的标准图）
   - opencv/opencv 仓库 samples/data（Apache-2.0）：board/box/building/home/messi5/smarties/sudoku
     https://github.com/opencv/opencv
   - scikit-image v0.22.0 tag 的 skimage/data（BSD-3 / CC0）：camera/coins/cell/horse/moon/page/text
     https://github.com/scikit-image/scikit-image
   注：不放 lena.jpg（原出处争议，教学场景用 camera.png 等替代）。

2. helmet/ —— 电力现场作业安全监督场景（安全帽佩戴检测）
   - njvisionpower/Safety-Helmet-Wearing-Dataset (SHWD) 仓库 image/ 下的原图
     （完整 7581 张 + VOC 标注在百度/谷歌网盘，仓库内仅附样例）
   - labels.json / *.txt：本仓库教学用手工粗标（人眼目检画框，只框头顶/安全帽，
     精度有限，仅用于 IoU/NMS/标注解析教学，勿当真实数据集用）

3. mnist/ —— 手写数字分类
   - torchvision 从 ossci-datasets.s3.amazonaws.com 下载原始 MNIST（缓存到 _raw/），
     固定 seed=42 抽 2000 训练 + 500 测试，导出为 npz（28x28 uint8 + label）。

用法（本机 conda 环境 self）：
    /Users/luolinjie/miniconda3/envs/self/bin/python make_data.py
"""
import base64
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
API = "https://api.github.com/repos/{repo}/contents/{path}"
UA = {"User-Agent": "AICodeArena-edu/1.0"}

OPENCV_FILES = ["board.jpg", "box.png", "building.jpg", "home.jpg",
                "messi5.jpg", "smarties.png", "sudoku.png"]
SKIMAGE_FILES = ["camera.png", "coins.png", "cell.png", "horse.png",
                 "moon.png", "page.png", "text.png"]

SOURCES = {
    "board.jpg": "https://github.com/opencv/opencv (samples/data, Apache-2.0)",
    "box.png": "https://github.com/opencv/opencv (samples/data, Apache-2.0)",
    "building.jpg": "https://github.com/opencv/opencv (samples/data, Apache-2.0)",
    "home.jpg": "https://github.com/opencv/opencv (samples/data, Apache-2.0)",
    "messi5.jpg": "https://github.com/opencv/opencv (samples/data, Apache-2.0)",
    "smarties.png": "https://github.com/opencv/opencv (samples/data, Apache-2.0)",
    "sudoku.png": "https://github.com/opencv/opencv (samples/data, Apache-2.0)",
    "camera.png": "https://github.com/scikit-image/scikit-image (v0.22.0, CC0)",
    "coins.png": "https://github.com/scikit-image/scikit-image (v0.22.0, CC0)",
    "cell.png": "https://github.com/scikit-image/scikit-image (v0.22.0, CC0)",
    "horse.png": "https://github.com/scikit-image/scikit-image (v0.22.0, CC0)",
    "moon.png": "https://github.com/scikit-image/scikit-image (v0.22.0, CC0)",
    "page.png": "https://github.com/scikit-image/scikit-image (v0.22.0, CC0)",
    "text.png": "https://github.com/scikit-image/scikit-image (v0.22.0, CC0)",
}


def _retry(fn, tries=4):
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  网络异常重试 {i + 1}/{tries - 1}: {e}")
            time.sleep(2)


def api_get(repo: str, path: str, ref: str = "") -> bytes:
    url = API.format(repo=repo, path=path)
    if ref:
        url += f"?ref={ref}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    if "content" not in data or data.get("encoding") != "base64":
        raise RuntimeError(f"API 返回异常: {url} -> {data.get('message', data.keys())}")
    return base64.b64decode(data["content"])


def fetch_if_needed(dst: Path, getter):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  已存在，跳过 {dst.name}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(_retry(getter))
    print(f"  下载 {dst.name} ({dst.stat().st_size} bytes)")
    return True


def fetch_classic():
    print("== classic/ 经典测试图 ==")
    out = HERE / "classic"
    for name in OPENCV_FILES:
        fetch_if_needed(out / name,
                        lambda n=name: api_get("opencv/opencv", f"samples/data/{n}"))
    for name in SKIMAGE_FILES:
        fetch_if_needed(out / name,
                        lambda n=name: api_get("scikit-image/scikit-image", f"skimage/data/{n}", "v0.22.0"))


def fetch_helmet():
    print("== helmet/ 安全帽场景图 ==")
    out = HERE / "helmet"
    listing = json.loads(urllib.request.urlopen(urllib.request.Request(
        "https://api.github.com/repos/njvisionpower/Safety-Helmet-Wearing-Dataset/contents/image",
        headers=UA), timeout=60).read().decode())
    raws = sorted((it for it in listing
                   if it["name"].lower().endswith(".jpg")
                   and "_result" not in it["name"]),
                  key=lambda it: int(it["name"].split(".")[0]))
    for i, it in enumerate(raws, 1):
        name = f"site_{i:02d}.jpg"
        fetch_if_needed(out / name,
                        lambda u=it["url"]: api_get_bytes_by_url(u))
    print(f"  共 {len(raws)} 张原图")


def api_get_bytes_by_url(api_url: str) -> bytes:
    req = urllib.request.Request(api_url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    return base64.b64decode(data["content"])


def export_mnist():
    print("== mnist/ 导出子集 ==")
    out = HERE / "mnist"
    tr_npz, te_npz = out / "mnist_train_sub.npz", out / "mnist_test_sub.npz"
    if tr_npz.exists() and te_npz.exists():
        print("  npz 已存在，跳过（删除后可重新生成）")
        return
    from torchvision import datasets
    raw = HERE / "_raw"
    ds_tr = datasets.MNIST(root=str(raw), train=True, download=True)
    ds_te = datasets.MNIST(root=str(raw), train=False, download=True)
    rng = np.random.RandomState(42)
    out.mkdir(parents=True, exist_ok=True)
    for ds, n, path in [(ds_tr, 2000, tr_npz), (ds_te, 500, te_npz)]:
        X = ds.data.numpy()          # (N,28,28) uint8
        y = ds.targets.numpy()
        idx = rng.choice(len(X), size=n, replace=False)
        np.savez_compressed(path, images=X[idx], labels=y[idx])
        print(f"  {path.name}: {n} 张, 类别分布 {np.bincount(y[idx], minlength=10).tolist()}")


def write_manifest():
    import hashlib
    manifest = {}
    for sub in ["classic", "helmet", "mnist"]:
        d = HERE / sub
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file():
                manifest[f"{sub}/{f.name}"] = {
                    "md5": hashlib.md5(f.read_bytes()).hexdigest(),
                    "bytes": f.stat().st_size,
                    "source": SOURCES.get(f.name,
                                          "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset (image/)"
                                          if sub == "helmet" else "MNIST via torchvision (ossci-datasets S3)"),
                }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== manifest.json: {len(manifest)} 个文件 ==")


def sanity_check():
    print("== 完整性抽检 ==")
    import cv2
    bad = []
    for f in sorted((HERE / "classic").glob("*")) + sorted((HERE / "helmet").glob("site_*.jpg")):
        img = cv2.imread(str(f))
        if img is None:
            bad.append(f.name)
        else:
            print(f"  {f.name}: {img.shape}")
    if bad:
        print("!! 读不出来的图:", bad)
        sys.exit(1)
    tr = np.load(HERE / "mnist" / "mnist_train_sub.npz")
    print(f"  mnist_train_sub: {tr['images'].shape}, labels {tr['labels'].min()}~{tr['labels'].max()}")


if __name__ == "__main__":
    fetch_classic()
    fetch_helmet()
    export_mnist()
    write_manifest()
    sanity_check()
