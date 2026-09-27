#!/usr/bin/env python3
"""生成 helmet/ 的教学标注：labels.json（像素 xyxy）+ YOLO txt（归一化 cxcywh）。

类别：0 = head（未戴安全帽的头部），1 = helmet（已戴安全帽的头部）。
注意：这是**教学用手工粗标**（人眼目检画框，误差约 ±10px），
只覆盖 4 张样例图，仅用于 ch09 的 IoU / NMS / 标注解析教学，
不能当真实训练集使用。完整 7581 张 + VOC 标注见 SHWD 官方网盘。
"""
import json
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent / "helmet"

# 类名顺序即 YOLO class id 顺序
CLASSES = ["head", "helmet"]

# 教学粗标：{图片: [(class_id, x1, y1, x2, y2), ...]}，像素坐标 xyxy（网格目检校准）
LABELS = {
    # 站会场景：白帽为主 + 一名黄帽，最干净的演示图
    "site_05.jpg": [
        (1, 0, 198, 38, 238),
        (1, 58, 232, 104, 270),
        (1, 178, 248, 228, 288),
        (1, 268, 207, 300, 237),
        (1, 318, 232, 372, 278),
        (1, 433, 177, 467, 202),   # 黄帽（蓝衣）
        (1, 455, 227, 505, 268),
        (1, 583, 222, 632, 260),
        (1, 680, 200, 726, 240),
        (1, 725, 195, 770, 235),
    ],
    # 车间小图：黄帽为主
    "site_03.jpg": [
        (1, 47, 186, 66, 199),
        (1, 107, 179, 126, 194),
        (1, 254, 176, 271, 191),
        (1, 329, 183, 346, 197),
        (1, 379, 169, 396, 183),
        (1, 395, 168, 415, 185),
    ],
    # 工地：红/蓝/白帽混合，含未戴帽人员
    "site_07.jpg": [
        (1, 85, 87, 125, 118),
        (1, 178, 107, 214, 140),
        (1, 248, 78, 292, 115),
        (1, 345, 88, 372, 115),
        (1, 412, 95, 448, 122),
        (0, 492, 92, 528, 125),    # 未戴帽（灰发）
    ],
    # 街头人流：负样本，只有 head
    "site_01.jpg": [
        (0, 208, 307, 255, 350),
        (0, 207, 397, 258, 440),
        (0, 63, 247, 115, 292),
        (0, 474, 218, 520, 262),
        (0, 963, 188, 1012, 232),
        (0, 795, 412, 848, 458),
        (0, 428, 597, 492, 648),
    ],
}


def main():
    boxes_by_img = {}
    for name, objs in LABELS.items():
        img = cv2.imread(str(HERE / name))
        if img is None:
            raise FileNotFoundError(name)
        h, w = img.shape[:2]
        recs = []
        for cid, x1, y1, x2, y2 in objs:
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, f"{name} 框越界: {(x1, y1, x2, y2)}"
            recs.append({
                "class_id": cid,
                "class_name": CLASSES[cid],
                "bbox_xyxy": [x1, y1, x2, y2],
            })
        boxes_by_img[name] = {"width": w, "height": h, "objects": recs}
        # YOLO txt：class cx cy w h（归一化，保留 6 位）
        lines = []
        for cid, x1, y1, x2, y2 in objs:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (HERE / name.replace(".jpg", ".txt")).write_text("\n".join(lines) + "\n")
        print(f"{name}: {len(recs)} 框 ({sum(1 for r in recs if r['class_id']==1)} helmet / "
              f"{sum(1 for r in recs if r['class_id']==0)} head)")

    (HERE / "labels.json").write_text(
        json.dumps({"classes": CLASSES, "images": boxes_by_img},
                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("labels.json 写入完成（教学粗标，误差 ±10px）")


if __name__ == "__main__":
    main()
