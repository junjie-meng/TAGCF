#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_single_domain_attr_edges_v4c.py

从 lightgcn 的 v4c 风格 preference.txt 为单个 domain 生成 RLMRec 用的 attr_edges.pkl。
逻辑与 data/transfer.py 一致，但支持命令行参数。

示例：
    python scripts/build_single_domain_attr_edges_v4c.py \\
        --dataset yelp \\
        --lightgcn_root /root/lightgcn/data \\
        --output_dir /root/RLMRec/data
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Build single-domain attr_edges.pkl from lightgcn v4c preference.txt"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Lightgcn dataset name, e.g. amazon-book, yelp, Office",
    )
    parser.add_argument(
        "--lightgcn_root",
        type=str,
        default="/root/lightgcn/data",
        help="Lightgcn data root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/RLMRec/data",
        help="Output directory for attr_edges.pkl",
    )
    args = parser.parse_args()

    data_dir = Path(args.lightgcn_root) / args.dataset
    attr_path = data_dir / "preference.txt"
    id2user_path = data_dir / "id2user.json"
    id2item_path = data_dir / "id2item.json"

    if not attr_path.exists():
        raise FileNotFoundError(f"preference.txt not found: {attr_path}")

    attr_edges = []
    attr_counter = Counter()
    with open(attr_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            item, attr = int(parts[0]), int(parts[1])
            attr_edges.append((item, attr))
            attr_counter[attr] += 1

    print(f"[INFO] {args.dataset}: {len(attr_edges)} edges loaded")

    item_ids, attr_ids = zip(*attr_edges)
    item_ids = np.array(item_ids)
    attr_ids = np.array(attr_ids)

    with open(id2item_path, "r", encoding="utf-8") as f:
        id2item = json.load(f)
    with open(id2user_path, "r", encoding="utf-8") as f:
        id2user = json.load(f)

    n_users = len(id2user)
    n_items = len(id2item)
    user_item_num = n_users + n_items
    n_attrs = max(attr_ids) + 1

    print(f"[INFO] n_users={n_users}, n_items={n_items}, n_entities={user_item_num}, n_attrs={n_attrs}")

    attr_edges_mat = coo_matrix(
        (np.ones(len(item_ids), dtype=np.float32), (item_ids, attr_ids)),
        shape=(user_item_num, n_attrs),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / args.dataset / "attr_edges.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(attr_edges_mat, f)

    print(f"[INFO] Saved attr_edges.pkl -> {output_path}")
    print(f"[INFO] shape={attr_edges_mat.shape}, nnz={attr_edges_mat.nnz}")


if __name__ == "__main__":
    main()
