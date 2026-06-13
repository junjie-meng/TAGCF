#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cross_domain_attr_edges_v4c.py

从 lightgcn 的 v4c 风格 preference.txt 构建 RLMRec 跨域 attr_edges.pkl。

逻辑：
    1. 读取每个域的 lightgcn preference.txt（entity_id attr_id）
    2. 读取 lightgcn 的 id2user.json / id2item.json 得到该域的 n_users / n_items
    3. 汇总所有域的 attribute，构建跨域共享 attribute 词表
    4. 为每个域生成 attr_edges.pkl（scipy.sparse.coo_matrix）
       shape = (n_users + n_items, n_global_attrs)

示例：
    python scripts/build_cross_domain_attr_edges_v4c.py \\
        --lightgcn_root /root/lightgcn/data \\
        --output_dir /root/RLMRec/data/cross_domain_v4c \\
        --domains amazon-book yelp Office \\
        --domain_map '{"amazon-book":"amazon","yelp":"yelp","Office":"Office"}'
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix


def load_id_maps(lightgcn_root, dataset):
    """读取 lightgcn 的 id2user / id2item，返回 n_users, n_items。"""
    root = Path(lightgcn_root) / dataset
    with open(root / "id2user.json", "r", encoding="utf-8") as f:
        id2user = json.load(f)
    with open(root / "id2item.json", "r", encoding="utf-8") as f:
        id2item = json.load(f)
    return len(id2user), len(id2item)


def load_preference_edges(pref_path):
    """读取 preference.txt，返回 [(entity_id, attr_id), ...]。"""
    edges = []
    with open(pref_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            entity_id, attr_id = int(parts[0]), int(parts[1])
            edges.append((entity_id, attr_id))
    return edges


def main():
    parser = argparse.ArgumentParser(
        description="Build cross-domain attr_edges.pkl from lightgcn v4c preference.txt"
    )
    parser.add_argument(
        "--lightgcn_root",
        type=str,
        default="/root/lightgcn/data",
        help="lightgcn data root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/RLMRec/data/cross_domain_v4c",
        help="Output directory for attr_edges.pkl and shared vocab",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["amazon-book", "yelp", "Office"],
        help="Lightgcn domain names to process",
    )
    parser.add_argument(
        "--domain_map",
        type=str,
        default='{"amazon-book":"amazon","yelp":"yelp","Office":"Office"}',
        help='JSON dict mapping lightgcn domain name -> RLMRec domain name',
    )
    args = parser.parse_args()

    domain_map = json.loads(args.domain_map)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 第一遍：收集每个域的属性边 ----------
    domain_edges = {}
    for lg_ds in args.domains:
        pref_path = Path(args.lightgcn_root) / lg_ds / "preference.txt"
        if not pref_path.exists():
            print(f"[WARN] {pref_path} not found, skip {lg_ds}")
            continue

        edges = load_preference_edges(pref_path)
        domain_edges[lg_ds] = edges
        unique_attrs = len(set(a for _, a in edges))
        print(f"[INFO] {lg_ds}: {len(edges)} edges, {unique_attrs} unique attrs")

    if not domain_edges:
        print("[ERROR] No valid preference.txt found")
        return

    # ---------- 构建跨域共享 attribute 词表 ----------
    all_attrs = sorted(set(attr for edges in domain_edges.values() for _, attr in edges))
    global_attr2id = {attr: idx for idx, attr in enumerate(all_attrs)}
    print(f"[INFO] Global shared attributes: {len(all_attrs)}")

    vocab_path = out_dir / "shared_attr2id_v4c.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(global_attr2id, f, indent=2)
    print(f"[INFO] Saved shared vocab: {vocab_path}")

    # 统计频次
    attr_counter = Counter()
    for edges in domain_edges.values():
        for _, attr in edges:
            attr_counter[attr] += 1
    stats_path = out_dir / "shared_attr_counter_v4c.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dict(attr_counter.most_common()), f, indent=2)
    print(f"[INFO] Saved attr frequency stats: {stats_path}")

    # ---------- 第二遍：为每个域生成 attr_edges.pkl ----------
    print("\n[INFO] Building attr_edges.pkl ...")
    for lg_ds in args.domains:
        if lg_ds not in domain_edges:
            continue

        rlm_ds = domain_map.get(lg_ds, lg_ds)
        n_users, n_items = load_id_maps(args.lightgcn_root, lg_ds)
        edges = domain_edges[lg_ds]

        rows = []
        cols = []
        for entity_id, attr in edges:
            rows.append(entity_id)
            cols.append(global_attr2id[attr])

        attr_edges = coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_users + n_items, len(all_attrs)),
        )

        ds_out = out_dir / rlm_ds
        ds_out.mkdir(parents=True, exist_ok=True)
        out_path = ds_out / "attr_edges.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(attr_edges, f)

        print(
            f"[INFO] {lg_ds} -> {rlm_ds}: "
            f"users={n_users}, items={n_items}, "
            f"shape={attr_edges.shape}, nnz={attr_edges.nnz} -> {out_path}"
        )

    print("\n[INFO] Done")


if __name__ == "__main__":
    main()
