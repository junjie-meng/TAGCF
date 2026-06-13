import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

"""\n分析 amazon_layers.txt 日志文件：\n1. 识别每个运行的 model / dataset / num_layers (来自 Command 行)\n2. 读取对应的 Best Epoch 最终测试指标 (recall / ndcg arrays)\n3. 按 layer_num 汇总并作图 (默认只针对 dataset=amazon)\n4. 输出 CSV 与图片。\n\n日志示例片段：\nCommand: python encoder/train_encoder.py --model lightgcn --dataset amazon --cuda 0 --emb_size 64 --num_layers 2\n...\nBest Epoch 153. Final test result: {'recall': array([0.0628, 0.0994, 0.1489]), 'ndcg': array([0.0638, 0.0763, 0.0926])}.\n"""

# 默认日志路径（可被命令行覆盖）
LOG_PATH = Path(__file__).parent / 'amazon_layers.txt'
OUT_DIR = Path(__file__).parent

# Regex patterns
# 解析命令行（更通用，允许可变参数顺序）
COMMAND_RE = re.compile(
    r'^Command:\s+python\s+[^\n]*?train_encoder\.py(?P<args>.*)$'
)
# 匹配单个参数 --key value
ARG_RE = re.compile(r'--(\w+)\s+([^\s]+)')

# 解析 Best Epoch 最终结果
FINAL_RESULT_RE = re.compile(r"Best Epoch (\d+)\. Final test result: (\{.*?\})\.")

# 可选：从配置行里兜底读取 layer_num ( 'layer_num': 3 )
CONFIG_LAYER_RE = re.compile(r"'layer_num':\s*(\d+)")


def _extract_args(arg_str: str) -> Dict[str, str]:
    return {k: v for k, v in ARG_RE.findall(arg_str)}


def parse_log(path: Path, target_dataset: Optional[str] = 'amazon') -> pd.DataFrame:
    current_model: Optional[str] = None
    current_layer: Optional[int] = None
    current_dataset: Optional[str] = None
    # 标记是否已经捕获到一次命令，避免跨 run 污染
    have_command = False
    records: List[Dict] = []

    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()

            # 1. 捕获 Command 行
            m_cmd = COMMAND_RE.match(line)
            if m_cmd:
                args = _extract_args(m_cmd.group('args'))
                current_model = args.get('model')
                current_dataset = args.get('dataset')
                # num_layers 参数有时可能没有（看配置），后面可以兜底
                if 'num_layers' in args:
                    try:
                        current_layer = int(args['num_layers'])
                    except ValueError:
                        current_layer = None
                else:
                    current_layer = None
                have_command = True
                continue

            # 2. 如果还没有拿到 layer，并且当前行里含有配置中的 'layer_num': 可兜底
            if have_command and current_layer is None:
                m_layer = CONFIG_LAYER_RE.search(line)
                if m_layer:
                    current_layer = int(m_layer.group(1))
                    continue

            # 3. 捕获 Best Epoch 行
            m_final = FINAL_RESULT_RE.search(line)
            if m_final and have_command and current_model is not None:
                # dataset 过滤（若指定）
                if target_dataset and current_dataset != target_dataset:
                    # 重置状态跳过
                    current_model = None
                    current_dataset = None
                    current_layer = None
                    have_command = False
                    continue

                epoch = int(m_final.group(1))
                payload = m_final.group(2)
                recall_match = re.search(r"'recall': array\(\[([^\]]+)\]\)", payload)
                ndcg_match = re.search(r"'ndcg': array\(\[([^\]]+)\]\)", payload)
                if not (recall_match and ndcg_match and current_layer is not None):
                    # 状态重置，继续后续 run
                    current_model = None
                    current_dataset = None
                    current_layer = None
                    have_command = False
                    continue

                def to_floats(s: str):
                    return [float(x.strip()) for x in s.split(',') if x.strip()]

                recall_vals = to_floats(recall_match.group(1))
                ndcg_vals = to_floats(ndcg_match.group(1))

                if len(recall_vals) >= 3 and len(ndcg_vals) >= 3:
                    records.append({
                        'dataset': current_dataset,
                        'model': current_model,
                        'layer_num': current_layer,
                        'best_epoch': epoch,
                        'recall@5': recall_vals[0],
                        'recall@10': recall_vals[1],
                        'recall@20': recall_vals[2],
                        'ndcg@5': ndcg_vals[0],
                        'ndcg@10': ndcg_vals[1],
                        'ndcg@20': ndcg_vals[2],
                    })
                # 重置状态，准备下一组
                current_model = None
                current_dataset = None
                current_layer = None
                have_command = False

    df = pd.DataFrame(records)
    if df.empty:
        return df
    # 去重：同一 model+layer 可能多次（只保留指标最好的 recall@20）
    df.sort_values(['model', 'layer_num', 'recall@20'], ascending=[True, True, False], inplace=True)
    df = df.drop_duplicates(subset=['model', 'layer_num'], keep='first')
    return df.sort_values(['model', 'layer_num'])


def plot_metric(df: pd.DataFrame, metric: str, prefix: str = ''):
    plt.figure(figsize=(6, 4))
    for model, g in df.groupby('model'):
        plt.plot(g['layer_num'], g[metric], marker='o', label=model)
    plt.xlabel('Layer Number')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Layer Number')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    name_prefix = f'{prefix}_' if prefix else ''
    out_file = OUT_DIR / f'{name_prefix}{metric}_vs_layer.png'
    plt.savefig(out_file, dpi=150)
    print(f'Saved {out_file}')


def plot_improvement(df: pd.DataFrame, metric_base: str, prefix: str = ''):
    """绘制指定基础指标的绝对与相对提升曲线 (相对提升为比值，而不是百分号)。"""
    abs_col = f'{metric_base}_improve_abs'
    pct_col = f'{metric_base}_improve_pct'
    if abs_col not in df.columns or pct_col not in df.columns:
        print(f'Skip improvement plotting: columns {abs_col} / {pct_col} missing.')
        return

    # 绝对提升
    plt.figure(figsize=(6,4))
    for model, g in df.groupby('model'):
        plt.plot(g['layer_num'], g[abs_col], marker='o', label=model)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.xlabel('Layer Number')
    plt.ylabel(f'{metric_base} Absolute Δ')
    plt.title(f'{metric_base} Absolute Improvement vs Layer Number')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    name_prefix = f'{prefix}_' if prefix else ''
    out_file_abs = OUT_DIR / f'{name_prefix}{metric_base}_improve_abs_vs_layer.png'
    plt.savefig(out_file_abs, dpi=150)
    print(f'Saved {out_file_abs}')

    # 相对提升
    plt.figure(figsize=(6,4))
    for model, g in df.groupby('model'):
        # 转换为百分比显示更直观，但值保留原列（fraction）
        plt.plot(g['layer_num'], g[pct_col] * 100.0, marker='o', label=model)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.xlabel('Layer Number')
    plt.ylabel(f'{metric_base} Relative Δ (%)')
    plt.title(f'{metric_base} Relative Improvement vs Layer Number')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_file_pct = OUT_DIR / f'{name_prefix}{metric_base}_improve_pct_vs_layer.png'
    plt.savefig(out_file_pct, dpi=150)
    print(f'Saved {out_file_pct}')


def plot_improvement_combined(df: pd.DataFrame, metric_base: str, prefix: str = ''):
    """在同一张图中绘制绝对提升 (左 Y) 与 相对百分比提升 (右 Y)。"""
    abs_col = f'{metric_base}_improve_abs'
    pct_col = f'{metric_base}_improve_pct'
    if abs_col not in df.columns or pct_col not in df.columns:
        print(f'Skip combined improvement plotting: columns {abs_col} / {pct_col} missing.')
        return

    name_prefix = f'{prefix}_' if prefix else ''
    fig, ax_abs = plt.subplots(figsize=(7, 4))
    ax_pct = ax_abs.twinx()

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    for idx, (model, g) in enumerate(df.groupby('model')):
        color = colors[idx % len(colors)] if colors else None
        ax_abs.plot(g['layer_num'], g[abs_col], marker='o', label=f'{model} abs', color=color)
        ax_pct.plot(g['layer_num'], g[pct_col] * 100.0, marker='x', linestyle='--', label=f'{model} rel%', color=color)

    ax_abs.axhline(0, color='gray', linewidth=0.8)
    ax_abs.set_xlabel('Layer Number')
    ax_abs.set_ylabel(f'{metric_base} Absolute Δ')
    ax_pct.set_ylabel(f'{metric_base} Relative Δ (%)')
    ax_abs.set_title(f'{metric_base} Improvement (Absolute & Relative) vs Layer Number')

    # 合并图例
    lines_abs, labels_abs = ax_abs.get_legend_handles_labels()
    lines_pct, labels_pct = ax_pct.get_legend_handles_labels()
    ax_abs.legend(lines_abs + lines_pct, labels_abs + labels_pct, fontsize=8)

    ax_abs.grid(alpha=0.3)
    fig.tight_layout()
    out_file = OUT_DIR / f'{name_prefix}{metric_base}_improve_combined_vs_layer.png'
    fig.savefig(out_file, dpi=150)
    print(f'Saved {out_file}')


def infer_dataset_from_filename(path: Path) -> Optional[str]:
    stem = path.stem  # e.g. amazon_layers
    # 常见命名: amazon_layers / yelp_layers / office_layers
    for cand in ['amazon', 'yelp', 'office', 'steam']:
        if stem.startswith(cand):
            return cand
    return None


MODEL_FAMILIES: List[Tuple[str, List[str]]] = [
    ("lightgcn_family", ["lightgcn", "lightgcn_a", "rgcn"]),
    ("sgl_family", ["sgl", "sgl_a", "sgl_rgcn"]),
    ("simgcl_family", ["simgcl", "simgcl_av1", "simgcl_rgcn"]),
]


def plot_families(df: pd.DataFrame, metric: str):
    for fam_name, members in MODEL_FAMILIES:
        fam_df = df[df['model'].isin(members)]
        if fam_df.empty:
            print(f'Family {fam_name} has no parsed members in log, skip.')
            continue
        plt.figure(figsize=(6,4))
        for model, g in fam_df.groupby('model'):
            plt.plot(g['layer_num'], g[metric], marker='o', label=model)
        plt.xlabel('layer_num')
        plt.ylabel(metric)
        plt.title(f'{fam_name} {metric} vs layer_num')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = OUT_DIR / f'{fam_name}_{metric}_vs_layer.png'
        plt.savefig(out_path, dpi=150)
        print(f'Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Analyse layer influence from log file.')
    parser.add_argument('--log', type=Path, default=LOG_PATH, help='Path to layers log file.')
    parser.add_argument('--dataset', type=str, default='auto', help="Dataset filter: e.g. amazon / yelp / office; 'auto' 从文件名推断; 'none' 不过滤")
    parser.add_argument('--models', type=str, default='', help="仅保留这些模型(逗号分隔)，例如: lightgcn,lightgcn_rgcn")
    parser.add_argument('--layer-min', type=int, default=None, help='最小 layer 过滤 (含)')
    parser.add_argument('--layer-max', type=int, default=None, help='最大 layer 过滤 (含)')
    parser.add_argument('--no-families', action='store_true', help='Disable family plots.')
    args = parser.parse_args()

    log_path: Path = args.log
    if not log_path.exists():
        raise FileNotFoundError(f'Log file not found: {log_path}')

    if args.dataset == 'auto':
        ds = infer_dataset_from_filename(log_path)
        if ds is None:
            print('无法自动推断数据集，将不做过滤 (dataset=None)。')
        target_dataset = ds
    elif args.dataset == 'none':
        target_dataset = None
    else:
        target_dataset = args.dataset

    df = parse_log(log_path, target_dataset=target_dataset)
    if df.empty:
        print('No records parsed from log – please check the log format or regex.')
        return

    # 模型过滤
    if args.models:
        keep_models = {m.strip() for m in args.models.split(',') if m.strip()}
        df = df[df['model'].isin(keep_models)]

    # 层数范围过滤
    if args.layer_min is not None:
        df = df[df['layer_num'] >= args.layer_min]
    if args.layer_max is not None:
        df = df[df['layer_num'] <= args.layer_max]

    if df.empty:
        print('After applying model/layer filters, no data remains.')
        return

    # 计算相对提升（相对于当前过滤后每个模型的最小 layer）
    def add_improvements(metric: str):
        base_vals = df.groupby('model')[metric].transform('first')
        df[f'{metric}_improve_abs'] = df[metric] - base_vals
        df[f'{metric}_improve_pct'] = (df[f'{metric}_improve_abs'] / base_vals).round(4)

    add_improvements('recall@20')
    add_improvements('ndcg@20')

    print('Parsed layer results (after filters):')
    print(df)

    # 绘图（针对 amazon）
    prefix = target_dataset if target_dataset else 'all'
    plot_metric(df, 'recall@20', prefix=prefix)
    plot_metric(df, 'ndcg@20', prefix=prefix)
    # 绘制改进曲线（单独与组合）
    plot_improvement(df, 'recall@20', prefix=prefix)
    plot_improvement(df, 'ndcg@20', prefix=prefix)
    plot_improvement_combined(df, 'recall@20', prefix=prefix)
    plot_improvement_combined(df, 'ndcg@20', prefix=prefix)
    # 家族图（如果有多模型且未禁用）
    if not args.no_families:
        plot_families(df, 'recall@20')
        plot_families(df, 'ndcg@20')

    # 输出 CSV
    # 输出 CSV 文件名包含数据集
    csv_suffix = target_dataset if target_dataset else 'all'
    csv_path = OUT_DIR / f'{csv_suffix}_layer_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'Saved results to {csv_path}')

if __name__ == '__main__':
    main()
