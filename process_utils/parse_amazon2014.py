#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 Amazon2014 实验结果
author: you
"""
import json
import os
import pathlib
import argparse
from collections import defaultdict
import pandas as pd

METRICS = ["Recall@20", "NDCG@20"]  # 关心的指标

def parse_one_json(path: pathlib.Path):
    """
    解析单个 json 文件，返回 list[dict]
    每个 dict 包含: dataset, backbone, loss, Recall@20, NDCG@20, param
    """
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] {path} 读取失败: {e}")
        return []

    # 文件名形如 ItemRec-amazon2014-health-LightGCN-BPR.json
    parts = path.stem.split("-")  # ['ItemRec','amazon2014','health','LightGCN','BPR']
    if len(parts) < 5:
        print(f"[WARN] {path} 文件名格式不对，跳过")
        return []

    dataset = parts[2]  # health / electronic / book
    backbone = parts[3]  # LightGCN / MF / XSimGCL
    loss = parts[4]  # BPR / SL / PSL / Softmax / ...

    records = []
    for trial in data:
        try:
            value = json.loads(trial["value"])
            row = {
                "dataset": dataset,
                "backbone": backbone,
                "loss": loss,
                "Recall@20": float(value["Recall@20"]),
                "NDCG@20": float(value["NDCG@20"]),
                "param": trial["parameter"],
            }
            records.append(row)
        except Exception as e:
            print(f"[WARN] {path} 某条 trial 解析失败: {e}")
            continue
    return records

def gather_all_records(root_dir: str):
    root = pathlib.Path(root_dir)
    all_records = []
    for json_path in root.rglob("*.json"):
        all_records.extend(parse_one_json(json_path))
    return all_records

def pick_best(records):
    """
    按 dataset-backbone-loss 分组，每组保留 NDCG@20 最大的一条
    """
    group = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["backbone"], r["loss"])
        group[key].append(r)

    best = []
    for key, rows in group.items():
        top = max(rows, key=lambda x: x["NDCG@20"])
        best.append(top)
    return best

def load_author_results(path: str):
    """
    加载作者的实验结果
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        # 转换为更方便的格式
        author_dict = {}
        for dataset_data in data:
            for dataset, dataset_content in dataset_data.items():
                # 将数据集名映射到我们的格式
                dataset_name = dataset.lower()  # Health -> health, Electronic -> electronic, Book -> book
                if dataset_name == "book":
                    dataset_name = "book"  # 保持一致
                elif dataset_name == "electronic":
                    dataset_name = "electronic"
                elif dataset_name == "health":
                    dataset_name = "health"
                
                for backbone, backbone_content in dataset_content.items():
                    for loss, metrics in backbone_content.items():
                        # 处理一些loss名称的映射
                        loss_name = loss
                        if loss == "SL":
                            loss_name = "Softmax"  # 假设SL对应Softmax
                        elif loss == "SL@20 (Ours)":
                            loss_name = "SLatK"  # 假设这是我们的SLatK方法
                        
                        key = (dataset_name, backbone, loss_name)
                        author_dict[key] = metrics
        
        return author_dict
    except Exception as e:
        print(f"[WARN] 加载作者结果失败: {e}")
        return {}

def to_markdown_table(best_records):
    df = pd.DataFrame(best_records)
    # 按论文常用顺序排序
    df = df.sort_values(["dataset", "backbone", "loss"])
    # 只保留关心的列
    df = df[["dataset", "backbone", "loss", "Recall@20", "NDCG@20"]]
    # 设置打印精度
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    return df.to_markdown(index=False)

def create_comparison_table(best_records, author_dict):
    """
    创建比较表格
    """
    comparison_data = []
    
    for record in best_records:
        dataset = record["dataset"]
        backbone = record["backbone"]
        loss = record["loss"]
        our_recall = record["Recall@20"]
        our_ndcg = record["NDCG@20"]
        
        # 查找作者结果
        key = (dataset, backbone, loss)
        author_result = author_dict.get(key)
        
        if author_result:
            author_recall = author_result["Recall@20"]
            author_ndcg = author_result["NDCG@20"]
            
            recall_diff = our_recall - author_recall
            ndcg_diff = our_ndcg - author_ndcg
            
            comparison_data.append({
                "dataset": dataset,
                "backbone": backbone,
                "loss": loss,
                "Our_Recall@20": our_recall,
                "Author_Recall@20": author_recall,
                "Recall_Diff": recall_diff,
                "Our_NDCG@20": our_ndcg,
                "Author_NDCG@20": author_ndcg,
                "NDCG_Diff": ndcg_diff
            })
        else:
            # 如果没有找到对应的作者结果，标记为N/A
            comparison_data.append({
                "dataset": dataset,
                "backbone": backbone,
                "loss": loss,
                "Our_Recall@20": our_recall,
                "Author_Recall@20": "N/A",
                "Recall_Diff": "N/A",
                "Our_NDCG@20": our_ndcg,
                "Author_NDCG@20": "N/A",
                "NDCG_Diff": "N/A"
            })
    
    return comparison_data

def to_comparison_markdown_table(comparison_data):
    """
    将比较数据转换为markdown格式的表格，并将差值 <= 0.01 的结果标红
    """
    df = pd.DataFrame(comparison_data)
    # 按数据集、骨干网络、损失函数排序
    df = df.sort_values(["dataset", "backbone", "loss"])
    
    # 处理差值列，将 <= 0.01 的值标红
    def format_diff(value):
        if value == "N/A":
            return value
        try:
            diff_val = float(value)
            if abs(diff_val) > 0.01:
                if diff_val > 0:
                    return f'<span style="color:green">{diff_val:.4f}</span>'
                else:
                    return f'<span style="color:red">{diff_val:.4f}</span>'
            else:
                return f"{diff_val:.4f}"
        except:
            return str(value)
    
    # 格式化其他数值列
    def format_number(value):
        if value == "N/A":
            return value
        try:
            return f"{float(value):.4f}"
        except:
            return str(value)
    
    # 应用格式化
    df["Recall_Diff"] = df["Recall_Diff"].apply(format_diff)
    df["NDCG_Diff"] = df["NDCG_Diff"].apply(format_diff)
    df["Our_Recall@20"] = df["Our_Recall@20"].apply(format_number)
    df["Author_Recall@20"] = df["Author_Recall@20"].apply(format_number)
    df["Our_NDCG@20"] = df["Our_NDCG@20"].apply(format_number)
    df["Author_NDCG@20"] = df["Author_NDCG@20"].apply(format_number)
    
    # 生成markdown表格
    return df.to_markdown(index=False, tablefmt="pipe")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiment_results", help="含 json 的根目录")
    parser.add_argument("--out_table", default="result_analysis/our_result.md", help="输出 markdown 表格")
    parser.add_argument("--out_csv", default="result_analysis/our_result.csv", help="输出 csv（可选）")
    parser.add_argument("--author_results", default="result_analysis/author_results.json", help="作者结果文件路径")
    parser.add_argument("--compare_csv", default="result_analysis/compare.csv", help="比较结果输出路径")
    parser.add_argument("--compare_md", default="result_analysis/compare.md", help="比较结果markdown输出路径")
    args = parser.parse_args()

    records = gather_all_records(args.root)
    print(f"共解析 {len(records)} 条 trial 记录")
    best = pick_best(records)
    print(f"汇总后 {len(best)} 条最优记录")

    # 生成原有的markdown表格和csv
    md = to_markdown_table(best)
    with open(args.out_table, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown 表格已写入 {args.out_table}")

    if args.out_csv:
        pd.DataFrame(best).to_csv(args.out_csv, index=False, float_format="%.4f")
        print(f"CSV 已写入 {args.out_csv}")

    # 加载作者结果并生成比较表格
    author_dict = load_author_results(args.author_results)
    if author_dict:
        comparison_data = create_comparison_table(best, author_dict)
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较结果 CSV
        comparison_df.to_csv(args.compare_csv, index=False, float_format="%.4f")
        print(f"比较结果CSV已写入 {args.compare_csv}")
        
        # 生成并保存比较结果 Markdown
        comparison_md = to_comparison_markdown_table(comparison_data)
        with open(args.compare_md, "w", encoding="utf-8") as f:
            f.write("# 实验结果对比\n\n")
            f.write("以下是我们的实验结果与作者结果的对比：\n\n")
            f.write(comparison_md)
            f.write("\n\n## 说明\n")
            f.write("- Our_Recall@20/Our_NDCG@20: 我们的实验结果\n")
            f.write("- Author_Recall@20/Author_NDCG@20: 作者的实验结果\n")
            f.write("- Recall_Diff/NDCG_Diff: 差值（我们的结果 - 作者的结果）\n")
            f.write("- 正值表示我们的结果更好，负值表示作者的结果更好\n")
            f.write("- 差值绝对值 ≤ 0.01 的结果未标色，差值 > 0.01 的结果标绿，差值 < -0.01 的结果标红\n")
        print(f"比较结果Markdown已写入 {args.compare_md}")
        
        # 打印一些统计信息
        valid_comparisons = [row for row in comparison_data if row["Recall_Diff"] != "N/A"]
        if valid_comparisons:
            recall_improvements = [row["Recall_Diff"] for row in valid_comparisons if row["Recall_Diff"] > 0]
            ndcg_improvements = [row["NDCG_Diff"] for row in valid_comparisons if row["NDCG_Diff"] > 0]
            print(f"找到 {len(valid_comparisons)} 个有效比较")
            print(f"Recall@20 改进的情况: {len(recall_improvements)}/{len(valid_comparisons)}")
            print(f"NDCG@20 改进的情况: {len(ndcg_improvements)}/{len(valid_comparisons)}")
    else:
        print("[WARN] 无法加载作者结果，跳过比较")

if __name__ == "__main__":
    main()

'''
python process_utils/parse_amazon2014.py --root experiment_results --out_table result_analysis/our_result.md --out_csv result_analysis/our_result.csv --compare_csv result_analysis/compare.csv --compare_md result_analysis/compare.md
'''