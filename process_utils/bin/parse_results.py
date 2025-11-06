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
    每个 dict 包含: dataset, backbone, loss, Recall@20, NDCG@20, param, 以及各种NDCG@K
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
            
            # 添加各种 NDCG@K 指标
            for k in [5, 10, 20, 50, 75, 100]:
                ndcg_key = f"NDCG@{k}"
                if ndcg_key in value:
                    row[ndcg_key] = float(value[ndcg_key])
                else:
                    print(f"[WARN] {path} trial 缺少 {ndcg_key} 指标")
                    row[ndcg_key] = None
            
            records.append(row)
        except Exception as e:
            print(f"[WARN] {path} 某条 trial 解析失败: {e}")
            continue
    return records

def gather_all_records(root_dir: str):
    root = pathlib.Path(root_dir)
    all_records = []
    for json_path in root.rglob("*.json"):
        # 跳过隐藏目录（以点开头的目录）
        if any(part.startswith('.') for part in json_path.parts):
            # print(f"[INFO] 跳过隐藏目录中的文件: {json_path}")
            continue
        all_records.extend(parse_one_json(json_path))
    return all_records

def pick_best(records):
    """
    按 dataset-backbone-loss 分组，每组保留 NDCG@20 最大的一条
    对于 SLatK，特殊处理：先筛选 k=20 的 trial，再选 NDCG@20 最高的
    """
    group = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["backbone"], r["loss"])
        group[key].append(r)

    best = []
    for key, rows in group.items():
        dataset, backbone, loss = key
        
        if loss == "SLatK":
            # 对于 SLatK，先筛选出 k=20 的 trial
            k20_trials = []
            for row in rows:
                param = row.get("param", {})
                # 检查参数中的 k 值是否为 20
                if param.get("k") == 20 or param.get("K") == 20:
                    k20_trials.append(row)
            
            if k20_trials:
                # 从 k=20 的 trial 中选择 NDCG@20 最高的
                top = max(k20_trials, key=lambda x: x["NDCG@20"])
            else:
                print(f"[WARN] {key} 没有找到 k=20 的 trial，使用所有 trial 中 NDCG@20 最高的")
                top = max(rows, key=lambda x: x["NDCG@20"])
        else:
            # 对于其他 loss，直接选择 NDCG@20 最高的
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
        imp_dict = {}  # 存储 Imp% 信息，带baseline标注
        
        # 新格式: data["results"][backbone][dataset][method] = [recall, ndcg]
        for backbone, backbone_content in data["results"].items():
            for dataset, dataset_content in backbone_content.items():
                # 将数据集名映射到我们的格式
                dataset_name = dataset.lower()  # Health -> health, Electronic -> electronic, Book -> book
                
                # 提取原始 Imp% 信息
                if "Imp. %" in dataset_content:
                    raw_imp = dataset_content["Imp. %"]
                    
                    # 找到该组合中除了"SL@20 (Ours)"和"Imp. %"之外的所有方法，找最佳baseline
                    baselines = {}
                    for method, values in dataset_content.items():
                        if method in ["SL@20 (Ours)", "Imp. %"]:
                            continue
                        if isinstance(values, list) and len(values) >= 2:
                            # 处理方法名映射，用于与我们的记录匹配
                            method_name = method
                            if method == "SL":
                                method_name = "Softmax"
                            baselines[method_name] = values
                    
                    # 找到Recall@20和NDCG@20的最佳baseline
                    best_recall_method = "N/A"
                    best_ndcg_method = "N/A"
                    best_recall_value = 0
                    best_ndcg_value = 0
                    
                    for method, values in baselines.items():
                        if values[0] > best_recall_value:  # Recall@20
                            best_recall_value = values[0]
                            best_recall_method = method
                        if values[1] > best_ndcg_value:  # NDCG@20
                            best_ndcg_value = values[1]
                            best_ndcg_method = method
                    
                    # 为作者的Imp%添加baseline标注
                    annotated_imp = []
                    if isinstance(raw_imp, list) and len(raw_imp) >= 2:
                        # Recall Imp%
                        recall_imp = raw_imp[0]
                        if recall_imp != "N/A" and best_recall_method != "N/A":
                            annotated_imp.append(f"{recall_imp} (vs {best_recall_method})")
                        else:
                            annotated_imp.append(recall_imp)
                        
                        # NDCG Imp%
                        ndcg_imp = raw_imp[1]
                        if ndcg_imp != "N/A" and best_ndcg_method != "N/A":
                            annotated_imp.append(f"{ndcg_imp} (vs {best_ndcg_method})")
                        else:
                            annotated_imp.append(ndcg_imp)
                    else:
                        annotated_imp = raw_imp
                    
                    imp_key = (dataset_name, backbone)
                    imp_dict[imp_key] = annotated_imp
                
                for loss, metrics in dataset_content.items():
                    # 跳过改进百分比行
                    if loss == "Imp. %":
                        continue
                    
                    # 处理一些loss名称的映射
                    loss_name = loss
                    if loss == "SL":
                        loss_name = "Softmax"  # 假设SL对应Softmax
                    elif loss == "SL@20 (Ours)":
                        loss_name = "SLatK"  # 假设这是我们的SLatK方法
                    
                    # metrics 是一个列表 [recall_value, ndcg_value]
                    if isinstance(metrics, list) and len(metrics) >= 2:
                        key = (dataset_name, backbone, loss_name)
                        author_dict[key] = {
                            "Recall@20": metrics[0],
                            "NDCG@20": metrics[1]
                        }
        
        return author_dict, imp_dict
    except Exception as e:
        print(f"[WARN] 加载作者结果失败: {e}")
        return {}, {}

def load_author_table3_results(path: str):
    """
    加载作者的 Table 3 实验结果（MF backbone 的 NDCG@K 结果）
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        author_table3_dict = {}
        table3_imp_dict = {}  # 存储 Table 3 的 Imp% 信息，带baseline标注
        k_values = [5, 10, 20, 50, 75, 100]  # 对应 D@5, D@10, D@20, D@50, D@75, D@100
        
        for dataset in data["results"]:
            dataset_name = dataset.lower()  # Health -> health, Electronic -> electronic
            
            # 提取原始 Imp% 信息并添加baseline标注
            if "Imp.%" in data["results"][dataset]:
                raw_imp = data["results"][dataset]["Imp.%"]
                
                # 找到该数据集中除了"SL@K (Ours)"和"Imp.%"之外的所有方法的NDCG值
                baselines = {}
                for method, values in data["results"][dataset].items():
                    if method in ["SL@K (Ours)", "Imp.%"]:
                        continue
                    if isinstance(values, list):
                        # 处理方法名映射
                        method_name = method
                        if method == "SL":
                            method_name = "Softmax"
                        baselines[method_name] = values
                
                # 为每个K值找到最佳baseline并标注
                annotated_imp = []
                if isinstance(raw_imp, list):
                    for i, (k, imp_val) in enumerate(zip(k_values, raw_imp)):
                        if i < len(raw_imp) and imp_val != "N/A":
                            # 找到该K值下的最佳baseline
                            best_method = "N/A"
                            best_value = 0
                            for method, values in baselines.items():
                                if i < len(values) and values[i] > best_value:
                                    best_value = values[i]
                                    best_method = method
                            
                            if best_method != "N/A":
                                annotated_imp.append(f"{imp_val} (vs {best_method})")
                            else:
                                annotated_imp.append(imp_val)
                        else:
                            annotated_imp.append(imp_val if i < len(raw_imp) else "N/A")
                else:
                    annotated_imp = raw_imp
                
                table3_imp_dict[dataset_name] = annotated_imp
            
            for method, values in data["results"][dataset].items():
                # 跳过改进百分比行
                if method == "Imp.%":
                    continue
                
                # 处理方法名映射
                loss_name = method
                if method == "SL":
                    loss_name = "Softmax"
                elif method == "SL@K (Ours)":
                    loss_name = "SLatK"
                
                # 存储每个 K 值的结果
                for i, k in enumerate(k_values):
                    if i < len(values):
                        key = (dataset_name, "MF", loss_name, f"NDCG@{k}")
                        author_table3_dict[key] = values[i]
        
        return author_table3_dict, table3_imp_dict
    except Exception as e:
        print(f"[WARN] 加载作者 Table 3 结果失败: {e}")
        return {}, {}

def to_markdown_table(best_records):
    df = pd.DataFrame(best_records)
    # 按论文常用顺序排序
    df = df.sort_values(["dataset", "backbone", "loss"])
    # 只保留关心的列
    df = df[["dataset", "backbone", "loss", "Recall@20", "NDCG@20"]]
    # 设置打印精度
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    return df.to_markdown(index=False)

def calculate_our_improvement(best_records):
    """
    计算我们复现结果中SL@K方法相对于最好baseline的提升百分比（与作者计算方法一致）
    """
    our_imp_dict = {}
    best_baseline_dict = {}  # 存储最好baseline的信息
    
    # 按 dataset-backbone 分组
    groups = defaultdict(list)
    for record in best_records:
        key = (record["dataset"], record["backbone"])
        groups[key].append(record)
    
    for (dataset, backbone), records in groups.items():
        # 找到 SLatK 方法的结果
        slatk_record = next((r for r in records if r["loss"] == "SLatK"), None)
        if not slatk_record:
            continue
        
        # 找到其他方法的结果
        other_records = [r for r in records if r["loss"] != "SLatK"]
        if not other_records:
            continue
        
        # 找到最好的baseline（Recall@20和NDCG@20分别找最高的）
        best_recall_baseline = max(other_records, key=lambda x: x["Recall@20"])
        best_ndcg_baseline = max(other_records, key=lambda x: x["NDCG@20"])
        
        # 计算相对于最好baseline的提升百分比
        if best_recall_baseline["Recall@20"] > 0:
            recall_imp = (slatk_record["Recall@20"] - best_recall_baseline["Recall@20"]) / best_recall_baseline["Recall@20"] * 100
            recall_imp_str = f"{recall_imp:+.2f}% (vs {best_recall_baseline['loss']})"
        else:
            recall_imp = 0
            recall_imp_str = f"{recall_imp:+.2f}%"
            
        if best_ndcg_baseline["NDCG@20"] > 0:
            ndcg_imp = (slatk_record["NDCG@20"] - best_ndcg_baseline["NDCG@20"]) / best_ndcg_baseline["NDCG@20"] * 100
            ndcg_imp_str = f"{ndcg_imp:+.2f}% (vs {best_ndcg_baseline['loss']})"
        else:
            ndcg_imp = 0
            ndcg_imp_str = f"{ndcg_imp:+.2f}%"
        
        key = (dataset, backbone)
        our_imp_dict[key] = [recall_imp_str, ndcg_imp_str]
        best_baseline_dict[key] = {
            "recall_baseline": best_recall_baseline["loss"],
            "ndcg_baseline": best_ndcg_baseline["loss"]
        }
    
    return our_imp_dict, best_baseline_dict

def create_comparison_table(best_records, author_dict, imp_dict):
    """
    创建比较表格
    """
    comparison_data = []
    
    # 计算我们的提升百分比
    our_imp_dict, best_baseline_dict = calculate_our_improvement(best_records)
    
    for record in best_records:
        dataset = record["dataset"]
        backbone = record["backbone"]
        loss = record["loss"]
        our_recall = record["Recall@20"]
        our_ndcg = record["NDCG@20"]
        
        # 查找作者结果
        key = (dataset, backbone, loss)
        author_result = author_dict.get(key)
        
        # 查找 Imp% 信息（仅对 SLatK 方法）
        imp_key = (dataset, backbone)
        author_imp_info = imp_dict.get(imp_key, ["N/A", "N/A"])
        our_imp_info = our_imp_dict.get(imp_key, ["N/A", "N/A"])
        
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
                "NDCG_Diff": ndcg_diff,
                "Author_Recall_Imp%": author_imp_info[0] if loss == "SLatK" else "",
                "Author_NDCG_Imp%": author_imp_info[1] if loss == "SLatK" else "",
                "Our_Recall_Imp%": our_imp_info[0] if loss == "SLatK" else "",
                "Our_NDCG_Imp%": our_imp_info[1] if loss == "SLatK" else ""
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
                "NDCG_Diff": "N/A",
                "Author_Recall_Imp%": "",
                "Author_NDCG_Imp%": "",
                "Our_Recall_Imp%": our_imp_info[0] if loss == "SLatK" else "",
                "Our_NDCG_Imp%": our_imp_info[1] if loss == "SLatK" else ""
            })
    
    return comparison_data

def to_comparison_markdown_table(comparison_data):
    """
    将比较数据转换为markdown格式的表格，按dataset+backbone分组，并用横线分隔
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
    
    # 格式化Imp%列，将数字标成蓝色
    def format_imp(value):
        if value == "" or value == "N/A":
            return value
        try:
            return f'<span style="color:blue">{value}</span>'
        except:
            return str(value)
    
    # 应用格式化
    df["Recall_Diff"] = df["Recall_Diff"].apply(format_diff)
    df["NDCG_Diff"] = df["NDCG_Diff"].apply(format_diff)
    df["Our_Recall@20"] = df["Our_Recall@20"].apply(format_number)
    df["Author_Recall@20"] = df["Author_Recall@20"].apply(format_number)
    df["Our_NDCG@20"] = df["Our_NDCG@20"].apply(format_number)
    df["Author_NDCG@20"] = df["Author_NDCG@20"].apply(format_number)
    df["Author_Recall_Imp%"] = df["Author_Recall_Imp%"].apply(format_imp)
    df["Author_NDCG_Imp%"] = df["Author_NDCG_Imp%"].apply(format_imp)
    df["Our_Recall_Imp%"] = df["Our_Recall_Imp%"].apply(format_imp)
    df["Our_NDCG_Imp%"] = df["Our_NDCG_Imp%"].apply(format_imp)
    
    # 手动生成分组的markdown表格
    markdown_lines = []
    
    # 表头
    headers = ["dataset", "backbone", "loss", "Our_Recall@20", "Author_Recall@20", "Recall_Diff", 
               "Our_NDCG@20", "Author_NDCG@20", "NDCG_Diff", "Author_Recall_Imp%", "Author_NDCG_Imp%", 
               "Our_Recall_Imp%", "Our_NDCG_Imp%"]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|" + "|".join([" --- " for _ in headers]) + "|"
    
    markdown_lines.append(header_line)
    markdown_lines.append(separator_line)
    
    # 按dataset+backbone分组
    current_group = None
    for idx, row in df.iterrows():
        group_key = (row["dataset"], row["backbone"])
        
        # 如果是新的组，添加分隔线（除了第一组）
        if current_group is not None and current_group != group_key:
            # 添加一行分隔线
            separator_row = "|" + "|".join([" --- " for _ in headers]) + "|"
            markdown_lines.append(separator_row)
        
        current_group = group_key
        
        # 构建数据行
        row_values = []
        for col in headers:
            row_values.append(str(row[col]))
        
        data_line = "| " + " | ".join(row_values) + " |"
        markdown_lines.append(data_line)
    
    return "\n".join(markdown_lines)

def calculate_our_table3_improvement(best_records):
    """
    计算我们复现结果中SL@K方法相对于最好baseline在Table 3中的提升百分比（与作者计算方法一致）
    """
    our_table3_imp_dict = {}
    table3_baseline_dict = {}  # 存储最好baseline的信息
    k_values = [5, 10, 20, 50, 75, 100]
    
    # 筛选出 MF backbone 且在 health/electronic 数据集的结果
    mf_records = [r for r in best_records 
                  if r["backbone"] == "MF" and r["dataset"] in ["health", "electronic"]]
    
    # 按数据集分组
    for dataset in ["health", "electronic"]:
        dataset_records = [r for r in mf_records if r["dataset"] == dataset]
        
        # 找到 SLatK 方法的结果
        slatk_record = next((r for r in dataset_records if r["loss"] == "SLatK"), None)
        if not slatk_record:
            continue
        
        # 找到其他方法的结果
        other_records = [r for r in dataset_records if r["loss"] != "SLatK"]
        if not other_records:
            continue
        
        # 对每个 K 值计算相对于最好baseline的提升百分比
        improvements = []
        baselines = []
        for k in k_values:
            ndcg_key = f"NDCG@{k}"
            slatk_value = slatk_record.get(ndcg_key)
            
            if slatk_value is None or slatk_value == "N/A":
                improvements.append("N/A")
                baselines.append("N/A")
                continue
            
            # 找到该K值下最好的baseline及其对应的方法
            best_baseline_value = 0
            best_baseline_method = "N/A"
            for other_record in other_records:
                other_value = other_record.get(ndcg_key)
                if other_value is not None and other_value != "N/A" and other_value > best_baseline_value:
                    best_baseline_value = other_value
                    best_baseline_method = other_record["loss"]
            
            if best_baseline_value > 0:
                imp = (slatk_value - best_baseline_value) / best_baseline_value * 100
                improvements.append(f"{imp:+.2f}% (vs {best_baseline_method})")
                baselines.append(best_baseline_method)
            else:
                improvements.append("N/A")
                baselines.append("N/A")
        
        our_table3_imp_dict[dataset] = improvements
        table3_baseline_dict[dataset] = baselines
    
    return our_table3_imp_dict, table3_baseline_dict

def create_table3_comparison(best_records, author_table3_dict, table3_imp_dict):
    """
    创建 Table 3 风格的比较表格（MF backbone 的 NDCG@K 结果）
    """
    # 筛选出 MF backbone 且在 health/electronic 数据集的结果
    mf_records = [r for r in best_records 
                  if r["backbone"] == "MF" and r["dataset"] in ["health", "electronic"]]
    
    k_values = [5, 10, 20, 50, 75, 100]
    datasets = ["health", "electronic"]
    
    # 计算我们的提升百分比
    our_table3_imp_dict, table3_baseline_dict = calculate_our_table3_improvement(best_records)
    
    # 获取所有可用的损失函数
    available_losses = list(set([r["loss"] for r in mf_records]))
    available_losses.sort()
    
    table3_data = []
    
    for dataset in datasets:
        for loss in available_losses:
            row_data = {
                "dataset": dataset.capitalize(),
                "loss": loss,
            }
            
            # 找到对应的记录
            record = next((r for r in mf_records 
                         if r["dataset"] == dataset and r["loss"] == loss), None)
            
            # 获取 Imp% 信息（仅对 SLatK 方法）
            author_imp_info = table3_imp_dict.get(dataset, ["N/A"] * 6)
            our_imp_info = our_table3_imp_dict.get(dataset, ["N/A"] * 6)
            
            if record:
                # 添加我们的结果和作者结果
                for i, k in enumerate(k_values):
                    ndcg_key = f"NDCG@{k}"
                    our_value = record.get(ndcg_key, "N/A")
                    
                    # 查找作者结果
                    author_key = (dataset, "MF", loss, ndcg_key)
                    author_value = author_table3_dict.get(author_key, "N/A")
                    
                    row_data[f"Our_NDCG@{k}"] = our_value
                    row_data[f"Author_NDCG@{k}"] = author_value
                    
                    # 计算差值
                    if our_value != "N/A" and author_value != "N/A":
                        diff = our_value - author_value
                        row_data[f"Diff_NDCG@{k}"] = diff
                    else:
                        row_data[f"Diff_NDCG@{k}"] = "N/A"
                    
                    # 添加作者 Imp% 信息（仅对 SLatK）
                    if loss == "SLatK" and i < len(author_imp_info):
                        row_data[f"Author_Imp%_NDCG@{k}"] = author_imp_info[i]
                    else:
                        row_data[f"Author_Imp%_NDCG@{k}"] = ""
                    
                    # 添加我们的 Imp% 信息（仅对 SLatK）
                    if loss == "SLatK" and i < len(our_imp_info):
                        row_data[f"Our_Imp%_NDCG@{k}"] = our_imp_info[i]
                    else:
                        row_data[f"Our_Imp%_NDCG@{k}"] = ""
            else:
                # 如果没有找到记录，填充 N/A
                for k in k_values:
                    row_data[f"Our_NDCG@{k}"] = "N/A"
                    row_data[f"Author_NDCG@{k}"] = "N/A"
                    row_data[f"Diff_NDCG@{k}"] = "N/A"
                    row_data[f"Author_Imp%_NDCG@{k}"] = ""
                    row_data[f"Our_Imp%_NDCG@{k}"] = ""
            
            table3_data.append(row_data)
    
    return table3_data

def to_table3_markdown(table3_data):
    """
    将 Table 3 数据转换为 markdown 格式，按dataset分组（Table 3只包含MF backbone）
    """
    if not table3_data:
        return "# Table 3 比较结果\n\n没有找到相关数据。\n"
    
    k_values = [5, 10, 20, 50, 75, 100]
    
    # 按数据集分组
    from collections import defaultdict
    grouped_data = defaultdict(list)
    for row in table3_data:
        key = row["dataset"]
        grouped_data[key].append(row)
    
    # 按key排序
    sorted_groups = sorted(grouped_data.items())
    
    markdown = "# Table 3: MF Backbone NDCG@K 比较结果\n\n"
    
    # 创建统一的表格头
    header = "| Dataset | Method |"
    for k in k_values:
        header += f" Our_NDCG@{k} | Author_NDCG@{k} | Diff | Author_Imp% | Our_Imp% |"
    header += "\n"
    
    # 创建分隔线
    separator = "|---------|--------|"
    for k in k_values:
        separator += "------------|---------------|------|-------------|-----------|"
    separator += "\n"
    
    markdown += header + separator
    
    # 添加数据行，按组分隔
    first_group = True
    for dataset_name, group_data in sorted_groups:
        
        # 如果不是第一组，添加分隔线
        if not first_group:
            sep_line = "|---------|--------|"
            for k in k_values:
                sep_line += "------------|---------------|------|-------------|-----------|"
            sep_line += "\n"
            markdown += sep_line
        first_group = False
        
        # 添加数据行
        for row in group_data:
            line = f"| {dataset_name} | {row['loss']} |"
            for k in k_values:
                our_val = row[f"Our_NDCG@{k}"]
                author_val = row[f"Author_NDCG@{k}"]
                diff_val = row[f"Diff_NDCG@{k}"]
                author_imp_val = row[f"Author_Imp%_NDCG@{k}"]
                our_imp_val = row[f"Our_Imp%_NDCG@{k}"]
                
                # 格式化数值
                our_str = f"{our_val:.4f}" if our_val != "N/A" else "N/A"
                author_str = f"{author_val:.4f}" if author_val != "N/A" else "N/A"
                
                if diff_val != "N/A":
                    if diff_val > 0.01:
                        diff_str = f'<span style="color:green">{diff_val:.4f}</span>'
                    elif diff_val < -0.01:
                        diff_str = f'<span style="color:red">{diff_val:.4f}</span>'
                    else:
                        diff_str = f"{diff_val:.4f}"
                else:
                    diff_str = "N/A"
                
                # 格式化Imp%，将数字标成蓝色
                if author_imp_val and author_imp_val != "N/A":
                    author_imp_str = f'<span style="color:blue">{author_imp_val}</span>'
                else:
                    author_imp_str = ""
                
                if our_imp_val and our_imp_val != "N/A":
                    our_imp_str = f'<span style="color:blue">{our_imp_val}</span>'
                else:
                    our_imp_str = ""
                
                line += f" {our_str} | {author_str} | {diff_str} | {author_imp_str} | {our_imp_str} |"
            line += "\n"
            markdown += line
    
    markdown += "## 说明\n"
    markdown += "- Our_NDCG@K: 我们的实验结果\n"
    markdown += "- Author_NDCG@K: 作者的实验结果\n"
    markdown += "- Diff: 差值（我们的结果 - 作者的结果）\n"
    markdown += "- Author_Imp%: 作者论文中SL@K相对于最好baseline的提升百分比（仅对SLatK方法显示）\n"
    markdown += "- Our_Imp%: 我们复现结果中SL@K相对于最好baseline的提升百分比（仅对SLatK方法显示）\n"
    markdown += "- 正值（绿色）表示我们的结果更好，负值（红色）表示作者的结果更好\n"
    markdown += "- 差值绝对值 ≤ 0.01 的结果未标色\n"
    
    return markdown

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiment_results", help="含 json 的根目录")
    parser.add_argument("--output_dir", default="result_analysis", help="输出文件夹路径")
    args = parser.parse_args()
    
    # 确保输出文件夹存在
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义输出文件路径
    compare_csv = output_dir / "table2_compare.csv"
    compare_md = output_dir / "table2_compare.md"
    table3_csv = output_dir / "table3_compare.csv"
    table3_md = output_dir / "table3_compare.md"
    author_table2_path = output_dir / "author_table2.json"
    author_table3_path = output_dir / "author_table3.json"

    records = gather_all_records(args.root)
    print(f"共解析 {len(records)} 条 trial 记录")
    best = pick_best(records)
    print(f"汇总后 {len(best)} 条最优记录")

    # 加载作者结果并生成比较表格
    author_dict, imp_dict = load_author_results(str(author_table2_path))
    if author_dict:
        comparison_data = create_comparison_table(best, author_dict, imp_dict)
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较结果 CSV
        comparison_df.to_csv(compare_csv, index=False, float_format="%.4f")
        print(f"比较结果CSV已写入 {compare_csv}")
        
        # 生成并保存比较结果 Markdown
        comparison_md = to_comparison_markdown_table(comparison_data)
        with open(compare_md, "w", encoding="utf-8") as f:
            f.write("# 实验结果对比\n\n")
            f.write("以下是我们的实验结果与作者结果的对比：\n\n")
            f.write(comparison_md)
            f.write("\n\n## 说明\n")
            f.write("- Our_Recall@20/Our_NDCG@20: 我们的实验结果\n")
            f.write("- Author_Recall@20/Author_NDCG@20: 作者的实验结果\n")
            f.write("- Recall_Diff/NDCG_Diff: 差值（我们的结果 - 作者的结果）\n")
            f.write("- Author_Recall_Imp%/Author_NDCG_Imp%: 作者论文中SL@K相对于最好baseline的提升百分比（仅对SLatK方法显示）\n")
            f.write("- Our_Recall_Imp%/Our_NDCG_Imp%: 我们复现结果中SL@K相对于最好baseline的提升百分比（仅对SLatK方法显示）\n")
            f.write("- 正值表示我们的结果更好，负值表示作者的结果更好\n")
            f.write("- 差值绝对值 ≤ 0.01 的结果未标色，差值 > 0.01 的结果标绿，差值 < -0.01 的结果标红\n")
        print(f"比较结果Markdown已写入 {compare_md}")
    else:
        print("[WARN] 无法加载作者结果，跳过比较")
    
    # 加载作者 Table 3 结果并生成对比
    author_table3_dict, table3_imp_dict = load_author_table3_results(str(author_table3_path))
    if author_table3_dict:
        table3_data = create_table3_comparison(best, author_table3_dict, table3_imp_dict)
        
        # 保存 Table 3 比较结果 CSV
        if table3_data:
            table3_df = pd.DataFrame(table3_data)
            table3_df.to_csv(table3_csv, index=False, float_format="%.4f")
            print(f"Table 3 比较结果CSV已写入 {table3_csv}")
        
        # 生成并保存 Table 3 比较结果 Markdown
        table3_md_content = to_table3_markdown(table3_data)
        with open(table3_md, "w", encoding="utf-8") as f:
            f.write(table3_md_content)
        print(f"Table 3 比较结果Markdown已写入 {table3_md}")
    else:
        print("[WARN] 无法加载作者 Table 3 结果，跳过 Table 3 比较")

if __name__ == "__main__":
    main()

'''
python process_utils/parse_results.py --root experiment_results --output_dir result_analysis
'''