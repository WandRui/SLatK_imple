#!/usr/bin/env python3
import pathlib
import pandas as pd
import re

# 1. 读入你刚才生成的汇总表（CSV 或 MD 均可，这里用 CSV 演示）
summary_csv = "result_analysis/our_result.csv"          # 上一步脚本 --out_csv 生成的文件
df = pd.read_csv(summary_csv)

# 2. 收集表格里已有的 (dataset, backbone, loss) 集合
in_table = set()
for _, row in df.iterrows():
    in_table.add((row.dataset, row.backbone, row.loss))

# 3. 扫描目录下所有 json 文件，收集实际存在的三元组
root_dir = pathlib.Path("experiment_results")
on_disk = set()
for json_path in root_dir.rglob("*.json"):
    # 文件名形如 ItemRec-amazon2014-health-LightGCN-BPR.json
    parts = json_path.stem.split("-")
    if len(parts) < 5:
        continue
    dataset, backbone, loss = parts[2], parts[3], parts[4]
    on_disk.add((dataset, backbone, loss))

# 4. 求差异
missing_in_table   = on_disk - in_table   # 磁盘有但表格没有 → 可能漏汇总
redundant_in_table = in_table - on_disk   # 表格有但磁盘没有 → 可能文件名对不上或已被删

print("========== 磁盘有但汇总表缺失(数据为空) ==========")
for tri in sorted(missing_in_table):
    print(tri)

print("\n========== 汇总表有但磁盘找不到 ==========")
for tri in sorted(redundant_in_table):
    print(tri)

print(f"\n总计：磁盘实际 {len(on_disk)} 组，汇总表 {len(in_table)} 组")