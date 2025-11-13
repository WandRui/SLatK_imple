#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python process_utils/parse_results_refactored.py --root experiment_results --output_dir result_analysis/comparison
'''

'''
zip -r zipped_result/comparisonV3.zip result_analysis/comparison
'''

"""
Summary of Amazon2014 Experiment Results - Refactored Version
author: you
"""
import json
import os
import pathlib
import argparse
import csv
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

METRICS = ["Recall@20", "NDCG@20"]  # Metrics of interest


class ExperimentRecord:
    """Experiment record data class"""
    
    def __init__(self, dataset: str, backbone: str, loss: str, metrics: Dict[str, float], 
                 param: Dict[str, Any]):
        self.dataset = dataset
        self.backbone = backbone
        self.loss = loss
        self.metrics = metrics
        self.param = param
    
    def get_metric(self, metric_name: str) -> Optional[float]:
        """Get specified metric value"""
        return self.metrics.get(metric_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            "dataset": self.dataset,
            "backbone": self.backbone,
            "loss": self.loss,
            "param": self.param
        }
        result.update(self.metrics)
        return result


class ExperimentParser:
    """Experiment result parser"""
    
    def __init__(self):
        self.k_values = [5, 10, 20, 50, 75, 100]
    
    def parse_one_json(self, path: pathlib.Path) -> List[ExperimentRecord]:
        """
        Parse a single json file and return a list of ExperimentRecord
        """
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] {path} read failed: {e}")
            return []

        # Filename format: 
        # ItemRec-amazon2014-health-LightGCN-BPR.json (for amazon2014 datasets)
        # ItemRec-gowalla-LightGCN-BPR.json (for gowalla dataset)
        parts = path.stem.split("-")
        if len(parts) < 4:
            print(f"[WARN] {path} filename format incorrect, skipping")
            return []

        if len(parts) >= 5:
            # Amazon2014 format: ItemRec-amazon2014-health-LightGCN-BPR.json
            dataset = parts[2]  # health / electronic / book
            backbone = parts[3]  # LightGCN / MF / XSimGCL
            loss = parts[4]  # BPR / SL / PSL / Softmax / ...
        else:
            # Gowalla format: ItemRec-gowalla-LightGCN-BPR.json
            dataset = parts[1]  # gowalla
            backbone = parts[2]  # LightGCN / MF / XSimGCL
            loss = parts[3]  # BPR / SL / PSL / Softmax / ...

        records = []
        for trial in data:
            try:
                value = json.loads(trial["value"])
                metrics = {
                    "Recall@20": float(value["Recall@20"]),
                    "NDCG@20": float(value["NDCG@20"]),
                }
                
                # Add various NDCG@K metrics
                for k in self.k_values:
                    ndcg_key = f"NDCG@{k}"
                    if ndcg_key in value:
                        metrics[ndcg_key] = float(value[ndcg_key])
                    else:
                        print(f"[WARN] {path} trial missing {ndcg_key} metric")
                        metrics[ndcg_key] = None
                
                record = ExperimentRecord(dataset, backbone, loss, metrics, trial["parameter"])
                records.append(record)
            except Exception as e:
                print(f"[WARN] {path} some trial parsing failed: {e}")
                continue
        
        return records

    def gather_all_records(self, root_dir: str) -> List[ExperimentRecord]:
        """Collect all experiment records"""
        root = pathlib.Path(root_dir)
        all_records = []
        for json_path in root.rglob("*.json"):
            # Skip hidden directories (starting with dot)
            if any(part.startswith('.') for part in json_path.parts):
                continue
            all_records.extend(self.parse_one_json(json_path))
        return all_records


class BestRecordSelector:
    """Best record selector"""
    
    def pick_best(self, records: List[ExperimentRecord]) -> List[ExperimentRecord]:
        """
        Group by dataset-backbone-loss, keep the one with highest NDCG@20 in each group
        For SLatK, special handling: first filter trials with k=20, then select the one with highest NDCG@20
        """
        group = defaultdict(list)
        for r in records:
            key = (r.dataset, r.backbone, r.loss)
            group[key].append(r)

        best = []
        for key, rows in group.items():
            dataset, backbone, loss = key
            
            if loss == "SLatK":
                # For SLatK, first filter trials with k=20
                k20_trials = []
                for row in rows:
                    param = row.param
                    # Check if k value in parameters is 20
                    if param.get("k") == 20 or param.get("K") == 20:
                        k20_trials.append(row)
                
                if k20_trials:
                    # Select the one with highest NDCG@20 from k=20 trials
                    top = max(k20_trials, key=lambda x: x.get_metric("NDCG@20"))
                else:
                    print(f"[WARN] {key} no k=20 trial found, using the one with highest NDCG@20 from all trials")
                    top = max(rows, key=lambda x: x.get_metric("NDCG@20"))
            else:
                # For other losses, directly select the one with highest NDCG@20
                top = max(rows, key=lambda x: x.get_metric("NDCG@20"))
            
            best.append(top)
        return best

    def pick_best_for_table3_extended(self, records: List[ExperimentRecord]) -> List[ExperimentRecord]:
        """
        Group by dataset-backbone-loss, keep the one with highest NDCG@20 in each group
        For SLatK, select the best trial across all K values (not restricted to k=20)
        This is used for Table 3 extended analysis to find the optimal SL@K configuration
        """
        group = defaultdict(list)
        for r in records:
            key = (r.dataset, r.backbone, r.loss)
            group[key].append(r)

        best = []
        for key, rows in group.items():
            dataset, backbone, loss = key
            
            # For all losses including SLatK, directly select the one with highest NDCG@20
            # This allows SLatK to use its optimal K value rather than being restricted to K=20
            top = max(rows, key=lambda x: x.get_metric("NDCG@20"))
            best.append(top)
        return best


class AuthorResultLoader:
    """Author result loader"""
    
    def load_author_results(self, path: str) -> Tuple[Dict[Tuple[str, str, str], Dict[str, float]], 
                                                      Dict[Tuple[str, str], List[str]]]:
        """
        Load author's experiment results
        Returns: (author_dict, imp_dict)
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert to more convenient format
            author_dict = {}
            imp_dict = {}  # Store Imp% information with baseline annotation
            
            # New format: data["results"][backbone][dataset][method] = [recall, ndcg]
            for backbone, backbone_content in data["results"].items():
                for dataset, dataset_content in backbone_content.items():
                    # Map dataset name to our format
                    dataset_name = dataset.lower()  # Health -> health, Electronic -> electronic, Book -> book, Gowalla -> gowalla
                    
                    # Extract original Imp% information
                    if "Imp. %" in dataset_content:
                        raw_imp = dataset_content["Imp. %"]
                        
                        # Find all methods except "SL@20 (Ours)" and "Imp. %" in this combination, find best baseline
                        baselines = {}
                        for method, values in dataset_content.items():
                            if method in ["SL@20 (Ours)", "Imp. %"]:
                                continue
                            if isinstance(values, list) and len(values) >= 2:
                                # Handle method name mapping for matching with our records
                                method_name = method
                                if method == "SL":
                                    method_name = "Softmax"
                                elif method == "SONG@20":
                                    method_name = "SONGatK"
                                baselines[method_name] = values
                        
                        # Find best baseline for Recall@20 and NDCG@20
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
                        
                        # Add baseline annotation to author's Imp%
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
                        # Skip improvement percentage rows
                        if loss == "Imp. %":
                            continue
                        
                        # Handle some loss name mappings
                        loss_name = loss
                        if loss == "SL":
                            loss_name = "Softmax"  # Assume SL corresponds to Softmax
                        elif loss == "SL@20 (Ours)":
                            loss_name = "SLatK"  # Assume this is our SLatK method
                        elif loss == "SONG@20":
                            loss_name = "SONGatK"  # SONG@20 corresponds to our SONGatK method
                        
                        # metrics is a list [recall_value, ndcg_value]
                        if isinstance(metrics, list) and len(metrics) >= 2:
                            key = (dataset_name, backbone, loss_name)
                            author_dict[key] = {
                                "Recall@20": metrics[0],
                                "NDCG@20": metrics[1]
                            }
            
            return author_dict, imp_dict
        except Exception as e:
            print(f"[WARN] Failed to load author results: {e}")
            return {}, {}

    def load_author_table3_results(self, path: str) -> Tuple[Dict[Tuple[str, str, str, str], float], 
                                                             Dict[str, List[str]]]:
        """
        Load author's Table 3 experiment results (MF backbone NDCG@K results)
        Returns: (author_table3_dict, table3_imp_dict)
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            
            author_table3_dict = {}
            table3_imp_dict = {}  # Store Table 3 Imp% information with baseline annotation
            k_values = [5, 10, 20, 50, 75, 100]  # Corresponding to D@5, D@10, D@20, D@50, D@75, D@100
            
            for dataset in data["results"]:
                dataset_name = dataset.lower()  # Health -> health, Electronic -> electronic
                
                # Extract original Imp% information and add baseline annotation
                if "Imp.%" in data["results"][dataset]:
                    raw_imp = data["results"][dataset]["Imp.%"]
                    
                    # Find NDCG values for all methods except "SL@K (Ours)" and "Imp.%" in this dataset
                    baselines = {}
                    for method, values in data["results"][dataset].items():
                        if method in ["SL@K (Ours)", "Imp.%"]:
                            continue
                        if isinstance(values, list):
                            # Handle method name mapping
                            method_name = method
                            if method == "SL":
                                method_name = "Softmax"
                            elif method == "SONG@20":
                                method_name = "SONGatK"
                            baselines[method_name] = values
                    
                    # Find best baseline for each K value and annotate
                    annotated_imp = []
                    if isinstance(raw_imp, list):
                        for i, (k, imp_val) in enumerate(zip(k_values, raw_imp)):
                            if i < len(raw_imp) and imp_val != "N/A":
                                # Find best baseline for this K value
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
                    # Skip improvement percentage rows
                    if method == "Imp.%":
                        continue
                    
                    # Handle method name mapping
                    loss_name = method
                    if method == "SL":
                        loss_name = "Softmax"
                    elif method == "SL@K (Ours)":
                        loss_name = "SLatK"
                    elif method == "SONG@20":
                        loss_name = "SONGatK"
                    
                    # Store results for each K value
                    for i, k in enumerate(k_values):
                        if i < len(values):
                            key = (dataset_name, "MF", loss_name, f"NDCG@{k}")
                            author_table3_dict[key] = values[i]
            
            return author_table3_dict, table3_imp_dict
        except Exception as e:
            print(f"[WARN] Failed to load author Table 3 results: {e}")
            return {}, {}

    def load_author_table4_results(self, path: str) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
        """
        Load author's Table 4 experiment results (Performance of SL@K method with different K values on various NDCG@K')
        Returns: (dataset, method_type, metric_k) -> [{"method_variant": "SL@5", "value": 0.1080}, ...] dictionary
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            
            author_table4_dict = {}
            k_values = [5, 10, 20, 50, 75, 100]  # Corresponding to D@5, D@10, D@20, D@50, D@75, D@100
            
            for dataset_name, dataset_results in data["results"].items():
                dataset_key = dataset_name.lower()  # Health -> health, Electronic -> electronic
                
                for method_name, values in dataset_results.items():
                    # Store results for each NDCG@K value
                    for i, k in enumerate(k_values):
                        if i < len(values):
                            # Use unified method type "SLatK" to group different K value variants
                            key = (dataset_key, "SLatK", f"NDCG@{k}")
                            if key not in author_table4_dict:
                                author_table4_dict[key] = []
                            author_table4_dict[key].append({
                                "method_variant": method_name,  # Save original method name to distinguish different K values
                                "value": values[i]
                            })
            
            return author_table4_dict
        except Exception as e:
            print(f"[WARN] Failed to load author Table 4 results: {e}")
            return {}


class ImprovementCalculator:
    """Improvement percentage calculator"""
    
    def calculate_our_improvement(self, best_records: List[ExperimentRecord]) -> Tuple[Dict[Tuple[str, str], List[str]], 
                                                                                      Dict[Tuple[str, str], Dict[str, str]]]:
        """
        Calculate improvement percentage of SL@K method relative to best baseline in our reproduction results (consistent with author's calculation method)
        Returns: (our_imp_dict, best_baseline_dict)
        """
        our_imp_dict = {}
        best_baseline_dict = {}  # Store best baseline information
        
        # Group by dataset-backbone
        groups = defaultdict(list)
        for record in best_records:
            key = (record.dataset, record.backbone)
            groups[key].append(record)
        
        for (dataset, backbone), records in groups.items():
            # Find SLatK method results
            slatk_record = next((r for r in records if r.loss == "SLatK"), None)
            if not slatk_record:
                continue
            
            # Find other method results
            other_records = [r for r in records if r.loss != "SLatK"]
            if not other_records:
                continue
            
            # Find best baseline (find highest for Recall@20 and NDCG@20 separately)
            best_recall_baseline = max(other_records, key=lambda x: x.get_metric("Recall@20"))
            best_ndcg_baseline = max(other_records, key=lambda x: x.get_metric("NDCG@20"))
            
            # Calculate improvement percentage relative to best baseline
            if best_recall_baseline.get_metric("Recall@20") > 0:
                recall_imp = (slatk_record.get_metric("Recall@20") - best_recall_baseline.get_metric("Recall@20")) / best_recall_baseline.get_metric("Recall@20") * 100
                recall_imp_str = f"{recall_imp:+.2f}% (vs {best_recall_baseline.loss})"
            else:
                recall_imp = 0
                recall_imp_str = f"{recall_imp:+.2f}%"
                
            if best_ndcg_baseline.get_metric("NDCG@20") > 0:
                ndcg_imp = (slatk_record.get_metric("NDCG@20") - best_ndcg_baseline.get_metric("NDCG@20")) / best_ndcg_baseline.get_metric("NDCG@20") * 100
                ndcg_imp_str = f"{ndcg_imp:+.2f}% (vs {best_ndcg_baseline.loss})"
            else:
                ndcg_imp = 0
                ndcg_imp_str = f"{ndcg_imp:+.2f}%"
            
            key = (dataset, backbone)
            our_imp_dict[key] = [recall_imp_str, ndcg_imp_str]
            best_baseline_dict[key] = {
                "recall_baseline": best_recall_baseline.loss,
                "ndcg_baseline": best_ndcg_baseline.loss
            }
        
        return our_imp_dict, best_baseline_dict

    def calculate_our_table3_improvement(self, best_records: List[ExperimentRecord]) -> Tuple[Dict[str, List[str]], 
                                                                                              Dict[str, List[str]]]:
        """
        Calculate improvement percentage of SL@K method relative to best baseline in Table 3 for our reproduction results (consistent with author's calculation method)
        Returns: (our_table3_imp_dict, table3_baseline_dict)
        Note: Only process datasets that have author table3 results (health, electronic)
        """
        our_table3_imp_dict = {}
        table3_baseline_dict = {}  # Store best baseline information
        k_values = [5, 10, 20, 50, 75, 100]
        
        # Filter MF backbone results on health/electronic datasets (only datasets with author table3 results)
        mf_records = [r for r in best_records 
                      if r.backbone == "MF" and r.dataset in ["health", "electronic"]]
        
        # Group by dataset (only process datasets with author table3 results)
        for dataset in ["health", "electronic"]:
            dataset_records = [r for r in mf_records if r.dataset == dataset]
            
            # Find SLatK method results
            slatk_record = next((r for r in dataset_records if r.loss == "SLatK"), None)
            if not slatk_record:
                continue
            
            # Find other method results
            other_records = [r for r in dataset_records if r.loss != "SLatK"]
            if not other_records:
                continue
            
            # Calculate improvement percentage relative to best baseline for each K value
            improvements = []
            baselines = []
            for k in k_values:
                ndcg_key = f"NDCG@{k}"
                slatk_value = slatk_record.get_metric(ndcg_key)
                
                if slatk_value is None or slatk_value == "N/A":
                    improvements.append("N/A")
                    baselines.append("N/A")
                    continue
                
                # Find best baseline for this K value and corresponding method
                best_baseline_value = 0
                best_baseline_method = "N/A"
                for other_record in other_records:
                    other_value = other_record.get_metric(ndcg_key)
                    if other_value is not None and other_value != "N/A" and other_value > best_baseline_value:
                        best_baseline_value = other_value
                        best_baseline_method = other_record.loss
                
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


class ComparisonTableGenerator:
    """Comparison table generator"""
    
    def __init__(self, improvement_calculator: ImprovementCalculator):
        self.improvement_calculator = improvement_calculator
    
    def create_comparison_table(self, best_records: List[ExperimentRecord], 
                               author_dict: Dict[Tuple[str, str, str], Dict[str, float]], 
                               imp_dict: Dict[Tuple[str, str], List[str]]) -> List[Dict[str, Any]]:
        """
        Create comparison table
        """
        comparison_data = []
        
        # Calculate our improvement percentages
        our_imp_dict, _ = self.improvement_calculator.calculate_our_improvement(best_records)
        
        for record in best_records:
            dataset = record.dataset
            backbone = record.backbone
            loss = record.loss
            our_recall = record.get_metric("Recall@20")
            our_ndcg = record.get_metric("NDCG@20")
            
            # Look up author results
            key = (dataset, backbone, loss)
            author_result = author_dict.get(key)
            
            # Look up Imp% information (only for SLatK method)
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
                # If no corresponding author result found, mark as N/A
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

    def create_table3_comparison(self, best_records: List[ExperimentRecord], 
                                author_table3_dict: Dict[Tuple[str, str, str, str], float], 
                                table3_imp_dict: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Create Table 3 style comparison table (MF backbone NDCG@K results)
        Note: Only process datasets that have author table3 results (health, electronic)
        """
        # Filter MF backbone results on health/electronic datasets (only datasets with author table3 results)
        mf_records = [r for r in best_records 
                      if r.backbone == "MF" and r.dataset in ["health", "electronic"]]
        
        k_values = [5, 10, 20, 50, 75, 100]
        datasets = ["health", "electronic"]  # Only datasets with author table3 results
        
        # Calculate our improvement percentages
        our_table3_imp_dict, _ = self.improvement_calculator.calculate_our_table3_improvement(best_records)
        
        # Get all available loss functions
        available_losses = list(set([r.loss for r in mf_records]))
        available_losses.sort()
        
        table3_data = []
        
        for dataset in datasets:
            for loss in available_losses:
                row_data = {
                    "dataset": dataset.capitalize(),
                    "loss": loss,
                }
                
                # Find corresponding record
                record = next((r for r in mf_records 
                             if r.dataset == dataset and r.loss == loss), None)
                
                # Get Imp% information (only for SLatK method)
                author_imp_info = table3_imp_dict.get(dataset, ["N/A"] * 6)
                our_imp_info = our_table3_imp_dict.get(dataset, ["N/A"] * 6)
                
                if record:
                    # Add our results and author results
                    for i, k in enumerate(k_values):
                        ndcg_key = f"NDCG@{k}"
                        our_value = record.get_metric(ndcg_key)
                        if our_value is None:
                            our_value = "N/A"
                        
                        # Look up author results
                        author_key = (dataset, "MF", loss, ndcg_key)
                        author_value = author_table3_dict.get(author_key, "N/A")
                        
                        row_data[f"Our_NDCG@{k}"] = our_value
                        row_data[f"Author_NDCG@{k}"] = author_value
                        
                        # Calculate difference
                        if our_value != "N/A" and author_value != "N/A":
                            diff = our_value - author_value
                            row_data[f"Diff_NDCG@{k}"] = diff
                        else:
                            row_data[f"Diff_NDCG@{k}"] = "N/A"
                        
                        # Add author Imp% information (only for SLatK)
                        if loss == "SLatK" and i < len(author_imp_info):
                            row_data[f"Author_Imp%_NDCG@{k}"] = author_imp_info[i]
                        else:
                            row_data[f"Author_Imp%_NDCG@{k}"] = ""
                        
                        # Add our Imp% information (only for SLatK)
                        if loss == "SLatK" and i < len(our_imp_info):
                            row_data[f"Our_Imp%_NDCG@{k}"] = our_imp_info[i]
                        else:
                            row_data[f"Our_Imp%_NDCG@{k}"] = ""
                else:
                    # If no record found, fill with N/A
                    for k in k_values:
                        row_data[f"Our_NDCG@{k}"] = "N/A"
                        row_data[f"Author_NDCG@{k}"] = "N/A"
                        row_data[f"Diff_NDCG@{k}"] = "N/A"
                        row_data[f"Author_Imp%_NDCG@{k}"] = ""
                        row_data[f"Our_Imp%_NDCG@{k}"] = ""
                
                table3_data.append(row_data)
        
        return table3_data

    def create_table4_comparison(self, all_records: List[ExperimentRecord], 
                                author_table4_dict: Dict[Tuple[str, str, str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Create Table 4 style comparison table (Performance comparison of SL@K method with different K values)
        Note: This requires using all original records, not filtered optimal records, because we need SLatK experiment results with different K values
        Note: Only process datasets that have author table4 results (health, electronic)
        """
        # Filter SLatK method results on health/electronic datasets (only datasets with author table4 results)
        slatk_records = [r for r in all_records 
                         if r.loss == "SLatK" and r.dataset in ["health", "electronic"]]
        
        k_values = [5, 10, 20, 50, 75, 100]
        datasets = ["health", "electronic"]  # Only datasets with author table4 results
        
        table4_data = []
        
        # If no SLatK experiment results, we can still generate a comparison table based on author data
        if not slatk_records:
            print("[INFO] No SLatK experiment results found, showing only author data as reference")
        
        for dataset in datasets:
            # Find corresponding records
            dataset_records = [r for r in slatk_records if r.dataset == dataset]
            
            # Create a row of data for each NDCG@K
            for k in k_values:
                row_data = {
                    "dataset": dataset.capitalize(),
                    "metric": f"NDCG@{k}",
                }
                
                # Get our results - collect from SLatK records with different K parameter values
                our_values = {}
                if dataset_records:
                    for record in dataset_records:
                        ndcg_key = f"NDCG@{k}"
                        our_value = record.get_metric(ndcg_key)
                        if our_value is not None:
                            # Distinguish different SL@K variants based on K value in experiment parameters
                            param_k = record.param.get("k", record.param.get("K", 20))  # Default to 20
                            variant_name = f"SL@{param_k}"
                            
                            # If same variant has multiple records, choose the best result
                            if variant_name in our_values:
                                if our_value > our_values[variant_name]:
                                    our_values[variant_name] = our_value
                            else:
                                our_values[variant_name] = our_value
                
                # Get author results
                author_key = (dataset, "SLatK", f"NDCG@{k}")
                author_results = author_table4_dict.get(author_key, [])
                
                # Collect all variants
                all_variants = set()
                for author_result in author_results:
                    all_variants.add(author_result["method_variant"])
                for variant in our_values.keys():
                    all_variants.add(variant)
                
                # If no variants, skip this row
                if not all_variants:
                    continue
                
                # Sort variant names
                sorted_variants = sorted(all_variants, key=lambda x: (
                    999 if "∞" in x else int(x.split("@")[1]) if "@" in x and x.split("@")[1].isdigit() else 0
                ))
                
                for variant in sorted_variants:
                    # Get our results
                    our_value = our_values.get(variant, "N/A")
                    
                    # Get author results
                    author_value = "N/A"
                    for author_result in author_results:
                        if author_result["method_variant"] == variant:
                            author_value = author_result["value"]
                            break
                    
                    # Calculate difference
                    if our_value != "N/A" and author_value != "N/A":
                        diff = our_value - author_value
                    else:
                        diff = "N/A"
                    
                    row_data[f"Our_{variant}"] = our_value
                    row_data[f"Author_{variant}"] = author_value
                    row_data[f"Diff_{variant}"] = diff
                
                table4_data.append(row_data)
        
        return table4_data

    def create_extended_ndcg_table(self, best_records: List[ExperimentRecord], 
                                   backbone: str, extended_records: List[ExperimentRecord]) -> List[Dict[str, Any]]:
        """
        Create extended NDCG@K table for specified backbone (XSimGCL or LightGCN)
        Similar to Table 3 but for different backbone architectures
        Note: Only process datasets health, electronic (same as Table 3)
        Note: Uses extended_records which allows SLatK to use optimal K value (not restricted to K=20)
        """
        # Filter specified backbone results on health/electronic datasets
        backbone_records = [r for r in extended_records 
                           if r.backbone == backbone and r.dataset in ["health", "electronic"]]
        
        k_values = [5, 10, 20, 50, 75, 100]
        datasets = ["health", "electronic"]
        
        # Calculate improvement percentages for this backbone
        our_imp_dict = {}
        
        # Group by dataset
        for dataset in datasets:
            dataset_records = [r for r in backbone_records if r.dataset == dataset]
            
            # Find SLatK method results
            slatk_record = next((r for r in dataset_records if r.loss == "SLatK"), None)
            if not slatk_record:
                continue
            
            # Find other method results
            other_records = [r for r in dataset_records if r.loss != "SLatK"]
            if not other_records:
                continue
            
            # Calculate improvement percentage relative to best baseline for each K value
            improvements = []
            for k in k_values:
                ndcg_key = f"NDCG@{k}"
                slatk_value = slatk_record.get_metric(ndcg_key)
                
                if slatk_value is None or slatk_value == "N/A":
                    improvements.append("N/A")
                    continue
                
                # Find best baseline for this K value and corresponding method
                best_baseline_value = 0
                best_baseline_method = "N/A"
                for other_record in other_records:
                    other_value = other_record.get_metric(ndcg_key)
                    if other_value is not None and other_value != "N/A" and other_value > best_baseline_value:
                        best_baseline_value = other_value
                        best_baseline_method = other_record.loss
                
                if best_baseline_value > 0:
                    imp = (slatk_value - best_baseline_value) / best_baseline_value * 100
                    improvements.append(f"{imp:+.2f}% (vs {best_baseline_method})")
                else:
                    improvements.append("N/A")
            
            our_imp_dict[dataset] = improvements
        
        # Get all available loss functions
        available_losses = list(set([r.loss for r in backbone_records]))
        available_losses.sort()
        
        extended_table_data = []
        
        for dataset in datasets:
            for loss in available_losses:
                row_data = {
                    "dataset": dataset.capitalize(),
                    "backbone": backbone,
                    "loss": loss,
                }
                
                # Find corresponding record
                record = next((r for r in backbone_records 
                             if r.dataset == dataset and r.loss == loss), None)
                
                # Get our Imp% information (only for SLatK method)
                our_imp_info = our_imp_dict.get(dataset, ["N/A"] * 6)
                
                if record:
                    # Add our results
                    for i, k in enumerate(k_values):
                        ndcg_key = f"NDCG@{k}"
                        our_value = record.get_metric(ndcg_key)
                        if our_value is None:
                            our_value = "N/A"
                        
                        row_data[f"NDCG@{k}"] = our_value
                        
                        # Add our Imp% information (only for SLatK)
                        if loss == "SLatK" and i < len(our_imp_info):
                            row_data[f"Imp%_NDCG@{k}"] = our_imp_info[i]
                        else:
                            row_data[f"Imp%_NDCG@{k}"] = ""
                else:
                    # If no record found, fill with N/A
                    for k in k_values:
                        row_data[f"NDCG@{k}"] = "N/A"
                        row_data[f"Imp%_NDCG@{k}"] = ""
                
                extended_table_data.append(row_data)
        
        return extended_table_data

    def create_backbone_table4_analysis(self, all_records: List[ExperimentRecord], 
                                        backbone: str) -> List[Dict[str, Any]]:
        """
        Create Table 4 style analysis for specified backbone (Performance comparison of SL@K method with different K values)
        Similar to create_table4_comparison but for any backbone (not just MF with author comparison)
        Note: Only process datasets health, electronic (same as original Table 4)
        """
        # Filter SLatK method results for specified backbone on health/electronic datasets
        slatk_records = [r for r in all_records 
                         if r.loss == "SLatK" and r.backbone == backbone and r.dataset in ["health", "electronic"]]
        
        k_values = [5, 10, 20, 50, 75, 100]
        datasets = ["health", "electronic"]
        
        table4_data = []
        
        # If no SLatK experiment results, return empty
        if not slatk_records:
            print(f"[INFO] No SLatK experiment results found for {backbone} backbone")
            return table4_data
        
        for dataset in datasets:
            # Find corresponding records for this dataset and backbone
            dataset_records = [r for r in slatk_records if r.dataset == dataset]
            
            if not dataset_records:
                continue
                
            # Create a row of data for each NDCG@K
            for k in k_values:
                row_data = {
                    "dataset": dataset.capitalize(),
                    "backbone": backbone,
                    "metric": f"NDCG@{k}",
                }
                
                # Get our results - collect from SLatK records with different K parameter values
                our_values = {}
                for record in dataset_records:
                    ndcg_key = f"NDCG@{k}"
                    our_value = record.get_metric(ndcg_key)
                    if our_value is not None:
                        # Distinguish different SL@K variants based on K value in experiment parameters
                        k_param = record.param.get("k") or record.param.get("K")
                        if k_param is not None:
                            variant = f"SL@{k_param}"
                            our_values[variant] = our_value
                
                # If no variants found for this metric, skip this row
                if not our_values:
                    continue
                
                # Sort variant names by K value
                sorted_variants = sorted(our_values.keys(), key=lambda x: (
                    999 if "∞" in x else int(x.split("@")[1]) if "@" in x and x.split("@")[1].isdigit() else 0
                ))
                
                # Add results for each variant
                for variant in sorted_variants:
                    our_value = our_values[variant]
                    row_data[f"NDCG@{k}_{variant}"] = our_value
                
                table4_data.append(row_data)
        
        return table4_data


class MarkdownFormatter:
    """Markdown formatter"""
    
    def to_markdown_table(self, best_records: List[ExperimentRecord]) -> str:
        """Generate basic Markdown table"""
        record_dicts = [record.to_dict() for record in best_records]
        df = pd.DataFrame(record_dicts)
        # Sort by common paper order
        df = df.sort_values(["dataset", "backbone", "loss"])
        # Keep only columns of interest
        df = df[["dataset", "backbone", "loss", "Recall@20", "NDCG@20"]]
        # Set print precision
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        return df.to_markdown(index=False)

    def to_comparison_markdown_table(self, comparison_data: List[Dict[str, Any]]) -> str:
        """
        Convert comparison data to markdown table format, grouped by dataset+backbone, separated by horizontal lines
        """
        df = pd.DataFrame(comparison_data)
        # Sort by dataset, backbone, loss function
        df = df.sort_values(["dataset", "backbone", "loss"])
        
        # Handle difference columns, mark values <= 0.01 in red
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
        
        # Format other numeric columns
        def format_number(value):
            if value == "N/A":
                return value
            try:
                return f"{float(value):.4f}"
            except:
                return str(value)
        
        # Format Imp% columns, mark numbers in blue
        def format_imp(value):
            if value == "" or value == "N/A":
                return value
            try:
                return f'<span style="color:blue">{value}</span>'
            except:
                return str(value)
        
        # Apply formatting
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
        
        # Manually generate grouped markdown table
        markdown_lines = []
        
        # Header
        headers = ["dataset", "backbone", "loss", "Our_Recall@20", "Author_Recall@20", "Recall_Diff", 
                   "Our_NDCG@20", "Author_NDCG@20", "NDCG_Diff", "Author_Recall_Imp%", "Author_NDCG_Imp%", 
                   "Our_Recall_Imp%", "Our_NDCG_Imp%"]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "|" + "|".join([" --- " for _ in headers]) + "|"
        
        markdown_lines.append(header_line)
        markdown_lines.append(separator_line)
        
        # Group by dataset+backbone
        current_group = None
        for idx, row in df.iterrows():
            group_key = (row["dataset"], row["backbone"])
            
            # If it's a new group, add separator line (except for first group)
            if current_group is not None and current_group != group_key:
                # Add a separator line
                separator_row = "|" + "|".join([" --- " for _ in headers]) + "|"
                markdown_lines.append(separator_row)
            
            current_group = group_key
            
            # Build data row
            row_values = []
            for col in headers:
                row_values.append(str(row[col]))
            
            data_line = "| " + " | ".join(row_values) + " |"
            markdown_lines.append(data_line)
        
        return "\n".join(markdown_lines)

    def to_table3_markdown(self, table3_data: List[Dict[str, Any]]) -> str:
        """
        Convert Table 3 data to markdown format, grouped by dataset (Table 3 only includes MF backbone)
        """
        if not table3_data:
            return "# Table 3 Comparison Results\n\nNo relevant data found.\n"
        
        k_values = [5, 10, 20, 50, 75, 100]
        
        # Group by dataset
        grouped_data = defaultdict(list)
        for row in table3_data:
            key = row["dataset"]
            grouped_data[key].append(row)
        
        # Sort by key
        sorted_groups = sorted(grouped_data.items())
        
        markdown = "# Table 3: MF Backbone NDCG@K Comparison Results\n\n"
        
        # Create unified table header
        header = "| Dataset | Method |"
        for k in k_values:
            header += f" Our_NDCG@{k} | Author_NDCG@{k} | Diff | Author_Imp% | Our_Imp% |"
        header += "\n"
        
        # Create separator line
        separator = "|---------|--------|"
        for k in k_values:
            separator += "------------|---------------|------|-------------|-----------|"
        separator += "\n"
        
        markdown += header + separator
        
        # Add data rows, separated by groups
        first_group = True
        for dataset_name, group_data in sorted_groups:
            
            # If not first group, add separator line
            if not first_group:
                sep_line = "|---------|--------|"
                for k in k_values:
                    sep_line += "------------|---------------|------|-------------|-----------|"
                sep_line += "\n"
                markdown += sep_line
            first_group = False
            
            # Add data rows
            for row in group_data:
                line = f"| {dataset_name} | {row['loss']} |"
                for k in k_values:
                    our_val = row[f"Our_NDCG@{k}"]
                    author_val = row[f"Author_NDCG@{k}"]
                    diff_val = row[f"Diff_NDCG@{k}"]
                    author_imp_val = row[f"Author_Imp%_NDCG@{k}"]
                    our_imp_val = row[f"Our_Imp%_NDCG@{k}"]
                    
                    # Format numerical values
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
                    
                    # Format Imp%, mark numbers in blue
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
        
        markdown += "## Notes\n"
        markdown += "- Our_NDCG@K: Our experiment results\n"
        markdown += "- Author_NDCG@K: Author's experiment results\n"
        markdown += "- Diff: Difference (our results - author's results)\n"
        markdown += "- Author_Imp%: Author's improvement percentage of SL@K relative to best baseline (only shown for SLatK method)\n"
        markdown += "- Our_Imp%: Our reproduction improvement percentage of SL@K relative to best baseline (only shown for SLatK method)\n"
        markdown += "- Positive values (green) indicate our results are better, negative values (red) indicate author's results are better\n"
        markdown += "- Results with absolute difference ≤ 0.01 are not colored\n"
        
        return markdown

    def to_table4_markdown(self, table4_data: List[Dict[str, Any]]) -> str:
        """
        Convert Table 4 data to markdown format (Performance comparison of SL@K method with different K values)
        """
        if not table4_data:
            return "# Table 4 Comparison Results\n\nNo relevant data found.\n"
        
        # Group by dataset
        grouped_data = defaultdict(list)
        for row in table4_data:
            key = row["dataset"]
            grouped_data[key].append(row)
        
        # Sort by key
        sorted_groups = sorted(grouped_data.items())
        
        markdown = "# Table 4: Performance Comparison of SL@K Method with Different K Values\n\n"
        
        # Extract all SL@K variants from first row data
        if table4_data:
            first_row = table4_data[0]
            variants = []
            for key in first_row.keys():
                if key.startswith("Our_SL@") or key.startswith("Our_SL ("):
                    variant = key[4:]  # Remove "Our_" prefix
                    variants.append(variant)
            
            # Sort variants
            variants.sort(key=lambda x: (999 if "∞" in x else int(x.split("@")[1]) if "@" in x else 0))
            
            # Create table header
            header = "| Dataset | Metric |"
            for variant in variants:
                header += f" Our_{variant} | Author_{variant} | Diff |"
            header += "\n"
            
            # Create separator line
            separator = "|---------|--------|"
            for variant in variants:
                separator += "-----------|-------------|------|"
            separator += "\n"
            
            markdown += header + separator
            
            # Add data rows, separated by groups
            first_group = True
            for dataset_name, group_data in sorted_groups:
                
                # If not first group, add separator line
                if not first_group:
                    sep_line = "|---------|--------|"
                    for variant in variants:
                        sep_line += "-----------|-------------|------|"
                    sep_line += "\n"
                    markdown += sep_line
                first_group = False
                
                # Add data rows
                for row in group_data:
                    line = f"| {dataset_name} | {row['metric']} |"
                    for variant in variants:
                        our_val = row.get(f"Our_{variant}", "N/A")
                        author_val = row.get(f"Author_{variant}", "N/A")
                        diff_val = row.get(f"Diff_{variant}", "N/A")
                        
                        # Format numerical values
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
                        
                        line += f" {our_str} | {author_str} | {diff_str} |"
                    line += "\n"
                    markdown += line
        
        markdown += "\n## Notes\n"
        markdown += "- Our_SL@K: Our experiment results\n"
        markdown += "- Author_SL@K: Author's experiment results\n"
        markdown += "- Diff: Difference (our results - author's results)\n"
        markdown += "- Positive values (green) indicate our results are better, negative values (red) indicate author's results are better\n"
        markdown += "- Results with absolute difference ≤ 0.01 are not colored\n"
        markdown += "- SL@K indicates SL method trained with K value, SL (@∞) indicates infinite K value (standard SL)\n"
        
        return markdown

    def to_extended_ndcg_markdown(self, extended_data: List[Dict[str, Any]], backbone: str) -> str:
        """
        Convert extended NDCG@K data to markdown format (for XSimGCL or LightGCN backbone)
        """
        if not extended_data:
            return f"# Table 3 Extended: {backbone} Backbone NDCG@K Results\n\nNo relevant data found.\n"
        
        k_values = [5, 10, 20, 50, 75, 100]
        
        # Group by dataset
        grouped_data = defaultdict(list)
        for row in extended_data:
            key = row["dataset"]
            grouped_data[key].append(row)
        
        # Sort by key
        sorted_groups = sorted(grouped_data.items())
        
        markdown = f"# Table 3 Extended: {backbone} Backbone NDCG@K Results\n\n"
        markdown += f"This table shows NDCG@K results for {backbone} backbone on Health and Electronic datasets, "
        markdown += "extending the analysis beyond MF backbone to test generalizability.\n\n"
        
        # Create unified table header
        header = "| Dataset | Method |"
        for k in k_values:
            header += f" NDCG@{k} | Imp% |"
        header += "\n"
        
        # Create separator line
        separator = "|---------|--------|"
        for k in k_values:
            separator += "----------|------|"
        separator += "\n"
        
        markdown += header + separator
        
        # Add data rows, separated by groups
        first_group = True
        for dataset_name, group_data in sorted_groups:
            
            # If not first group, add separator line
            if not first_group:
                sep_line = "|---------|--------|"
                for k in k_values:
                    sep_line += "----------|------|"
                sep_line += "\n"
                markdown += sep_line
            first_group = False
            
            # Add data rows
            for row in group_data:
                line = f"| {dataset_name} | {row['loss']} |"
                for k in k_values:
                    our_val = row[f"NDCG@{k}"]
                    imp_val = row[f"Imp%_NDCG@{k}"]
                    
                    # Format numerical values
                    our_str = f"{our_val:.4f}" if our_val != "N/A" else "N/A"
                    
                    # Format Imp%, mark numbers in blue
                    if imp_val and imp_val != "N/A":
                        imp_str = f'<span style="color:blue">{imp_val}</span>'
                    else:
                        imp_str = ""
                    
                    line += f" {our_str} | {imp_str} |"
                line += "\n"
                markdown += line
        
        markdown += f"\n## Notes\n"
        markdown += f"- NDCG@K: Our experiment results using {backbone} backbone\n"
        markdown += f"- Imp%: Our improvement percentage of SL@K relative to best baseline for {backbone} backbone (only shown for SLatK method)\n"
        markdown += f"- This table extends Table 3 analysis to {backbone} backbone to test the generalizability of the SL@K method\n"
        markdown += f"- Only Health and Electronic datasets are included (same as original Table 3)\n"
        markdown += f"- Results help understand whether SL@K improvements are specific to MF backbone or generalizable to other architectures\n"
        
        return markdown

    def to_backbone_table4_markdown(self, table4_data: List[Dict[str, Any]], backbone: str) -> str:
        """
        Convert backbone-specific Table 4 data to markdown format (Performance analysis of SL@K method with different K values)
        """
        if not table4_data:
            return f"# Table 4 Extended: {backbone} Backbone SL@K Performance Analysis\n\nNo relevant data found.\n"
        
        # Group by dataset
        grouped_data = defaultdict(list)
        for row in table4_data:
            key = row["dataset"]
            grouped_data[key].append(row)
        
        # Sort by key
        sorted_groups = sorted(grouped_data.items())
        
        markdown = f"# Table 4 Extended: {backbone} Backbone SL@K Performance Analysis\n\n"
        markdown += f"This table shows the performance of SL@K method with different K values using {backbone} backbone "
        markdown += "on Health and Electronic datasets, extending the analysis beyond MF backbone.\n\n"
        
        # Extract all SL@K variants from data
        all_variants = set()
        for row in table4_data:
            for key in row.keys():
                if key.startswith("NDCG@") and "_SL@" in key:
                    # Extract variant name (e.g., "SL@5" from "NDCG@20_SL@5")
                    variant = key.split("_")[1]
                    all_variants.add(variant)
        
        if not all_variants:
            return f"# Table 4 Extended: {backbone} Backbone SL@K Performance Analysis\n\nNo SL@K variants found.\n"
        
        # Sort variants by K value
        sorted_variants = sorted(all_variants, key=lambda x: (
            999 if "∞" in x else int(x.split("@")[1]) if "@" in x and x.split("@")[1].isdigit() else 0
        ))
        
        # Create table header
        header = "| Dataset | Metric |"
        for variant in sorted_variants:
            header += f" {variant} |"
        header += "\n"
        
        # Create separator line
        separator = "|---------|--------|"
        for variant in sorted_variants:
            separator += "----------|"
        separator += "\n"
        
        markdown += header + separator
        
        # Add data rows, separated by groups
        first_group = True
        for dataset_name, group_data in sorted_groups:
            
            # If not first group, add separator line
            if not first_group:
                sep_line = "|---------|--------|"
                for variant in sorted_variants:
                    sep_line += "----------|"
                sep_line += "\n"
                markdown += sep_line
            first_group = False
            
            # Add data rows
            for row in group_data:
                line = f"| {dataset_name} | {row['metric']} |"
                for variant in sorted_variants:
                    value_key = f"{row['metric']}_{variant}"
                    value = row.get(value_key, "N/A")
                    
                    # Format numerical values
                    value_str = f"{value:.4f}" if value != "N/A" else "N/A"
                    line += f" {value_str} |"
                line += "\n"
                markdown += line
        
        markdown += f"\n## Notes\n"
        markdown += f"- SL@K: Results of SL@K method with different K values using {backbone} backbone\n"
        markdown += f"- This table extends Table 4 analysis to {backbone} backbone to test the generalizability across different K values\n"
        markdown += f"- Only Health and Electronic datasets are included (same as original Table 4)\n"
        markdown += f"- Results help understand how different K values affect performance on different backbone architectures\n"
        markdown += f"- SL@K indicates SL method trained with K value\n"
        
        return markdown


class ExperimentResultAnalyzer:
    """Experiment result analyzer main class"""
    
    def __init__(self):
        self.parser = ExperimentParser()
        self.selector = BestRecordSelector()
        self.author_loader = AuthorResultLoader()
        self.improvement_calculator = ImprovementCalculator()
        self.table_generator = ComparisonTableGenerator(self.improvement_calculator)
        self.formatter = MarkdownFormatter()
    
    def analyze_and_generate_reports(self, root_dir: str, output_dir: str):
        """
        Analyze experiment results and generate reports
        """
        # Ensure output folder exists
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define output file paths
        compare_csv = output_path / "table2_compare.csv"
        compare_md = output_path / "table2_compare.md"
        table3_csv = output_path / "table3_compare.csv"
        table3_md = output_path / "table3_compare.md"
        table4_csv = output_path / "table4_compare.csv"
        table4_md = output_path / "table4_compare.md"
        
        # Author result files remain in result_analysis directory
        author_results_path = pathlib.Path("result_analysis")
        author_table2_path = author_results_path / "author_table2.json"
        author_table3_path = author_results_path / "author_table3.json"
        author_table4_path = author_results_path / "author_table4.json"

        # Parse experiment records
        records = self.parser.gather_all_records(root_dir)
        print(f"Parsed {len(records)} trial records in total")
        
        # Select best records
        best = self.selector.pick_best(records)
        print(f"Summarized {len(best)} optimal records (using SL@20 for SLatK)")
        
        # Select best records for extended analysis (allows optimal K for SLatK)
        best_extended = self.selector.pick_best_for_table3_extended(records)
        print(f"Summarized {len(best_extended)} optimal records for extended analysis (using optimal SL@K)")

        # Generate Table2 comparison report (includes all datasets: health, electronic, gowalla, book)
        self._generate_table2_comparison(best, str(author_table2_path), compare_csv, compare_md)
        
        # Generate Table3 comparison report (only health, electronic)
        # Note: Table 3 uses optimal SL@K (not restricted to k=20) to match author's SL@K approach
        self._generate_table3_comparison(best_extended, str(author_table3_path), table3_csv, table3_md)
        
        # Generate Table4 comparison report (only health, electronic)
        self._generate_table4_comparison(records, str(author_table4_path), table4_csv, table4_md)

        # Generate extended NDCG@K comparison for XSimGCL and LightGCN backbones
        # (to test generalizability of results beyond MF backbone)
        xsimgcl_csv = output_path / "table3_xsimgcl_ndcg.csv"
        xsimgcl_md = output_path / "table3_xsimgcl_ndcg.md"
        lightgcn_csv = output_path / "table3_lightgcn_ndcg.csv"
        lightgcn_md = output_path / "table3_lightgcn_ndcg.md"
        
        self._generate_extended_ndcg_comparison(best, best_extended, "XSimGCL", xsimgcl_csv, xsimgcl_md)
        self._generate_extended_ndcg_comparison(best, best_extended, "LightGCN", lightgcn_csv, lightgcn_md)

        # Generate extended Table 4 analysis for XSimGCL and LightGCN backbones
        # (to test SL@K performance with different K values across different architectures)
        xsimgcl_table4_csv = output_path / "table4_xsimgcl_analysis.csv"
        xsimgcl_table4_md = output_path / "table4_xsimgcl_analysis.md"
        lightgcn_table4_csv = output_path / "table4_lightgcn_analysis.csv"
        lightgcn_table4_md = output_path / "table4_lightgcn_analysis.md"
        
        self._generate_backbone_table4_analysis(records, "XSimGCL", xsimgcl_table4_csv, xsimgcl_table4_md)
        self._generate_backbone_table4_analysis(records, "LightGCN", lightgcn_table4_csv, lightgcn_table4_md)

    def _generate_table2_comparison(self, best_records: List[ExperimentRecord], 
                                   author_table2_path: str, compare_csv: pathlib.Path, 
                                   compare_md: pathlib.Path):
        """Generate Table2 comparison report (includes all datasets: health, electronic, gowalla, book)"""
        # Load author results and generate comparison table
        author_dict, imp_dict = self.author_loader.load_author_results(author_table2_path)
        if author_dict:
            comparison_data = self.table_generator.create_comparison_table(best_records, author_dict, imp_dict)
            comparison_df = pd.DataFrame(comparison_data)
            
            # Save comparison results CSV
            comparison_df.to_csv(compare_csv, index=False, float_format="%.4f")
            print(f"Comparison results CSV written to {compare_csv}")
            
            # Generate and save comparison results Markdown
            comparison_md_content = self.formatter.to_comparison_markdown_table(comparison_data)
            with open(compare_md, "w", encoding="utf-8") as f:
                f.write("# Experiment Results Comparison\n\n")
                f.write("The following is a comparison between our experiment results and the author's results:\n\n")
                f.write(comparison_md_content)
                f.write("\n\n## Notes\n")
                f.write("- Our_Recall@20/Our_NDCG@20: Our experiment results\n")
                f.write("- Author_Recall@20/Author_NDCG@20: Author's experiment results\n")
                f.write("- Recall_Diff/NDCG_Diff: Difference (our results - author's results)\n")
                f.write("- Author_Recall_Imp%/Author_NDCG_Imp%: Author's improvement percentage of SL@K relative to best baseline (only shown for SLatK method)\n")
                f.write("- Our_Recall_Imp%/Our_NDCG_Imp%: Our reproduction improvement percentage of SL@K relative to best baseline (only shown for SLatK method)\n")
                f.write("- Positive values indicate our results are better, negative values indicate author's results are better\n")
                f.write("- Results with absolute difference ≤ 0.01 are not colored, differences > 0.01 are marked green, differences < -0.01 are marked red\n")
            print(f"Comparison results Markdown written to {compare_md}")
        else:
            print("[WARN] Unable to load author results, skipping comparison")

    def _generate_table3_comparison(self, best_records: List[ExperimentRecord], 
                                   author_table3_path: str, table3_csv: pathlib.Path, 
                                   table3_md: pathlib.Path):
        """Generate Table3 comparison report"""
        # Load author Table 3 results and generate comparison
        author_table3_dict, table3_imp_dict = self.author_loader.load_author_table3_results(author_table3_path)
        if author_table3_dict:
            table3_data = self.table_generator.create_table3_comparison(best_records, author_table3_dict, table3_imp_dict)
            
            # Save Table 3 comparison results CSV
            if table3_data:
                table3_df = pd.DataFrame(table3_data)
                table3_df.to_csv(table3_csv, index=False, float_format="%.4f")
                print(f"Table 3 comparison results CSV written to {table3_csv}")
            
            # Generate and save Table 3 comparison results Markdown
            table3_md_content = self.formatter.to_table3_markdown(table3_data)
            with open(table3_md, "w", encoding="utf-8") as f:
                f.write(table3_md_content)
            print(f"Table 3 comparison results Markdown written to {table3_md}")
        else:
            print("[WARN] Unable to load author Table 3 results, skipping Table 3 comparison")

    def _generate_table4_comparison(self, all_records: List[ExperimentRecord], 
                                   author_table4_path: str, table4_csv: pathlib.Path, 
                                   table4_md: pathlib.Path):
        """Generate Table4 comparison report"""
        # Load author Table 4 results and generate comparison
        author_table4_dict = self.author_loader.load_author_table4_results(author_table4_path)
        if author_table4_dict:
            table4_data = self.table_generator.create_table4_comparison(all_records, author_table4_dict)
            
            # Save Table 4 comparison results CSV
            if table4_data:
                table4_df = pd.DataFrame(table4_data)
                table4_df.to_csv(table4_csv, index=False, float_format="%.4f")
                print(f"Table 4 comparison results CSV written to {table4_csv}")
            
            # Generate and save Table 4 comparison results Markdown
            table4_md_content = self.formatter.to_table4_markdown(table4_data)
            with open(table4_md, "w", encoding="utf-8") as f:
                f.write(table4_md_content)
            print(f"Table 4 comparison results Markdown written to {table4_md}")
        else:
            print("[WARN] Unable to load author Table 4 results, skipping Table 4 comparison")

    def _generate_extended_ndcg_comparison(self, best_records: List[ExperimentRecord], 
                                          extended_records: List[ExperimentRecord], 
                                          backbone: str, extended_csv: pathlib.Path, 
                                          extended_md: pathlib.Path):
        """Generate extended NDCG@K comparison for specified backbone (XSimGCL or LightGCN)"""
        extended_data = self.table_generator.create_extended_ndcg_table(best_records, backbone, extended_records)
        
        # Save extended NDCG@K results CSV
        if extended_data:
            extended_df = pd.DataFrame(extended_data)
            extended_df.to_csv(extended_csv, index=False, float_format="%.4f")
            print(f"{backbone} backbone extended NDCG@K results CSV written to {extended_csv}")
        else:
            print(f"[WARN] No {backbone} backbone data found, skipping {backbone} extended table")
        
        # Generate and save extended NDCG@K results Markdown
        extended_md_content = self.formatter.to_extended_ndcg_markdown(extended_data, backbone)
        with open(extended_md, "w", encoding="utf-8") as f:
            f.write(extended_md_content)
        print(f"{backbone} backbone extended NDCG@K results Markdown written to {extended_md}")

    def _generate_backbone_table4_analysis(self, all_records: List[ExperimentRecord], 
                                          backbone: str, table4_csv: pathlib.Path, 
                                          table4_md: pathlib.Path):
        """Generate Table 4 style analysis for specified backbone (XSimGCL or LightGCN)"""
        table4_data = self.table_generator.create_backbone_table4_analysis(all_records, backbone)
        
        # Save Table 4 analysis results CSV
        if table4_data:
            table4_df = pd.DataFrame(table4_data)
            table4_df.to_csv(table4_csv, index=False, float_format="%.4f")
            print(f"{backbone} backbone Table 4 analysis CSV written to {table4_csv}")
        else:
            print(f"[WARN] No {backbone} backbone SL@K data found, skipping {backbone} Table 4 analysis")
        
        # Generate and save Table 4 analysis Markdown
        table4_md_content = self.formatter.to_backbone_table4_markdown(table4_data, backbone)
        with open(table4_md, "w", encoding="utf-8") as f:
            f.write(table4_md_content)
        print(f"{backbone} backbone Table 4 analysis Markdown written to {table4_md}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiment_results", help="Root directory containing json files")
    parser.add_argument("--output_dir", default="result_analysis/comparison", help="Output folder path")
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ExperimentResultAnalyzer()
    analyzer.analyze_and_generate_reports(args.root, args.output_dir)


if __name__ == "__main__":
    main()