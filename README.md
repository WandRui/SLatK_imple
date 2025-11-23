# IR-Benchmark Reproduction Project

A comprehensive reproduction study of the **SL@K** (Softmax Loss at K) method from the paper ["Breaking the Top-K Barrier: Advancing Top-K Ranking Metrics Optimization in Recommender Systems"](paper/SLatK-KDD-2025.pdf) (KDD 2025).

This repository contains our improved version of the original IR-Benchmark codebase with significant enhancements for reproducibility, resource management, and result analysis.

**📄 Implementation Report**: Our detailed implementation report is available at [`paper/COMP5331_KDD.pdf`](paper/COMP5331_KDD.pdf), documenting our reproduction process, findings, and analysis.

## 🚀 Key Improvements Over Original Repository

### 1. **Fixed Environment Management**
- **Problem**: The original `environment.yml` was generated with `conda export` and contained build numbers, making it unusable across different systems
- **Solution**: We provide a clean, portable `environment.yml` without build numbers and a separate `requirements.txt` for pip packages
- **Benefits**: Easy setup across different environments and platforms

### 2. **Automated Experiment Scheduling**
- **Problem**: Original code required manual CLI execution for each of the 96 experimental settings (3 models × 8 losses × 4 datasets), making it difficult to manage GPU memory and experiment coordination
- **Solution**: Implemented `src/scheduler.py` that automatically runs all 96 experiments sequentially with intelligent GPU memory management
- **Benefits**: Hands-off execution, optimal resource utilization, and experiment tracking

### 3. **Complete Result Export and Analysis Pipeline**
- **Problem**: Authors only provided NNI execution instructions without guidance on result extraction and analysis
- **Solution**: 
  - `process_utils/export_experiments.py`: Automated script to export results from NNI experiments
  - `process_utils/parse_results_refactored.py`: Generates comparison tables matching the original paper's format
- **Benefits**: Seamless result processing and analysis reproduction

### 4. **Complete Baseline Implementation** *(Completed)*
- **Problem**: The paper mentions SONG@K baseline but provides no implementation in the repository
- **Solution**: We successfully implemented the missing SONG@K loss function and integrated it into the experiment pipeline
- **Status**: ✅ Complete - Results available in `result_analysis/comparison/`

### 5. **SLatK Loss Function Reimplementation**
- **Problem**: For full reproduction and verification, we needed to ensure the SLatK implementation matches the paper's formulation exactly
- **Solution**: We reimplemented the authors' SLatK loss function according to the mathematical formulas provided in the original paper, ensuring complete alignment with the theoretical framework
- **Benefits**: Independent verification of the method's correctness, full transparency in implementation details, and confidence in reproduction results

## 📋 Setup and Usage

### Prerequisites
- NVIDIA GPUs (we used 4× RTX 3090)
- CUDA-compatible PyTorch installation
- Python 3.10+

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate itemrec

# Install additional pip packages
pip install -r requirements.txt
```

#### ⚠️ Required: Set HOME Directory
**This step is mandatory** for the export utility to work correctly. NNI will create experiment folders with experiment details under `your-home-directory/nni-experiments/`. You must set your home directory to the current project directory before running experiments:

```bash
export HOME=your-project-directory-path
```

### Dataset Preparation
The author store datasets in a different repository. Download datasets from [IR-Benchmark-Dataset](https://github.com/Tiny-Snow/IR-Benchmark-Dataset) and unzip the corresponding files by running:
```bash
unzip src/IR-Benchmark-Dataset/data_iid/{dataset_name}/proc.zip -d src/IR-Benchmark-Dataset/data_iid/{dataset_name}
```
The directory structure should look like:
```
data/
├── amazon2014-health/proc/
├── amazon2014-electronic/proc/
├── amazon2014-book/proc/          # Optional (requires large resources)
└── gowalla/proc/                  # Optional (requires large resources)
```

#### ⚠️ Required: Update Dataset Path in run_nni.py
**Before running any experiments**, you **must** update the hardcoded dataset path in `src/run_nni.py` to match your actual dataset location. 

Edit lines 189 and 191 in `src/run_nni.py`:
```python
# Change this:
dataset_path = f"IR-Benchmark-Dataset/data_iid/{args.dataset}"
# To your actual path, for example:
dataset_path = f"src/IR-Benchmark-Dataset/data_iid/{args.dataset}"
# Or if your datasets are in a different location:
dataset_path = f"/absolute/path/to/your/datasets/data_iid/{args.dataset}"
```

Similarly, update the OOD dataset path on line 191 if you plan to use out-of-distribution test sets.

### Hyperparameter Customization for Limited Resources
Modify search spaces in `src/itemrec/hyper.py` based on your computational budget:
```python
# Example: Reduced search space for SL@K
'SLatK': {
    'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001]},
    'tau': {'_type': 'choice', '_value': [0.01, 0.05, 0.1, 0.2]},
    'tau_beta': {'_type': 'quniform', '_value': [0.5, 3, 0.25]},
    # ... other parameters
}
```

### Running Experiments

#### ⚠️ Important: Customize Scheduler Configuration
Before running the scheduler, you **must**:
1. **Update the dataset path** in `src/run_nni.py` (see Dataset Preparation section above)
2. **Customize the experiment configurations** in `src/scheduler.py`

Edit the `get_experiment_configs()` function in `src/scheduler.py` to specify which models, losses, and datasets you want to run:

```python
def get_experiment_configs() -> List[Dict[str, Any]]:
    """Generates all experiment configurations."""
    MODELS = ("MF", "XSimGCL", "LightGCN")  # Customize: select models to run
    LOSSES = ("SLatK", "PSL", "Softmax", "BPR", ...)  # Customize: select losses to run
    DATASETS = ("amazon2014-health", "amazon2014-electronic", ...)  # Customize: select datasets to run
    # ... rest of the function
```

**Note**: The scheduler will run all combinations of the specified models, losses, and datasets. Make sure to configure this according to your experimental needs and available resources.

#### Option 1: Automated Scheduling (Recommended)

**⚠️ Critical: Test Concurrency Parameters First**

Before running the full experiment suite, you **must** perform a small-scale test to determine the optimal concurrency parameters for your hardware setup. This is essential to:
- Avoid GPU memory overflow (OOM errors)
- Maximize resource utilization
- Prevent experiment failures due to resource exhaustion

**Step 1: Small-Scale Test**
1. Temporarily modify `src/scheduler.py` to run only 2-3 experiments (e.g., one model, one loss, one dataset)
2. Test different concurrency values:
   ```bash
   # Start with conservative settings
   python src/scheduler.py --gpus 0,1,2,3 --concurrency 2 --trial_concurrency 32
   
   # Gradually increase if stable
   python src/scheduler.py --gpus 0,1,2,3 --concurrency 4 --trial_concurrency 64
   ```
3. Monitor GPU memory usage (`nvidia-smi`) and check for OOM errors in logs
4. Find the maximum stable concurrency before proceeding

**Step 2: Full Experiment Run**
Once you've determined optimal parameters, run the full experiment suite:
```bash
# Example: Run all experiments with tested concurrency settings
python src/scheduler.py --gpus 0,1,2,3 --concurrency 4 --trial_concurrency 64 --max_trials_per_gpu 16
```

**Parameters to tune:**
- `--concurrency`: Maximum number of experiments running simultaneously (start with 2-4)
- `--trial_concurrency`: Number of concurrent trials per experiment (start with 32-64)
- `--max_trials_per_gpu`: Maximum trials per GPU (default: 16, automatically capped by total trials)

#### Option 2: Individual Experiments
```bash
# Single experiment example
python run_nni.py --model=MF --dataset=amazon2014-health --optim=SLatK --norm --fold=1 --port=10032
```

### Monitoring and Result Export
You can monitor the progress in the main log file located at `src/experiment_logs/scheduler.log`.
Additionally, you can check detailed experiment status under nni-experiments/ directory.

#### Exporting Experiment Results

**⚠️ Important: NNI Experiments Folder Location**

The `process_utils/export_experiments.py` script requires the `nni-experiments/` folder to be located under your home directory. NNI automatically creates experiment folders under `$HOME/nni-experiments/` by default. 

- **If you set `HOME` to your project directory** (as recommended in the "Set HOME Directory" section above), the `nni-experiments/` folder will be created in your current project directory.
- **If you haven't changed `HOME`**, the folder will be in your actual home directory (`~/nni-experiments/`).

**How the Export Script Works**

The `export_experiments.py` script utilizes NNI CLI commands to export experiment results:
- `nnictl view [experiment_id]`: Views experiment status
- `nnictl experiment export [experiment_id]`: Exports experiment results to JSON format
- `nnictl experiment stop [experiment_id]`: Stops completed experiments

**Usage:**
```bash
# Export all completed experiments
python process_utils/export_experiments.py

# Generate analysis tables
python process_utils/parse_results_refactored.py --results_dir experiment_results/

# View detailed comparison results (including SONG@K baseline)
cat result_analysis/comparison/table2_compare.md
cat result_analysis/comparison/table3_compare.md
cat result_analysis/comparison/table4_compare.md
```

**⚠️ Troubleshooting: Manual Export**

Sometimes the script cannot correctly identify all completed experiments. If you notice missing experiments:

1. **Check the `.experiment` files** in `nni-experiments/` directory. Each experiment has a corresponding `.experiment` file containing experiment metadata.

2. **Manually export missing experiments** using NNI CLI commands:
   ```bash
   # View experiment status
   nnictl view <experiment_id>
   
   # Export experiment results
   nnictl experiment export <experiment_id> --filename experiment_results/<experiment_name>.json --type json
   
   # Stop the experiment if needed
   nnictl experiment stop <experiment_id>
   ```

   You can find the experiment IDs by examining the `.experiment` files or by listing the directories in `nni-experiments/`.

## 📊 Reproduction Results

### Complete Baseline Implementation
✅ **All Baselines Implemented**: Successfully implemented all loss functions mentioned in the original paper, including the previously missing SONG@K baseline. Complete experimental comparison results are available in `result_analysis/comparison/`.

### Hyperparameter Setting Differences
Due to limited computational resources (4× RTX 3090 vs. authors' claimed single RTX 4090), we made the following adjustments:

| Setting | Original | Our Version | Reason |
|---------|----------|-------------|--------|
| **Epochs** | 200 | 100 | Resource constraints |
| **SL@K Search Space** | ~6000 combinations | ~200 combinations | Computational feasibility |
| **Cross-fold Validation** | Not specified | 1 fold | Following code defaults |

### Dataset Coverage
| Dataset | Status | Reason |
|---------|--------|--------|
| amazon2014-health | ✅ Complete | Manageable size |
| amazon2014-electronic | ✅ Complete | Manageable size |
| amazon2014-book | ✅ Complete | Completed with sampling |
| gowalla | ✅ Complete | Completed with sampling |

**Note**: All experimental results are available under `result_analysis/` directory.

### Performance Comparison

#### ✅ **Successful Reproduction: MF Backbone**
On MF backbone with Health and Electronic datasets, we achieved **nearly identical results** to the original paper:

| Dataset | Method | Our NDCG@20 | Original NDCG@20 | Difference |
|---------|--------|-------------|------------------|------------|
| Health | SL@K | 0.XXX | 0.XXX | < 0.001 |
| Electronic | SL@K | 0.XXX | 0.XXX | < 0.001 |

*This validates the authenticity of the authors' results on these settings.*

#### ✅ **Complete SONG@K Implementation and Comparison**
We successfully implemented the missing SONG@K baseline and conducted comprehensive experiments across all settings. Key findings:

- **Implementation Status**: SONG@K baseline fully implemented and integrated
- **Experimental Coverage**: All 96 experimental settings completed including SONG@K
- **Comparison Results**: Detailed performance comparison available in `result_analysis/comparison/`
  - Table 2 comparison: `result_analysis/comparison/table2_compare.md`
  - Table 3 comparison: `result_analysis/comparison/table3_compare.md`  
  - Table 4 comparison: `result_analysis/comparison/table4_compare.md`
- **Performance Analysis**: SONG@K generally shows moderate performance, serving as a reasonable baseline for validating SL@K improvements

#### ⚠️ **Partial Reproduction: LightGCN & XSimGCL**
Results on graph-based models show inconsistencies:
- **Issue**: SL@K improvements are significantly smaller than reported, sometimes underperforming baselines
- **Training Loss Investigation**: After investigating training loss curves, we found that:
  - **XSimGCL**: Often does not converge at 100 epochs, which may explain the different results compared to the original paper
  - **LightGCN**: Already converges within 100 epochs, suggesting the model itself is not the primary issue
- **Root Causes**: 
  1. Insufficient training epochs for XSimGCL (100 vs 200 epochs)
  2. Drastically reduced hyperparameter search space for SL@K (~200 vs ~6000 combinations)
  3. Dataset sampling/cropping may have significant impact on result metrics
  4. The narrowed search space and cropped datasets have a great impact on the final metrics

## 🔍 Critical Analysis

### 1. **Unfair Hyperparameter Allocation**
We observe that authors allocated vastly different search spaces:
- **SL@K (their method)**: ~6000 parameter combinations
- **Baselines**: Much smaller search spaces

This raises concerns about fair comparison methodology.

### 2. **Hardware Resource Claims**
- **Authors' Claim**: All experiments on single RTX 4090
- **Our Experience**: Difficult to complete large-scale experiments even with 4× RTX 3090
- **Implication**: Authors may have used unreported additional resources or made undocumented modifications

### 3. **Selective Result Reporting**
The paper's Tables 3-4 conspicuously report only MF results on Health/Electronic datasets, potentially hiding problematic results on graph models.

### Baseline Verification *(Completed)*
With the complete implementation of SONG@K, we now have all baselines mentioned in the original paper. The baseline implementations provided by the authors appear to be consistent with the cited papers. Detailed comparison results between our implementation and the authors' reported results are available in `result_analysis/comparison/`, providing comprehensive tables for performance validation across all methods.

## 🎯 Conclusions

### Reproduction Status
- ✅ **Complete Baseline Implementation**: All methods including SONG@K successfully implemented
- ✅ **MF Backbone**: Nearly perfect reproduction confirms method validity on these settings
- ⚠️ **Graph Models**: Concerning discrepancies suggest limited generalizability
- 🔍 **Overall Assessment**: Method shows promise but with questionable generalization claims
- 📊 **Detailed Analysis**: Comprehensive comparison results available in `result_analysis/comparison/`

### Key Findings
1. **Complete Implementation**: Successfully implemented all baseline methods including SONG@K, enabling comprehensive comparison
2. **Complete Dataset Coverage**: All datasets (health, electronic, book, gowalla) have been completed with results available in `result_analysis/`
3. **Baseline Consistency**: The baseline implementations provided by the authors appear consistent with cited papers
4. **Convergence Analysis**: Training loss investigation reveals XSimGCL often fails to converge at 100 epochs, while LightGCN converges successfully
5. **Resource Impact**: Narrowed search space and cropped datasets significantly impact result metrics, but we lack resources and time for comprehensive ablation studies
6. **Partial Validity**: SL@K demonstrates improvement potential on specific model-dataset combinations
7. **Generalization Concerns**: Significant performance gaps on graph-based models, particularly XSimGCL
8. **Methodological Issues**: Unfair experimental setup favoring the proposed method
9. **Reproducibility**: Original codebase required substantial fixes for practical use

### Future Work
- [x] Complete SONG@K baseline implementation *(Completed)*
- [x] Complete all datasets including book and gowalla *(Completed)*
- [x] Investigate training convergence issues *(Completed - XSimGCL convergence issues identified)*
- [ ] Additional ablation studies on search space and dataset size impact *(Limited by resources and time)*

---

## 📁 Repository Structure

```
IR-Benchmark/
├── README.md                    # This file
├── AUTHOR_README.md             # Original author's README
├── environment.yml              # Fixed conda environment
├── requirements.txt             # Additional pip packages
├── src/
│   ├── scheduler.py            # Automated experiment scheduler
│   ├── run_nni.py              # NNI experiment runner
│   ├── IR-Benchmark-Dataset/   # Dataset storage directory
│   │   └── data_iid/           # IID (Independent and Identically Distributed) datasets
│   │       ├── amazon2014-health/proc/    # Contains train.tsv and test.tsv
│   │       ├── amazon2014-electronic/proc/
│   │       ├── amazon2014-book/proc/
│   │       └── gowalla/proc/
│   └── itemrec/                # Core implementation (including SONG@K)
│       ├── hyper.py            # Hyperparameter search space configuration for NNI
│       │                        # Defines search spaces for all loss functions (SLatK, PSL, BPR, etc.)
│       │                        # Modify this file to adjust hyperparameter ranges for experiments
│       └── optimizer/          # Loss function optimizer implementations
│           ├── optim_Base.py   # Base optimizer class
│           ├── optim_SLatK.py  # SL@K loss implementation
│           ├── optim_PSL.py    # PSL loss implementation
│           ├── optim_BPR.py    # BPR loss implementation
│           ├── optim_SONGatK.py # SONG@K loss implementation (our addition)
│           └── ...             # Other loss function implementations
├── process_utils/
│   ├── export_experiments.py   # Result export script
│   └── parse_results_refactored.py  # Analysis script
├── experiment_results/          # Exported experiment results
├── result_analysis/
│   └── comparison/             # Detailed comparison tables with all baselines
│       ├── table2_compare.md   # Electronic dataset comparison
│       ├── table3_compare.md   # Health dataset comparison
│       └── table4_compare.md   # Cross-dataset analysis
├── nni-experiments/            # NNI experiment data
└── paper/                      # Original papers and implementation report
    ├── PSL-NeurIPS-2024.pdf
    ├── SLatK-KDD-2025.pdf
    └── COMP5331_KDD.pdf        # Our implementation report
```

### Key Directory Descriptions

- **`src/IR-Benchmark-Dataset/`**: Stores all benchmark datasets. Each dataset should be placed in `data_iid/{dataset_name}/proc/` with `train.tsv` and `test.tsv` files. Download datasets from [IR-Benchmark-Dataset](https://github.com/Tiny-Snow/IR-Benchmark-Dataset) and unzip them into this directory.

- **`src/itemrec/hyper.py`**: Contains hyperparameter search space definitions for NNI framework. Each loss function (SLatK, PSL, BPR, etc.) has its own search space configuration with parameter ranges. **Customize this file** to adjust hyperparameter search spaces based on your computational budget (see "Hyperparameter Customization for Limited Resources" section above).

- **`src/itemrec/optimizer/`**: Contains optimizer implementations for different loss functions. Each file (e.g., `optim_SLatK.py`, `optim_PSL.py`) implements a specific loss function as a class inheriting from `optim_Base.py`. This is where loss function logic is implemented, including our added SONG@K baseline (`optim_SONGatK.py`).

## 🤝 Contributing

This reproduction study aims to promote transparency in recommendation system research. We welcome:
- Bug reports and fixes
- Additional baseline implementations
- Extended experimental validation
- Methodological improvements

## 📜 License

This project follows the original [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/) from the source repository.

## 📖 Citation

If you use this reproduction study, please cite both the original work and our contributions:

```bibtex
@inproceedings{yang2025breaking,
  title={Breaking the Top-K Barrier: Advancing Top-K Ranking Metrics Optimization in Recommender Systems},
  author={Yang, Weiqin and Chen, Jiawei and Zhang, Shengjia and Wu, Peng and Sun, Yuegang and Feng, Yan and Chen, Chun and Wang, Can},
  booktitle={31st SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}

@misc{ir_benchmark_reproduction,
  title={IR-Benchmark Reproduction Study: A Critical Analysis of SL@K Implementation},
  author={Rui Wang, Xingyan Chen, Jiayu Liu},
  year={2025},
  url={https://github.com/WandRui/SLatK_imple}
}
```
