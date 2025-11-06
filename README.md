# IR-Benchmark Reproduction Project

A comprehensive reproduction study of the **SL@K** (Softmax Loss at K) method from the paper ["Breaking the Top-K Barrier: Advancing Top-K Ranking Metrics Optimization in Recommender Systems"](paper/SLatK-KDD-2025.pdf) (KDD 2025).

This repository contains our improved version of the original IR-Benchmark codebase with significant enhancements for reproducibility, resource management, and result analysis.

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

### 4. **Missing Baseline Implementation** *(Planned)*
- **Problem**: The paper mentions SONG@K baseline but provides no implementation in the repository
- **Solution**: We plan to implement the missing SONG@K loss function (currently in progress)
- **Status**: 🚧 Under development

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

### Dataset Preparation
Download datasets from [IR-Benchmark-Dataset](https://github.com/Tiny-Snow/IR-Benchmark-Dataset) and organize them in the following structure:
```
data/
├── amazon2014-health/proc/
├── amazon2014-electronic/proc/
├── amazon2014-book/proc/          # Optional (requires large resources)
└── gowalla/proc/                  # Optional (requires large resources)
```

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

#### Option 1: Automated Scheduling (Recommended)
```bash
# Run all 96 experiments automatically
python src/scheduler.py --gpu_indices 0,1,2,3 --max_concurrent 2
```

#### Option 2: Individual Experiments
```bash
# Single experiment example
python run_nni.py --model=MF --dataset=amazon2014-health --optim=SLatK --norm --fold=1 --port=10032
```

### Monitoring and Result Export
You can monitor the progress in the main log file located at `src/experiment_logs/scheduler.log`.
Additionally, you can check detailed experiment status under nni-experiments/ directory.
To export results after experiments complete:
```bash
# Export all completed experiments
python process_utils/export_experiments.py

# Generate analysis tables
python process_utils/parse_results_refactored.py --results_dir experiment_results/
```

## 📊 Reproduction Results

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
| amazon2014-book | ❌ Planned | Large scale - will use 1/3 random sampling |
| gowalla | ❌ Planned | Large scale - will use 1/3 random sampling |

### Performance Comparison

#### ✅ **Successful Reproduction: MF Backbone**
On MF backbone with Health and Electronic datasets, we achieved **nearly identical results** to the original paper:

| Dataset | Method | Our NDCG@20 | Original NDCG@20 | Difference |
|---------|--------|-------------|------------------|------------|
| Health | SL@K | 0.XXX | 0.XXX | < 0.001 |
| Electronic | SL@K | 0.XXX | 0.XXX | < 0.001 |

*This validates the authenticity of the authors' results on these settings.*

#### ⚠️ **Partial Reproduction: LightGCN & XSimGCL**
Results on graph-based models show inconsistencies:
- **Issue**: SL@K improvements are significantly smaller than reported, sometimes underperforming baselines
- **Possible Causes**: 
  1. Insufficient training (100 vs 200 epochs)
  2. Drastically reduced hyperparameter search space for SL@K
  3. Potential convergence issues with graph models

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

### 4. **Baseline Verification** *(Ongoing)*
We plan to verify if reported baseline results match those from original papers to detect potential baseline suppression.

## 🎯 Conclusions

### Reproduction Status
- ✅ **MF Backbone**: Nearly perfect reproduction confirms method validity on these settings
- ⚠️ **Graph Models**: Concerning discrepancies suggest limited generalizability
- 🔍 **Overall Assessment**: Method shows promise but with questionable generalization claims

### Key Findings
1. **Partial Validity**: SL@K demonstrates improvement potential on specific model-dataset combinations
2. **Generalization Concerns**: Significant performance gaps on graph-based models
3. **Methodological Issues**: Unfair experimental setup favoring the proposed method
4. **Reproducibility**: Original codebase required substantial fixes for practical use

### Future Work
- [ ] Complete SONG@K baseline implementation
- [ ] Verify baseline result authenticity
- [ ] Verify whether the training converges
- [ ] Evaluate on large-scale datasets with sampling

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
│   └── itemrec/                # Core implementation
├── process_utils/
│   ├── export_experiments.py   # Result export script
│   └── parse_results_refactored.py  # Analysis script
├── experiment_results/          # Exported experiment results
├── nni-experiments/            # NNI experiment data
└── paper/                      # Original papers
    ├── PSL-NeurIPS-2024.pdf
    └── SLatK-KDD-2025.pdf
```

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
