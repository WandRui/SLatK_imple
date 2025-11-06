import json
from itemrec.hyper import get_search_space
from run_nni import get_total_trials

with open("memory_test_results/memory_test.json", "r") as f:
    data = json.load(f)

results = []
for entry in data:
    optim = entry["optim"]
    search_space = get_search_space(optim)
    total_trials = get_total_trials(search_space)
    entry["maxTrial"] = total_trials
    results.append(entry)

print(len(results))

MODELS = ("MF", "LightGCN", "XSimGCL")
LOSSES = ("BPR", "GuidedRec", "LLPAUC", "Softmax", "AdvInfoNCE", "BSL", "PSL", "SLatK")
DATASETS = ("amazon2014-health", "amazon2014-electronic", "amazon2014-book", "gowalla")

# Check which combinations are missing
existing_combinations = set((entry["model"], entry["dataset"], entry["optim"]) for entry in results)
missing_combinations = []
for model in MODELS:
    for dataset in DATASETS:
        for optim in LOSSES:
            if (model, dataset, optim) not in existing_combinations:
                missing_combinations.append((model, dataset, optim))
print("Missing combinations:")
for combo in missing_combinations:
    print(combo)

with open("memory_test_results/memory_test.json", "w") as f:
    json.dump(results, f, indent=4)