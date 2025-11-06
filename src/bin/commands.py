import itertools
import json

memory_test_path = "src/memory_test_results/memory_test.json"

def generate_all_commands():
    """生成所有需要运行的NNI命令"""
    
    # 定义所有实验配置
    MODELS = ("MF", "LightGCN", "XSimGCL")
    LOSSES = ("BPR", "GuidedRec", "LLPAUC", "Softmax", "AdvInfoNCE", "BSL", "PSL", "SLatK")
    DATASETS = ("amazon2014-health", "amazon2014-electronic", "amazon2014-book", "gowalla")
    
    commands = []
    port = 50000
    
    for dataset, model, optim in itertools.product(DATASETS, MODELS, LOSSES):
        # 构建命令，留空GPU和并发数让你自己填
        cmd = f"python run_nni.py --model {model} --dataset {dataset} --optim {optim} --fold 1 --num_epochs 100 --norm --gpu_index  --max_trials_per_gpu  --trial_concurrency 256 --port {port} > experiment_logs/{dataset}_{model}_{optim}.log 2>&1 &"
        
        commands.append({
            'command': cmd,
            'port': port,
            'dataset': dataset,
            'model': model,
            'optim': optim
        })
        port += 1
    
    return commands

def save_commands_to_file(commands):
    """将所有命令保存到一个文件中"""
    
    # with open('all_commands.txt', 'w') as f:
    #     f.write("# 所有实验命令 - 共{}个实验\n".format(len(commands)))
    #     f.write("# 使用方法: 将每个命令中的 '--gpu_index ' 替换为GPU编号，如 '--gpu_index 4'\n")
    #     f.write("#          将每个命令中的 '--trial_concurrency ' 替换为并发数，如 '--trial_concurrency 16'\n")
    #     f.write("# 示例: python run_nni.py --model MF --dataset amazon2014-health --optim BPR --fold 1 --num_epochs 200 --norm --gpu_index 4 --trial_concurrency 16 --port 40000\n\n")
        
    #     for i, cmd_info in enumerate(commands):
    #         f.write("# 实验 {}: {} - {} - {}\n".format(
    #             i+1, cmd_info['dataset'], cmd_info['model'], cmd_info['optim']
    #         ))
    #         f.write(cmd_info['command'] + "\n\n")
    
    with open(memory_test_path, 'r') as f:
        memory_data = json.load(f)

    def find_maxTrial(dataset, model, optim):
        for entry in memory_data:
            if (entry['dataset'] == dataset and 
                entry['model'] == model and 
                entry['optim'] == optim):
                return entry.get('maxTrial', None)
        return None

    with open('commands_only.txt', 'w') as f:
        for cmd_info in commands:
            max_trial = find_maxTrial(cmd_info['dataset'], cmd_info['model'], cmd_info['optim'])
            if max_trial is not None:
                f.write(f"MaxTrial: {max_trial}\n")
            f.write(cmd_info['command'] + "\n\n")

def print_summary(commands):
    """打印命令摘要"""
    print(f"\n=== 命令生成完成 ===")
    print(f"总实验数: {len(commands)}")
    
    # 统计信息
    datasets = set(cmd['dataset'] for cmd in commands)
    models = set(cmd['model'] for cmd in commands)
    optims = set(cmd['optim'] for cmd in commands)
    
    print(f"数据集 ({len(datasets)}): {', '.join(datasets)}")
    print(f"模型 ({len(models)}): {', '.join(models)}") 
    print(f"优化器 ({len(optims)}): {', '.join(optims)}")
    
    print(f"\n生成的文件:")
    print(f"- all_commands.txt: 包含所有命令和详细注释")
    print(f"- commands_only.txt: 只包含命令，没有注释")
    
    print("\n端口号范围: 40000 - {}".format(40000 + len(commands) - 1))
    
    print(f"\n使用建议:")
    print(f"1. 查看 all_commands.txt 了解所有命令")
    print(f"2. 用文本编辑器批量替换 '--gpu_index ' 为 '--gpu_index 4' (或其他GPU编号)")
    print(f"3. 用文本编辑器批量替换 '--trial_concurrency ' 为 '--trial_concurrency 16' (或其他并发数)")
    print(f"4. 复制命令到终端执行，或使用以下方法批量执行:")
    print(f"   # 方法1: 逐行执行")
    print(f"   while read line; do eval $line; done < commands_only.txt")
    print(f"   # 方法2: 使用GNU parallel并行执行")
    print(f"   cat commands_only.txt | parallel -j4")

if __name__ == "__main__":
    # 生成所有命令
    commands = generate_all_commands()
    
    # 保存到文件
    save_commands_to_file(commands)
    
    # 打印摘要
    print_summary(commands)