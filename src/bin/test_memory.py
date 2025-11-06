import os
import sys
import time
import subprocess
import argparse
import signal
import json
import csv
from collections import deque
from typing import List, Dict, Any, Optional
from nni.experiment import Experiment
from tqdm import tqdm
from itemrec.port_handle import find_available_port
import pynvml as nvml


class GPUMemoryMonitor:
    """GPU显存监控类"""
    
    def __init__(self):
        nvml.nvmlInit()
        
    def get_gpu_memory_usage(self, gpu_id: int) -> Dict[str, int]:
        """获取指定GPU的显存使用情况
        
        Args:
            gpu_id: GPU索引
            
        Returns:
            包含总显存、已用显存、空闲显存的字典 (单位: MB)
        """
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                'total_mb': memory_info.total // (1024 * 1024),
                'used_mb': memory_info.used // (1024 * 1024),
                'free_mb': memory_info.free // (1024 * 1024)
            }
        except Exception as e:
            print(f"Error getting GPU {gpu_id} memory info: {e}")
            return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0}
    
    def get_all_gpus_memory(self, gpu_ids: List[int]) -> Dict[int, Dict[str, int]]:
        """获取所有指定GPU的显存使用情况"""
        return {gpu_id: self.get_gpu_memory_usage(gpu_id) for gpu_id in gpu_ids}


class MemoryTestScheduler:
    """显存测试调度器"""
    
    def __init__(self, args):
        self.args = args
        self.memory_monitor = GPUMemoryMonitor()
        self.running_experiments: Dict[int, Dict[str, Any]] = {}
        self.memory_results: List[Dict[str, Any]] = []
        self.baseline_memory: Dict[int, int] = {}  # 基准显存占用 (GPU ID -> 基准显存MB)
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.cleanup)
        
    def get_experiment_configs(self) -> List[Dict[str, Any]]:
        """生成所有实验配置"""
        MODELS = ("MF", "LightGCN", "XSimGCL")
        LOSSES = ("BPR", "GuidedRec", "LLPAUC", "Softmax", "AdvInfoNCE", "BSL", "PSL", "SLatK")
        DATASETS = ("amazon2014-health", "amazon2014-electronic", "amazon2014-book", "gowalla")
        
        configs = []
        for dataset in DATASETS:
            for model in MODELS:
                for optim in LOSSES:
                    config = {
                        "model": model,
                        "dataset": dataset,
                        "optim": optim,
                        "norm": True,
                        "fold": 1,
                        "num_epochs": 200,
                    }
                    configs.append(config)
        return configs
    
    def record_baseline_memory(self):
        """记录基准显存占用量（实验开始前）"""
        print("Recording baseline GPU memory usage...")
        baseline = self.memory_monitor.get_all_gpus_memory(self.args.gpus)
        for gpu_id, memory_info in baseline.items():
            self.baseline_memory[gpu_id] = memory_info['used_mb']
            print(f"  GPU {gpu_id}: Baseline memory = {memory_info['used_mb']} MB")
        print()
    
    def launch_experiment(self, config: Dict[str, Any], gpu_id: int, port: int) -> Optional[subprocess.Popen]:
        """启动单个实验
        
        Args:
            config: 实验配置
            gpu_id: 分配的GPU ID
            port: NNI端口
            
        Returns:
            启动的进程对象，失败时返回None
        """
        print(f"--- Launching Memory Test Experiment ---")
        print(f"Config: {config['dataset']}-{config['model']}-{config['optim']}")
        print(f"GPU: {gpu_id}, Port: {port}")
        
        cmd = [
            sys.executable, "run_nni.py",
            "--model", config['model'],
            "--dataset", config['dataset'],
            "--optim", config['optim'],
            "--fold", str(config['fold']),
            "--num_epochs", str(config['num_epochs']),
            "--port", str(port),
            "--gpu_index", str(gpu_id),
            "--trial_concurrency", "1",  # 固定为1来测试单个trial的显存占用
            "--max_trials_per_gpu", "1",  # 固定为1
        ]
        
        if config.get('norm', False):
            cmd.append("--norm")
        if 'num_layers' in config:
            cmd.extend(["--num_layers", str(config['num_layers'])])
        
        try:
            # 创建日志目录和文件
            log_dir = "memory_test_logs"
            os.makedirs(log_dir, exist_ok=True)
            log_filename = f"{log_dir}/{config['dataset']}_{config['model']}_{config['optim']}.log"
            log_file = open(log_filename, 'w')
            
            # 启动进程
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            
            # 记录实验信息
            self.running_experiments[port] = {
                "process": process,
                "gpu_id": gpu_id,
                "config": config,
                "log_file": log_file,
                "start_time": time.time(),
                "memory_recorded": False
            }
            
            return process
            
        except Exception as e:
            print(f"ERROR: Failed to launch experiment: {e}")
            return None
    
    def check_and_record_memory(self, port: int, experiment_info: Dict[str, Any]):
        """检查并记录实验的显存占用"""
        if experiment_info['memory_recorded']:
            return
            
        # 等待实验启动并稳定运行
        elapsed = time.time() - experiment_info['start_time']
        if elapsed < self.args.wait_time:
            return
            
        gpu_id = experiment_info['gpu_id']
        config = experiment_info['config']
        
        # 记录显存使用情况
        print(f"--- Recording Memory for {config['model']}-{config['dataset']}-{config['optim']} ---")
        
        current_memory = self.memory_monitor.get_gpu_memory_usage(gpu_id)
        current_memory_mb = current_memory['used_mb']
        baseline_used = self.baseline_memory.get(gpu_id, 0)
        experiment_memory_usage = current_memory_mb - baseline_used
        
        print(f"Current memory usage: {current_memory_mb} MB")
        
        # 记录结果
        result = {
            'dataset': config['dataset'],
            'model': config['model'],
            'optim': config['optim'],
            'gpu_id': gpu_id,
            'port': port,
            'baseline_memory_mb': baseline_used,
            'current_memory_mb': current_memory_mb,
            'experiment_memory_mb': experiment_memory_usage,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.memory_results.append(result)
        experiment_info['memory_recorded'] = True
        
        print(f"GPU {gpu_id}: Baseline = {baseline_used} MB, Current = {current_memory_mb} MB")
        print(f"Experiment Memory Usage = {experiment_memory_usage:.1f} MB")
        print("----------------------------------------")
    
    def stop_experiment(self, port: int):
        """停止指定端口的实验"""
        experiment_info = self.running_experiments.get(port)
        if not experiment_info:
            return
            
        try:
            # 首先尝试通过NNI API停止
            experiment = Experiment.connect(port)
            experiment.stop()
            print(f"Stopped experiment on port {port} via NNI API")
        except Exception as e:
            print(f"Could not stop via NNI API (port {port}): {e}")
            # 直接终止进程
            try:
                experiment_info['process'].terminate()
                print(f"Terminated process for port {port}")
            except Exception as pe:
                print(f"Could not terminate process for port {port}: {pe}")
        
        # 使用 nnictl stop --port 确保彻底停止指定端口的NNI实验
        try:
            result = subprocess.run(['nnictl', 'stop', '--port', str(port)], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"Successfully ran 'nnictl stop --port {port}'")
            else:
                print(f"nnictl stop --port {port} returned non-zero exit code: {result.returncode}")
                if result.stderr:
                    print(f"stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Warning: 'nnictl stop --port {port}' timed out")
        except Exception as e:
            print(f"Error running 'nnictl stop --port {port}': {e}")
        
        # 清理资源
        try:
            experiment_info['log_file'].close()
        except:
            pass
        
        del self.running_experiments[port]
    
    def save_results(self):
        """保存测试结果到文件"""
        # 确保输出目录存在
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        # 保存为JSON格式
        json_filename = os.path.join(self.args.output_dir, f"memory_test_results_{timestamp}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.memory_results, f, indent=2, ensure_ascii=False)
        print(f"Memory test results saved to: {json_filename}")
        
        # 保存为CSV格式，便于分析
        csv_filename = os.path.join(self.args.output_dir, f"memory_test_results_{timestamp}.csv")
        if self.memory_results:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.memory_results[0].keys())
                writer.writeheader()
                writer.writerows(self.memory_results)
            print(f"Memory test results saved to: {csv_filename}")
            
            # 打印摘要统计
            self.print_memory_summary()
    
    def print_memory_summary(self):
        """打印显存使用摘要"""
        print("\n--- Memory Usage Summary ---")
        
        # 按模型分组统计
        model_stats = {}
        for result in self.memory_results:
            model = result['model']
            memory_mb = result['experiment_memory_mb']
            
            if model not in model_stats:
                model_stats[model] = []
            model_stats[model].append(memory_mb)
        
        for model, memory_list in model_stats.items():
            avg_memory = sum(memory_list) / len(memory_list)
            max_memory = max(memory_list)
            min_memory = min(memory_list)
            print(f"{model}: Avg={avg_memory:.1f}MB, Max={max_memory:.1f}MB, Min={min_memory:.1f}MB")
        
        # 按损失函数分组统计
        print("\n--- Memory Usage by Loss Function ---")
        loss_stats = {}
        for result in self.memory_results:
            loss = result['optim']
            memory_mb = result['experiment_memory_mb']
            
            if loss not in loss_stats:
                loss_stats[loss] = []
            loss_stats[loss].append(memory_mb)
        
        for loss, memory_list in loss_stats.items():
            avg_memory = sum(memory_list) / len(memory_list)
            print(f"{loss}: Avg={avg_memory:.1f}MB")
        
        print("----------------------------------------")
    
    def cleanup(self, signum=None, frame=None):
        """清理函数"""
        print("\nCleaning up all running experiments...")
        
        for port in list(self.running_experiments.keys()):
            self.stop_experiment(port)
        
        # 保存已收集的结果
        if self.memory_results:
            self.save_results()
        
        print("Cleanup complete.")
        sys.exit(0)
    
    def run(self):
        """运行主循环"""
        print("--- Memory Test Scheduler Started ---")
        
        # 记录基准显存
        self.record_baseline_memory()
        
        # 获取所有实验配置
        configs = self.get_experiment_configs()
        configs_queue = deque(configs)
        
        print(f"Total configurations to test: {len(configs)}")
        print(f"Available GPUs: {self.args.gpus}")
        print(f"Max concurrent experiments: {self.args.concurrency}")
        print(f"Wait time before recording memory: {self.args.wait_time}s")
        print("----------------------------------------")
        
        next_port = self.args.start_port
        pbar = tqdm(total=len(configs), desc="Memory Testing Progress", unit="config")
        
        while configs_queue or self.running_experiments:
            
            # 检查已完成的实验
            for port, experiment_info in list(self.running_experiments.items()):
                # 记录显存使用情况（如果还没记录的话）
                self.check_and_record_memory(port, experiment_info)
                
                # 检查实验是否完成或需要停止
                return_code = experiment_info['process'].poll()
                elapsed = time.time() - experiment_info['start_time']
                
                # 如果已经记录了显存且运行时间超过阈值，或者进程已结束，则停止实验
                if (experiment_info['memory_recorded'] and elapsed > self.args.max_run_time) or return_code is not None:
                    config = experiment_info['config']
                    
                    if return_code is not None:
                        status = "COMPLETED" if return_code == 0 else f"FAILED (Exit: {return_code})"
                    else:
                        status = "STOPPED (Memory recorded)"
                    
                    print(f"Experiment finished: {config['model']}-{config['dataset']}-{config['optim']} - {status}")
                    
                    self.stop_experiment(port)
                    pbar.update(1)
            
            # 启动新实验
            while len(self.running_experiments) < self.args.concurrency and configs_queue:
                config = configs_queue.popleft()
                
                # 选择GPU（轮询方式）
                gpu_id = self.args.gpus[len(self.running_experiments) % len(self.args.gpus)]
                
                # 获取可用端口
                port = find_available_port(next_port)
                next_port = port + 1
                
                # 启动实验
                process = self.launch_experiment(config, gpu_id, port)
                if process is None:
                    pbar.update(1)  # 跳过失败的实验
                
                time.sleep(2)  # 给实验启动一些时间
            
            # 等待一段时间再检查
            if self.running_experiments:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running: {len(self.running_experiments)}, Pending: {len(configs_queue)}")
                time.sleep(self.args.check_interval)
        
        # 保存最终结果
        print("\n--- All Memory Tests Completed ---")
        self.save_results()
        pbar.close()


def main():
    parser = argparse.ArgumentParser(description="GPU Memory Usage Test for NNI Experiments")
    parser.add_argument('--gpus', 
                       type=lambda s: [int(item) for item in s.split(',')], 
                       required=True, 
                       help='Comma-separated list of GPU indices to use (e.g., "0,1,2,3")')
    parser.add_argument('--concurrency', 
                       type=int, 
                       default=4, 
                       help='Maximum number of concurrent experiments for memory testing')
    parser.add_argument('--start_port', 
                       type=int, 
                       default=60000, 
                       help='Starting port number for NNI experiments')
    parser.add_argument('--wait_time', 
                       type=int, 
                       default=60, 
                       help='Seconds to wait before recording memory usage after experiment start')
    parser.add_argument('--max_run_time', 
                       type=int, 
                       default=10, 
                       help='Maximum time to let each experiment run after memory is recorded (seconds)')
    parser.add_argument('--check_interval', 
                       type=int, 
                       default=120, 
                       help='Seconds between status checks')
    parser.add_argument('--output_dir', 
                       type=str, 
                       default='memory_test_results', 
                       help='Directory to save output files')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.concurrency > len(args.gpus):
        print(f"Warning: concurrency ({args.concurrency}) is greater than available GPUs ({len(args.gpus)})")
    
    scheduler = MemoryTestScheduler(args)
    scheduler.run()


if __name__ == "__main__":
    main()


'''
python test_memory.py \
--gpus 4,5,6,7 \
--concurrency 4 \
--start_port 50000 \
--wait_time 60 \
--max_run_time 10 \
--output_dir memory_test_results
'''