#!/bin/bash
# Example CLI Usage
python -u -m itemrec \
  --log=test.log \
  --save_dir=test_save \
  --seed=1234 \
  model \
    --emb_size=32 \
    --num_epochs=10 \
    MF \
  dataset \
    --data_path=IR-Benchmark-Dataset/data_iid/amazon2014-health \
    --batch_size=128 \
  optim \
    --lr=0.01 \
    BPR

# Example NNI commands to run
export HOME=/csproject/rwangcn/projects/IR-Benchmark
python run_nni.py \
  --model=LightGCN \
  --num_layers=2 \
  --dataset=gowalla \
  --optim=SLatK \
  --norm \
  --fold=5 \
  --num_epochs=200 \
  --port=50000

# Run the scheduler
python -u scheduler.py \
  --gpus 0,1,7 \
  --concurrency 3 \
  --max_trials_per_gpu 20 \
  --trial_concurrency 64 \
  --start_port 34080 \
  --check_interval 60 > experiment_logs/scheduler_book.log

# Cleanup commands
pkill -f "python run_nni.py"
pkill -f "python -u -m itemrec"
pkill -f "python scheduler.py"
rm -rf /project/rwangcn/projects/IR-Benchmark/src/nni_save/*
rm -rf /project/rwangcn/projects/IR-Benchmark/src/experiment_logs/*
rm -rf /project/rwangcn/projects/IR-Benchmark/nni-experiments/*
rm -rf /project/rwangcn/projects/IR-Benchmark/nni-experiments/.experiment


python -u test_memory.py \
  --gpus 6,7 \
  --concurrency 2 \
  --start_port 50000 \
  --wait_time 180 \
  --max_run_time 10 \
  --output_dir memory_test_results > memory_test_logs/test_memory.log


pkill -f "python run_nni.py"
pkill -f "python -u -m itemrec"
rm -rf /project/rwangcn/projects/IR-Benchmark/src/memory_test_logs
rm -rf /project/rwangcn/projects/IR-Benchmark/src/nni-experiments
rm -rf /project/rwangcn/projects/IR-Benchmark/src/memory_test_save
# rm -rf /project/rwangcn/projects/IR-Benchmark/src/memory_test_results/*

# Scheduler_plus.py
python -u scheduler_plus.py \
  --gpus 4,5,6,7 \
  --max_memory_per_gpu 23000 \
  --max_experiments_per_gpu 4 \
  --trial_concurrency 64 \
  --start_port 40000 \
  --check_interval 180 > experiment_logs/scheduler_plus.log

# Cleanup commands
pkill -f "python run_nni.py"
pkill -f "python -u -m itemrec"
rm -rf /project/rwangcn/projects/IR-Benchmark/src/nni_save/*
rm -rf /project/rwangcn/projects/IR-Benchmark/src/experiment_logs/*
rm -rf /project/rwangcn/projects/IR-Benchmark/nni-experiments/*
