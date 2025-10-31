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
  --gpus 4,5,6,7 \
  --concurrency 4 \
  --trial_concurrency 32 \
  --max_trials_per_gpu 16 \
  --start_port 10000 \
  --check_interval 180 > experiment_logs/scheduler.log

# Cleanup commands
pkill -f "python run_nni.py"
pkill -f "python -u -m itemrec"
pkill -f "python scheduler.py"
rm -rf /project/rwangcn/projects/IR-Benchmark/src/nni_save/*
rm -rf /project/rwangcn/projects/IR-Benchmark/src/experiment_logs/*
rm -rf /project/rwangcn/projects/IR-Benchmark/nni-experiments