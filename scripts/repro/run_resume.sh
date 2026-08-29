#!/bin/bash
# 续跑：跳过已完成的 ETTh1/ETTh2/ETTm1，只跑 Traffic/Weather 2 个数据集（单 GPU 串行）
export PATH="/d/anaconda/envs/basicTS:$PATH"
cd /e/ME-LLM || exit 1

SCRIPTS=(
  MELLM_Traffic
  MELLM_Weather
)

for s in "${SCRIPTS[@]}"; do
  echo "===== [$s] START $(date '+%F %T') ====="
  bash "scripts/repro/${s}.sh" > "logs/${s}.log" 2>&1
  code=$?
  echo "===== [$s] DONE exit=$code $(date '+%F %T') ====="
done

echo "ALL_DONE"
