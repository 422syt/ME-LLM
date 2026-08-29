#!/bin/bash
# 串行跑 scripts/repro/ 下全部 9 个训练脚本（单 GPU，不能并行）
export PATH="/d/anaconda/envs/basicTS:$PATH"
cd /e/ME-LLM || exit 1

SCRIPTS=(
  MELLM_ETTh1
  MELLM_ETTh2
  MELLM_ETTm1
  MELLM_ETTm2
  MELLM_ECL
  MELLM_Traffic
  MELLM_Weather
  MELLM_M4
  MELLM_ETTh1_ETTh2
)

for s in "${SCRIPTS[@]}"; do
  echo "===== [$s] START $(date '+%F %T') ====="
  bash "scripts/repro/${s}.sh" > "logs/${s}.log" 2>&1
  code=$?
  echo "===== [$s] DONE exit=$code $(date '+%F %T') ====="
done

echo "ALL_DONE"
