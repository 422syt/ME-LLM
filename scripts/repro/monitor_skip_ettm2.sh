#!/bin/bash
# 等 ETTm1 完成后，停掉当前总控（跳过 ETTm2），然后直接跑 Traffic + Weather
export PATH="/d/anaconda/envs/basicTS:$PATH"
cd /e/ME-LLM || exit 1

RESULT="results/long_term_forecast_ETTm1_512_96_MELLM_ETTm1_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Exp_0-MELLM-ETTm1.json"
echo "[monitor] 等待 ETTm1 完成..."
while [ ! -f "$RESULT" ]; do sleep 60; done
echo "[monitor] $(date '+%F %T') ETTm1 完成，停止总控（跳过 ETTm2）"
taskkill //F //T //PID 26472 2>&1
sleep 5
echo "[monitor] 开始跑 Traffic"
bash scripts/repro/MELLM_Traffic.sh > logs/MELLM_Traffic.log 2>&1
echo "[monitor] Traffic exit=$?"
echo "[monitor] 开始跑 Weather"
bash scripts/repro/MELLM_Weather.sh > logs/MELLM_Weather.log 2>&1
echo "[monitor] Weather exit=$?"
echo "[monitor] ALL_DONE"
