#!/bin/bash
# 等 Weather 完成后，自动跑 Traffic（d_model=16 的 channel_num bug 已修复）
export PATH="/d/anaconda/envs/basicTS:$PATH"
cd /e/ME-LLM || exit 1

RESULT="results/long_term_forecast_weather_512_96_MELLM_Weather_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_Exp_0-MELLM-Weather.json"
echo "[watcher] 等待 Weather 完成..."
while [ ! -f "$RESULT" ]; do sleep 60; done
echo "[watcher] $(date '+%F %T') Weather 完成，开始跑 Traffic（已修复 d_model=16）"
bash scripts/repro/MELLM_Traffic.sh > logs/MELLM_Traffic.log 2>&1
echo "[watcher] Traffic exit=$?"
echo "[watcher] ALL_DONE"
