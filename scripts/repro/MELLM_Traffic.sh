model_name=MELLM
train_epochs=10
learning_rate=0.01
llama_layers=12

batch_size=24
d_model=16
d_ff=32

comment='MELLM-Traffic'

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Traffic/ \
  --data_path traffic.csv \
  --model_id traffic_512_96 \
  --model $model_name \
  --data Traffic \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate 0.01 \
  --llm_model BERT \
  --llm_dim 768 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --use_amp \
  --model_comment $comment
