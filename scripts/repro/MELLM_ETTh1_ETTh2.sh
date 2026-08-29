model_name=MELLM
train_epochs=10
learning_rate=0.01
llama_layers=12

batch_size=24
d_model=32
d_ff=128

comment='MELLM-ETTh1_ETTh2'

python run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path_pretrain ETTh1/ETTh1.csv \
  --data_path ETTh2/ETTh2.csv \
  --model_id ETTh1_ETTh2_512_96 \
  --model $model_name \
  --data_pretrain ETTh1 \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model BERT \
  --llm_dim 768 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --use_amp \
  --model_comment $comment
