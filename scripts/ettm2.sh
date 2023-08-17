# Multivariate
# 96
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm2.csv \
--model_id cetrnet_ETTm2_96_96 \
--model CETRNet \
--dataset ETTm2 \
--features M \
--seq_len 336 \
--pred_len 96 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 16 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20  > logs/ettm2/cetrnet_ettm2_pl96_sl336_bs16_nl3.log

# 192
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm2.csv \
--model_id cetrnet_ETTm2_192_192 \
--model CETRNet \
--dataset ETTm2 \
--features M \
--seq_len 336 \
--pred_len 192 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 32 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm2/cetrnet_ettm2_pl192_sl336_bs32_nl3.log

# 336
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm2.csv \
--model_id cetrnet_ETTm2_336_336 \
--model CETRNet \
--dataset ETTm2 \
--features M \
--seq_len 720 \
--pred_len 336 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 64 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm2/cetrnet_ettm2_pl336_sl720_bs64_nl3.log

# 720
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm2.csv \
--model_id cetrnet_ETTm2_720_720 \
--model CETRNet \
--dataset ETTm2 \
--features M \
--seq_len 720 \
--pred_len 720 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 128 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm2/cetrnet_ettm2_pl720_sl720_bs128_nl3.log

# Univariate
# 96
nohup python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm2.csv \
--model_id cetrnet_ETTm2_96_96 \
--model CETRNet \
--dataset ETTm2 \
--features S \
--seq_len 336 \
--pred_len 96 \
--enc_in 1 \
--des 'Exp' \
--itr 1 \
--batch_size 128 \
--learning_rate 0.01 \
--num_levels 1 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm2/S_cetrnet_ettm2_pl96_sl336_bs128_nl1.log &

# 192
nohup python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm2.csv \
--model_id cetrnet_ETTm2_192_192 \
--model CETRNet \
--dataset ETTm2 \
--features S \
--seq_len 336 \
--pred_len 192 \
--enc_in 1 \
--des 'Exp' \
--itr 1 \
--batch_size 32 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm2/S_cetrnet_ettm2_pl192_sl336_bs32_nl3.log &

# 336
nohup python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm2.csv \
--model_id cetrnet_ETTm2_336_336 \
--model CETRNet \
--dataset ETTm2 \
--features S \
--seq_len 336 \
--pred_len 336 \
--enc_in 1 \
--des 'Exp' \
--itr 1 \
--batch_size 64 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm2/S_cetrnet_ettm2_pl336_sl336_bs64_nl3.log &

# 720
nohup python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm2.csv \
--model_id cetrnet_ETTm2_720_720 \
--model CETRNet \
--dataset ETTm2 \
--features S \
--seq_len 336 \
--pred_len 720 \
--enc_in 1 \
--des 'Exp' \
--itr 1 \
--batch_size 64 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm2/S_cetrnet_ettm2_pl720_sl336_bs64_nl3.log &