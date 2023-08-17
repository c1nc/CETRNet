# Multivariate
# 96
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm1.csv \
--model_id cetrnet_ETTm1_96_96 \
--model CETRNet \
--dataset ETTm1 \
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
--train_epochs 20 > logs/ettm1/cetrnet_ettm1_pl96_sl336_bs16_nl3.log

# 192
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm1.csv \
--model_id cetrnet_ETTm1_192_192 \
--model CETRNet \
--dataset ETTm1 \
--features M \
--seq_len 336 \
--pred_len 192 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 128 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20  > logs/ettm1/cetrnet_ettm1_pl192_sl336_bs128_nl3.log

# 336
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm1.csv \
--model_id cetrnet_ETTm1_336_336 \
--model CETRNet \
--dataset ETTm1 \
--features M \
--seq_len 336 \
--pred_len 336 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 128 \
--learning_rate 0.01 \
--num_levels 1 \
--dropout 0.05 \
--train_epochs 20  > logs/ettm1/cetrnet_ettm1_pl336_sl336_bs128_nl1.log
 
# 720
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm1.csv \
--model_id cetrnet_ETTm1_720_720 \
--model CETRNet \
--dataset ETTm1 \
--features M \
--seq_len 736 \
--pred_len 720 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 64 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20  > logs/ettm1/cetrnet_ettm1_pl720_sl736_bs64_nl3.log

# Univariate
# 96
nohup python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm1.csv \
--model_id cetrnet_ETTm1_96_96 \
--model CETRNet \
--dataset ETTm1 \
--features S \
--seq_len 336 \
--pred_len 96 \
--enc_in 1 \
--des 'Exp' \
--itr 1 \
--batch_size 16 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20  > logs/ettm1/S_cetrnet_ettm1_pl096_sl336_bs16_nl3.log &

# 192
nohup python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm1.csv \
--model_id cetrnet_ETTm1_192_192 \
--model CETRNet \
--dataset ETTm1 \
--features S \
--seq_len 336 \
--pred_len 192 \
--enc_in 1 \
--des 'Exp' \
--itr 1 \
--batch_size 128 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm1/S_cetrnet_ettm1_pl192_sl336_bs128_nl3.log &

# 336
nohup python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm1.csv \
--model_id cetrnet_ETTm1_336_336 \
--model CETRNet \
--dataset ETTm1 \
--features S \
--seq_len 336 \
--pred_len 336 \
--enc_in 1 \
--des 'Exp' \
--itr 1 \
--batch_size 128 \
--learning_rate 0.01 \
--num_levels 1 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm1/S_cetrnet_ettm1_pl336_sl336_bs128_nl3.log &
 
# 720
nohup python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ETT \
--data_path ETTm1.csv \
--model_id cetrnet_ETTm1_720_720 \
--model CETRNet \
--dataset ETTm1 \
--features S \
--seq_len 736 \
--pred_len 720 \
--enc_in 1 \
--des 'Exp' \
--itr 1 \
--batch_size 64 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ettm1/S_cetrnet_ettm1_pl720_sl736_bs64_nl3.log &