# 96
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path traffic.csv \
--model_id cetrnet_tra_336_96 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 336 \
--pred_len 96 \
--enc_in 862 \
--des 'Exp' \
--itr 1 \
--batch_size 8 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/traffic/cetrnet_traffic_pl96_sl336_bs8_nl3.log

# 192
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path traffic.csv \
--model_id cetrnet_tra_336_192 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 720 \
--pred_len 192 \
--enc_in 862  \
--des 'Exp' \
--itr 1 \
--batch_size 64 \
--learning_rate 0.01 \
--num_levels 2 \
--dropout 0.05 \
--train_epochs 20 > logs/traffic/cetrnet_traffic_pl192_sl720_bs64_nl2.log

# 336
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path traffic.csv \
--model_id cetrnet_tra_336_336 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 720 \
--pred_len 336 \
--enc_in 862  \
--des 'Exp' \
--itr 1 \
--batch_size 8 \
--learning_rate 0.01 \
--num_levels 2 \
--dropout 0.05 \
--train_epochs 20 > logs/traffic/cetrnet_traffic_pl336_sl720_bs8_nl2.log

# 720
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path traffic.csv \
--model_id cetrnet_tra_336_720 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 720 \
--pred_len 720 \
--enc_in 862  \
--des 'Exp' \
--itr 1 \
--batch_size 8 \
--learning_rate 0.01 \
--num_levels 2 \
--dropout 0.05 \
--train_epochs 20 > logs/traffic/cetrnet_traffic_pl720_sl720_bs8_nl2.log