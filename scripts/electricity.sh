# 96
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path electricity.csv \
--model_id cetrnet_ele_96_96 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 336 \
--pred_len 96 \
--enc_in 321 \
--des 'Exp' \
--itr 1 \
--batch_size 32 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20  > logs/eletricity/cetrnet_eletricity_pl96_sl336_bs32_nl3.log &

# 192
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path electricity.csv \
--model_id cetrnet_ele_192_192 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 720 \
--pred_len 192 \
--enc_in 321 \
--des 'Exp' \
--itr 1 \
--batch_size 32 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/eletricity/cetrnet_eletricity_pl192_sl720_bs32_nl3.log &

# 336
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path electricity.csv \
--model_id cetrnet_ele_336_336 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 720 \
--pred_len 336 \
--enc_in 321 \
--des 'Exp' \
--itr 1 \
--batch_size 16 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/eletricity/cetrnet_eletricity_pl336_sl720_bs16_nl3.log &

# 720
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path electricity.csv \
--model_id cetrnet_ele_1440_720 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 1440 \
--pred_len 720 \
--enc_in 321 \
--des 'Exp' \
--itr 1 \
--batch_size 16 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 15 > logs/eletricity/cetrnet_eletricity_pl720_sl1440_bs16_nl3.log &