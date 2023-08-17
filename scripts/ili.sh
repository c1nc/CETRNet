# 24
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path national_illness.csv \
--model_id cetrnet_ili_96_24 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 96 \
--pred_len 24 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 16 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ili/cetrnet_ili_pl24_sl96_bs16_nl3.log

# 36
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path national_illness.csv \
--model_id cetrnet_ili_96_24 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 96 \
--pred_len 36 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 16 \
--learning_rate 0.01 \
--num_levels 4 \
--dropout 0.05 \
--train_epochs 20 > logs/ili/cetrnet_ili_pl36_sl96_bs16_nl4.log

# 48
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path national_illness.csv \
--model_id cetrnet_ili_96_24 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 96 \
--pred_len 48 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 32 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/ili/cetrnet_ili_pl48_sl96_bs32_nl3.log

# 60
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/Others \
--data_path national_illness.csv \
--model_id cetrnet_ili_96_24 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 96 \
--pred_len 60 \
--enc_in 7 \
--des 'Exp' \
--itr 1 \
--batch_size 32 \
--learning_rate 0.01 \
--num_levels 4 \
--dropout 0.05 \
--train_epochs 20 > logs/ili/cetrnet_ili_pl60_sl96_bs32_nl4.log