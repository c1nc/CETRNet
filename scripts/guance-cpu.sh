# 96
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/GUANCE \
--data_path GUANCE-CPU.csv \
--model_id cetrnet_gccpu_96_96 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 336 \
--pred_len 96 \
--enc_in 5 \
--des 'Exp' \
--itr 1 \
--batch_size 32 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/cpu/cetrnet_cpu_pl96_sl336_bs32_nl3.log

# 192
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/GUANCE \
--data_path GUANCE-CPU.csv \
--model_id cetrnet_gccpu_192_192 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 336 \
--pred_len 192 \
--enc_in 5 \
--des 'Exp' \
--itr 1 \
--batch_size 32 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/cpu/cetrnet_cpu_pl192_sl336_bs32_nl3.log

# 336
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/GUANCE \
--data_path GUANCE-CPU.csv \
--model_id cetrnet_gccpu_336_336 \
--model CETRNet \
--dataset custom \
--features M \
--seq_len 336 \
--pred_len 336 \
--enc_in 5 \
--des 'Exp' \
--itr 1 \
--batch_size 16 \
--learning_rate 0.01 \
--num_levels 3 \
--dropout 0.05 \
--train_epochs 20 > logs/cpu/cetrnet_cpu_pl336_sl336_bs16_nl3.log

# 720
python3 -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/GUANCE \
--data_path GUANCE-CPU.csv \
--model_id cetrnet_gccpu_336_336 \
--model CETRNet \
--dataset custom \
--features M \
--patience 4 \
--seq_len 96 \
--pred_len 720 \
--enc_in 5 \
--des 'Exp' \
--itr 1 \
--batch_size 128 \
--learning_rate 0.01 \
--num_levels 2 \
--dropout 0.05 \
--train_epochs 20 > logs/cpu/cetrnet_cpu_pl720_sl96_bs128_nl2.log