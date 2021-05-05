python ./STGAT/train_gan_ver2.py \
  --dataset_name "trajectory_combined_cls" \
  --delim tab \
  --pred_len 12 \
  --noise_dim 16 \
  --noise_type gaussian \
  --traj_lstm_input_size 2 \
  --traj_lstm_hidden_size 32 \
  --cls_embedding_dim 8 \
  --heads "4,1" \
  --hidden-units 16 \
  --graph_network_out_dims 32 \
  --graph_lstm_hidden_size 32 \
  --mlp_dim 512 \
  --dropout 0.3 \
  --alpha 0.2 \
  --batch_size 64 \
  --batch_norm 1 \
  --num_epochs 200 \
  --best_k 20 \
  --gpu_num 0 \
  --g_lr 1e-3 \
  --d_lr 1e-3 \
  --ckpt_interval 10 \
  --val_interval 5 \
  --print_every 100 \
  --clipping_threshold_g 1.5 \
  --clipping_threshold_d 1.5 \
  --verbose