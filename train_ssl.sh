resume_path=null
num_nodes=1
epoch=32
batch=4
exp=smartpretrain_bz${batch}_epoch${epoch}

python train_qcnet_ssl.py \
     --root /path/to/av2 \
     --resume_path ${resume_path} \
     --num_nodes ${num_nodes} \
     --train_batch_size ${batch} --val_batch_size ${batch} --test_batch_size ${batch} \
     --exp_name ${exp} \
     --max_epochs ${epoch} --T_max ${epoch} \
     --lr 1.5e-3 --lr_init 1e-4 --lr_last 1e-6 \
     --weight_decay 1e-1 --weight_decay_init 1e-2 \
     --contra_t 0.07 --contra_momentum 0.996 --warmup 15 \
     --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 \
     --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 \
     --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 \
     --load_av1 0 --load_waymo 0 \
     --num_workers 8 \
     --is_pretrain 1 \
