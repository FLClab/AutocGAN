#!/usr/bin/env bash

python train_derived.py \
-gen_bs 4 \
-dis_bs 4 \
--dataset stl10 \
--bottom_width 12 \
--eval_batch_size 10 \
--img_size 96 \
--max_iter 50000 \
--gen_model shared_cgan \
--dis_model shared_cgan \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--arch 1 1 0 1 0 0 0 1 1 1 0 0 0 3 \
--n_classes 10 \
--exp_name derive_stl96
