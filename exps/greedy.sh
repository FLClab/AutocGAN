#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=150Gb
#SBATCH --gres=gpu:v100:1
#SBATCH --output=autocgan.log
#SBATCH --err=autocgan_err.log
#SBATCH --mail-user=frederic.beaupre.3@ulaval.ca
#SBATCH --partition=gpu_96h

python search_cgan.py \
-gen_bs 128 \
-dis_bs 128 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--gen_model shared_cgan \
--dis_model shared_cgan \
--controller controller \
--latent_dim 128 \
--gf_dim 128 \
--df_dim 64 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--ctrl_sample_batch 1 \
--num_candidate 4 \
--topk 4 \
--shared_epoch 15 \
--grow_step1 15 \
--grow_step2 35 \
--max_search_iter 65 \
--ctrl_step 30 \
--exp_name autocgan_search2 \
--n_classes 10 