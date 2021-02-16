#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=train_stgat

#SBATCH --output=/home/kq708907/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=huzjkevin@gmail.com

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=3G

#SBATCH --gres=gpu:pascal:1

#SBATCH --time=1-00:00:00

#SBATCH --signal=TERM@120

#SBATCH --partition=c18g



source $HOME/.zshrc
conda activate kevin

WS_DIR="$HOME/Projects/STGAT"
SCRIPT="train.py"

cd ${WS_DIR}

# wandb on

srun --unbuffered python ${SCRIPT} --dataset_name nuscenes
