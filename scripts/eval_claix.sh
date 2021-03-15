#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=eval_stgat_full_nusc

#SBATCH --ntasks=3

#SBATCH --output=/home/kq708907/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=huzjkevin@gmail.com

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=3G

#SBATCH --gres=gpu:pascal:1

#SBATCH --time=00:30:00

#SBATCH --signal=TERM@120

#SBATCH --partition=c18g



source $HOME/.zshrc
conda activate kevin

WS_DIR="$HOME/Projects/STGAT"
SCRIPT="STGAT/evaluate_model.py"

cd ${WS_DIR}

# wandb on

# srun -n 1 python ${SCRIPT} --dataset_name trajectory_human --resume ./checkpoint_trajectory_human_model_best.pth.tar
# srun -n 1 python ${SCRIPT} --dataset_name trajectory_vehicle --resume ./checkpoint_trajectory_vehicle_model_best.pth.tar
# srun -n 1 python ${SCRIPT} --dataset_name trajectory_bicycle --resume ./checkpoint_trajectory_bicycle_model_best.pth.tar
srun -n 1 python ${SCRIPT} --dataset_name trajectory_combined --resume ./checkpoint_trajectory_combined_model_best.pth.tar

wait