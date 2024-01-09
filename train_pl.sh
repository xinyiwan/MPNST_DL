#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH -p hm
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /trinity/home/xwan/MPNST_DL/logs/out_%j.log
#SBATCH -e /trinity/home/xwan/MPNST_DL/logs/error_%j.log

# Load the modules

module purge

source "/tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env"
source venv/bin/activate  # commented out by conda initialize
module load Python/3.9.5-GCCcore-10.3.0

python3 run_pl.py input/configs/config003.yaml
