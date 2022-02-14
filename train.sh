#!/bin/bash
#SBATCH -o job.%j_btrain.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J flow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 

python -u  train.py 