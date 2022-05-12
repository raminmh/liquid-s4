#!/bin/bash
#
#SBATCH --job-name=s4_2
#SBATCH --output=s4_2.txt
#
#number of CPUs to be used
#SBATCH --ntasks=1
#SBATCH -c 6
#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=96:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=64G

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=A10

#Send emails when a job starts, it is finished or it exits
#SBATCH --mail-user=mlechner@ist.ac.at
#SBATCH --mail-type=END,FAIL
#
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV


module load python/3.8.12
module load cuda/11.2.2
module load cudnn/8.2.4.15
#module load cudnn/8.1.0.77
#module load cudnn/8.1.1.33



cd $HOME/state-spaces
source venv/bin/activate

# trainer.gpus=2
python -m train wandb=null experiment=s4-lra-pathx