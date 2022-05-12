#!/bin/bash
#
#SBATCH --job-name=s4_3
#SBATCH --output=s4_3.txt
#
#number of CPUs to be used
#SBATCH --ntasks=1
#SBATCH -c 6
#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=128:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=64G

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GTX1080Ti

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

cd $HOME/s4
source venv/bin/activate


# mkdir -p $HOME/s4/tmp/tmp_$SLURM_ARRAY_TASK_ID
# export TMPDIR=$HOME/s4/tmp/tmp_$SLURM_ARRAY_TASK_ID
# python3 -m train model.layer.poly=true pipeline=mnist model=s4 
### S4 Paper:
# Listops 58.35
# IMDB: 76.02
# AAN: 87.09
# CIFAR: 87.26
# Pathfinder: 86.05
# Path-X: 88.10

# listops: test_acc: 0.567
# imdb: 0.76816
# pathfindeR: 0.495

# 4 gpus:
# listops best 0.54650
# imdb  0.74128
# cifar: 0.84100
# pathfindeR 0.75270

# trainer.gpus=2
if [ $SLURM_ARRAY_TASK_ID = 6 ]
then
  python -m train model.layer.poly=true experiment=s4-lra-cifar model.layer.postact=glu model.layer.bidirectional=true optimizer.weight_decay=0.01 trainer.max_epochs=160 
elif [ $SLURM_ARRAY_TASK_ID = 7 ]
then
  python -m train model.layer.poly=true experiment=s4-lra-cifar model.layer.postact=glu model.layer.bidirectional=true optimizer.weight_decay=0.005 trainer.max_epochs=160 
  python -m train experiment=s4-lra-cifar model.layer.postact=glu model.layer.bidirectional=true optimizer.weight_decay=0.005 trainer.max_epochs=160
elif [ $SLURM_ARRAY_TASK_ID = 8 ]
then
  python -m train model.layer.poly=true experiment=s4-lra-cifar model.layer.postact=glu model.layer.bidirectional=true optimizer.weight_decay=0.01 trainer.max_epochs=160 scheduler.patience=20 
elif [ $SLURM_ARRAY_TASK_ID = 9 ]
then
  python -m train model.layer.poly=true experiment=s4-lra-pathfinder 
  python -m train experiment=s4-lra-pathfinder
elif [ $SLURM_ARRAY_TASK_ID = 10 ]
then
  python -m train model.layer.poly=true experiment=s4-lra-pathfinder optimizer.lr=0.01 
elif [ $SLURM_ARRAY_TASK_ID = 11 ]
then
  python -m train model.layer.poly=true experiment=s4-lra-pathfinder optimizer.lr=0.006 
elif [ $SLURM_ARRAY_TASK_ID = 12 ]
then
  python -m train model.layer.poly=true experiment=s4-lra-pathfinder optimizer.lr=0.004 scheduler.patience=20 
elif [ $SLURM_ARRAY_TASK_ID = 13 ]
then
  python -m train model.layer.poly=true experiment=s4-lra-aan &> results/aan.txt
fi

# rm -r $HOME/s4/tmp/tmp_$SLURM_ARRAY_TASK_ID