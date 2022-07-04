# papeR: S4_v1: 87.26, v2: 88.5, liquid: 90.95
python3 -m train wandb=null experiment=s4-lra-cifar-new # 4.2/it/s  top score:  0.8600000143051147
python3 -m train wandb=null experiment=s4-lra-cifar-new model.layer.liquid=1 # 3.04 it/s
python3 -m train wandb=null experiment=s4-lra-cifar-new model.layer.liquid=2 # 2.91 it/s top score:  0.8611999750137329
python3 -m train wandb=null experiment=s4-lra-cifar-new model.layer.liquid=3 # it/s
python3 -m train wandb=null experiment=s4-lra-cifar-new optimizer.weight_decay=0.01 trainer.max_epochs=200
python3 -m train wandb=null experiment=s4-lra-cifar-new optimizer.weight_decay=0.01 trainer.max_epochs=300
python3 -m train wandb=null experiment=s4-lra-cifar-new model.d_model=192 model.layer.postact=glu model.layer.bidirectional=true optimizer.weight_decay=0.01 trainer.max_epochs=200
python3 -m train wandb=null experiment=s4-lra-cifar-new optimizer.lr=0.01 optimizer.weight_decay=0.01 trainer.max_epochs=200