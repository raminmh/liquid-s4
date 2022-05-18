#python -m train experiment=s4-lra-cifar-new
python -m train wandb=null experiment=s4-lra-cifar-new optimizer.weight_decay=0.01 trainer.max_epochs=200
python -m train wandb=null experiment=s4-lra-cifar-new optimizer.weight_decay=0.01 trainer.max_epochs=300
python -m train wandb=null experiment=s4-lra-cifar-new model.d_model=192 model.layer.postact=glu model.layer.bidirectional=true optimizer.weight_decay=0.01 trainer.max_epochs=200
python -m train wandb=null experiment=s4-lra-cifar-new optimizer.lr=0.01 optimizer.weight_decay=0.01 trainer.max_epochs=200