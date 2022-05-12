# python -m train model.layer.poly=true experiment=s4-lra-cifar model.layer.postact=glu model.layer.bidirectional=true optimizer.weight_decay=0.01 trainer.max_epochs=160 
# python -m train model.layer.poly=true experiment=s4-lra-cifar model.layer.postact=glu model.layer.bidirectional=true optimizer.weight_decay=0.01 trainer.max_epochs=160 scheduler.patience=20 
python -m train model.layer.poly=true experiment=s4-lra-cifar model.layer.postact=glu model.layer.bidirectional=true optimizer.weight_decay=0.01 trainer.max_epochs=200 scheduler.patience=20 
