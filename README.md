# CurvatureGeneration_FSL

This repository is the implementation of ICCV 2021 paper: "Curvature Generation in Curved Spaces for Few-Shot Learning".

We provide the PyTorch code about few-shot learning. Our method reuqires Python 3.6, Pytorch1.0+, and tensorboardX.

Training our model requires one GPU with 11GB.

Here we provide the code of training our model on the Mini-ImageNet dataset.

1. Download the images of the Mini-ImageNet dataset and put these images into the 'data/miniimagenet' folder.

2. Prepare the pre-trained model. Download BigResNet12 models from 'https://github.com/Sha-Lab/FEAT'.
 
3. Run the corresponding train_protonet.py.

Concretely,

For 1-shot inductive tasks, please run:
-------
```
python inductive_oneshot/train_protonet.py --c_lr 0.001 --lr 1e-7 --dataset MiniImageNet --dim 640 --gamma 0.1 --max_epoch 100 --step_size 40 --model bigres12 --query 15 --rerank 80 --validation_way 5 --way 10 --shot 1 --load_init_weight --setting 'inductive' --multihead 20 --divide 1
```

For 5-shot inductive tasks, please run:
-------
```
python inductive_multishot/train_protonet.py --c_lr 0.01 --lr 1e-6 --dataset MiniImageNet --dim 640 --gamma 0.1 --max_epoch 100 --step_size 40 --model bigres12 --query 15 --rerank 11 --validation_way 5 --way 10 --shot 5 --load_init_weight --setting 'inductive' --multihead 1 --divide 200
```

For 1-shot transductive tasks, please run:
-------
```
python transductive_oneshot/train_protonet.py --hyperbolic --c_lr 0.01 --lr 1e-7 --dataset MiniImageNet --dim 640 --gamma 0.1 --max_epoch 200 --step_size 40 --model newres12 --query 15 --rerank 60 --validation_way 5 --way 10 --shot 1 --load_init_weight --divide 100  --l 0.0000001
```

For 5-shot transductive tasks, please run:
-------
```
python transductive_multishot/train_protonet.py --hyperbolic --c_lr 0.01 --lr 1e-7 --dataset MiniImageNet --dim 640 --gamma 0.1 --max_epoch 200 --step_size 40 --model newres12 --query 15 --rerank 60 --validation_way 5 --way 10 --shot 5 --load_init_weight --divide 1  --l 0.0000001
```

# Contact
gaozhi_2017@bit.edu.cn

