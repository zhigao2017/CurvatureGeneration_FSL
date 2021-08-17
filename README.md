# CurvatureGeneration_FSL

This repository is the implementation of ICCV 2021 paper: "Curvature Generation in Curved Spaces for Few-Shot Learning".

We provide the PyTorch code about few-shot learning. Our method reuqires Python 3.6, Pytorch1.0+, and tensorboardX.

Training our model requires one GPU with 11GB.

Here we provide the code of training our model on the Mini-ImageNet dataset.

1. Download the images of the Mini-ImageNet dataset and put these images into the 'data/miniimagenet' folder.

2. Prepare the pre-trained model. Download BigResNet12 models from 'https://github.com/cyvius96/few-shot-meta-baseline'.
 
3. Run the corresponding train_protonet.py.
