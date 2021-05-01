# NIR-ISL2021-Transfer-learning
This repository is winning scheme of one of top three in [NIR-ISL 2021](https://sites.google.com/view/nir-isl2021/home) based on pytorch implementation.
## Transfer Learning
![Alt text](/pic/transfer.png)
## Requirements
* python 3.6
* pytorch 1.5+
* scikit-learn
* scikit-image
* albumentations
* opencv-python
* tensorboardX
* segmentation_models_pytorch
## Usage
We upload all the tensorboard records in the training process and the final optimal model to [Baidu Cloud](https://pan.baidu.com/s/1C0D_PtN5s55rKn0azjGPLg) (x7nz). Replace the downloaded folder with the trained_models folder in the root directory. All training codes are only considered to be completed on one GPU.
### Data split
Five-fold cross-validation was used in our experiment. The split result we use is saved in the'. pkl' file under the' data' path. You can also get your own split result through the following code:
    python --data_dir data_dir --save_dir save_dir
### Training
