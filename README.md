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
We upload all the tensorboard records in the training process and the final optimal model to [Baidu Cloud](https://pan.baidu.com/s/1C0D_PtN5s55rKn0azjGPLg) (x7nz). Replace the downloaded folder with the trained_models folder in the root directory. All training codes are only considered to be completed on one GPU (RTX 2080Ti).
### Data split
Five-fold cross-validation was used in our experiment. The split result we use is saved in the'. pkl' file under the' data' path. You can also get your own split result through the following code:

    python --data_dir data_dir --save_dir save_dir
There are 'CASIA-Iris-Africa', 'CASIA-Iris-Asia' and 'CASIA-Iris-Mobile-V1.0' three folders under data_dir.

### Training
#### Baseline model
Training baseline model commands are as follows:

    # for semgentation
    # CASIA-Iris-Africa
    python train_seg.py --gpu 0 --bs 16 --name baseline --epoch 500 --net unet_resnet34 --fold 0 --dataset Africa
    # CASIA-Iris-Asia
    python train_seg.py --gpu 0 --bs 16 --name baseline --epoch 500 --net unet_resnet34 --fold 0 --dataset Asia
    # CASIA-Iris-Mobile-V1.0
    python train_seg.py --gpu 0 --bs 28 --name baseline --epoch 400 --net unet_resnet34 --fold 0 --dataset M1
    
    # for location
    # CASIA-Iris-Africa
    python train_local_inner_outer.py --gpu 0 --bs 16 --name baseline --epoch 500 --net unet_resnet34 --fold 0 --dataset Africa
    # CASIA-Iris-Asia
    python train_local_inner_outer.py --gpu 0 --bs 16 --name baseline --epoch 500 --net unet_resnet34 --fold 0 --dataset Asia
    # CASIA-Iris-Mobile-V1.0
    python train_local_inner_outer.py --gpu 0 --bs 28 --name baseline --epoch 400 --net unet_resnet34 --fold 0 --dataset M1
Each code execution only trains one fold. Train fold 0, 1, 2, 3 and 4 in turn.

#### Transfer learning model 
Firstly, all the data are used to train an iris pre-trained model.

    # for semgentation
    python train_seg.py --gpu 0 --bs 22 --name baseline --epoch 500 --net unet_resnet34 --fold 0 --dataset all
    
    # for location
    python train_local_inner_outer.py --gpu 0 --bs 22 --name baseline --epoch 500 --net unet_resnet34 --fold 0 --dataset all
Then fine-tune the iris pre-trained model on the subdatasets.

    # for semgentation
    # CASIA-Iris-Africa
    python finetune_seg.py --gpu 0 --bs 16 --name finetune --epoch 200 --net unet_resnet34 --fold 0 --dataset Africa --pretrained trained_models/All/Seg/baseline_UNet_ResNet34/bs22_epoch500_fold0/best_acc.pth
    # CASIA-Iris-Asia
    python finetune_seg.py --gpu 0 --bs 16 --name finetune --epoch 200 --net unet_resnet34 --fold 0 --dataset Asia --pretrained trained_models/All/Seg/baseline_UNet_ResNet34/bs22_epoch500_fold0/best_acc.pth
    # CASIA-Iris-Mobile-V1.0
    python finetune_seg.py --gpu 0 --bs 28 --name finetune --epoch 100 --net unet_resnet34 --fold 0 --dataset M1 --pretrained trained_models/All/Seg/baseline_UNet_ResNet34/bs22_epoch500_fold0/best_acc.pth
    
    # for location
    # CASIA-Iris-Africa
    python finetune_local_inner_outer.py --gpu 0 --bs 16 --name finetune --epoch 200 --net unet_resnet34 --fold 0 --dataset Africa --pretrained trained_models/All/Local_inner_outer/bsaeline_UNet_ResNet34/bs20_epoch500_fold0/best_model.pth
    # CASIA-Iris-Asia
    python finetune_local_inner_outer.py --gpu 0 --bs 16 --name finetune --epoch 200 --net unet_resnet34 --fold 0 --dataset Asia --pretrained trained_models/All/Local_inner_outer/bsaeline_UNet_ResNet34/bs20_epoch500_fold0/best_model.pth
    # CASIA-Iris-Mobile-V1.0
    python finetune_local_inner_outer.py --gpu 0 --bs 28 --name finetune --epoch 000 --net unet_resnet34 --fold 0 --dataset M1 --pretrained trained_models/All/Local_inner_outer/bsaeline_UNet_ResNet34/bs20_epoch500_fold0/best_model.pth

### Validation set evaluation

    # for semgentation
    python predict_seg_val.py --gpu 0 --net unet_resnet34 --fold 0 --model_path model_path --dataset subdataset
    
    # for location
    python predict_local_iris_val.py --gpu 0 --net unet_resnet34 --fold 0 --model_path model_path --dataset subdataset
**model_path** specifies the path of the corresponding'. pth' file. As with the above training command, **subdataset** chooses from chooses from 'Africa', 'Asia' and 'M1'.
 
### Test set prediction
Ensure that after downloading the trained code from Baidu Cloud and placing it in the specified location, execute the following codes in turn, and you can get the completely consistent test set prediction results provided by us in NIR-ISL2021.

    python predict_seg_Africa.py --gpu 0
    python predict_seg_Asia.py --gpu 0
    python predict_seg_M1.py --gpu 0
    python predict_local_Africa.py --gpu 0
    python predict_local_Asia.py --gpu 0
    python predict_local_M1.py --gpu 0
The prediction results are saved in *NIR-ISL2021_predictions* folder. If you want to use your own trained model for prediction, you only need to modify the **model_path** in the'. py' file.

### External data testing
![Alt text](/pic/pipeline.png)
We also wrote prediction codes for external data. For a given iris picture, the iris mask and inner and outer contours can be obtained by the following code. We provide three models for you to choose from 'Africa', 'Asia' and 'M1'. At the same time, it provides the prediction of CPU version (**--gpu none**).

    python predict_one_img.py --gpu 0 --img_path img_path --save_dir save_dir --model model
**img_path** is the path of a picture and **save_dir** is the folder for saving predicted pictures.

## Contributing
Yiwen Zhang
Tianbao Liu
