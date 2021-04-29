import os
from dataset_seg import Dataset_train, Dataset_val
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn as nn
import numpy as np
import torch
import math
import segmentation_models_pytorch as smp
from loss_function.pytorch_loss_function import dice_BCE_loss
from loss_function.DICE import dice1
import shutil
from torchvision.utils import make_grid
import argparse
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--bs', type=int, default=8, help='batch size')
parser.add_argument('--name', type=str, default='without_cycle', help='net name')
parser.add_argument('--epoch', type=int, default=200, help='all_epochs')
parser.add_argument('--net', type=str, default='baseline', help='net')
parser.add_argument('--fold', type=int, default=0, help='fold of cross validation')
parser.add_argument('--dataset', type=str, default='Asia', help='Asia/Africa/M1/All')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

data_root = 'data'
if args.dataset.lower() == 'asia':
    split_file = 'data/CASIA_Iris_Asia_split_20210305.pkl'
    description = 'Asia'
    input_size = (480, 640)
    resize = True
elif args.dataset.lower() == 'africa':
    split_file = 'data/CASIA_Iris_Africa_split_20210305.pkl'
    description = 'Africa'
    input_size = (384, 640)
    resize = True
elif args.dataset.lower() == 'm1':
    split_file = 'data/CASIA_Iris_M1_split_20210305.pkl'
    description = 'M1'
    input_size = (416, 416)
    resize = True
elif args.dataset.lower() == 'all':
    split_file = 'data/CASIA_Iris_All_split_20210305.pkl'
    description = 'All'
    input_size = (512, 512)
    resize = False

lr_max = 0.0002
L2 = 0.0001

save_name = 'bs{}_epoch{}_fold{}'.format(args.bs, args.epoch, args.fold)
save_dir = os.path.join('trained_models/{}/Seg/{}_{}'.format(description, args.name, args.net), save_name)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)
train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
print(os.path.join(save_dir))

print('data loading')
train_data = Dataset_train(data_root=data_root, split_file=split_file, size=input_size, fold=args.fold, resize=resize)
train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
val_data = Dataset_val(data_root=data_root, split_file=split_file, size=input_size, fold=args.fold, resize=resize)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

print('model loading')
if args.net.lower() == 'unet_resnet34':
    net = smp.Unet('resnet34', in_channels=1, classes=1, activation=None).cuda()
if args.net.lower() == 'unet_resnet101':
    net = smp.Unet('resnet101', in_channels=1, classes=1, activation=None).cuda()

train_data_len = train_data.len
val_data_len = val_data.len
print('train_lenth: %i  val_lenth: %i' % (train_data_len, val_data_len))

criterion = dice_BCE_loss(0.5, 0.5)
optimizer = optim.Adam(net.parameters(), lr=lr_max, weight_decay=L2)
scheduler = MultiStepLR(optimizer, milestones=[int((5 / 10) * args.epoch),
                                               int((8 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)
best_acc = 0
IMAGE = []
SEG = []

print('training')
for epoch in range(args.epoch):
    net.train()
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    print('lr for this epoch:', lr)
    epoch_train_loss = []
    epoch_train_acc = []
    for i, (inputs, segs) in enumerate(train_dataloader):
        inputs, segs = inputs.float().cuda(), segs.float().cuda()
        optimizer.zero_grad()
        result = net(inputs)
        result = torch.sigmoid(result)
        train_loss = criterion(result, segs)
        train_loss.backward()
        optimizer.step()
        pred_train = result.cpu().float()
        pred_train[pred_train <= 0.5] = 0
        pred_train[pred_train > 0.5] = 1
        acc = ((pred_train == segs.cpu()).sum().float() / pred_train.numel()).detach().numpy()
        epoch_train_acc.append(acc)
        epoch_train_loss.append(train_loss.item())
        print('[%d/%d, %5d/%d] train_loss: %.3f train_acc: %.3f' % (epoch + 1, args.epoch, i + 1,
                                                                     math.ceil(train_data_len / args.bs),
                                                                     train_loss.item(), acc))
    scheduler.step()

    net.eval()
    epoch_val_loss = []
    epoch_val_acc = []
    PREDICTION = []
    with torch.no_grad():
        for i, (inputs, segs) in enumerate(val_dataloader):
            inputs, segs = inputs.float().cuda(), segs.float().cuda()
            result = net(inputs)
            result = torch.sigmoid(result)
            val_loss = criterion(result, segs)
            pred_val = result.cpu().float()
            pred_val[pred_val <= 0.5] = 0
            pred_val[pred_val > 0.5] = 1
            acc = ((pred_val == segs.cpu()).sum().float() / pred_val.numel()).detach().numpy()
            epoch_val_acc.append(acc)
            epoch_val_loss.append(val_loss.item())
            if i in [1, 3, 6, 9] and epoch == 0:
                IMAGE.append(inputs[0:1, :, :, :].cpu())
                SEG.append(segs[0:1, :, :, :].cpu())
            if i in [1, 3, 6, 9] and epoch % (args.epoch // 10) == 0:
                PREDICTION.append(pred_val[0:1, :, :, :])
    epoch_train_loss = np.mean(epoch_train_loss)
    epoch_train_acc = np.mean(epoch_train_acc)
    epoch_val_loss = np.mean(epoch_val_loss)
    epoch_val_acc = np.mean(epoch_val_acc)
    print('[%d/%d] train_loss: %.3f  val_loss: %.3f  val_acc: %.3f' % (epoch + 1, args.epoch, epoch_train_loss,
                                                                        epoch_val_loss, epoch_val_acc))

    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('loss', epoch_train_loss, epoch)
    train_writer.add_scalar('acc', epoch_train_acc, epoch)
    val_writer.add_scalar('loss', epoch_val_loss, epoch)
    val_writer.add_scalar('acc', epoch_val_acc, epoch)
    val_writer.add_scalar('best_acc', best_acc, epoch)
    if epoch == 0:
        IMAGE = torch.cat(IMAGE, dim=0)
        SEG = torch.cat(SEG, dim=0)
        IMAGE = make_grid(IMAGE, 2, normalize=True)
        SEG = make_grid(SEG, 2, normalize=True)
        val_writer.add_image('IMAGE', IMAGE, epoch)
        val_writer.add_image('SEG', SEG, epoch)
    if epoch % (args.epoch // 10) == 0:
        PREDICTION = torch.cat(PREDICTION, dim=0)
        PREDICTION = make_grid(PREDICTION, 2, normalize=True)
        val_writer.add_image('PREDICTION', PREDICTION, epoch)
    if epoch + 1 == args.epoch:
        torch.save(net.state_dict(),
                   os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(net.state_dict(),
                   os.path.join(save_dir, 'best_acc.pth'))
train_writer.close()
val_writer.close()
print('saved_model_name:', save_dir)
