import os
from dataset_local_inner_outer import Dataset_train, Dataset_val
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
import math
from Networks.Local_network import Local_UNet
from loss_function.pytorch_loss_function import dice_BCE_loss
from loss_function.DICE import dice1
import shutil
from torchvision.utils import make_grid
import argparse
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--bs', type=int, default=8, help='batch size')
parser.add_argument('--name', type=str, default='finetune', help='save dir name')
parser.add_argument('--epoch', type=int, default=200, help='all_epochs')
parser.add_argument('--net', type=str, default='unet_resnet34', help='net')
parser.add_argument('--fold', type=int, default=0, help='fold of cross validation')
parser.add_argument('--pretrained', type=str, help='pretrained model path')
parser.add_argument('--dataset', type=str, default='Asia', help='Asia/Africa/M1/all')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

data_root = 'data'
if args.dataset.lower() == 'asia':
    split_file = 'data/CASIA_Iris_Asia_split.pkl'
    description = 'Asia'
    input_size = (480, 640)
    resize = True
elif args.dataset.lower() == 'africa':
    split_file = 'data/CASIA_Iris_Africa_split.pkl'
    description = 'Africa'
    input_size = (384, 640)
    resize = True
elif args.dataset.lower() == 'm1':
    split_file = 'data/CASIA_Iris_M1_split.pkl'
    description = 'M1'
    input_size = (416, 416)
    resize = True
elif args.dataset.lower() == 'all':
    split_file = 'data/CASIA_Iris_All_split.pkl'
    description = 'All'
    input_size = (512, 512)
    resize = False


lr_max = 0.00002
L2 = 0.00001

save_name = 'bs{}_epoch{}_fold{}'.format(args.bs, args.epoch, args.fold)
save_dir = os.path.join('trained_models/{}/Local_inner_outer/{}_{}'.format(description, args.name, args.net), save_name)
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
    net = Local_UNet(encoder_name='resnet34', in_channels=1, out_channels_1=1, out_channels_2=1).cuda()
if args.net.lower() == 'unet_resnet101':
    net = Local_UNet(encoder_name='resnet101', in_channels=1, out_channels_1=1, out_channels_2=1).cuda()

net.load_state_dict(torch.load(args.pretrained))

train_data_len = train_data.len
val_data_len = val_data.len
print('train_lenth: %i  val_lenth: %i' % (train_data_len, val_data_len))

criterion = dice_BCE_loss(0.5, 0.5)
optimizer = optim.Adam(net.parameters(), lr=lr_max, weight_decay=L2)
scheduler = MultiStepLR(optimizer, milestones=[int((5 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)

best_inner_dice = 0
best_outer_dice = 0
IMAGE = []
LOCAL_INNER = []
LOCAL_OUTER = []

print('training')
for epoch in range(args.epoch):
    net.train()
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    print('lr for this epoch:', lr)
    epoch_train_totalloss = []
    epoch_train_innerloss = []
    epoch_train_innerdice = []
    epoch_train_outerloss = []
    epoch_train_outerdice = []
    for i, (inputs, local_inners, local_outers) in enumerate(train_dataloader):
        inputs, local_inners, local_outers = inputs.float().cuda(), \
                                                   local_inners.float().cuda(), local_outers.float().cuda()
        optimizer.zero_grad()
        result_local_inner, result_local_outer = net(inputs)
        result_local_inner = torch.sigmoid(result_local_inner)
        result_local_outer = torch.sigmoid(result_local_outer)
        innerloss = criterion(result_local_inner, local_inners)
        outerloss = criterion(result_local_outer, local_outers)
        total_loss = 0.5 * innerloss + 0.5 * outerloss
        total_loss.backward()
        optimizer.step()

        pred_inner = result_local_inner.cpu().float()
        pred_inner[pred_inner <= 0.5] = 0
        pred_inner[pred_inner > 0.5] = 1
        inner_dice = dice1(local_inners.cpu(), pred_inner).detach().numpy()
        pred_outer = result_local_outer.cpu().float()
        pred_outer[pred_outer <= 0.5] = 0
        pred_outer[pred_outer > 0.5] = 1
        outer_dice = dice1(local_outers.cpu(), pred_outer).detach().numpy()
        epoch_train_totalloss.append(total_loss.item())
        epoch_train_innerloss.append(innerloss.item())
        epoch_train_innerdice.append(inner_dice)
        epoch_train_outerloss.append(outerloss.item())
        epoch_train_outerdice.append(outer_dice)
        print('[%d/%d, %5d/%d] train_total_loss: %.3f inner_dice: %.3f outer_dice: %.3f' %
              (epoch + 1, args.epoch, i + 1, math.ceil(train_data_len / args.bs), total_loss.item(),
               inner_dice, outer_dice))
    scheduler.step()

    net.eval()
    epoch_val_totalloss = []
    epoch_val_innerloss = []
    epoch_val_innerdice = []
    epoch_val_outerloss = []
    epoch_val_outerdice = []
    PREDICTION_LOCAL_INNER = []
    PREDICTION_LOCAL_OUTER = []
    with torch.no_grad():
        for i, (inputs, local_inners, local_outers) in enumerate(val_dataloader):
            inputs, local_inners, local_outers = inputs.float().cuda(), \
                                                       local_inners.float().cuda(), local_outers.float().cuda()
            result_local_inner, result_local_outer = net(inputs)
            result_local_inner = torch.sigmoid(result_local_inner)
            result_local_outer = torch.sigmoid(result_local_outer)
            innerloss = criterion(result_local_inner, local_inners)
            outerloss = criterion(result_local_outer, local_outers)
            total_loss = 0.5 * innerloss + 0.5 * outerloss

            pred_inner = result_local_inner.cpu().float()
            pred_inner[pred_inner <= 0.5] = 0
            pred_inner[pred_inner > 0.5] = 1
            inner_dice = dice1(local_inners.cpu(), pred_inner).detach().numpy()
            pred_outer = result_local_outer.cpu().float()
            pred_outer[pred_outer <= 0.5] = 0
            pred_outer[pred_outer > 0.5] = 1
            outer_dice = dice1(local_outers.cpu(), pred_outer).detach().numpy()
            epoch_val_totalloss.append(total_loss.item())
            epoch_val_innerloss.append(innerloss.item())
            epoch_val_innerdice.append(inner_dice)
            epoch_val_outerloss.append(outerloss.item())
            epoch_val_outerdice.append(outer_dice)
            if i in [0, 1, 2, 3] and epoch == 0:
                IMAGE.append(inputs[0:1, :, :, :].cpu())
                LOCAL_INNER.append(local_inners[0:1, :, :, :].cpu())
                LOCAL_OUTER.append(local_outers[0:1, :, :, :].cpu())
            if i in [0, 1, 2, 3] and epoch % (args.epoch // 10) == 0:
                PREDICTION_LOCAL_INNER.append(pred_inner[0:1, :, :, :])
                PREDICTION_LOCAL_OUTER.append(pred_outer[0:1, :, :, :])
    epoch_train_totalloss = np.mean(epoch_train_totalloss)
    epoch_train_innerloss = np.mean(epoch_train_innerloss)
    epoch_train_innerdice = np.mean(epoch_train_innerdice)
    epoch_train_outerloss = np.mean(epoch_train_outerloss)
    epoch_train_outerdice = np.mean(epoch_train_outerdice)

    epoch_val_totalloss = np.mean(epoch_val_totalloss)
    epoch_val_innerloss = np.mean(epoch_val_innerloss)
    epoch_val_innerdice = np.mean(epoch_val_innerdice)
    epoch_val_outerloss = np.mean(epoch_val_outerloss)
    epoch_val_outerdice = np.mean(epoch_val_outerdice)
    print('[%d/%d] train_totalloss: %.3f  val_totalloss: %.3f val_innerdice: %.3f val_outerdice: %.3f'
          % (epoch + 1, args.epoch, epoch_train_totalloss, epoch_val_totalloss, epoch_val_innerdice,
             epoch_val_outerdice))

    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('total_loss', epoch_train_totalloss, epoch)
    train_writer.add_scalar('inner_loss', epoch_train_innerloss, epoch)
    train_writer.add_scalar('outer_loss', epoch_train_outerloss, epoch)
    train_writer.add_scalar('inner_dice', epoch_train_innerdice, epoch)
    train_writer.add_scalar('outer_dice', epoch_train_outerdice, epoch)

    val_writer.add_scalar('total_loss', epoch_val_totalloss, epoch)
    val_writer.add_scalar('inner_loss', epoch_val_innerloss, epoch)
    val_writer.add_scalar('outer_loss', epoch_val_outerloss, epoch)
    val_writer.add_scalar('inner_dice', epoch_val_innerdice, epoch)
    val_writer.add_scalar('outer_dice', epoch_val_outerdice, epoch)
    val_writer.add_scalar('best_innerdice', best_inner_dice, epoch)
    val_writer.add_scalar('best_outerdice', best_outer_dice, epoch)
    if epoch == 0:
        IMAGE = torch.cat(IMAGE, dim=0)
        LOCAL_INNER = torch.cat(LOCAL_INNER, dim=0)
        LOCAL_OUTER = torch.cat(LOCAL_OUTER, dim=0)
        IMAGE = make_grid(IMAGE, 2, normalize=True)
        LOCAL_INNER = make_grid(LOCAL_INNER, 2, normalize=True)
        LOCAL_OUTER = make_grid(LOCAL_OUTER, 2, normalize=True)
        val_writer.add_image('IMAGE', IMAGE, epoch)
        val_writer.add_image('LOCAL_INNER', LOCAL_INNER, epoch)
        val_writer.add_image('LOCAL_OUTER', LOCAL_OUTER, epoch)
    if epoch % (args.epoch // 10) == 0:
        PREDICTION_LOCAL_INNER = torch.cat(PREDICTION_LOCAL_INNER, dim=0)
        PREDICTION_LOCAL_OUTER = torch.cat(PREDICTION_LOCAL_OUTER, dim=0)
        PREDICTION_LOCAL_INNER = make_grid(PREDICTION_LOCAL_INNER, 2, normalize=True)
        PREDICTION_LOCAL_OUTER = make_grid(PREDICTION_LOCAL_OUTER, 2, normalize=True)
        val_writer.add_image('PREDICTION_LOCAL_INNER', PREDICTION_LOCAL_INNER, epoch)
        val_writer.add_image('PREDICTION_LOCAL_OUTER', PREDICTION_LOCAL_OUTER, epoch)
    if epoch + 1 == args.epoch:
        torch.save(net.state_dict(),
                   os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))
    if epoch_val_innerdice > best_inner_dice:
        best_inner_dice = epoch_val_innerdice
        torch.save(net.state_dict(),
                   os.path.join(save_dir, 'best_inner_dice.pth'))
        torch.save(net.state_dict(),
                   os.path.join(save_dir, 'best_model.pth'))
    if epoch_val_outerdice > best_outer_dice:
        best_outer_dice = epoch_val_outerdice
        torch.save(net.state_dict(),
                   os.path.join(save_dir, 'best_outer_dice.pth'))
        torch.save(net.state_dict(),
                   os.path.join(save_dir, 'best_model.pth'))
train_writer.close()
val_writer.close()
print('saved_model_name:', save_dir)
