
import os,sys
import torch.utils.data
from configs.config_setting import setting_train_config
from torch.utils.tensorboard import SummaryWriter
from src.base_function import get_logger,ce_loss,focal_loss,train,evaluation
from torch.optim.lr_scheduler import  CosineAnnealingLR
import time

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

parse_config = setting_train_config()
# print("parse_config:",parse_config)

EPOCHS = parse_config.n_epochs


#############################################
#----------Preparing all save_path----------#
#############################################
if parse_config.arch == 'MPBA_Net':
    parse_config.exp_name += '_{}_{}_{}_e{}'.format(parse_config.trans,
                                                    parse_config.point_pred,
                                                    parse_config.cross,
                                                    parse_config.ppl)
exp_name = parse_config.dataset + '/' + parse_config.exp_name + '_loss_' + str(
    parse_config.seg_loss) + '_aug_' + str(parse_config.aug) + '/fold_' + str(
        parse_config.fold)

os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
writer = SummaryWriter('logs/{}/log'.format(exp_name))
save_path = 'logs/{}/model/best.pkl'.format(exp_name)
latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)


##############################
#----------GPU init----------#.
##############################
os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
device_ids = range(torch.cuda.device_count())
torch.set_num_threads(8)


#######################################
#----------Preparing dataset----------#
#######################################
if parse_config.dataset == 'isic2018':
    from dataset.isic2018 import myDataset
    dataset = myDataset(split='train', aug=parse_config.aug)
    dataset2 = myDataset(split='valid', aug=False)

elif parse_config.dataset == 'isic2016':
    from dataset.isic2016 import myDataset
    dataset = myDataset(split='train', aug=parse_config.aug)
    dataset2 = myDataset(split='valid', aug=False)

elif parse_config.dataset == 'isic2017':
    from dataset.isic2017 import myDataset
    dataset = myDataset(split='train', aug=parse_config.aug)
    dataset2 = myDataset(split='valid', aug=False)



###############################################
#在linux系统中可以使用多个子进程加载数据，而在windows系统中不能。
#所以在windows中要将DataLoader中的num_workers设置为0或者采用默认为0的设置
###########################################
#----------Load Train & val data----------#
###########################################
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=parse_config.bt_size,
                                           shuffle=False,
                                           num_workers=1,
                                           pin_memory=True,
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset2,
                                         batch_size=8,  #parse_config.bt_size
                                         shuffle=False,  #True
                                         num_workers=2,
                                         pin_memory=True,
                                         drop_last=False)  #True



#######################################
#----------Prepareing Model-----------#
#######################################
if parse_config.arch == 'MPBA_Net':
    if parse_config.trans == 1:
        from src.FusionNetworks import MPBA_Net
        model = MPBA_Net(1, parse_config.net_layer, parse_config.point_pred,
                    parse_config.ppl).cuda()
    else:
        from src.base import DeepLabV3
        model = DeepLabV3(1, parse_config.net_layer).cuda()

if len(device_ids) > 1:  # Multi-Card Training
    model = torch.nn.DataParallel(model).cuda()


#########################################
#----------Prepareing opt, sch----------#
#########################################
optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)
scheduler = CosineAnnealingLR(optimizer, T_max=20)


#####################################
#----------Creating logger----------#
#####################################
dir_path = 'logs/{}/model/'.format(exp_name)
txt_path_loss = os.path.join(dir_path + 'loss.txt')
txt_path_evaluation = os.path.join(dir_path + 'evaluation.txt')
logging_loss = get_logger('loss', txt_path_loss)
logging_evaluation = get_logger('evaluation', txt_path_evaluation)


#################################
#----------Choose Point_loss----------#
#################################
Choose_Point_loss = [focal_loss, ce_loss][parse_config.seg_loss]



def main():
    max_iou = 0

    min_loss = 10
    min_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        # 打印学习率 lr

        this_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('Learning Rate', this_lr, epoch)
        start = time.time()
        train(epoch,
              model,
              train_loader,
              parse_config,
              Choose_Point_loss,
              optimizer,
              writer,
              logging_loss)
        dice, iou, loss = evaluation(epoch,
                                     val_loader,
                                     model,
                                     parse_config,
                                     Choose_Point_loss,
                                     writer,
                                     logging_evaluation)
        # scheduler.step(loss)
        scheduler.step()

        if loss < min_loss:
            min_epoch = epoch
            min_loss = loss
        else:
            if epoch - min_epoch >= parse_config.patience:
                print('Early stopping!')
                break
        if iou > max_iou:
            max_iou = iou
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break
        torch.save(model.state_dict(), latest_path)


        time_elapsed = time.time() - start
        print('Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
              format(epoch, time_elapsed // 60, time_elapsed % 60))



if __name__ == '__main__':

    print('==> Building model..\n')
    main()
