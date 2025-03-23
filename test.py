
import torch,os
import numpy as np
from tqdm import tqdm
from configs.config_setting import setting_test_config
from src.base_function import get_logger
from medpy.metric.binary import hd95, dc, jc, assd,specificity,sensitivity,precision
from src.utils import load_model

parse_config = setting_test_config()
print("parse_config:",parse_config)

##############################
#----------GPU init----------#
##############################
os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#######################################
#----------Preparing dataset----------#
#######################################
if parse_config.dataset == 'isic2018':
    from dataset.isic2018 import myDataset
    dataset = myDataset('test', aug=False)

###########################################
#--------------Load Test data-------------#
###########################################
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=8,
                                          pin_memory=True,
                                          drop_last=False,
                                          shuffle=False)


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


#######################################
#----------Load Test Model------------#
#######################################
dir_path = os.path.dirname(
    os.path.abspath(__file__)) + "/logs/{}/{}/fold_{}/".format(
        parse_config.dataset, parse_config.log_name, parse_config.fold)
model = load_model(model, dir_path + 'model/best.pkl')

#####################################
#----------Creating logger----------#
#####################################
txt_path_parameter = os.path.join(dir_path + 'parameter.txt')
logging_parameter = get_logger('loss', txt_path_parameter)



def test():
    model.eval()
    num = 0

    dice_value = 0
    jc_value = 0
    precision_value = 0
    sp_value = 0
    se_value = 0
    hd95_value = 0
    assd_value = 0

    labels = []
    pres = []

    for batch_idx, batch_data in tqdm(enumerate(test_loader)):
        data = batch_data['image'].to(device).float()
        label = batch_data['label'].to(device).float()
        with torch.no_grad():
            if parse_config.arch == 'transfuse':
                _, _, output = model(data)
            elif parse_config.point_pred == 0:
                output = model(data)
            elif parse_config.point_pred == 1:
                output, _ = model(data)
            output = torch.sigmoid(output)
            output = output.cpu().numpy() > 0.5
        label = label.cpu().numpy()
        assert (output.shape == label.shape)
        labels.append(label)
        pres.append(output)
    labels = np.concatenate(labels, axis=0)
    pres = np.concatenate(pres, axis=0)

    print(labels.shape, pres.shape)
    for _id in range(labels.shape[0]):
        dice_ave = dc(labels[_id], pres[_id])
        jc_ave = jc(labels[_id], pres[_id])
        precision_ave = precision(labels[_id], pres[_id])
        sp_ave = specificity(labels[_id], pres[_id])
        se_ave = sensitivity(labels[_id], pres[_id])
        print("dice_ave the %d Picture = %f" %(_id+1, dice_ave))
        try:
            hd95_ave = hd95(labels[_id], pres[_id])
            assd_ave = assd(labels[_id], pres[_id])
        except RuntimeError:
            num += 1
            hd95_ave = 0
            assd_ave = 0

        dice_value += dice_ave
        jc_value += jc_ave
        precision_value += precision_ave
        sp_value += sp_ave
        se_value += se_ave
        hd95_value += hd95_ave
        assd_value += assd_ave


    dice_average = dice_value / (labels.shape[0] - num)
    jc_average = jc_value / (labels.shape[0] - num)
    precision_average = precision_value / (labels.shape[0] - num)
    sp_average = sp_value / (labels.shape[0] - num)
    se_average = se_value / (labels.shape[0] - num)
    hd95_average = hd95_value / (labels.shape[0] - num)
    assd_average = assd_value / (labels.shape[0] - num)

    logging_parameter.info('Dice value of test dataset  : %f' % (dice_average))
    logging_parameter.info('IoU value of test dataset  : %f' % (jc_average))
    logging_parameter.info('precision value of test dataset  : %f' % (precision_average))
    logging_parameter.info('SP value of test dataset  : %f' % (sp_average))
    logging_parameter.info('SE value of test dataset  : %f' % (se_average))
    logging_parameter.info('Hd95 value of test dataset  : %f' % (hd95_average))
    logging_parameter.info('Assd value of test dataset  : %f' % (assd_average))

    print("Average dice value of evaluation dataset = ", dice_average)
    print("Average IoU value of evaluation dataset = ", jc_average)
    print("Average precision value of evaluation dataset = ", precision_average)
    print("Average SP value of evaluation dataset = ", sp_average)
    print("Average SE value of evaluation dataset = ", se_average)
    print("Average hd95 value of evaluation dataset = ", hd95_average)
    print("Average assd value of evaluation dataset = ", assd_average)

    print("Picture number = ", (labels.shape[0] - num))

    return dice_average


if __name__ == '__main__':
    test()
