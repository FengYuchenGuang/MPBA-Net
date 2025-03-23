import argparse
def setting_train_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='MPBA_Net')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--net_layer', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='isic_test') #isic_test isic2016 isic2017 isic2018
    parser.add_argument('--exp_name', type=str, default='MPBA_Net')
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--lr_seg', type=float, default=1e-4)  # 0.0003
    parser.add_argument('--n_epochs', type=int, default=200)  # 100
    parser.add_argument('--bt_size', type=int, default=32)  # 8
    parser.add_argument('--seg_loss', type=int, default=0, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--patience', type=int, default=500)  # 50

    # pre-train
    parser.add_argument('--pre', type=int, default=0)

    # transformer
    parser.add_argument('--trans', type=int, default=1)

    # point constrain
    parser.add_argument('--point_pred', type=int, default=1)
    parser.add_argument('--ppl', type=int, default=6)

    # cross-scale framework
    parser.add_argument('--cross', type=int, default=0)
    parse_config = parser.parse_args()

    return parse_config


def setting_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name',
                        type=str,
                        default='MPBA_Net_1_1_0_e6_loss_0_aug_1')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='isic_test')

    parser.add_argument('--arch', type=str, default='MPBA_Net')
    parser.add_argument('--net_layer', type=int, default=50)
    # pre-train
    parser.add_argument('--pre', type=int, default=0)

    # transformer
    parser.add_argument('--trans', type=int, default=1)

    # point constrain
    parser.add_argument('--point_pred', type=int, default=1)
    parser.add_argument('--ppl', type=int, default=6)

    # cross-scale framework
    parser.add_argument('--cross', type=int, default=0)
    parse_config = parser.parse_args()

    return parse_config

