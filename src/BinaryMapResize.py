import cv2
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def process_isic_npy(
        dim=(512, 512), save_dir='../Newdata/npyPic/isic2018/'):
    process_name = ["Test", "Train", "Validation"]

    for mkdir in process_name:
        usesave_dir = save_dir
        image_dir_path = '../isic2018/{}/Image'.format(mkdir)
        mask_dir_path = '../isic2018/{}/Label'.format(mkdir)
        usesave_dir = usesave_dir+mkdir+'/'

        image_path_list = os.listdir(image_dir_path)
        mask_path_list = os.listdir(mask_dir_path)

        image_path_list = list(filter(lambda x: x[-3:] == 'png', image_path_list))
        mask_path_list = list(filter(lambda x: x[-3:] == 'png', mask_path_list))

        image_path_list.sort()
        mask_path_list.sort()

        print(len(image_path_list), len(mask_path_list))

        # ISIC Dataset
        for image_path, mask_path in zip(image_path_list, mask_path_list):
            if image_path[-3:] == 'png':
                print(image_path)
                assert os.path.basename(image_path) == os.path.basename(mask_path)
                _id = os.path.basename(image_path)[:-4]
                image_path = os.path.join(image_dir_path, image_path)
                mask_path = os.path.join(mask_dir_path, mask_path)
                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

                dim = (512, 512)
                image_new = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
                image_new = np.array(image_new, dtype=np.uint8)
                mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)
                mask_new = cv2.blur(mask_new,(3,3))
                mask_new = np.array(mask_new, dtype=np.uint8)


                save_dir_path = usesave_dir + '/Image/'
                os.makedirs(save_dir_path, exist_ok=True)
                np.save(os.path.join(save_dir_path, str(_id) + '.npy'), image_new)

                save_dir_path = usesave_dir + '/Label/'
                os.makedirs(save_dir_path, exist_ok=True)
                np.save(os.path.join(save_dir_path, str(_id) + '.npy'), mask_new)


def process_isic2018(
        dim=(512, 512), save_dir='../Newdata/re_isic2018/'):
    process_name = ["Test", "Train", "Validation"]


    for mkdir in process_name:
        usesave_dir = save_dir
        image_dir_path = '../isic2018/{}/Image'.format(mkdir)
        mask_dir_path = '../isic2018/{}/Label'.format(mkdir)
        usesave_dir = usesave_dir + mkdir + '/'

        image_path_list = os.listdir(image_dir_path)
        mask_path_list = os.listdir(mask_dir_path)

        image_path_list = list(filter(lambda x: x[-3:] == 'png', image_path_list))
        mask_path_list = list(filter(lambda x: x[-3:] == 'png', mask_path_list))

        image_path_list.sort()
        mask_path_list.sort()

        print(len(image_path_list), len(mask_path_list))

        # ISIC Dataset
        for image_path, mask_path in zip(image_path_list, mask_path_list):
            if image_path[-3:] == 'png':
                print(image_path)
                assert os.path.basename(image_path) == os.path.basename(mask_path)
                _id = os.path.basename(image_path)[:-4]
                image_path = os.path.join(image_dir_path, image_path)
                mask_path = os.path.join(mask_dir_path, mask_path)
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

                dim = (512, 512)
                image_new = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
                image_new = np.array(image_new, dtype=np.uint8)
                mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)
                mask_new = cv2.blur(mask_new,(3,3))
                mask_new = np.array(mask_new, dtype=np.uint8)


                save_dir_path = usesave_dir + '/Image/'
                os.makedirs(save_dir_path, exist_ok=True)
                # save png
                cv2.imwrite(os.path.join(save_dir_path, str(_id) + '.png'),image_new)

                save_dir_path = usesave_dir + '/Label/'
                os.makedirs(save_dir_path, exist_ok=True)
                # save png
                cv2.imwrite(os.path.join(save_dir_path, str(_id) + '.png'),mask_new)


if __name__ == '__main__':
    # process_isic2018()
    process_isic_npy()