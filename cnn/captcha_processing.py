# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from config import N_CHAR, CHAR_DICT, TRAIN_DIR, VALID_DIR, TEST_DIR
from config import IMAGES_H, IMAGES_W, CHANNEL


def create_x_images(picture):
    """

    :param picture: 传入验证码图片
    :return:
    """
    return img_to_array(picture) / 255


def create_y_labels(file_name):
    """

    :param file_name: 传入验证码标签名
    :return:
    """
    return file_name.split("_")[0]


def string_to_num(label_str):
    """

    :param label_str: 验证码标签
    :return:
    """
    return [CHAR_DICT[label] for label in label_str]


def create_train_model_data(file_list, mode):
    """

    :param file_list: 传入训练集文件中验证码图片
    :param mode 1.训练模式
                2.评估模型
    :return:
    """
    x_images = []
    y_labels = []
    # 加载图片
    if len(file_list) != 0:
        for p in file_list:
            picture = None
            if p != '.DS_Store':
                if mode == 1:
                    picture = load_img(TRAIN_DIR + "/" + p)
                if mode == 2:
                    picture = load_img(VALID_DIR + "/" + p)
                if picture is None:
                    raise Exception("picture is None!")
                else:
                    x_images.append(create_x_images(picture))
                    sn = string_to_num(create_y_labels(p))
                    y_labels.append(to_categorical(sn, num_classes=len(N_CHAR)))
        return np.array(x_images), np.array(y_labels)
    else:
        raise Exception("File_list is no len!")


def create_predict_model_data(p):
    """

    :param p: 传入待识别验证码
    :return: 多维数组
    """
    arr = create_x_images(load_img(TEST_DIR + "/" + p))
    return arr.reshape((-1, IMAGES_H, IMAGES_W, CHANNEL))
