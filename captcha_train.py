# -*- coding: utf-8 -*-
import os
from config import TRAIN_DIR
from config import IMAGES_H, IMAGES_W, CHANNEL, N_CLASS, N_CHAR
from config import LOSS, OPTIMIZER, METRICS, KEEP_PROB
from config import FILTERS_1, FILTERS_2, FILTERS_3, FILTERS_4
from cnn.captcha_processing import create_train_model_data
from cnn.captcha_model import CaptchaCnn


def train_run():
    # 读取训练验证码
    file_name_list = os.listdir(TRAIN_DIR)
    # 建立训练数据集
    print("Create train model data...")
    x_images, y_labels = create_train_model_data(file_name_list, mode=1)
    print("x_images: ", x_images.shape)
    print("y_labels: ", y_labels.shape)
    if x_images.shape[0] == 1:
        raise Exception("The train sample is too small!")
    else:
        # 建立模型对象
        captcha_cnn = CaptchaCnn()
        # 设置模型参数
        captcha_cnn.set_param(
            images_h=IMAGES_H,
            images_w=IMAGES_W,
            channel=CHANNEL,
            loss=LOSS,
            optimizer=OPTIMIZER,
            metrics=METRICS,
            keep_prob=KEEP_PROB,
            n_class=N_CLASS,
            n_char=N_CHAR
        )
        # 定义模型
        captcha_cnn.building_model(
            filters_1=FILTERS_1,
            filters_2=FILTERS_2,
            filters_3=FILTERS_3,
            filters_4=FILTERS_4
        )
        # 训练模型
        captcha_cnn.fit_model(
            x_images,
            y_labels,
            epochs=1,
            batch_size=128
        )
        # 模型保存
        captcha_cnn.model_save()

if __name__ == '__main__':
    train_run()
