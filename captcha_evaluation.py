# -*- coding: utf-8 -*-
import os
from config import VALID_DIR
from cnn.captcha_model import CaptchaCnn
from cnn.captcha_processing import create_train_model_data


def evaluation_run():
    # 读取测试验证码
    file_name_list = os.listdir(VALID_DIR)
    # 建立测试数据集
    print("Create train model data...")
    x_images, y_labels = create_train_model_data(file_name_list, mode=2)
    captcha_cnn = CaptchaCnn()
    # 加载训练好的模型
    captcha_cnn.model_load()
    # 模型评估
    captcha_cnn.model_evaluation(x_images, y_labels)

if __name__ == '__main__':
    evaluation_run()
