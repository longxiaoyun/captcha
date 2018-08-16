import os
import numpy as np
from config import TEST_DIR, CHAR_DICT
from cnn.captcha_model import CaptchaCnn
from cnn.captcha_processing import create_predict_model_data


def predict_run(p):
    """

    :param p: 验证码图片
    :return:
    """
    x_data = create_predict_model_data(p)
    captcha_cnn = CaptchaCnn()
    # 加载训练好的模型
    captcha_cnn.model_load()
    # 模型预测
    model_predict = captcha_cnn.model_predict(x_data)[0]
    if 1 in set(np.isnan(model_predict).ravel()):
        raise Exception("model predict values have NAN!")
    else:
        res = []
        ncd = {v: k for k, v in CHAR_DICT.items()}
        for i in range(4):
            res.append(ncd[np.where(model_predict[i]==np.max(model_predict[i]))[0].tolist()[0]])
    return str(res)

if __name__ == '__main__':
    file_name_list = os.listdir(TEST_DIR)
    for p in file_name_list:
        if p != '.DS_Store':
            print(predict_run(p))
