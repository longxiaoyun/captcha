from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from config import N_CHAR, N_CLASS, CHAR_DICT


def create_x_images(picture):
    """

    :param picture: 传入验证码图片
    :return:
    """
    return img_to_array(picture)


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


def create_train_model_data(file_list):
    """

    :param file_list: 传入训练集文件中验证码图片
    :return:
    """
    x_images = []
    y_labels = []
    # 加载图片
    for p in file_list:
        picture = load_img(p)
        x_images.append(create_x_images(picture))
        sn = string_to_num(create_y_labels(p))
        y_labels.append(to_categorical(sn, num_classes=N_CLASS * len(N_CHAR)))
    return x_images, y_labels


def create_predict_model_data(p):
    """

    :param p: 传入待识别验证码
    :return: 多维数组
    """
    return create_x_images(load_img(p))
