import math
import os
import random
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split

from data_gen.random_eraser import get_random_eraser


class GarbageDataSequence(Sequence):
    """垃圾分类数据流，每次batch返回batch_size大小数据
    model.fit_generator使用
    """
    def __init__(self, img_paths, labels, batch_size, img_size, use_aug):

        # 1、获取训练特征与目标值的合并结果 [batch_size, 1],  [batch_size, 40]   [batch_size, 41]
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1),
                              np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size  # (300, 300)
        self.use_aug = use_aug
        self.alpha = 0.2
        # 随机擦出方法
        self.eraser = get_random_eraser(s_h=0.3, pixel_level=True)

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    @staticmethod
    def center_img(img, size=None, fill_value=255):
        """改变图片尺寸到300x300，并且做填充使得图像处于中间位置
        """
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def preprocess_img(self, img_path):
        """处理每张图片，大小， 数据增强
        :param img_path:
        :return:
        """
        # 1、读取图片对应内容，做形状，内容处理, (h, w)
        img = Image.open(img_path)
        # [180, 200, 3]
        scale = self.img_size[0] / max(img.size[:2])
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        img = img.convert('RGB')
        img = np.array(img)

        # 2、数据增强：如果是训练集进行数据增强操作
        if self.use_aug:

            # 1、随机擦处
            img = self.eraser(img)

            # 2、翻转
            datagen = ImageDataGenerator(
                width_shift_range=0.05,
                height_shift_range=0.05,
                horizontal_flip=True,
                vertical_flip=True,
            )
            img = datagen.random_transform(img)

        # 4、处理一下形状 【300， 300， 3】
        # 改变到[300, 300] 建议不要进行裁剪操作，变形操作，保留数据增强之后的效果，填充到300x300
        img = self.center_img(img, self.img_size[0])
        return img

    def __getitem__(self, idx):

        # 1、获取当前批次idx对应的特征值和目标值
        batch_x = self.x_y[idx * self.batch_size: self.batch_size * (idx + 1), 0]
        batch_y = self.x_y[idx * self.batch_size: self.batch_size * (idx + 1), 1:]

        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)

        # 2、mixup
        batch_x, batch_y = self.mixup(batch_x, batch_y)

        # 3、归一化处理
        batch_x = self.preprocess_input(batch_x)

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.x_y)

    def mixup(self, batch_x, batch_y):
        """
        数据混合mixup
        :param batch_x: 要mixup的batch_X
        :param batch_y: 要mixup的batch_y
        :return: mixup后的数据
        """
        size = self.batch_size
        l = np.random.beta(self.alpha, self.alpha, size)

        X_l = l.reshape(size, 1, 1, 1)
        y_l = l.reshape(size, 1)

        X1 = batch_x
        Y1 = batch_y
        X2 = batch_x[::-1]
        Y2 = batch_y[::-1]

        X = X1 * X_l + X2 * (1 - X_l)
        Y = Y1 * y_l + Y2 * (1 - y_l)

        return X, Y

    def preprocess_input(self, x):
        """归一化处理样本特征值
        :param x:
        :return:
        """
        assert x.ndim in (3, 4)
        assert x.shape[-1] == 3

        MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        x = x - np.array(MEAN_RGB)
        x = x / np.array(STDDEV_RGB)

        return x


def smooth_labels(y, smooth_factor=0.1):

    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label_2 smoothing factor: ' + str(smooth_factor))
    return y


def data_from_sequence(train_data_dir, batch_size, num_classes, input_size):
    """读取本地数据到sequence
    :param train_data_dir: 训练数据目录
    :param batch_size: 批次大小
    :param num_classes: 总类别书40
    :param input_size: 输入图片大小(300, 300)
    :return:
    """
    # 1、读取txt文件，打乱文件顺序, .jpg, .txt
    label_files = [os.path.join(train_data_dir, filename) for filename
                   in os.listdir(train_data_dir) if filename.endswith('.txt')]

    random.shuffle(label_files)
    # 2、解析txt文件当中 特征值以及目标值（标签）
    img_paths = []
    labels = []

    for index, file_path in enumerate(label_files):
        with open(file_path, 'r') as f:
            line = f.readline()

        line_split = line.strip().split(', ')
        # line '*.jpg, 0'
        if len(line_split) != 2:
            print("% 文件格式出错", (file_path))
            continue

        img_name = line_split[0]
        label = int(line_split[1])

        # 最后保存到所有的列表当中
        img_paths.append(os.path.join(train_data_dir, img_name))
        labels.append(label)

    # print(img_paths, labels)
    # 3、目标标签类别ont_hot编码转换， 平滑处理
    labels = to_categorical(labels, num_classes)
    labels = smooth_labels(labels)

    # 分割训练集合验证集合
    train_img_paths, validation_img_paths, train_labels, validation_labels \
        = train_test_split(img_paths, labels, test_size=0.15, random_state=0)
    # print(validation_img_paths)
    # print(train_labels, validation_labels)
    print("总共样本数： %d , 训练样本数： %d, 验证样本数： %d" % (len(img_paths), len(train_img_paths), len(validation_img_paths)))

    # 4、Sequence调用测试
    train_sequence = GarbageDataSequence(train_img_paths, train_labels, batch_size, [input_size, input_size], use_aug=True)
    validation_sequence = GarbageDataSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size], use_aug=False)

    return train_sequence, validation_sequence


if __name__ == '__main__':
    train_data_dir = "../data/garbage_classify/train_data"
    batch_size = 32

    train_sequence, validation_sequence = data_from_sequence(train_data_dir, batch_size, num_classes=40, input_size=300)

    for i in range(100):
        print("第 %d 批次数据" % i)
        batch_x, batch_y = train_sequence.__getitem__(i)
        print(batch_x, batch_y)



