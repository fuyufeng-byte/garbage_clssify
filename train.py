import multiprocessing
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

from efficientnet import model as EfficientNet
from data_gen import data_from_sequence
from utils.lr_scheduler import WarmUpCosineDecayScheduler
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 注意关闭默认的eager模式
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("data_url", type=str, default='./data/train_data', help="data dir", nargs='?')
parser.add_argument("train_url", type=str, default='./garbage_ckpt/', help="save model dir", nargs='?')
parser.add_argument("num_classes", type=int, default=40, help="num_classes", nargs='?')
parser.add_argument("input_size", type=int, default=300, help="input_size", nargs='?')
parser.add_argument("batch_size", type=int, default=16, help="batch_size", nargs='?')
parser.add_argument("learning_rate", type=float, default=0.001, help="learning_rate", nargs='?')
parser.add_argument("max_epochs", type=int, default=30, help="max_epochs", nargs='?')
parser.add_argument("deploy_script_path", type=str, default='', help="deploy_script_path", nargs='?')
parser.add_argument("test_data_url", type=str, default='', help="test_data_url", nargs='?')


def model_fn(param):
    """修改符合垃圾分类的模型
    :param param: 命令行参数
    :return:
    """
    base_model = EfficientNet.EfficientNetB3(include_top=False,
                                             input_shape=(param.input_size, param.input_size, 3),
                                             classes=param.num_classes)

    x = base_model.output
    # 自定义修改40个分类的后面基层
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(param.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def train_model(param):
    """训练模型逻辑
    :param param: 各种参数命令行
    :return:
    """
    # 1、读取sequence数据
    train_sequence, validation_sequence = data_from_sequence(param.data_url, param.batch_size, param.num_classes, param.input_size)

    # 2、建立模型，修改模型指定训练相关参数
    model = model_fn(param)

    optimizer = Adam(lr=param.learning_rate)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    # 模型修改
    # 模型训练优化器指定
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    model.summary()

    # 3、指定相关回调函数
    # Tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=1, write_graph=True, write_images=True)

    # modelcheckpoint
    # （3）模型保存相关参数
    check = tf.keras.callbacks.ModelCheckpoint(param.train_url + 'weights_{epoch:02d}-{val_acc:.2f}.h5',
                                               monitor='val_acc',
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='auto',
                                               period=1)

    # 余弦退回warmup
    # 得到总样本数
    sample_count = len(train_sequence)
    batch_size = param.batch_size

    # 第二阶段学习率以及总步数
    learning_rate_base = param.learning_rate
    total_steps = int(param.max_epochs * sample_count) / batch_size
    # 计算第一阶段的步数需要多少 warmup_steps
    warmup_epoch = 5
    warmup_steps = int(warmup_epoch * sample_count) / batch_size



    warm_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0,)

    # 4、训练步骤
    model.fit_generator(
        train_sequence,
        steps_per_epoch=int(sample_count / batch_size),  # 一个epoch需要多少步 ， 1epoch sample_out 140000多样本， 140000 / 16 = 步数
        epochs=param.max_epochs,
        verbose=1,
        callbacks=[check, tensorboard, warm_lr],
        validation_data=validation_sequence,
        max_queue_size=10,
        workers=int(multiprocessing.cpu_count() * 0.7),
        # use_multiprocessing=True,
        shuffle=True
    )

    return None


if __name__ == '__main__':
    args = parser.parse_args()
    train_model(args)
