# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../cutedl')

'''
使用卷积神经网络实现的手写数字识别模型
'''

from datasets.mnist import Mnist

'''
加载手写数字数据集
'''
mnist = Mnist('../datasets/mnist')
ds_train, ds_test = mnist.load(64)

import pdb
import numpy as np

from cutedl.model import Model
from cutedl.session import Session
from cutedl import session, losses, optimizers, utils
from cutedl import nn_layers as nn
from cutedl import cnn_layers as cnn

import fit_tools

report_path = "./pics/mnist-recoginze-"
model_path = "./model/mnist-recoginze-"

'''
训练模型
'''
def fit01(name, optimizer):
    inshape = ds_train.data.shape[1:]
    #pdb.set_trace()
    model = Model([
                cnn.Conv2D(32, (5,5), inshape=inshape),
                cnn.MaxPool2D((2,2), strides=(2,2)),
                cnn.Conv2D(64, (5,5)),
                cnn.MaxPool2D((2,2), strides=(2,2)),
                nn.Flatten(),
                nn.Dense(1024),
                nn.Dropout(0.5),
                nn.Dense(10)
            ])
    model.assemble()

    sess = Session(model,
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizer
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'val_loss', 20)

    def save_and_report(history):
        #pdb.set_trace()
        fit_tools.fit_report(history, report_path+name+".png")
        model.save(model_path+name)

    #pdb.set_trace()
    history = sess.fit(ds_train, 200, val_data=ds_test, val_steps=100,
                        listeners=[
                            stop_fit,
                            session.FitListener('val_end', callback=lambda h: fit_tools.accuracy(sess, h)),
                            session.FitListener('epoch_end', callback=save_and_report)
                        ]
                    )

    save_and_report(history)


if '__main__' == __name__:
    fit01('1', optimizers.Adam())
