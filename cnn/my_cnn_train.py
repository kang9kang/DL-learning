import numpy as np
from my_function import smooth_curve
from my_cnet import SimpleConvNet
from mnist import load_mnist
from my_optimizer import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


(x_train,t_train),(x_test,t_test) = load_mnist(flatten=False)

network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

"""
epoch 全覆盖次数
mini_batch_size 批处理数据数
train_size 训练数据数
iter_per_epoch 一次全覆盖批处理次数
max_iter 整个训练批处理次数
optimizer 梯度更新选择Adam算法
current_epoch 目前进行的epoch次数
"""
epoch = 20
mini_batch_size = 100
train_size = x_train.shape[0]
iter_per_epoch = max(train_size/mini_batch_size,1)
iter_per_epoch = int(iter_per_epoch)#变为整数
max_iter = epoch*iter_per_epoch
optimizer = Adam(lr = 0.001)
current_epoch = 0
"""
画图参数

"""
train_loss_list = []
train_acc_list = []
test_acc_list = []


print("开始训练请等待...")
for i in range(max_iter):
    batch_mask = np.random.choice(train_size,mini_batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch,t_batch)
    grads = optimizer.update(network.params,grads)

    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    if i %iter_per_epoch==0 :
        current_epoch += 1
        #取1000个数据计算正确率(节省时间)
        x_train_simple,t_train_simple = x_train[:1000],t_train[:1000]
        x_test_sample,t_test_sample = x_test[:1000],t_test[:1000]

        train_acc = network.accuracy(x_train_simple,t_train_simple)
        test_acc = network.accuracy(x_test_sample,t_test_sample)
        if current_epoch == 20 :
            cm = confusion_matrix(t_test_sample,np.argmax(network.predict(x_test_sample), axis=1))
            cmn = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cmn = np.around(cmn,decimals=2)

            plt.figure(figsize=(8, 8))
            sns.heatmap(cmn, annot=True, cmap='Blues')

            plt.ylim(0, 10)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("=== epoch : "+str(current_epoch)+", train acc:"+str(train_acc)+",test acc:"+str(test_acc)+" ===")
# network.save_parms("params.pkl")
print("训练结束，您的损失函数值已经降低到"+str(train_loss_list[-1])+"下面开始作图")
"""
画图
"""
plt.figure("loss")
x = np.arange(len(train_loss_list))
y = np.array(smooth_curve(train_loss_list))
plt.plot(x,y)
plt.xlabel("mini_batch")
plt.ylabel("loss")


plt.figure("accuracy")
x = np.arange(len(train_acc_list))
y1 = np.array(train_acc_list)
y2 = np.array(test_acc_list)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(x,y1,label="train_accuracy")
plt.plot(x,y2,label="test_accuracy")
plt.legend()

plt.show()
