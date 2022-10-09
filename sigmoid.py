import numpy as np
import matplotlib.pylab as plt

def sigmoid(x): #即使输入数组也可以，广播功能
    return 1/(1+np.exp(-x))

x=np.arange(-5.0,5.0,0.1)
y=1.5 * sigmoid(x)

plt.plot(x,y)
plt.ylim(-0.1,1.6) #指定y轴范围
plt.show()

