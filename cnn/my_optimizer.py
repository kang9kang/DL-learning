import numpy as np

class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0
        self.eps = 1e-8

    def update(self,params,grads):
        if self.m is None:
            self.m,self.v = {},{}
            for key,val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)