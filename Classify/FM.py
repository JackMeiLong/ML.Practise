# coding:UTF-8
'''
Date:20171010
@author: meilong
'''
import numpy as np
from random import normalvariate  # 正态分布

class FM:
    
    def __init__(self,solver='sgd',max_iter=20000,k=2,alpha=0.1):
        self.solver=solver
        self.max_iter=max_iter
        self.k=k
        self.alpha=alpha
        self.w0=0
        self.w=[]
        self.v=[]
        
    def initialize_v(n, k):
        '''初始化交叉项
        input:  n(int)特征的个数
                k(int)FM模型的交叉向量维度
        output: v(mat):交叉项的系数权重
        '''
        v = np.mat(np.zeros((n, k)))
    
        for i in xrange(n):
            for j in xrange(k):
                # 利用正态分布生成每一个权重
                v[i, j] = normalvariate(0, 0.2)
    return v
    
    def SGD(X, y, k, max_iter, alpha):
    '''利用随机梯度下降法训练FM模型
    input:  dataMatrix(mat)特征
            classLabels(mat)标签
            k(int)v的维数
            max_iter(int)最大迭代次数
            alpha(float)学习率
    output: w0(float),w(mat),v(mat):权重
    '''
        m, n = np.shape(X)
        # 1、初始化参数
        w = np.zeros((n, 1))  # 其中n是特征的个数
        w0 = 0  # 偏置项
        v = initialize_v(n, self.k)  # 初始化V
        alpha=self.alpha
        max_iter=self.max_iter
        
        # 2、训练
        for it in xrange(max_iter):
            for x in xrange(m):  # 随机优化，对每一个样本而言的
                inter_1 = X[x] * v
                inter_2 = np.multiply(X[x], X[x]) * np.multiply(v, v)  # multiply对应元素相乘
            # 完成交叉项
                interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
                p = w0 + X[x] * w + interaction  # 计算预测的输出
                loss = sigmoid(y[x] * p[0, 0]) - 1
        
                w0 = w0 - alpha * loss * y[x]
                for i in xrange(n):
                    if X[x, i] != 0:
                        w[i, 0] = w[i, 0] - alpha * loss * y[x] * X[x, i]
                        
                        for j in xrange(k):
                            v[i, j] = v[i, j] - alpha * loss * y[x] *(X[x, i] * inter_1[0, j] -v[i, j] * X[x, i] * X[x, i])
            
    # 3、返回最终的FM模型的参数
    return w0, w, v  
    
    def fit(X,y):
        if self.solver.upper()=='SGD':
            w0, w, v=self.SGD(X,y)
            self.w0=w0
            self.w=w
            self.v=v
            params={}
            params['w0']=w0
            params['w']=w
            params['v']=v
            
            return params
        else:
            return None
    
    def predict(X):
    '''得到预测值
    input:  dataMatrix(mat)特征
            w(int)常数项权重
            w0(int)一次项权重
            v(float)交叉项权重
    output: result(list)预测的结果
    '''
        w0=self.w0
        w=self.w
        v=self.v
        
        m = np.shape(X)[0]
        result = []
        for x in xrange(m):
            inter_1 = X[x] * v
            inter_2 = np.multiply(X[x], X[x]) * np.multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            p = w0 + X[x] * w + interaction  # 计算预测的输出        
            pre = sigmoid(p[0, 0])        
            result.append(pre)        
    return result

    def getAccuracy(predict, classLabels):
        m = len(predict)
        allItem = 0
        error = 0
        for i in xrange(m):
            allItem += 1
            if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
                error += 1
                elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
                    error += 1
            else:
                continue
    return float(error) / allItem

if __name__=='__main__':
    fm=FM()
    params=fm.fit(X,y)
    result=fm.predict(X)
    accuracy=fm.getAccuracy(predict,y)
